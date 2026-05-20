from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EdgeFusionNPUConfig:
    """Static deployment contract for the EdgeFusionNPU separator core."""

    n_chan: int = 1
    n_src: int = 3
    n_freq: int = 1025
    hidden_channels: int = 24
    num_blocks: int = 6
    time_kernel: int = 3
    freq_kernel: int = 3
    low_freq_boost: float = 0.35
    mask_scale: float = 1.25
    memory_mode: str = "conv"
    use_band_bottleneck: bool = False
    band_stride: int = 2
    band_channels: int = 8
    capacity_channels: int = 0
    token_capacity_channels: int = 0
    token_capacity_layers: int = 0
    token_capacity_stride: int = 2
    token_capacity_stages: int = 8

    @property
    def in_channels(self) -> int:
        return 2 * self.n_chan

    @property
    def out_channels(self) -> int:
        return self.n_src * self.n_chan

    @property
    def context_size(self) -> int:
        return self.time_kernel - 1

    @property
    def state_channels(self) -> int:
        return self.hidden_channels * self.num_blocks


class LowFrequencyBias(nn.Module):
    """Static low-frequency emphasis inspired by mel/bandsplit priors."""

    def __init__(self, n_freq: int, strength: float):
        super().__init__()
        freq = torch.linspace(0.0, 1.0, steps=n_freq, dtype=torch.float32)
        gain = 1.0 + strength * (1.0 - freq).pow(2.0)
        self.register_buffer("gain", gain.view(1, 1, n_freq, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gain


class EdgeFusionBlock(nn.Module):
    """
    One causal block combining narrow-band time memory and cross-band mixing.

    The block intentionally uses only NPU-stable primitives: Conv2d, Slice,
    Concat, Add, Mul, ReLU, and Sigmoid. It avoids scalar Gather, rank-3 MatMul,
    PReLU, dynamic shape construction, and STFT/iSTFT inside the graph.
    """

    def __init__(
        self,
        channels: int,
        *,
        time_kernel: int,
        freq_kernel: int,
        memory_mode: str = "conv",
        capacity_channels: int = 0,
    ):
        super().__init__()
        if time_kernel < 2:
            raise ValueError("time_kernel must be >= 2 for a causal cache.")
        if freq_kernel % 2 != 1:
            raise ValueError("freq_kernel must be odd.")
        if (time_kernel - 1) >= 14:
            raise ValueError("NPU rule expects (kernel_size - 1) * dilation < 14.")
        if memory_mode not in {"conv", "ssm_lite"}:
            raise ValueError("memory_mode must be 'conv' or 'ssm_lite'.")
        if capacity_channels < 0:
            raise ValueError("capacity_channels must be non-negative.")

        self.channels = channels
        self.context_size = time_kernel - 1
        self.memory_mode = memory_mode

        self.time_dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, time_kernel),
            groups=channels,
            bias=True,
        )
        self.freq_dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=(freq_kernel, 1),
            padding=(freq_kernel // 2, 0),
            groups=channels,
            bias=True,
        )
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.capacity = (
            nn.Sequential(
                nn.Conv2d(channels, capacity_channels, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv2d(capacity_channels, channels, kernel_size=1, bias=True),
            )
            if capacity_channels > 0
            else nn.Identity()
        )
        self.act = nn.ReLU()
        if memory_mode == "ssm_lite":
            self.state_decay_logit = nn.Parameter(torch.full((1, channels, 1, 1), 1.5))

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        full = torch.cat([state, x], dim=3)
        y = self.time_dw(full)
        y = self.freq_dw(y)
        y = self.act(self.mix(y))
        y = y * torch.sigmoid(self.gate(y))
        y = self.proj(y)
        out = self.act(x + y)
        out = self.act(out + self.capacity(out))
        if self.memory_mode == "ssm_lite":
            prev = state[:, :, :, self.context_size - 1 : self.context_size]
            decay = torch.sigmoid(self.state_decay_logit)
            new_cell = prev * decay + x * (1.0 - decay)
            next_state = torch.cat([state[:, :, :, 1:self.context_size], new_cell], dim=3)
        else:
            next_state = full[:, :, :, 1 : self.context_size + 1]
        return out, next_state


class BandTokenBottleneck(nn.Module):
    """Conv-only frequency compression branch for F=4k+1 STFT layouts."""

    def __init__(self, channels: int, *, n_freq: int, band_channels: int, stride: int):
        super().__init__()
        if stride < 2:
            raise ValueError("band_stride must be >= 2.")
        if n_freq <= 0:
            raise ValueError("n_freq must be positive.")
        if band_channels <= 0:
            raise ValueError("band_channels must be positive.")
        kernel = stride
        padding = stride // 2 - 1 if stride % 2 == 0 else stride // 2
        out_freq = (n_freq + 2 * padding - kernel) // stride + 1
        output_padding = n_freq - ((out_freq - 1) * stride - 2 * padding + kernel)
        if output_padding < 0 or output_padding >= stride:
            raise ValueError(
                f"n_freq={n_freq} is incompatible with stride={stride}; "
                "choose an STFT size whose bin count can round-trip through ConvTranspose2d."
            )
        self.down = nn.Conv2d(
            channels,
            band_channels,
            kernel_size=(kernel, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            bias=True,
        )
        self.mix = nn.Conv2d(band_channels, band_channels, kernel_size=1, bias=True)
        self.up = nn.ConvTranspose2d(
            band_channels,
            channels,
            kernel_size=(kernel, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            output_padding=(output_padding, 0),
            bias=True,
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act(self.down(x))
        z = self.act(self.mix(z))
        return self.act(x + self.up(z))


class TokenCapacityBottleneck(nn.Module):
    """Large parameter bank applied on aggressively compressed frequency tokens."""

    def __init__(
        self,
        channels: int,
        *,
        n_freq: int,
        token_channels: int,
        num_layers: int,
        stride: int,
        num_stages: int,
    ):
        super().__init__()
        if token_channels <= 0:
            raise ValueError("token_channels must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if stride != 2:
            raise ValueError("token_capacity_stride currently supports stride=2 only.")
        if num_stages <= 0:
            raise ValueError("token_capacity_stages must be positive.")

        down_layers: list[nn.Module] = []
        up_specs: list[tuple[int, int]] = []
        current_freq = n_freq
        for _ in range(num_stages):
            next_freq = (current_freq - stride) // stride + 1
            if next_freq < 1:
                raise ValueError(
                    f"token bottleneck over-compresses n_freq={n_freq}; reduce token_capacity_stages."
                )
            down_layers.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(stride, 1),
                    stride=(stride, 1),
                    groups=channels,
                    bias=True,
                )
            )
            up_specs.append((next_freq, current_freq))
            current_freq = next_freq

        self.down = nn.Sequential(*down_layers)
        capacity_layers: list[nn.Module] = [
            nn.Conv2d(channels, token_channels, kernel_size=1, bias=True),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            capacity_layers.extend(
                [
                    nn.Conv2d(token_channels, token_channels, kernel_size=1, bias=True),
                    nn.ReLU(),
                ]
            )
        capacity_layers.append(nn.Conv2d(token_channels, channels, kernel_size=1, bias=True))
        self.capacity = nn.Sequential(*capacity_layers)

        up_layers: list[nn.Module] = []
        for in_freq, out_freq in reversed(up_specs):
            output_padding = out_freq - ((in_freq - 1) * stride + stride)
            if output_padding < 0 or output_padding >= stride:
                raise ValueError(
                    f"Cannot reconstruct frequency size {out_freq} from {in_freq} with stride={stride}."
                )
            up_layers.append(
                nn.ConvTranspose2d(
                    channels,
                    channels,
                    kernel_size=(stride, 1),
                    stride=(stride, 1),
                    output_padding=(output_padding, 0),
                    bias=True,
                )
            )
        self.up = nn.Sequential(*up_layers)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.down(x)
        z = self.capacity(z)
        return self.act(x + self.up(z))


class EdgeFusionNPU(nn.Module):
    """
    Mask-estimation core for online edge audio separation.

    Input:
      x: [B, 2*n_chan, F, T] packed real/imag STFT frame, chunk, or clip.

    State:
      one packed 4D cache tensor [B, num_blocks*hidden_channels, F, time_kernel-1].

    Output:
      mask: [B, n_src*n_chan, F, T] real gain masks. The host runtime applies
      the masks to complex STFT bins and handles overlap-add iSTFT.
    """

    def __init__(self, config: EdgeFusionNPUConfig):
        super().__init__()
        self.config = config
        self.n_chan = config.n_chan
        self.n_src = config.n_src
        self.n_freq = config.n_freq
        self.hidden_channels = config.hidden_channels
        self.num_blocks = config.num_blocks
        self.context_size = config.context_size
        self.state_channels = config.state_channels
        self.register_buffer("mask_scale", torch.tensor(float(config.mask_scale), dtype=torch.float32))

        self.low_freq = LowFrequencyBias(config.n_freq, config.low_freq_boost)
        self.input_proj = nn.Conv2d(config.in_channels, config.hidden_channels, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList(
            [
                EdgeFusionBlock(
                    config.hidden_channels,
                    time_kernel=config.time_kernel,
                    freq_kernel=config.freq_kernel,
                    memory_mode=config.memory_mode,
                    capacity_channels=config.capacity_channels,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.band_bottleneck = (
            BandTokenBottleneck(
                config.hidden_channels,
                n_freq=config.n_freq,
                band_channels=config.band_channels,
                stride=config.band_stride,
            )
            if config.use_band_bottleneck
            else nn.Identity()
        )
        self.token_capacity = (
            TokenCapacityBottleneck(
                config.hidden_channels,
                n_freq=config.n_freq,
                token_channels=config.token_capacity_channels,
                num_layers=config.token_capacity_layers,
                stride=config.token_capacity_stride,
                num_stages=config.token_capacity_stages,
            )
            if config.token_capacity_channels > 0 or config.token_capacity_layers > 0
            else nn.Identity()
        )
        self.head = nn.Sequential(
            nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(config.hidden_channels, config.out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def init_states(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.state_channels,
            self.n_freq,
            self.context_size,
            device=device,
            dtype=dtype,
        )

    def _forward_frame(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(self.low_freq(x))
        next_states: list[torch.Tensor] = []
        for block_idx, block in enumerate(self.blocks):
            c0 = block_idx * self.hidden_channels
            c1 = c0 + self.hidden_channels
            h, next_state = block(h, state[:, c0:c1, :, :])
            next_states.append(next_state)
        h = self.band_bottleneck(h)
        h = self.token_capacity(h)
        mask = self.head(h) * self.mask_scale
        return mask, torch.cat(next_states, dim=1)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.onnx.is_in_onnx_export():
            if x.shape[1] != self.config.in_channels:
                raise ValueError(f"expected {self.config.in_channels} input channels, got {x.shape[1]}")
            if x.shape[2] != self.n_freq:
                raise ValueError(f"expected {self.n_freq} frequency bins, got {x.shape[2]}")
            if x.shape[3] < 1:
                raise ValueError("expected at least one frame.")
            if state.shape[1] != self.state_channels:
                raise ValueError(f"expected {self.state_channels} state channels, got {state.shape[1]}")
            if state.shape[2] != self.n_freq:
                raise ValueError(f"expected state frequency bins {self.n_freq}, got {state.shape[2]}")
            if state.shape[3] != self.context_size:
                raise ValueError(f"expected state context {self.context_size}, got {state.shape[3]}")

        if x.shape[3] == 1:
            return self._forward_frame(x, state)

        masks: list[torch.Tensor] = []
        next_state = state
        for frame_idx in range(x.shape[3]):
            mask, next_state = self._forward_frame(x[:, :, :, frame_idx : frame_idx + 1], next_state)
            masks.append(mask)
        return torch.cat(masks, dim=3), next_state


class EdgeFusionNPUExportWrapper(nn.Module):
    """Single-frame wrapper for torch.onnx.export."""

    def __init__(self, model: EdgeFusionNPU):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model._forward_frame(x, state)


def build_edge_fusion_npu_preset(name: str, **overrides: object) -> EdgeFusionNPU:
    presets = {
        "tiny": EdgeFusionNPUConfig(hidden_channels=12, num_blocks=3, n_freq=1025, n_src=3, n_chan=1),
        "compact": EdgeFusionNPUConfig(hidden_channels=16, num_blocks=5, n_freq=513, n_src=3, n_chan=1),
        "balanced": EdgeFusionNPUConfig(hidden_channels=24, num_blocks=6, n_freq=1025, n_src=3, n_chan=1),
        "wide": EdgeFusionNPUConfig(hidden_channels=32, num_blocks=8, n_freq=1025, n_src=3, n_chan=1),
        "compact-v2-ssmlite": EdgeFusionNPUConfig(
            hidden_channels=16,
            num_blocks=5,
            n_freq=513,
            n_src=3,
            n_chan=1,
            memory_mode="ssm_lite",
        ),
        "compact-v2-bandtoken": EdgeFusionNPUConfig(
            hidden_channels=16,
            num_blocks=5,
            n_freq=513,
            n_src=3,
            n_chan=1,
            use_band_bottleneck=True,
            band_channels=8,
        ),
        "compact-v2-hybrid": EdgeFusionNPUConfig(
            hidden_channels=16,
            num_blocks=5,
            n_freq=513,
            n_src=3,
            n_chan=1,
            memory_mode="ssm_lite",
            use_band_bottleneck=True,
            band_channels=8,
        ),
        "balanced-v2-hybrid": EdgeFusionNPUConfig(
            hidden_channels=24,
            num_blocks=6,
            n_freq=257,
            n_src=3,
            n_chan=1,
            memory_mode="ssm_lite",
            use_band_bottleneck=True,
            band_channels=12,
        ),
        "big-v2-hybrid-2m": EdgeFusionNPUConfig(
            hidden_channels=24,
            num_blocks=5,
            n_freq=257,
            n_src=3,
            n_chan=1,
            memory_mode="ssm_lite",
            use_band_bottleneck=True,
            band_channels=12,
            token_capacity_channels=512,
            token_capacity_layers=8,
        ),
        "large-v2-hybrid-5m": EdgeFusionNPUConfig(
            hidden_channels=24,
            num_blocks=7,
            n_freq=257,
            n_src=3,
            n_chan=1,
            memory_mode="ssm_lite",
            use_band_bottleneck=True,
            band_channels=12,
            token_capacity_channels=512,
            token_capacity_layers=20,
        ),
    }
    if name not in presets:
        raise ValueError(f"Unknown EdgeFusionNPU preset {name!r}; choose one of {sorted(presets)}.")
    base = presets[name]
    cfg = EdgeFusionNPUConfig(**{**base.__dict__, **overrides})
    return EdgeFusionNPU(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
