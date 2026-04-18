"""
Online / realtime SFC variant for edge NPUs.

Design constraints (for ONNX + NPU deployment):
- Only 2D ops inside the model (Conv2d, elementwise, reductions, activations).
- All runtime tensors are <= 4D.
- ONNX export assumes a fixed number of frequency bins.

This refactor stays closer to the paper's structure than a plain 2D separator:
- compress spectral features from F bins to K band tokens,
- run separation blocks on the compressed axis,
- decode back from K band tokens to F bins,
- predict packed complex masks or mappings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _runtime_assert(condition: bool, message: str) -> None:
    if not torch.onnx.is_in_onnx_export() and not condition:
        raise AssertionError(message)


def _validate_npu_kernel_dilation_limit(
    kernel_size: int,
    dilation: int,
    *,
    axis: str,
    limit: int = 14,
) -> None:
    span = (kernel_size - 1) * dilation
    if span >= limit:
        raise ValueError(
            f"NPU constraint violated on {axis} axis: "
            f"(kernel_size - 1) * dilation = ({kernel_size} - 1) * {dilation} = {span} >= {limit}"
        )


class RMSNorm2d(nn.Module):
    """
    Frame-local RMSNorm for 4D tensors (B, C, T, F).

    The normalization is performed over the frequency axis only so that each
    output frame depends only on the current frame and never on future frames.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = (x * x).mean(dim=3, keepdim=True)
        x = x * torch.rsqrt(ms + self.eps)
        return x * self.weight.view(1, -1, 1, 1)


class CausalConv2d(nn.Module):
    """
    Causal conv along time (T), regular centered conv along frequency (F).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: tuple[int, int] = (3, 3),
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        kt, kf = kernel_size
        dt, df = dilation
        _validate_npu_kernel_dilation_limit(kt, dt, axis="time")
        _validate_npu_kernel_dilation_limit(kf, df, axis="frequency")
        self.pad_t = (kt - 1) * dt
        self.pad_f = ((kf - 1) * df) // 2
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kt, kf),
            dilation=(dt, df),
            padding=(0, self.pad_f),
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_t > 0:
            x = F.pad(x, (0, 0, self.pad_t, 0))
        return self.conv(x)

    def stream_context_frames(self) -> int:
        return self.pad_t

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if self.pad_t == 0:
            return torch.zeros(batch_size, self.conv.in_channels, 0, freq_bins, device=device, dtype=dtype)
        return torch.zeros(
            batch_size,
            self.conv.in_channels,
            self.pad_t,
            freq_bins,
            device=device,
            dtype=dtype,
        )

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pad_t == 0:
            empty = x.new_zeros(x.shape[0], x.shape[1], 0, x.shape[-1])
            return self.conv(x), empty

        if state is None:
            state = self.init_stream_state(
                x.shape[0],
                freq_bins=x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )

        _runtime_assert(state.ndim == 4, f"Expected 4D conv state, got {state.shape}")
        _runtime_assert(
            state.shape == (x.shape[0], x.shape[1], self.pad_t, x.shape[-1]),
            f"Invalid conv state shape {state.shape} for input {x.shape}",
        )

        x_cat = torch.cat([state, x], dim=2)
        new_state = x_cat[:, :, -self.pad_t :, :]
        return self.conv(x_cat), new_state


class SameConv2d(nn.Module):
    """
    Shape-preserving conv along both time and frequency axes.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: tuple[int, int] = (3, 3),
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        kt, kf = kernel_size
        dt, df = dilation
        _validate_npu_kernel_dilation_limit(kt, dt, axis="time")
        _validate_npu_kernel_dilation_limit(kf, df, axis="frequency")
        self.pad_t = (kt - 1) * dt
        self.pad_f = (kf - 1) * df
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(kt, kf),
            dilation=(dt, df),
            padding=0,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_t > 0 or self.pad_f > 0:
            pad_t_left = self.pad_t // 2
            pad_t_right = self.pad_t - pad_t_left
            pad_f_left = self.pad_f // 2
            pad_f_right = self.pad_f - pad_f_left
            x = F.pad(x, (pad_f_left, pad_f_right, pad_t_left, pad_t_right))
        return self.conv(x)


class OnlineConvBlock(nn.Module):
    def __init__(self, ch: int, expansion: int = 2, kernel_size: tuple[int, int] = (3, 3), causal: bool = True):
        super().__init__()
        hidden = ch * expansion
        Conv = CausalConv2d if causal else SameConv2d

        self.norm1 = RMSNorm2d(ch)
        self.pw1 = nn.Conv2d(ch, hidden * 2, kernel_size=1, bias=True)
        self.dw = Conv(hidden, hidden, kernel_size=kernel_size, groups=hidden, bias=True)
        self.pw2 = nn.Conv2d(hidden, ch, kernel_size=1, bias=True)
        self.norm2 = RMSNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        a, b = self.pw1(y).chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y = self.dw(y)
        y = F.silu(y)
        y = self.pw2(y)
        return self.norm2(x + y)

    def stream_context_frames(self) -> int:
        if isinstance(self.dw, CausalConv2d):
            return self.dw.stream_context_frames()
        return 0

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not isinstance(self.dw, CausalConv2d):
            raise RuntimeError("Streaming state is only supported when causal=True.")
        return self.dw.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self.dw, CausalConv2d):
            raise RuntimeError("forward_stream is only supported when causal=True.")

        y = self.norm1(x)
        a, b = self.pw1(y).chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y, new_state = self.dw.forward_stream(y, state)
        y = F.silu(y)
        y = self.pw2(y)
        return self.norm2(x + y), new_state


class BandSpec2d(nn.Module):
    """
    Frequency-band specification used by the compressor and decoder.

    Band supports and priors used by the compressor and decoder.
    """

    def __init__(
        self,
        n_freq: int,
        n_bands: int,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
    ):
        super().__init__()
        assert n_freq > 0
        assert n_bands > 0
        self.n_freq = n_freq
        self.n_bands = n_bands
        self.band_config = band_config

        starts, ends, basis = self._build_band_definition(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        freq_positions = torch.arange(n_freq, dtype=torch.float32)
        widths = (ends - starts).to(torch.float32)

        weight_sum = basis.sum(dim=1).clamp_min(1e-6)
        centers = (basis * freq_positions.view(1, -1)).sum(dim=1) / weight_sum

        self.register_buffer("starts", starts)
        self.register_buffer("ends", ends)
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)
        self.register_buffer("freq_positions", freq_positions)
        self.register_buffer("basis", basis.view(1, n_bands, 1, n_freq))

    @staticmethod
    def _build_band_definition(
        n_freq: int,
        n_bands: int,
        n_fft: int | None,
        sample_rate: int | None,
        band_config: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if n_fft is not None and sample_rate is not None:
            from spectral_feature_compression.core.model.bandit_split import get_band_specs

            band_specs, freq_weights, _ = get_band_specs(band_config, n_fft, sample_rate, n_bands=n_bands)
            starts = torch.tensor([start for start, _ in band_specs], dtype=torch.long)
            ends = torch.tensor([end for _, end in band_specs], dtype=torch.long)
            basis = torch.zeros(n_bands, n_freq, dtype=torch.float32)
            for band_idx, ((start, end), weights) in enumerate(zip(band_specs, freq_weights)):
                end = min(end, n_freq)
                width = end - start
                if width <= 0:
                    continue
                w = weights.to(torch.float32)
                if w.numel() != width:
                    w = F.interpolate(
                        w.view(1, 1, -1),
                        size=width,
                        mode="linear",
                        align_corners=False,
                    ).view(-1)
                basis[band_idx, start:end] = w
            return starts, ends, basis

        edges = torch.linspace(0, n_freq, steps=n_bands + 1)
        starts = torch.floor(edges[:-1]).to(torch.long)
        ends = torch.ceil(edges[1:]).to(torch.long)
        ends = torch.maximum(ends, starts + 1)
        ends = torch.clamp(ends, max=n_freq)

        basis = torch.zeros(n_bands, n_freq, dtype=torch.float32)
        freq_positions = torch.arange(n_freq, dtype=torch.float32)
        for k in range(n_bands):
            start = int(starts[k].item())
            end = int(ends[k].item())
            center = 0.5 * (start + end - 1)
            width = max(0.5 * (end - start), 1.0)
            values = 1.0 - torch.abs(freq_positions[start:end] - center) / width
            basis[k, start:end] = torch.clamp(values, min=0.0)
        return starts, ends, basis

    def band_bias(self) -> torch.Tensor:
        """
        Logit prior shaped as (1, K, 1, F).
        """
        peak = self.basis.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        bias = 2.0 * (self.basis / peak) - 1.0
        return bias

    def decode_basis(self) -> torch.Tensor:
        """
        Frequency reconstruction basis shaped as (1, K, 1, F).
        """
        basis = self.basis / self.basis.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return basis


class SpectralCompressor2d(nn.Module):
    """
    Compress full-resolution spectral features from F bins to K band tokens.

    This is a 2D-friendly approximation of query-based compression:
    - a static band bias encodes which frequencies each band should prefer,
    - an input-conditioned score map nudges the pooling weights per frame.
    """

    def __init__(self, channels: int, band_spec: BandSpec2d, causal: bool = True):
        super().__init__()
        self.channels = channels
        self.band_spec = band_spec
        self.n_bands = band_spec.n_bands
        Conv = CausalConv2d if causal else SameConv2d
        self.causal = causal

        self.norm = RMSNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.dw = Conv(channels, channels, kernel_size=(3, 3), groups=channels, bias=True)
        self.score = nn.Conv2d(channels, self.n_bands, kernel_size=1, bias=True)
        self.score_scale = nn.Parameter(torch.tensor(1.0))
        self.bias_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("band_bias", band_spec.band_bias())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.shape[-1] == self.band_spec.n_freq, f"{x.shape} vs {self.band_spec.n_freq}")
        h = self.dw(self.pw(self.norm(x)))
        scores = self.score(h) * self.score_scale + self.band_bias * self.bias_scale
        weights = torch.softmax(scores, dim=-1)
        batch, channels, n_frames, n_freq = h.shape

        # Batch matrix multiply over (time, batch) keeps the graph compact while
        # implementing per-frame soft pooling from F bins to K band tokens.
        h_btfc = h.permute(0, 2, 3, 1).reshape(batch * n_frames, n_freq, channels)
        w_btkf = weights.permute(0, 2, 1, 3).reshape(batch * n_frames, self.n_bands, n_freq)
        z_btkc = torch.bmm(w_btkf, h_btfc)
        return z_btkc.reshape(batch, n_frames, self.n_bands, channels).permute(0, 3, 1, 2)

    def stream_context_frames(self) -> int:
        if isinstance(self.dw, CausalConv2d):
            return self.dw.stream_context_frames()
        return 0

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not isinstance(self.dw, CausalConv2d):
            raise RuntimeError("Streaming state is only supported when causal=True.")
        return self.dw.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(self.dw, CausalConv2d):
            raise RuntimeError("forward_stream is only supported when causal=True.")

        _runtime_assert(x.shape[-1] == self.band_spec.n_freq, f"{x.shape} vs {self.band_spec.n_freq}")
        h = self.norm(x)
        h = self.pw(h)
        h, new_state = self.dw.forward_stream(h, state)
        scores = self.score(h) * self.score_scale + self.band_bias * self.bias_scale
        weights = torch.softmax(scores, dim=-1)
        batch, channels, n_frames, n_freq = h.shape
        h_btfc = h.permute(0, 2, 3, 1).reshape(batch * n_frames, n_freq, channels)
        w_btkf = weights.permute(0, 2, 1, 3).reshape(batch * n_frames, self.n_bands, n_freq)
        z_btkc = torch.bmm(w_btkf, h_btfc)
        z = z_btkc.reshape(batch, n_frames, self.n_bands, channels).permute(0, 3, 1, 2)
        return z, new_state


class LatentSeparator2d(nn.Module):
    """
    Separation stack operating on the compressed band axis.
    """

    def __init__(self, channels: int, n_layers: int, kernel_size: tuple[int, int], causal: bool = True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [OnlineConvBlock(channels, expansion=2, kernel_size=kernel_size, causal=causal) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x

    def stream_context_frames(self) -> int:
        return sum(blk.stream_context_frames() for blk in self.blocks)

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            blk.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype) for blk in self.blocks
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        states: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if states is None:
            states = tuple([None] * len(self.blocks))
        _runtime_assert(len(states) == len(self.blocks), f"{len(states)} vs {len(self.blocks)}")

        new_states = []
        for blk, state in zip(self.blocks, states):
            x, state = blk.forward_stream(x, state)
            new_states.append(state)
        return x, tuple(new_states)


class SpectralDecoder2d(nn.Module):
    """
    Decode compressed band tokens back to full-resolution spectral features.

    A static reconstruction basis provides band locality, while an input-derived
    gate lets each frame adjust how strongly each band contributes at each
    frequency bin.
    """

    def __init__(self, channels: int, band_spec: BandSpec2d):
        super().__init__()
        self.channels = channels
        self.band_spec = band_spec
        self.n_bands = band_spec.n_bands
        self.n_freq = band_spec.n_freq

        self.pre = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.band_gate = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.gate_scale = nn.Parameter(torch.tensor(1.0))
        self.basis_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("decode_basis", band_spec.decode_basis())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _runtime_assert(z.shape[-1] == self.n_bands, f"{z.shape} vs {self.n_bands}")
        h = self.pre(z)
        batch, channels, n_frames, n_bands = h.shape

        # Produce one dynamic gain per band token and frame, then combine it
        # with the static band basis and re-normalize across bands for each
        # output frequency bin.
        band_gain = torch.sigmoid(self.band_gate(h)) * self.gate_scale
        band_gain = band_gain.permute(0, 3, 2, 1)  # (B, K, T, 1)

        coeff = self.decode_basis * (self.basis_scale + band_gain)
        coeff = coeff / coeff.sum(dim=1, keepdim=True).clamp_min(1e-6)

        h_btck = h.permute(0, 2, 1, 3).reshape(batch * n_frames, channels, n_bands)
        coeff_btkf = coeff.permute(0, 2, 1, 3).reshape(batch * n_frames, n_bands, self.n_freq)
        y_btcf = torch.bmm(h_btck, coeff_btkf)
        return y_btcf.reshape(batch, n_frames, channels, self.n_freq).permute(0, 2, 1, 3)


class OnlineSFC2D(nn.Module):
    """
    2D-only realtime separator core with explicit spectral compression:

    input (B, 2*M, T, F)
      -> in_proj
      -> compressor: (B, D, T, F) -> (B, D, T, K)
      -> latent separator on K bands
      -> decoder: (B, D, T, K) -> (B, D, T, F)
      -> out_proj
      -> packed complex masking or mapping
    """

    def __init__(
        self,
        n_freq: int,
        n_bands: int = 64,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
        n_src: int = 2,
        n_chan: int = 1,
        d_model: int = 96,
        n_layers: int = 12,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
        masking: bool = True,
    ):
        super().__init__()
        self.n_freq = n_freq
        self.n_bands = n_bands
        self.band_config = band_config
        self.n_src = n_src
        self.n_chan = n_chan
        self.d_model = d_model
        self.n_layers = n_layers
        self.causal = causal
        self.masking = masking

        in_ch = 2 * n_chan
        out_ch = 2 * n_src * n_chan

        self.band_spec = BandSpec2d(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, d_model, kernel_size=1, bias=True),
            RMSNorm2d(d_model),
        )
        self.compressor = SpectralCompressor2d(channels=d_model, band_spec=self.band_spec, causal=causal)
        self.latent_separator = LatentSeparator2d(
            channels=d_model,
            n_layers=n_layers,
            kernel_size=kernel_size,
            causal=causal,
        )
        self.decoder = SpectralDecoder2d(channels=d_model, band_spec=self.band_spec)
        self.out_proj = nn.Conv2d(d_model, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")

        h = self.in_proj(x)
        z = self.compressor(h)
        z = self.latent_separator(z)
        h = self.decoder(z)
        y = self.out_proj(h)

        if self.masking:
            return apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)

        return y

    def stream_context_frames(self) -> int:
        if not self.causal:
            return 0
        return self.compressor.stream_context_frames() + self.latent_separator.stream_context_frames()

    def init_stream_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        comp = self.compressor.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype)
        sep = self.latent_separator.init_stream_state(
            batch_size,
            freq_bins=self.n_bands,
            device=device,
            dtype=dtype,
        )
        return (comp, *sep)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not self.causal:
            raise RuntimeError("forward_stream is only supported when causal=True.")

        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")

        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        _runtime_assert(len(state) == 1 + len(self.latent_separator.blocks), f"Unexpected state tuple: {len(state)}")
        comp_state = state[0]
        sep_state = state[1:]

        h = self.in_proj(x)
        z, new_comp_state = self.compressor.forward_stream(h, comp_state)
        z, new_sep_state = self.latent_separator.forward_stream(z, sep_state)
        h = self.decoder(z)
        y = self.out_proj(h)

        if self.masking:
            y = apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)

        return y, (new_comp_state, *new_sep_state)

    def init_input_history(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        history_frames = self.stream_context_frames()
        return torch.zeros(batch_size, 2 * self.n_chan, history_frames, self.n_freq, device=device, dtype=dtype)

    def forward_stream_recompute(
        self,
        x: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "Exact low-memory recomputation from raw input history is not implemented for this model. "
            "Use forward_stream with layer caches for strict realtime equivalence."
        )

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        if not self.causal:
            return 0
        states = self.init_stream_state(batch_size=batch_size, device=self.out_proj.weight.device, dtype=self.out_proj.weight.dtype)
        return sum(int(s.numel()) for s in states)

    def input_history_numel(self, batch_size: int = 1) -> int:
        return batch_size * 2 * self.n_chan * self.stream_context_frames() * self.n_freq

    def state_size_bytes(self, *, batch_size: int = 1, dtype: torch.dtype = torch.float16, mode: str = "layer_cache") -> int:
        element_size = torch.tensor([], dtype=dtype).element_size()
        if mode == "layer_cache":
            return self.layer_cache_numel(batch_size=batch_size) * element_size
        if mode == "input_history":
            return self.input_history_numel(batch_size=batch_size) * element_size
        raise ValueError(f"Unsupported state mode: {mode}")


def apply_packed_complex_mask(x: torch.Tensor, y: torch.Tensor, n_src: int, n_chan: int) -> torch.Tensor:
    """
    Packed complex multiply using 4D tensors only.
    """

    bsz, cin, _, _ = x.shape
    _, cout, _, _ = y.shape
    _runtime_assert(cin == 2 * n_chan, f"{cin} vs {2 * n_chan}")
    _runtime_assert(cout == 2 * n_src * n_chan, f"{cout} vs {2 * n_src * n_chan}")

    x_rep = x.repeat(1, n_src, 1, 1)
    xr = x_rep[:, 0::2, :, :]
    xi = x_rep[:, 1::2, :, :]
    mr = y[:, 0::2, :, :]
    mi = y[:, 1::2, :, :]

    out_r = xr * mr - xi * mi
    out_i = xr * mi + xi * mr

    # Avoid slice assignment here so ONNX export does not lower the packing step
    # to ScatterND. We instead interleave real/imag by stacking and reshaping.
    out = torch.stack([out_r, out_i], dim=2)
    return out.reshape(bsz, 2 * n_src * n_chan, y.shape[-2], y.shape[-1])


def pack_complex_stft_as_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Convert complex STFT (B, M, F, T) into packed real tensor (B, 2*M, T, F).
    """

    assert x.is_complex(), x.dtype
    x = x.transpose(-1, -2)
    xr = x.real
    xi = x.imag
    return torch.stack([xr, xi], dim=2).reshape(x.shape[0], 2 * x.shape[1], x.shape[2], x.shape[3])


def unpack_2d_to_complex_stft(y: torch.Tensor, n_src: int, n_chan: int) -> torch.Tensor:
    """
    Convert packed output (B, 2*N*M, T, F) back to complex STFT (B, N, M, F, T).
    """

    _runtime_assert(y.ndim == 4, str(y.shape))
    bsz, channels, n_frames, n_freq = y.shape
    _runtime_assert(channels == 2 * n_src * n_chan, f"{channels} vs {(n_src, n_chan)}")

    y = y.reshape(bsz, n_src, n_chan, 2, n_frames, n_freq)
    yr = y[:, :, :, 0, :, :]
    yi = y[:, :, :, 1, :, :]
    z = torch.complex(yr.to(torch.float32), yi.to(torch.float32))
    return z.transpose(-1, -2)
