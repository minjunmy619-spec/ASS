"""BandSFCNetNPU core.

This model combines the deployment-friendly SFC frequency transport with the
BandSCNetNPU separation prior:

* SFC-style F -> K compression and K -> F expansion keep the recurrent state
  tied to a small latent band axis.
* BandSCNet-style cross-band and narrow-band stages model local frequency and
  causal temporal structure on those latent bands.
* Optional frequency-pooled channel mixers add trainable capacity without
  increasing persistent streaming state.

The deployable core operates on packed-real 2D STFT tensors shaped
``[B, 2*n_chan, T, F]`` and returns packed complex estimates or raw mask logits
depending on ``masking``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from BandSCNetNPU.blocks import BoundedCausalAttn, CrossBandBlock, GatedAct, PooledChannelMixer
from spectral_feature_compression.core.model.online_crossattn_query_sfc_2d import (
    NPUSafeCrossAttnDecoder2d,
    NPUSafeCrossAttnEncoder2d,
)
from spectral_feature_compression.core.model.online_sfc_2d import (
    CausalConv2d,
    RMSNorm2d,
    _runtime_assert,
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)
from spectral_feature_compression.core.model.online_soft_band_query_sfc_2d import (
    SoftBandQueryCompressor2d,
    SoftBandQueryExpander2d,
)
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import (
    SoftBandCompressor2d,
    SoftBandSpec2d,
)


def _normalize_dilation_schedule(
    num_stages: int,
    dilation_cycle: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    if dilation_cycle is None:
        dilation_cycle = (1, 1, 2, 4, 6)
    cycle = tuple(int(d) for d in dilation_cycle)
    if len(cycle) == 0:
        raise ValueError("dilation_cycle must not be empty")
    if any(d <= 0 for d in cycle):
        raise ValueError(f"All dilations must be positive, got {cycle}")
    return tuple(cycle[idx % len(cycle)] for idx in range(num_stages))


def _apply_packed_complex_mask_no_repeat(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_src: int,
    n_chan: int,
) -> torch.Tensor:
    """Packed complex multiply without repeat/Tile-prone expansion."""

    _runtime_assert(x.shape[1] == 2 * n_chan, f"{x.shape[1]} vs {2 * n_chan}")
    _runtime_assert(y.shape[1] == 2 * n_src * n_chan, f"{y.shape[1]} vs {2 * n_src * n_chan}")
    outputs: list[torch.Tensor] = []
    for src_idx in range(n_src):
        for chan_idx in range(n_chan):
            in_r = x[:, 2 * chan_idx : 2 * chan_idx + 1, :, :]
            in_i = x[:, 2 * chan_idx + 1 : 2 * chan_idx + 2, :, :]
            mask_base = 2 * (src_idx * n_chan + chan_idx)
            mask_r = y[:, mask_base : mask_base + 1, :, :]
            mask_i = y[:, mask_base + 1 : mask_base + 2, :, :]
            outputs.append(in_r * mask_r - in_i * mask_i)
            outputs.append(in_r * mask_i + in_i * mask_r)
    return torch.cat(outputs, dim=1)


class DilatedNarrowBandBlock(nn.Module):
    """BandSCNet narrow-band temporal block with optional NPU-valid dilation."""

    def __init__(
        self,
        channels: int,
        *,
        time_kernel: int = 3,
        time_dilation: int = 1,
        use_attn: bool = False,
        attn_window: int = 16,
        num_heads: int = 4,
        head_dim: int = 8,
    ):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(f"channels must be even, got {channels}")
        self.channels = channels
        self.time_kernel = time_kernel
        self.time_dilation = time_dilation
        self.use_attn = use_attn

        self.norm = RMSNorm2d(channels)
        self.causal_dw = CausalConv2d(
            channels,
            2 * channels,
            kernel_size=(time_kernel, 1),
            dilation=(time_dilation, 1),
            groups=channels,
            bias=True,
        )
        self.act = GatedAct()
        self.attn = (
            BoundedCausalAttn(channels, window=attn_window, num_heads=num_heads, head_dim=head_dim)
            if use_attn
            else None
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        y = self.norm(x)
        y = self.causal_dw(y)
        y = self.act(y)
        if self.attn is not None:
            y = y + self.attn(x)
        y = self.pointwise(y)
        return x + y

    def stream_context_frames(self) -> int:
        return self.causal_dw.stream_context_frames()

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        dw_state = self.causal_dw.init_stream_state(
            batch_size,
            freq_bins=freq_bins,
            device=device,
            dtype=dtype,
        )
        if self.attn is None:
            return (dw_state,)
        attn_state = self.attn.init_stream_state(
            batch_size,
            freq_bins=freq_bins,
            device=device,
            dtype=dtype,
        )
        return (dw_state, attn_state)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        if state is None:
            state = self.init_stream_state(
                x.shape[0],
                freq_bins=x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )
        y = self.norm(x)
        y, new_dw_state = self.causal_dw.forward_stream(y, state[0])
        y = self.act(y)
        if self.attn is not None:
            attn_out, new_attn_state = self.attn.forward_stream(x, state[1])
            y = y + attn_out
            new_state: tuple[torch.Tensor, ...] = (new_dw_state, new_attn_state)
        else:
            new_state = (new_dw_state,)
        y = self.pointwise(y)
        return x + y, new_state


class CrossBandMixer(nn.Module):
    """Grouped cross-band mixer from the Causal BandSFC-CNB proposal.

    This is the named Proposal-B cross-band component.  With ``grouped=True``
    the frequency convolution is channel-wise, so it mixes neighbouring latent
    bands without dense channel mixing until the final 1x1 projection.
    """

    def __init__(self, channels: int, *, grouped: bool = True, freq_kernel: int = 3):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(f"channels must be even, got {channels}")
        if freq_kernel <= 0:
            raise ValueError(f"freq_kernel must be positive, got {freq_kernel}")
        if freq_kernel % 2 != 1:
            raise ValueError(f"freq_kernel must be odd, got {freq_kernel}")
        if (freq_kernel - 1) >= 14:
            raise ValueError(f"freq_kernel violates NPU span rule: {freq_kernel}")
        groups = channels if grouped else 1
        self.channels = int(channels)
        self.grouped = bool(grouped)
        self.freq_kernel = int(freq_kernel)
        self.norm = RMSNorm2d(channels)
        self.freq_conv = nn.Conv2d(
            channels,
            2 * channels,
            kernel_size=(1, freq_kernel),
            padding=(0, freq_kernel // 2),
            groups=groups,
            bias=True,
        )
        self.act = GatedAct()
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        y = self.norm(x)
        y = self.freq_conv(y)
        y = self.act(y)
        return self.pointwise(y)


class CausalFSMNBandMixer(nn.Module):
    """FSMN-style causal narrow-band memory mixer for Proposal B.

    The research sketch uses ``kernel_t=5`` and ``dilation_schedule=(1, 2, 4)``.
    The current repo validator rejects that exact last branch because
    ``(5 - 1) * 4 = 16 >= 14``.  This module keeps the exact API but validates
    every branch through :class:`CausalConv2d`, so deployable presets should use
    the nearest NPU-safe schedule, typically ``(1, 2, 3)``.

    Streaming stores one shared input-history cache of the maximum branch span
    instead of one cache per dilation branch.  That preserves exact full/stream
    parity while reducing persistent state.
    """

    def __init__(
        self,
        channels: int,
        *,
        kernel_t: int = 5,
        dilation_schedule: tuple[int, ...] | list[int] = (1, 2, 3),
    ):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(f"channels must be even, got {channels}")
        if kernel_t <= 0:
            raise ValueError(f"kernel_t must be positive, got {kernel_t}")
        schedule = tuple(int(v) for v in dilation_schedule)
        if len(schedule) == 0:
            raise ValueError("dilation_schedule must not be empty")
        if any(v <= 0 for v in schedule):
            raise ValueError(f"dilation values must be positive, got {schedule}")
        self.channels = int(channels)
        self.kernel_t = int(kernel_t)
        self.dilation_schedule = schedule
        self.max_context = max((self.kernel_t - 1) * dilation for dilation in schedule)

        self.norm = RMSNorm2d(channels)
        self.in_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.memory = nn.ModuleList(
            [
                CausalConv2d(
                    channels,
                    channels,
                    kernel_size=(self.kernel_t, 1),
                    dilation=(dilation, 1),
                    groups=channels,
                    bias=True,
                )
                for dilation in schedule
            ]
        )
        self.gate = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def _mix_branches_full(self, y: torch.Tensor) -> torch.Tensor:
        outputs = [branch(y) for branch in self.memory]
        mixed = outputs[0]
        for output in outputs[1:]:
            mixed = mixed + output
        return mixed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        y = self.in_proj(self.norm(x))
        y_mem = self._mix_branches_full(y)
        y = y_mem * torch.sigmoid(self.gate(y))
        return self.out_proj(y)

    def stream_context_frames(self) -> int:
        return int(self.max_context)

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, self.channels, self.max_context, freq_bins, device=device, dtype=dtype)

    def forward_stream(self, x: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        _runtime_assert(x.shape[2] == 1, f"Expected single-frame input, got T={x.shape[2]}")
        if state is None:
            state = self.init_stream_state(x.shape[0], freq_bins=x.shape[-1], device=x.device, dtype=x.dtype)

        y = self.in_proj(self.norm(x))
        history = torch.cat([state, y], dim=2)
        outputs: list[torch.Tensor] = []
        for branch in self.memory:
            # Apply the underlying valid conv over the shared history and keep
            # the newest output frame; this matches branch.forward(y_sequence)
            # for online single-frame stepping.
            outputs.append(branch.conv(history)[:, :, -1:, :])
        y_mem = outputs[0]
        for output in outputs[1:]:
            y_mem = y_mem + output
        y = y_mem * torch.sigmoid(self.gate(y))
        if self.max_context == 0:
            new_state = history[:, :, 0:0, :]
        else:
            new_state = history[:, :, -self.max_context :, :]
        return self.out_proj(y), new_state


class CompressedSelfAttentionFusion(nn.Module):
    """Stateless compressed-band self-attention fusion for Proposal B.

    Attention is applied across the compressed band axis independently for each
    frame, so it is causal in time and carries no streaming cache.
    """

    def __init__(self, channels: int, *, num_heads: int = 4, head_dim: int = 8):
        super().__init__()
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.norm = RMSNorm2d(channels)
        self.qkv_proj = nn.Conv2d(channels, 3 * self.inner_dim, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(self.inner_dim, channels, kernel_size=1, bias=True)
        self.scale = 1.0 / math.sqrt(float(self.head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        batch, _, frames, bands = x.shape
        qkv = self.qkv_proj(self.norm(x))
        q, k, v = qkv.chunk(3, dim=1)

        def _tokens(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.permute(0, 2, 3, 1).reshape(batch * frames, bands, self.inner_dim)

        q_tokens = _tokens(q)
        k_tokens = _tokens(k)
        v_tokens = _tokens(v)
        scores = torch.bmm(q_tokens, k_tokens.transpose(1, 2)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.bmm(attn, v_tokens)
        out = out.reshape(batch, frames, bands, self.inner_dim).permute(0, 3, 1, 2)
        return self.out_proj(out)

    def forward_stream(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class CausalCNBBlock(nn.Module):
    """Exact named Causal BandSFC-CNB block from Proposal B.

    The block follows the proposal flow:

    ``cross_band -> causal FSMN narrow-band memory -> compressed attention fusion``.
    """

    def __init__(
        self,
        d_model: int,
        *,
        freq_kernel: int = 3,
        kernel_t: int = 5,
        dilation_schedule: tuple[int, ...] | list[int] = (1, 2, 3),
        num_heads: int = 4,
        head_dim: int = 8,
    ):
        super().__init__()
        self.cross_band = CrossBandMixer(d_model, grouped=True, freq_kernel=freq_kernel)
        self.narrow_band = CausalFSMNBandMixer(
            d_model,
            kernel_t=kernel_t,
            dilation_schedule=dilation_schedule,
        )
        self.csa = CompressedSelfAttentionFusion(d_model, num_heads=num_heads, head_dim=head_dim)

    def forward(self, z: torch.Tensor, state: torch.Tensor | None = None):
        if state is not None:
            _runtime_assert(z.shape[2] == 1, f"Stateful CNB forward expects single-frame input, got T={z.shape[2]}")
            return self.forward_stream(z, state)
        z = z + self.cross_band(z)
        narrow_out = self.narrow_band(z)
        z = z + narrow_out
        return z + self.csa(z)

    def stream_context_frames(self) -> int:
        return self.narrow_band.stream_context_frames()

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self.narrow_band.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(self, z: torch.Tensor, state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        z = z + self.cross_band(z)
        narrow_out, state = self.narrow_band.forward_stream(z, state)
        z = z + narrow_out
        z = z + self.csa.forward_stream(z)
        return z, state


class BandSFCStage(nn.Module):
    """Cross-band, dilated narrow-band, and optional pooled capacity branch."""

    def __init__(
        self,
        channels: int,
        *,
        time_kernel: int,
        freq_kernel: int,
        time_dilation: int,
        use_attn: bool,
        attn_window: int,
        num_heads: int,
        head_dim: int,
        pooled_mixer_hidden: int,
    ):
        super().__init__()
        self.cross = CrossBandBlock(channels, freq_kernel=freq_kernel)
        self.narrow = DilatedNarrowBandBlock(
            channels,
            time_kernel=time_kernel,
            time_dilation=time_dilation,
            use_attn=use_attn,
            attn_window=attn_window,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.pooled_mixer = (
            PooledChannelMixer(channels, pooled_mixer_hidden) if pooled_mixer_hidden > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cross(x)
        x = self.narrow(x)
        return self.pooled_mixer(x)

    def stream_context_frames(self) -> int:
        return self.narrow.stream_context_frames()

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return self.narrow.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x = self.cross(x)
        x, state = self.narrow.forward_stream(x, state)
        return self.pooled_mixer(x), state


class SoftBandExpanderNoClip2d(nn.Module):
    """Soft-band K -> F expander that avoids ONNX Clip from clamp_min."""

    def __init__(self, channels: int, band_spec: SoftBandSpec2d):
        super().__init__()
        self.channels = channels
        self.band_spec = band_spec
        self.n_bands = band_spec.n_bands
        self.n_freq = band_spec.n_freq
        self.pre_norm = RMSNorm2d(channels)
        self.pre_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.band_gain = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.gain_scale = nn.Parameter(torch.tensor(1.0))
        self.basis_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("expansion_basis", band_spec.expansion_basis())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _runtime_assert(z.shape[-1] == self.n_bands, f"{z.shape} vs {self.n_bands}")
        h = self.pre_proj(self.pre_norm(z))
        h = h * torch.sigmoid(h)
        batch, channels, n_frames, n_bands = h.shape

        gains = 1.0 + torch.sigmoid(self.band_gain(h)) * self.gain_scale
        gains = gains.permute(0, 3, 2, 1)
        coeff = self.expansion_basis * (self.basis_scale + gains)
        denom = coeff.sum(dim=1, keepdim=True) + 1e-6
        coeff = coeff / denom

        h_btck = h.permute(0, 2, 1, 3).reshape(batch * n_frames, channels, n_bands)
        coeff_btkf = coeff.permute(0, 2, 1, 3).reshape(batch * n_frames, n_bands, self.n_freq)
        expanded_btcf = torch.bmm(h_btck, coeff_btkf)
        return expanded_btcf.reshape(batch, n_frames, channels, self.n_freq).permute(0, 2, 1, 3)


class BandSFCNetNPU(nn.Module):
    """Online NPU core with SFC transport and BandSCNet latent separation."""

    def __init__(
        self,
        n_freq: int,
        *,
        n_bands: int = 64,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
        n_src: int = 3,
        n_chan: int = 1,
        channels: int = 32,
        num_stages: int = 4,
        time_kernel: int = 3,
        freq_kernel: int = 3,
        dilation_cycle: tuple[int, ...] | list[int] | None = None,
        transport: str = "soft",
        query_type: str = "adaptive",
        routing_normalization: str = "softmax",
        use_attn: bool = False,
        attn_window: int = 16,
        num_heads: int = 4,
        head_dim: int = 8,
        pooled_mixer_hidden: int = 0,
        pooled_mixer_hidden_schedule: tuple[int, ...] | list[int] | None = None,
        stage_type: str = "band_sfc",
        cnb_kernel: int = 5,
        cnb_dilation_schedule: tuple[int, ...] | list[int] | None = None,
        causal: bool = True,
        masking: bool = True,
        residual_head: bool = False,
    ):
        super().__init__()
        if transport == "soft_band_query":
            transport = "soft_query"
        if transport == "cross_attention_query":
            transport = "crossattn_query"
        if transport not in {"soft", "soft_query", "crossattn", "crossattn_query"}:
            raise ValueError(
                "transport must be 'soft', 'soft_band_query', 'soft_query', "
                f"'crossattn', or 'crossattn_query', got {transport!r}"
            )
        if not causal:
            raise ValueError("BandSFCNetNPU currently targets causal online deployment only")
        if channels % 2 != 0:
            raise ValueError(f"channels must be even, got {channels}")
        if residual_head and not masking:
            raise ValueError("residual_head requires masking=True so mask and residual estimates can be fused")
        if stage_type not in {"band_sfc", "causal_cnb"}:
            raise ValueError(f"stage_type must be 'band_sfc' or 'causal_cnb', got {stage_type!r}")

        self.n_freq = n_freq
        self.n_bands = n_bands
        self.n_src = n_src
        self.n_chan = n_chan
        self.channels = channels
        self.num_stages = num_stages
        self.transport = transport
        self.causal = causal
        self.masking = masking
        self.residual_head = residual_head
        self.time_kernel = time_kernel
        self.freq_kernel = freq_kernel
        self.routing_normalization = routing_normalization
        self.use_attn = use_attn
        self.attn_window = attn_window
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pooled_mixer_hidden = pooled_mixer_hidden
        self.stage_type = stage_type
        self.cnb_kernel = int(cnb_kernel)
        self.cnb_dilation_schedule = tuple(int(v) for v in (cnb_dilation_schedule or (1, 2, 3)))
        self.dilation_schedule = _normalize_dilation_schedule(num_stages, dilation_cycle)
        if pooled_mixer_hidden_schedule is None:
            self.pooled_mixer_hidden_schedule = tuple(int(pooled_mixer_hidden) for _ in range(num_stages))
        else:
            self.pooled_mixer_hidden_schedule = tuple(int(v) for v in pooled_mixer_hidden_schedule)
            if len(self.pooled_mixer_hidden_schedule) != num_stages:
                raise ValueError(
                    f"pooled_mixer_hidden_schedule must have {num_stages} entries, "
                    f"got {self.pooled_mixer_hidden_schedule}"
                )
            if any(v < 0 for v in self.pooled_mixer_hidden_schedule):
                raise ValueError(
                    f"pooled_mixer_hidden_schedule values must be >= 0, got {self.pooled_mixer_hidden_schedule}"
                )

        in_ch = 2 * n_chan
        out_ch = 2 * n_src * n_chan
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size=1, bias=True),
            RMSNorm2d(channels),
        )
        band_spec = SoftBandSpec2d(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )

        if transport == "soft":
            self.encoder = SoftBandCompressor2d(
                channels=channels,
                band_spec=band_spec,
                kernel_size=(time_kernel, freq_kernel),
                causal=causal,
                normalization=routing_normalization,
            )
            self.decoder = SoftBandExpanderNoClip2d(channels=channels, band_spec=band_spec)
        elif transport == "soft_query":
            self.encoder = SoftBandQueryCompressor2d(
                channels=channels,
                band_spec=band_spec,
                kernel_size=(time_kernel, freq_kernel),
                causal=causal,
                normalization=routing_normalization,
            )
            self.decoder = SoftBandQueryExpander2d(channels=channels, band_spec=band_spec)
        else:
            self.encoder = NPUSafeCrossAttnEncoder2d(
                channels=channels,
                band_spec=band_spec,
                kernel_size=(time_kernel, freq_kernel),
                causal=causal,
                query_type=query_type,
                routing_normalization=routing_normalization,
            )
            self.decoder = NPUSafeCrossAttnDecoder2d(
                channels=channels,
                band_spec=band_spec,
                query_type=query_type,
                routing_normalization=routing_normalization,
            )

        if self.stage_type == "causal_cnb":
            self.stages = nn.ModuleList(
                [
                    CausalCNBBlock(
                        channels,
                        freq_kernel=freq_kernel,
                        kernel_t=self.cnb_kernel,
                        dilation_schedule=self.cnb_dilation_schedule,
                        num_heads=num_heads,
                        head_dim=head_dim,
                    )
                    for _ in range(num_stages)
                ]
            )
        else:
            self.stages = nn.ModuleList(
                [
                    BandSFCStage(
                        channels,
                        time_kernel=time_kernel,
                        freq_kernel=freq_kernel,
                        time_dilation=dilation,
                        use_attn=use_attn,
                        attn_window=attn_window,
                        num_heads=num_heads,
                        head_dim=head_dim,
                        pooled_mixer_hidden=stage_pooled_hidden,
                    )
                    for dilation, stage_pooled_hidden in zip(self.dilation_schedule, self.pooled_mixer_hidden_schedule)
                ]
            )
        self.out_proj = nn.Conv2d(channels, out_ch * (2 if residual_head else 1), kernel_size=1, bias=True)
        self._init_output_head()

    def _init_output_head(self) -> None:
        """Start from a conservative mixture-split mask instead of random complex gains."""

        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is None:
            return
        nn.init.zeros_(self.out_proj.bias)
        if not self.masking:
            return

        with torch.no_grad():
            source_gain = 1.0 / float(self.n_src)
            mask_channels = 2 * self.n_src * self.n_chan
            for src_idx in range(self.n_src):
                for chan_idx in range(self.n_chan):
                    real_idx = 2 * (src_idx * self.n_chan + chan_idx)
                    if real_idx < mask_channels:
                        self.out_proj.bias[real_idx] = source_gain

    def _encode(self, x: torch.Tensor):
        if self.transport == "soft":
            z, _weights = self.encoder(x)
            return z, None
        if self.transport == "soft_query":
            return self.encoder(x)
        return self.encoder(x)

    def _encode_stream(self, x: torch.Tensor, state: torch.Tensor | None):
        if self.transport == "soft":
            z, new_state = self.encoder.forward_stream(x, state)
            return z, None, new_state
        (z, side), new_state = self.encoder.forward_stream(x, state)
        return z, side, new_state

    def _decode(self, z: torch.Tensor, side: torch.Tensor | None) -> torch.Tensor:
        if self.transport == "soft":
            return self.decoder(z)
        if side is None:
            raise RuntimeError(f"{self.transport} transport requires a decoder side path")
        return self.decoder(z, side)

    def _project_output(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y = self.out_proj(h)
        if not self.masking:
            return y
        if self.residual_head:
            mask, residual = y.chunk(2, dim=1)
            masked = _apply_packed_complex_mask_no_repeat(
                x=x,
                y=mask,
                n_src=self.n_src,
                n_chan=self.n_chan,
            )
            return masked + residual
        return _apply_packed_complex_mask_no_repeat(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {tuple(x.shape)}")
        _runtime_assert(x.shape[1] == 2 * self.n_chan, f"{x.shape[1]} vs {2 * self.n_chan}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape[-1]} vs {self.n_freq}")

        h = self.in_proj(x)
        z, side = self._encode(h)
        for stage in self.stages:
            z = stage(z)
        h = self._decode(z, side)
        return self._project_output(x, h)

    def stream_context_frames(self) -> int:
        return self.encoder.stream_context_frames() + sum(stage.stream_context_frames() for stage in self.stages)

    def init_stream_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        enc = self.encoder.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype)
        sep = tuple(
            stage.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype)
            for stage in self.stages
        )
        return (enc, *sep)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape[-1]} vs {self.n_freq}")
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        h = self.in_proj(x)
        z, side, new_enc_state = self._encode_stream(h, state[0])
        new_sep_states: list[tuple[torch.Tensor, ...]] = []
        for stage, stage_state in zip(self.stages, state[1:]):
            z, new_stage_state = stage.forward_stream(z, stage_state)
            new_sep_states.append(new_stage_state)
        h = self._decode(z, side)
        y = self._project_output(x, h)
        return y, (new_enc_state, *new_sep_states)

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        states = self.init_stream_state(
            batch_size=batch_size,
            device=self.out_proj.weight.device,
            dtype=self.out_proj.weight.dtype,
        )
        return _tree_numel(states)

    def input_history_numel(self, batch_size: int = 1) -> int:
        return batch_size * 2 * self.n_chan * self.stream_context_frames() * self.n_freq

    def state_size_bytes(
        self,
        *,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
        mode: str = "layer_cache",
    ) -> int:
        element_size = torch.tensor([], dtype=dtype).element_size()
        if mode == "layer_cache":
            return self.layer_cache_numel(batch_size=batch_size) * element_size
        if mode == "input_history":
            return self.input_history_numel(batch_size=batch_size) * element_size
        raise ValueError(f"Unsupported state mode: {mode}")


class BandSFCNetNPUModel(nn.Module):
    """Complex-STFT wrapper matching the existing online model contract."""

    def __init__(self, core: BandSFCNetNPU):
        super().__init__()
        self.core = core
        self.n_src = core.n_src
        self.n_chan = core.n_chan

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x2d = pack_complex_stft_as_2d(x)
        y2d = self.core(x2d)
        return unpack_2d_to_complex_stft(y2d, n_src=self.n_src, n_chan=self.n_chan)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None):
        return self.core.init_stream_state(batch_size=batch_size, device=device, dtype=dtype)

    def forward_stream(self, x2d: torch.Tensor, state=None):
        return self.core.forward_stream(x2d, state)


def _tree_numel(tree) -> int:
    if isinstance(tree, torch.Tensor):
        return int(tree.numel())
    return sum(_tree_numel(item) for item in tree)
