"""Sparse Downsample / Upsample pyramid for Band-SCNet-NPU.

The frequency axis is split into three bands (low / mid / high) with
asymmetric processing depth and stride-chain downsampling, inspired by
Band-SCNet. All strides are 2 (NPU rule 6). No ``Conv1D`` ops.

The encoder produces three per-band feature maps which are then
concatenated on the F' axis before entering the separation network.
The decoder mirrors the encoder and uses ``ConvTranspose2d`` with
``kernel=(1,2), stride=(1,2)`` for upsampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn

from spectral_feature_compression.core.model.online_sfc_2d import (
    CausalConv2d,
    RMSNorm2d,
    _runtime_assert,
)

from .blocks import GatedAct


# --- band split utility ------------------------------------------------------


class BandSplit(NamedTuple):
    low: int   # F_l (already at full resolution)
    mid: int   # F_m (raw width before the /4 downsample)
    high: int  # F_h (raw width before the /16 downsample)


def split_bands(
    n_freq: int,
    ratios: tuple[float, float, float] = (0.175, 0.392, 0.433),
    high_multiple: int = 16,
    mid_multiple: int = 4,
    low_multiple: int = 4,
) -> BandSplit:
    """Split ``n_freq`` into (low, mid, high) widths.

    Every band is a clean multiple of its corresponding stride multiple so the
    stride-2 chains in the pyramid (2 strides on low/mid, 4 on high) leave
    integer widths. If ``n_freq`` cannot be cleanly split while keeping each
    band a multiple of its stride multiple, this function raises. The
    ``BandSCNetNPU`` wrapper is responsible for zero-padding the frequency
    axis up to a split-compatible width before calling the encoder.
    """
    if n_freq <= 0:
        raise ValueError(f"n_freq must be positive, got {n_freq}")
    r_low, r_mid, r_high = ratios
    if abs(r_low + r_mid + r_high - 1.0) > 1e-3:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    # Initial rounding to each band's multiple.
    high = max(high_multiple, (int(round(n_freq * r_high)) // high_multiple) * high_multiple)
    mid = max(mid_multiple, (int(round(n_freq * r_mid)) // mid_multiple) * mid_multiple)
    low = max(low_multiple, (int(round(n_freq * r_low)) // low_multiple) * low_multiple)

    residual = n_freq - (low + mid + high)
    # Bump `mid` (cheapest multiple) to swallow positive residual.
    if residual > 0:
        bump = (residual // mid_multiple) * mid_multiple
        mid += bump
        residual -= bump
        if residual > 0:
            bump = (residual // low_multiple) * low_multiple
            low += bump
            residual -= bump
    # Trim `mid` then `low` to eat negative residual.
    elif residual < 0:
        trim = min(mid - mid_multiple, ((-residual) // mid_multiple) * mid_multiple)
        mid -= max(trim, 0)
        residual = n_freq - (low + mid + high)
        if residual < 0:
            trim = min(low - low_multiple, ((-residual) // low_multiple) * low_multiple)
            low -= max(trim, 0)
            residual = n_freq - (low + mid + high)

    if residual != 0:
        raise ValueError(
            f"n_freq={n_freq} cannot be cleanly split into multiples "
            f"(low%{low_multiple}, mid%{mid_multiple}, high%{high_multiple}) "
            f"with ratios={ratios}. Closest clean split: "
            f"low={low}, mid={mid}, high={high} (sum={low + mid + high}). "
            f"Use BandSCNetNPU which zero-pads the F axis to a clean n_freq."
        )
    if low <= 0 or mid <= 0 or high <= 0:
        raise ValueError(
            f"band split produced non-positive band: n_freq={n_freq}, "
            f"low={low}, mid={mid}, high={high}"
        )
    return BandSplit(low=low, mid=mid, high=high)


def pad_n_freq_for_split(
    n_freq: int,
    *,
    ratios: tuple[float, float, float] = (0.175, 0.392, 0.433),
    high_multiple: int = 16,
    mid_multiple: int = 4,
    low_multiple: int = 4,
) -> int:
    """Return the smallest ``n_freq_padded >= n_freq`` that splits cleanly.

    Exists because STFT sizes are typically ``n_fft/2 + 1`` which is one
    beyond a power of two. We extend the top of the spectrum with zeros in
    the model and crop after the decoder.
    """
    # Search forward for the next compatible width. With ratio (0.175, 0.392,
    # 0.433) the gap between compatible widths is at most ~lcm(multiples),
    # bounded by 16, so the loop is cheap.
    for candidate in range(n_freq, n_freq + 32):
        try:
            split_bands(
                candidate,
                ratios=ratios,
                high_multiple=high_multiple,
                mid_multiple=mid_multiple,
                low_multiple=low_multiple,
            )
            return candidate
        except ValueError:
            continue
    raise ValueError(f"Could not find a compatible padded n_freq near {n_freq}")

    if low <= 0 or mid <= 0 or high <= 0:
        raise ValueError(
            f"band split produced non-positive band: n_freq={n_freq}, "
            f"low={low}, mid={mid}, high={high}"
        )
    if low + mid + high != n_freq:
        raise ValueError(
            f"band split widths must sum to n_freq: "
            f"low={low}, mid={mid}, high={high}, n_freq={n_freq}"
        )
    return BandSplit(low=low, mid=mid, high=high)


# --- conv block used in the pyramid -----------------------------------------


class _PyramidConvBlock(nn.Module):
    """RMSNorm -> depthwise causal time conv -> PReLU -> freq Conv2d -> GatedAct.

    Shape-preserving; the caller is responsible for any stride / channel-lift
    surrounding this block.
    """

    def __init__(self, channels: int, time_kernel: int = 5, freq_kernel: int = 3):
        super().__init__()
        if freq_kernel % 2 != 1:
            raise ValueError(f"freq_kernel must be odd, got {freq_kernel}")
        self.channels = channels
        self.time_kernel = time_kernel

        self.norm = RMSNorm2d(channels)
        self.dw = CausalConv2d(
            channels,
            channels,
            kernel_size=(time_kernel, 1),
            groups=channels,
            bias=True,
        )
        self.act_dw = nn.PReLU(num_parameters=channels)
        self.freq_conv = nn.Conv2d(
            channels,
            2 * channels,
            kernel_size=(1, freq_kernel),
            padding=(0, freq_kernel // 2),
            bias=True,
        )
        self.gate = GatedAct()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.dw(y)
        y = self.act_dw(y)
        y = self.freq_conv(y)
        y = self.gate(y)
        return x + y

    def stream_context_frames(self) -> int:
        return self.dw.stream_context_frames()

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        return self.dw.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.norm(x)
        y, new_state = self.dw.forward_stream(y, state)
        y = self.act_dw(y)
        y = self.freq_conv(y)
        y = self.gate(y)
        return x + y, new_state


# --- sparse downsample encoder ----------------------------------------------


@dataclass
class _BranchSpec:
    name: str
    raw_width: int          # input width before any downsampling
    num_conv_blocks: int    # number of shape-preserving ConvBlocks
    num_strides: int        # number of stride-2 reductions (0, 2, or 4)


class _EncoderBranch(nn.Module):
    """Per-band encoder branch: lift-to-C -> [ConvBlock | stride-2-conv]*."""

    def __init__(
        self,
        in_ch: int,
        channels: int,
        spec: _BranchSpec,
        time_kernel: int,
        freq_kernel: int,
    ):
        super().__init__()
        self.spec = spec
        self.lift = nn.Conv2d(in_ch, channels, kernel_size=1, bias=True)
        blocks: list[nn.Module] = []
        strides_left = spec.num_strides
        conv_blocks_left = spec.num_conv_blocks
        # Place all stride-2 reductions FIRST so the shape-preserving
        # ConvBlocks (which carry streaming state proportional to F') run at
        # the reduced resolution. This is essential for fitting the 192 KiB
        # DSP state quota.
        while strides_left > 0:
            blocks.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(1, 2),
                    stride=(1, 2),
                    padding=(0, 0),
                    bias=True,
                )
            )
            strides_left -= 1
        while conv_blocks_left > 0:
            blocks.append(_PyramidConvBlock(channels, time_kernel=time_kernel, freq_kernel=freq_kernel))
            conv_blocks_left -= 1
        self.blocks = nn.ModuleList(blocks)

    def out_width(self) -> int:
        return self.spec.raw_width // (2 ** self.spec.num_strides)

    # --- helpers to iterate stateful sub-blocks ---
    def _stateful_indices(self) -> list[int]:
        return [i for i, b in enumerate(self.blocks) if isinstance(b, _PyramidConvBlock)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return x

    def init_stream_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        states: list[torch.Tensor] = []
        current_width = self.spec.raw_width
        for blk in self.blocks:
            if isinstance(blk, _PyramidConvBlock):
                states.append(
                    blk.init_stream_state(batch_size, freq_bins=current_width, device=device, dtype=dtype)
                )
            else:
                # stride-2 conv -> width halves
                current_width //= 2
        return tuple(states)

    def forward_stream(
        self,
        x: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x = self.lift(x)
        new_states: list[torch.Tensor] = []
        state_iter = iter(states)
        for blk in self.blocks:
            if isinstance(blk, _PyramidConvBlock):
                x, new_s = blk.forward_stream(x, next(state_iter))
                new_states.append(new_s)
            else:
                x = blk(x)
        return x, tuple(new_states)


class SparseDownsampleEncoder(nn.Module):
    """Three-branch sparse encoder: low / mid / high with asymmetric depth."""

    def __init__(
        self,
        n_freq: int,
        *,
        in_channels: int,
        channels: int,
        time_kernel: int = 5,
        freq_kernel: int = 3,
        ratios: tuple[float, float, float] = (0.175, 0.392, 0.433),
        conv_blocks_per_branch: tuple[int, int, int] = (3, 2, 1),
        strides_per_branch: tuple[int, int, int] = (0, 2, 4),
    ):
        super().__init__()
        self.n_freq = n_freq
        self.channels = channels
        self.bands = split_bands(n_freq, ratios=ratios)
        nb_low, nb_mid, nb_high = conv_blocks_per_branch
        s_low, s_mid, s_high = strides_per_branch

        specs = [
            _BranchSpec("low", self.bands.low, num_conv_blocks=nb_low, num_strides=s_low),
            _BranchSpec("mid", self.bands.mid, num_conv_blocks=nb_mid, num_strides=s_mid),
            _BranchSpec("high", self.bands.high, num_conv_blocks=nb_high, num_strides=s_high),
        ]
        self.low = _EncoderBranch(in_channels, channels, specs[0], time_kernel, freq_kernel)
        self.mid = _EncoderBranch(in_channels, channels, specs[1], time_kernel, freq_kernel)
        self.high = _EncoderBranch(in_channels, channels, specs[2], time_kernel, freq_kernel)

    @property
    def out_widths(self) -> tuple[int, int, int]:
        return (self.low.out_width(), self.mid.out_width(), self.high.out_width())

    @property
    def concat_width(self) -> int:
        return sum(self.out_widths)

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape[-1]} vs {self.n_freq}")
        f_l = self.bands.low
        f_m = self.bands.mid
        low = x[..., :f_l]
        mid = x[..., f_l : f_l + f_m]
        high = x[..., f_l + f_m :]
        return low, mid, high

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low_raw, mid_raw, high_raw = self._split(x)
        return self.low(low_raw), self.mid(mid_raw), self.high(high_raw)

    def init_stream_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        return (
            self.low.init_stream_state(batch_size, device=device, dtype=dtype),
            self.mid.init_stream_state(batch_size, device=device, dtype=dtype),
            self.high.init_stream_state(batch_size, device=device, dtype=dtype),
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    ]:
        low_raw, mid_raw, high_raw = self._split(x)
        low_out, new_low_state = self.low.forward_stream(low_raw, state[0])
        mid_out, new_mid_state = self.mid.forward_stream(mid_raw, state[1])
        high_out, new_high_state = self.high.forward_stream(high_raw, state[2])
        return (low_out, mid_out, high_out), (new_low_state, new_mid_state, new_high_state)


# --- sparse upsample decoder -------------------------------------------------


class _DecoderBranch(nn.Module):
    """Per-band decoder branch: [ConvBlock | stride-2 TConv]* -> skip fusion."""

    def __init__(
        self,
        channels: int,
        spec: _BranchSpec,
        time_kernel: int,
        freq_kernel: int,
    ):
        super().__init__()
        self.spec = spec

        # Skip fusion: concat encoder output (which is at the same resolution as
        # the input to this branch) on channels and collapse back to C via 1x1.
        self.skip_fuse = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True)

        blocks: list[nn.Module] = []
        strides_left = spec.num_strides
        conv_blocks_left = spec.num_conv_blocks
        # Mirror the encoder: ConvBlocks (which carry streaming state
        # proportional to F') run at the reduced resolution FIRST, then the
        # TConv chain expands back up to the branch input width.
        while conv_blocks_left > 0:
            blocks.append(_PyramidConvBlock(channels, time_kernel=time_kernel, freq_kernel=freq_kernel))
            conv_blocks_left -= 1
        while strides_left > 0:
            blocks.append(
                nn.ConvTranspose2d(
                    channels,
                    channels,
                    kernel_size=(1, 2),
                    stride=(1, 2),
                    padding=(0, 0),
                    bias=True,
                )
            )
            strides_left -= 1
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.skip_fuse(torch.cat([x, skip], dim=1))
        for blk in self.blocks:
            x = blk(x)
        return x

    def init_stream_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        states: list[torch.Tensor] = []
        # After skip_fuse the width equals the encoder-output width for this
        # branch (= raw_width / 2**num_strides). Each TConv doubles it.
        current_width = self.spec.raw_width // (2 ** self.spec.num_strides)
        for blk in self.blocks:
            if isinstance(blk, _PyramidConvBlock):
                states.append(
                    blk.init_stream_state(batch_size, freq_bins=current_width, device=device, dtype=dtype)
                )
            else:
                current_width *= 2
        return tuple(states)

    def forward_stream(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x = self.skip_fuse(torch.cat([x, skip], dim=1))
        new_states: list[torch.Tensor] = []
        state_iter = iter(states)
        for blk in self.blocks:
            if isinstance(blk, _PyramidConvBlock):
                x, new_s = blk.forward_stream(x, next(state_iter))
                new_states.append(new_s)
            else:
                x = blk(x)
        return x, tuple(new_states)


class SparseUpsampleDecoder(nn.Module):
    """Inverse of ``SparseDownsampleEncoder``."""

    def __init__(
        self,
        n_freq: int,
        *,
        channels: int,
        time_kernel: int = 5,
        freq_kernel: int = 3,
        ratios: tuple[float, float, float] = (0.175, 0.392, 0.433),
        conv_blocks_per_branch: tuple[int, int, int] = (3, 2, 1),
        strides_per_branch: tuple[int, int, int] = (0, 2, 4),
    ):
        super().__init__()
        self.n_freq = n_freq
        self.channels = channels
        self.bands = split_bands(n_freq, ratios=ratios)
        nb_low, nb_mid, nb_high = conv_blocks_per_branch
        s_low, s_mid, s_high = strides_per_branch

        specs = [
            _BranchSpec("low", self.bands.low, num_conv_blocks=nb_low, num_strides=s_low),
            _BranchSpec("mid", self.bands.mid, num_conv_blocks=nb_mid, num_strides=s_mid),
            _BranchSpec("high", self.bands.high, num_conv_blocks=nb_high, num_strides=s_high),
        ]
        self.low = _DecoderBranch(channels, specs[0], time_kernel, freq_kernel)
        self.mid = _DecoderBranch(channels, specs[1], time_kernel, freq_kernel)
        self.high = _DecoderBranch(channels, specs[2], time_kernel, freq_kernel)

    def forward(
        self,
        z_low: torch.Tensor,
        z_mid: torch.Tensor,
        z_high: torch.Tensor,
        skip_low: torch.Tensor,
        skip_mid: torch.Tensor,
        skip_high: torch.Tensor,
    ) -> torch.Tensor:
        y_low = self.low(z_low, skip_low)
        y_mid = self.mid(z_mid, skip_mid)
        y_high = self.high(z_high, skip_high)
        return torch.cat([y_low, y_mid, y_high], dim=-1)

    def init_stream_state(
        self,
        batch_size: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        return (
            self.low.init_stream_state(batch_size, device=device, dtype=dtype),
            self.mid.init_stream_state(batch_size, device=device, dtype=dtype),
            self.high.init_stream_state(batch_size, device=device, dtype=dtype),
        )

    def forward_stream(
        self,
        z_low: torch.Tensor,
        z_mid: torch.Tensor,
        z_high: torch.Tensor,
        skip_low: torch.Tensor,
        skip_mid: torch.Tensor,
        skip_high: torch.Tensor,
        state: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    ]:
        y_low, new_low_state = self.low.forward_stream(z_low, skip_low, state[0])
        y_mid, new_mid_state = self.mid.forward_stream(z_mid, skip_mid, state[1])
        y_high, new_high_state = self.high.forward_stream(z_high, skip_high, state[2])
        out = torch.cat([y_low, y_mid, y_high], dim=-1)
        return out, (new_low_state, new_mid_state, new_high_state)
