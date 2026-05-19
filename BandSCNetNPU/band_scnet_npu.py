"""BandSCNetNPU — NPU-native 3-stem audio source separator.

See ``.kiro/specs/band-scnet-npu/design.md`` for the full design rationale.

Input/output contract (matches the rest of the online model family):
- input  x: ``[B, 2*n_chan, T, F]`` packed real/imag STFT
- output y: ``[B, n_src*n_chan, T, F]`` real-valued source-gain masks
  (passed through sigmoid), applied outside this module to the complex STFT
  by the host/DSP streaming runtime.

STFT / iSTFT are NOT part of this module (see spec FR-5a).
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from spectral_feature_compression.core.model.online_sfc_2d import (
    RMSNorm2d,
    _runtime_assert,
)

from .blocks import CrossBandBlock, NarrowBandBlock, PooledChannelMixer
from .sparse_io import (
    SparseDownsampleEncoder,
    SparseUpsampleDecoder,
    pad_n_freq_for_split,
    split_bands,
)

#Replace the nn.PReLU with a custom subgraph that is compatible with the NPU
class PReluSubgraph(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha.view(1, -1, 1, 1).to(dtype=x.dtype)
        return torch.relu(x) + a * torch.minimum(x, torch.zeros_like(x))


class _SeparationStage(nn.Module):
    """CrossBand then NarrowBand pair."""

    def __init__(
        self,
        channels: int,
        *,
        time_kernel: int,
        freq_kernel: int,
        use_attn: bool,
        attn_window: int,
        num_heads: int,
        head_dim: int,
        pooled_mixer_hidden: int = 0,
    ):
        super().__init__()
        self.cross = CrossBandBlock(channels, freq_kernel=freq_kernel)
        self.narrow = NarrowBandBlock(
            channels,
            time_kernel=time_kernel,
            use_attn=use_attn,
            attn_window=attn_window,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        self.pooled_mixer = (
            PooledChannelMixer(channels, pooled_mixer_hidden)
            if pooled_mixer_hidden > 0
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cross(x)
        x = self.narrow(x)
        x = self.pooled_mixer(x)
        return x

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
        x, new_state = self.narrow.forward_stream(x, state)
        x = self.pooled_mixer(x)
        return x, new_state


class _SourceMaskHead(nn.Module):
    """Final head: Conv2d -> PReLU -> Conv2d producing ``n_src*n_chan`` gain logits.

    The exported mask is real-valued; the streaming runtime converts it to a
    complex-valued multiplicative mask by duplicating the real gain onto both
    real and imag STFT channels (see ``apply_source_gain_mask_4d``).
    """

    def __init__(self, channels: int, *, n_src: int, n_chan: int):
        super().__init__()
        out_ch = n_src * n_chan
        self.norm = RMSNorm2d(channels)
        self.hidden = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act = nn.PReLU(num_parameters=channels)
        self.proj = nn.Conv2d(channels, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.hidden(y)
        y = self.act(y)
        return self.proj(y)


def apply_source_gain_mask_4d(
    x: torch.Tensor,
    mask_logits: torch.Tensor,
    *,
    n_src: int,
    n_chan: int,
) -> torch.Tensor:
    """Apply real-valued source gains to packed complex input, staying 4D.

    Matches the output layout of the rest of the online model family
    (``[B, 2*n_src*n_chan, T, F]``), so downstream STFT-unpack logic can be
    shared with Dolphin / Online SFC.
    """
    _runtime_assert(x.shape[1] == 2 * n_chan, f"{x.shape[1]} vs {2 * n_chan}")
    _runtime_assert(mask_logits.shape[1] == n_src * n_chan, f"{mask_logits.shape[1]} vs {n_src * n_chan}")
    gains = torch.sigmoid(mask_logits)
    outputs: list[torch.Tensor] = []
    for src_idx in range(n_src):
        for chan_idx in range(n_chan):
            gain = gains[:, src_idx * n_chan + chan_idx : src_idx * n_chan + chan_idx + 1, :, :]
            real = x[:, 2 * chan_idx : 2 * chan_idx + 1, :, :] * gain
            imag = x[:, 2 * chan_idx + 1 : 2 * chan_idx + 2, :, :] * gain
            outputs.extend([real, imag])
    return torch.cat(outputs, dim=1)


class BandSCNetNPUState(NamedTuple):
    sd: tuple  # encoder per-branch states
    sep: tuple[tuple[torch.Tensor, ...], ...]  # per-stage narrow-band states
    su: tuple  # decoder per-branch states


class BandSCNetNPU(nn.Module):
    """NPU-compatible Band-SCNet variant.

    Args:
        n_freq: number of STFT frequency bins (= n_fft//2 + 1)
        n_src: number of output stems (3 for Speech/Music/Effects)
        n_chan: number of input channels (1 for mono)
        channels: base channel width C
        num_stages: L in the design doc (number of CrossBand+NarrowBand pairs)
        time_kernel: Kt for causal depthwise time conv
        freq_kernel: Kf for cross-band freq conv
        use_attn: toggle bounded causal attention inside every NarrowBandBlock
        attn_window, num_heads, head_dim: attention config (used if use_attn)
        ratios: frequency band split ratios (low / mid / high)
        masking: wrap the output in ``apply_source_gain_mask_4d`` so the final
            shape is ``[B, 2*n_src*n_chan, T, F]``. When False the raw logits
            ``[B, n_src*n_chan, T, F]`` are returned for training losses.
    """

    def __init__(
        self,
        n_freq: int,
        *,
        n_src: int = 3,
        n_chan: int = 1,
        channels: int = 48,
        pyramid_channels: int | None = None,
        num_stages: int = 4,
        time_kernel: int = 5,
        freq_kernel: int = 3,
        pyramid_time_kernel: int | None = None,
        pyramid_freq_kernel: int | None = None,
        use_attn: bool = True,
        attn_window: int = 16,
        num_heads: int = 4,
        head_dim: int = 8,
        pooled_mixer_hidden: int = 0,
        ratios: tuple[float, float, float] = (0.175, 0.392, 0.433),
        pyramid_conv_blocks: tuple[int, int, int] = (3, 2, 1),
        pyramid_strides: tuple[int, int, int] = (0, 2, 4),
        masking: bool = True,
    ):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(f"channels must be even, got {channels}")
        if num_stages <= 0:
            raise ValueError(f"num_stages must be positive, got {num_stages}")
        if pyramid_channels is None:
            pyramid_channels = channels
        if pyramid_channels % 2 != 0:
            raise ValueError(f"pyramid_channels must be even, got {pyramid_channels}")
        if pyramid_time_kernel is None:
            pyramid_time_kernel = time_kernel
        if pyramid_freq_kernel is None:
            pyramid_freq_kernel = freq_kernel

        self.n_freq = n_freq
        # Some STFT sizes (e.g. 257, 2049) are not cleanly divisible by the
        # low/mid/high band multiples. Zero-pad the frequency axis up to the
        # nearest compatible width; crop the tail again after the decoder.
        self.n_freq_padded = pad_n_freq_for_split(
            n_freq,
            ratios=ratios,
        )
        self.n_freq_pad = self.n_freq_padded - n_freq
        self.n_src = n_src
        self.n_chan = n_chan
        self.channels = channels
        self.pyramid_channels = pyramid_channels
        self.num_stages = num_stages
        self.time_kernel = time_kernel
        self.freq_kernel = freq_kernel
        self.pyramid_time_kernel = pyramid_time_kernel
        self.pyramid_freq_kernel = pyramid_freq_kernel
        self.use_attn = use_attn
        self.attn_window = attn_window
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pooled_mixer_hidden = pooled_mixer_hidden
        self.ratios = ratios
        self.masking = masking

        in_ch = 2 * n_chan

        self.encoder = SparseDownsampleEncoder(
            n_freq=self.n_freq_padded,
            in_channels=in_ch,
            channels=pyramid_channels,
            time_kernel=pyramid_time_kernel,
            freq_kernel=pyramid_freq_kernel,
            ratios=ratios,
            conv_blocks_per_branch=pyramid_conv_blocks,
            strides_per_branch=pyramid_strides,
        )
        self.bands = self.encoder.bands
        self.out_widths = self.encoder.out_widths
        self.concat_width = self.encoder.concat_width

        # Channel adapter: pyramid -> separator
        self.pre_sep_proj = (
            nn.Conv2d(pyramid_channels, channels, kernel_size=1, bias=True)
            if pyramid_channels != channels
            else nn.Identity()
        )
        self.post_sep_proj = (
            nn.Conv2d(channels, pyramid_channels, kernel_size=1, bias=True)
            if pyramid_channels != channels
            else nn.Identity()
        )

        self.stages = nn.ModuleList(
            _SeparationStage(
                channels,
                time_kernel=time_kernel,
                freq_kernel=freq_kernel,
                use_attn=use_attn,
                attn_window=attn_window,
                num_heads=num_heads,
                head_dim=head_dim,
                pooled_mixer_hidden=pooled_mixer_hidden,
            )
            for _ in range(num_stages)
        )

        self.decoder = SparseUpsampleDecoder(
            n_freq=self.n_freq_padded,
            channels=pyramid_channels,
            time_kernel=pyramid_time_kernel,
            freq_kernel=pyramid_freq_kernel,
            ratios=ratios,
            conv_blocks_per_branch=pyramid_conv_blocks,
            strides_per_branch=pyramid_strides,
        )

        self.head = _SourceMaskHead(pyramid_channels, n_src=n_src, n_chan=n_chan)

    # -- helpers for cross-path split/concat --------------------------------

    def _split_sep_output(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w_low, w_mid, w_high = self.out_widths
        z_low = z[..., :w_low]
        z_mid = z[..., w_low : w_low + w_mid]
        z_high = z[..., w_low + w_mid :]
        return z_low, z_mid, z_high

    # -- full-sequence forward -----------------------------------------------

    def _pad_freq(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_freq_pad == 0:
            return x
        # pad the F axis with zeros on the right
        pad = x.new_zeros(x.shape[0], x.shape[1], x.shape[2], self.n_freq_pad)
        return torch.cat([x, pad], dim=-1)

    def _crop_freq(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_freq_pad == 0:
            return x
        return x[..., : self.n_freq]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {tuple(x.shape)}")
        _runtime_assert(x.shape[1] == 2 * self.n_chan, f"{x.shape[1]} vs {2 * self.n_chan}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape[-1]} vs {self.n_freq}")

        x_in = x
        x_pad = self._pad_freq(x)

        z_low_enc, z_mid_enc, z_high_enc = self.encoder(x_pad)
        z = torch.cat([z_low_enc, z_mid_enc, z_high_enc], dim=-1)
        z = self.pre_sep_proj(z)

        for stage in self.stages:
            z = stage(z)

        z = self.post_sep_proj(z)
        z_low, z_mid, z_high = self._split_sep_output(z)
        y = self.decoder(
            z_low,
            z_mid,
            z_high,
            z_low_enc,
            z_mid_enc,
            z_high_enc,
        )
        y = self._crop_freq(y)
        logits = self.head(y)
        if self.masking:
            return apply_source_gain_mask_4d(x_in, logits, n_src=self.n_src, n_chan=self.n_chan)
        return logits

    # -- streaming API -------------------------------------------------------

    def init_stream_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        sd_state = self.encoder.init_stream_state(batch_size, device=device, dtype=dtype)
        sep_states = tuple(
            stage.init_stream_state(batch_size, freq_bins=self.concat_width, device=device, dtype=dtype)
            for stage in self.stages
        )
        su_state = self.decoder.init_stream_state(batch_size, device=device, dtype=dtype)
        return BandSCNetNPUState(sd=sd_state, sep=sep_states, su=su_state)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: BandSCNetNPUState | tuple | None = None,
    ) -> tuple[torch.Tensor, BandSCNetNPUState]:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        _runtime_assert(x.shape[2] == 1, f"forward_stream expects T=1, got T={x.shape[2]}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape[-1]} vs {self.n_freq}")

        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)
        if not isinstance(state, BandSCNetNPUState):
            state = BandSCNetNPUState(*state)

        x_in = x
        x_pad = self._pad_freq(x)

        (z_low_enc, z_mid_enc, z_high_enc), new_sd_state = self.encoder.forward_stream(x_pad, state.sd)
        z = torch.cat([z_low_enc, z_mid_enc, z_high_enc], dim=-1)
        z = self.pre_sep_proj(z)

        new_sep_states: list[tuple[torch.Tensor, ...]] = []
        for stage, stage_state in zip(self.stages, state.sep):
            z, new_stage_state = stage.forward_stream(z, stage_state)
            new_sep_states.append(new_stage_state)

        z = self.post_sep_proj(z)
        z_low, z_mid, z_high = self._split_sep_output(z)
        y, new_su_state = self.decoder.forward_stream(
            z_low,
            z_mid,
            z_high,
            z_low_enc,
            z_mid_enc,
            z_high_enc,
            state.su,
        )
        y = self._crop_freq(y)
        logits = self.head(y)
        if self.masking:
            out = apply_source_gain_mask_4d(x_in, logits, n_src=self.n_src, n_chan=self.n_chan)
        else:
            out = logits
        return out, BandSCNetNPUState(sd=new_sd_state, sep=tuple(new_sep_states), su=new_su_state)

    # -- state size --------------------------------------------------------

    def state_size_bytes(
        self,
        *,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ) -> int:
        state = self.init_stream_state(batch_size=batch_size, dtype=dtype)
        return _tree_numel(state) * torch.tensor([], dtype=dtype).element_size()


# --- helpers ---------------------------------------------------------------


def _tree_numel(tree) -> int:
    if isinstance(tree, torch.Tensor):
        return int(tree.numel())
    if isinstance(tree, BandSCNetNPUState):
        return (
            _tree_numel(tree.sd)
            + _tree_numel(tree.sep)
            + _tree_numel(tree.su)
        )
    return sum(_tree_numel(item) for item in tree)


# --- ONNX streaming export wrapper ----------------------------------------


class BandSCNetNPUStreamingExportWrapper(nn.Module):
    """Flattens/unflattens the streaming state for ``torch.onnx.export``."""

    def __init__(
        self,
        core: BandSCNetNPU,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        from spectral_feature_compression.utils.onnx_streaming import flatten_tensor_tree

        self.core = core
        example_state = core.init_stream_state(batch_size=batch_size, dtype=dtype)
        flat_state, state_spec = flatten_tensor_tree(tuple(example_state))
        self.state_spec = state_spec
        self.state_tensor_count = len(flat_state)

    def forward(self, x: torch.Tensor, *flat_state: torch.Tensor):
        from spectral_feature_compression.utils.onnx_streaming import (
            flatten_tensor_tree,
            unflatten_tensor_tree,
        )

        state_tuple = unflatten_tensor_tree(flat_state, self.state_spec)
        state = BandSCNetNPUState(*state_tuple)
        y, new_state = self.core.forward_stream(x, state)
        flat_new_state, _ = flatten_tensor_tree(tuple(new_state))
        return (y, *flat_new_state)
