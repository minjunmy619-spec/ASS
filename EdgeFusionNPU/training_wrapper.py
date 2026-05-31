"""Training integration for EdgeFusionNPU.

The deployable core is a single-frame mask estimator with one packed state
tensor. This wrapper adapts it to the repo's existing ``OnlineModelWrapper``:
complex STFT chunks or clips in, complex source estimates out.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    build_frequency_preprocessor,
    build_pcen_preprocessor,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_sfc_2d import (
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)

from .edge_fusion_npu import build_edge_fusion_npu_preset


class EdgeFusionNPUOnlineModel(nn.Module):
    """Run the single-frame NPU core over a complex STFT chunk or clip.

    Training uses this wrapper with ``x`` shaped ``[B, M, F, T]``. The wrapper
    loops over the ``T`` frames and carries the packed state internally, so
    gradients can flow through the whole chunk/clip. Export still targets the
    underlying ``core`` directly as a single-frame ``x, state -> mask,
    next_state`` graph.
    """

    def __init__(
        self,
        core: nn.Module,
        n_src: int,
        n_chan: int,
        *,
        chunk_frames: int | None = None,
        detach_state_between_chunks: bool = False,
    ):
        super().__init__()
        self.core = core
        self.n_src = n_src
        self.n_chan = n_chan
        if chunk_frames is not None and chunk_frames <= 0:
            raise ValueError(f"chunk_frames must be positive or None, got {chunk_frames}")
        self.chunk_frames = chunk_frames
        self.detach_state_between_chunks = detach_state_between_chunks

    def init_state(self, x: torch.Tensor) -> torch.Tensor:
        return self.core.init_states(batch_size=x.shape[0], device=x.real.device, dtype=x.real.dtype)

    def _forward_chunk(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, n_chan, n_freq, n_frames = x.shape
        packed = torch.cat([x.real, x.imag], dim=1)
        mask, state = self.core(packed, state)
        mask = mask.reshape(batch, self.n_src, self.n_chan, n_freq, n_frames)
        return x.unsqueeze(1) * mask, state

    def forward(
        self,
        x: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        return_state: bool = False,
        detach_state: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: complex STFT [B, M, F, T]
        batch, n_chan, n_freq, n_frames = x.shape
        if not torch.onnx.is_in_onnx_export():
            if n_chan != self.n_chan:
                raise ValueError(f"expected {self.n_chan} channels, got {n_chan}")
            if n_freq != self.core.n_freq:
                raise ValueError(f"expected {self.core.n_freq} frequency bins, got {n_freq}")
            if initial_state is not None:
                expected_state = (batch, self.core.state_channels, n_freq, self.core.context_size)
                if tuple(initial_state.shape) != expected_state:
                    raise ValueError(f"expected initial_state shape {expected_state}, got {tuple(initial_state.shape)}")

        state = (
            self.init_state(x)
            if initial_state is None
            else initial_state.to(device=x.real.device, dtype=x.real.dtype)
        )
        if detach_state:
            state = state.detach()

        if self.chunk_frames is None or n_frames <= self.chunk_frames:
            est, state = self._forward_chunk(x, state)
        else:
            outputs = []
            for frame_start in range(0, n_frames, self.chunk_frames):
                frame_end = min(frame_start + self.chunk_frames, n_frames)
                chunk, state = self._forward_chunk(x[..., frame_start:frame_end], state)
                outputs.append(chunk)
                if self.training and self.detach_state_between_chunks:
                    state = state.detach()
            est = torch.cat(outputs, dim=-1)

        if return_state:
            return est, state
        return est


class EdgeFusionNPU2DPackedCoreAdapter(nn.Module):
    """Expose EdgeFusionNPU as a packed-2D core for shared freq preprocessing."""

    def __init__(
        self,
        core: nn.Module,
        n_src: int,
        n_chan: int,
        *,
        chunk_frames: int | None = None,
        detach_state_between_chunks: bool = False,
    ):
        super().__init__()
        self.core = core
        self.n_src = n_src
        self.n_chan = n_chan
        self.n_freq = core.n_freq
        self.masking = True
        if chunk_frames is not None and chunk_frames <= 0:
            raise ValueError(f"chunk_frames must be positive or None, got {chunk_frames}")
        self.chunk_frames = chunk_frames
        self.detach_state_between_chunks = detach_state_between_chunks

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None):
        return self.core.init_states(batch_size=batch_size, device=device, dtype=dtype)

    def stream_context_frames(self) -> int:
        return int(self.core.context_size)

    def _apply_core_masks(self, x2d: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, frames, n_freq = x2d.shape
        if channels != 2 * self.n_chan:
            raise ValueError(f"expected {2 * self.n_chan} packed channels, got {channels}")
        if n_freq != self.n_freq:
            raise ValueError(f"expected {self.n_freq} frequency bins, got {n_freq}")

        packed = x2d.permute(0, 1, 3, 2).contiguous()
        mask, new_state = self.core(packed, state)
        if not self.masking:
            return mask.permute(0, 1, 3, 2).contiguous(), new_state
        x_complex = unpack_2d_to_complex_stft(x2d, n_src=1, n_chan=self.n_chan).squeeze(1)
        mask = mask.reshape(batch, self.n_src, self.n_chan, n_freq, frames)
        est = x_complex.unsqueeze(1) * mask
        return pack_complex_stft_as_2d(est.reshape(batch, self.n_src * self.n_chan, n_freq, frames)), new_state

    def forward(self, x2d: torch.Tensor, **kwargs) -> torch.Tensor:
        state = self.init_stream_state(batch_size=x2d.shape[0], device=x2d.device, dtype=x2d.dtype)
        n_frames = x2d.shape[2]
        if self.chunk_frames is None or n_frames <= self.chunk_frames:
            y2d, _ = self._apply_core_masks(x2d, state)
            return y2d

        outputs = []
        for frame_start in range(0, n_frames, self.chunk_frames):
            frame_end = min(frame_start + self.chunk_frames, n_frames)
            y_chunk, state = self._apply_core_masks(x2d[:, :, frame_start:frame_end, :], state)
            outputs.append(y_chunk)
            if self.training and self.detach_state_between_chunks:
                state = state.detach()
        y2d = torch.cat(outputs, dim=2)
        return y2d

    def forward_stream(self, x2d: torch.Tensor, state=None):
        if state is None:
            state = self.init_stream_state(batch_size=x2d.shape[0], device=x2d.device, dtype=x2d.dtype)
        return self._apply_core_masks(x2d, state)


def build_edge_fusion_npu_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    preset: str = "tiny",
    scaling: bool = False,
    freq_preprocess_enabled: bool = False,
    freq_preprocess_keep_bins: int | None = None,
    freq_preprocess_target_bins: int | None = None,
    freq_preprocess_mode: str = "triangular",
    dc_bypass_enabled: bool = False,
    dc_policy: str = "zero",
    pcen_preprocess_enabled: bool = False,
    pcen_smooth_coef: float = 0.98,
    pcen_alpha: float = 0.5,
    pcen_delta: float = 2.0,
    pcen_root: float = 0.5,
    pcen_eps: float = 1e-6,
    pcen_gain_floor: float = 0.05,
    pcen_gain_ceiling: float = 20.0,
    chunk_frames: int | None = None,
    detach_state_between_chunks: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
) -> OnlineModelWrapper:
    full_n_freq = (n_fft // 2) + 1
    core_n_freq = resolve_preprocessed_n_freq(
        full_n_freq,
        enabled=freq_preprocess_enabled,
        keep_bins=freq_preprocess_keep_bins,
        target_bins=freq_preprocess_target_bins,
        dc_bypass_enabled=dc_bypass_enabled,
    )
    freq_preprocessor = build_frequency_preprocessor(
        full_n_freq,
        enabled=freq_preprocess_enabled,
        keep_bins=freq_preprocess_keep_bins,
        target_bins=freq_preprocess_target_bins,
        mode=freq_preprocess_mode,
        dc_bypass_enabled=dc_bypass_enabled,
    )
    pcen_preprocessor = build_pcen_preprocessor(
        n_chan=n_chan,
        enabled=pcen_preprocess_enabled,
        smooth_coef=pcen_smooth_coef,
        alpha=pcen_alpha,
        delta=pcen_delta,
        root=pcen_root,
        eps=pcen_eps,
        gain_floor=pcen_gain_floor,
        gain_ceiling=pcen_gain_ceiling,
    )
    core = build_edge_fusion_npu_preset(preset, n_freq=core_n_freq, n_src=n_src, n_chan=n_chan)
    if freq_preprocessor is None and pcen_preprocessor is None and not dc_bypass_enabled:
        model = EdgeFusionNPUOnlineModel(
            core=core,
            n_src=n_src,
            n_chan=n_chan,
            chunk_frames=chunk_frames,
            detach_state_between_chunks=detach_state_between_chunks,
        )
    else:
        model = FrequencyPreprocessedOnlineModel(
            core=EdgeFusionNPU2DPackedCoreAdapter(
                core=core,
                n_src=n_src,
                n_chan=n_chan,
                chunk_frames=chunk_frames,
                detach_state_between_chunks=detach_state_between_chunks,
            ),
            n_src=n_src,
            n_chan=n_chan,
            freq_preprocessor=freq_preprocessor,
            pcen_preprocessor=pcen_preprocessor,
            dc_bypass_enabled=dc_bypass_enabled,
            dc_policy=dc_policy,
        )
    return OnlineModelWrapper(
        model=model,
        n_fft=n_fft,
        hop_length=hop_length,
        fs=fs,
        scaling=scaling,
        css_segment_size=css_segment_size,
        css_shift_size=css_shift_size,
        css_batch_size=css_batch_size,
    )
