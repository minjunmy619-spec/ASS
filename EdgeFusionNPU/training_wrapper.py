"""Training integration for EdgeFusionNPU.

The deployable core is a single-frame mask estimator with one packed state
tensor. This wrapper adapts it to the repo's existing ``OnlineModelWrapper``:
complex STFT in, complex source estimates out.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper

from .edge_fusion_npu import build_edge_fusion_npu_preset


class EdgeFusionNPUOnlineModel(nn.Module):
    """Run the single-frame NPU core over a complex STFT sequence."""

    def __init__(self, core: nn.Module, n_src: int, n_chan: int):
        super().__init__()
        self.core = core
        self.n_src = n_src
        self.n_chan = n_chan

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: complex STFT [B, M, F, T]
        batch, n_chan, n_freq, n_frames = x.shape
        if not torch.onnx.is_in_onnx_export():
            if n_chan != self.n_chan:
                raise ValueError(f"expected {self.n_chan} channels, got {n_chan}")
            if n_freq != self.core.n_freq:
                raise ValueError(f"expected {self.core.n_freq} frequency bins, got {n_freq}")

        state = self.core.init_states(batch_size=batch, device=x.real.device, dtype=x.real.dtype)
        estimates: list[torch.Tensor] = []
        for frame_idx in range(n_frames):
            frame = x[:, :, :, frame_idx : frame_idx + 1]
            packed = torch.cat([frame.real, frame.imag], dim=1)
            mask, state = self.core(packed, state)
            mask = mask.reshape(batch, self.n_src, self.n_chan, n_freq, 1)
            estimates.append(frame.unsqueeze(1) * mask)
        return torch.cat(estimates, dim=-1)


def build_edge_fusion_npu_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    preset: str = "tiny",
    scaling: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
) -> OnlineModelWrapper:
    n_freq = (n_fft // 2) + 1
    core = build_edge_fusion_npu_preset(preset, n_freq=n_freq, n_src=n_src, n_chan=n_chan)
    model = EdgeFusionNPUOnlineModel(core=core, n_src=n_src, n_chan=n_chan)
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
