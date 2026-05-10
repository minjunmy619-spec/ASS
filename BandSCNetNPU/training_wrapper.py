"""Training integration wrapper for BandSCNetNPU.

Provides ``build_band_scnet_npu_system`` which returns an
``OnlineModelWrapper`` compatible with the existing aiaccel + hydra
training pipeline (same interface as ``build_online_sfc_system`` and friends).
"""
from __future__ import annotations

import torch.nn as nn

from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_sfc_2d import (
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)

from .presets import build_band_scnet_npu_preset


class BandSCNetNPUOnlineModel(nn.Module):
    """Thin adapter: complex STFT in → complex STFT out.

    Matches the ``model`` interface expected by ``OnlineModelWrapper``:
    - Input: complex STFT ``(B, M, F, T)``
    - Output: complex STFT ``(B, N, M, F, T)``

    Internally delegates to ``BandSCNetNPU.forward`` which operates on
    packed-real ``[B, 2M, T, F]`` tensors.
    """

    def __init__(self, core: nn.Module, n_src: int, n_chan: int):
        super().__init__()
        self.core = core
        self.n_src = n_src
        self.n_chan = n_chan

    def forward(self, x, **kwargs):
        # x: complex (B, M, F, T)
        x2d = pack_complex_stft_as_2d(x)  # (B, 2*M, T, F)
        y2d = self.core(x2d)
        return unpack_2d_to_complex_stft(y2d, n_src=self.n_src, n_chan=self.n_chan)


def build_band_scnet_npu_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    preset: str = "rt192k",
    scaling: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
) -> OnlineModelWrapper:
    """Build a drop-in ``OnlineModelWrapper`` system for training on DnR.

    Usage in hydra config::

        model:
          _target_: BandSCNetNPU.training_wrapper.build_band_scnet_npu_system
          n_fft: ${n_fft}
          hop_length: ${hop_length}
          fs: ${sr}
          n_src: ${n_src}
          n_chan: ${n_chan}
          preset: rt192k
    """
    n_freq = (n_fft // 2) + 1
    core = build_band_scnet_npu_preset(preset, n_freq=n_freq, n_src=n_src, n_chan=n_chan)
    model = BandSCNetNPUOnlineModel(core=core, n_src=n_src, n_chan=n_chan)
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
