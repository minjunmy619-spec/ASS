"""Training integration wrapper for BandSCNetNPU.

Provides ``build_band_scnet_npu_system`` which returns an
``OnlineModelWrapper`` compatible with the existing aiaccel + hydra
training pipeline (same interface as ``build_online_sfc_system`` and friends).
"""
from __future__ import annotations

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
    core = build_band_scnet_npu_preset(preset, n_freq=core_n_freq, n_src=n_src, n_chan=n_chan)
    if freq_preprocessor is None and pcen_preprocessor is None and not dc_bypass_enabled:
        model = BandSCNetNPUOnlineModel(core=core, n_src=n_src, n_chan=n_chan)
    else:
        model = FrequencyPreprocessedOnlineModel(
            core=core,
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
