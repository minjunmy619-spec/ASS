"""Training integration for BandSFCNetNPU."""

from __future__ import annotations

from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    build_frequency_preprocessor,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper

from .band_sfc_net_npu import BandSFCNetNPUModel
from .presets import build_band_sfc_net_npu_preset


def build_band_sfc_net_npu_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    preset: str = "safe",
    scaling: bool = False,
    freq_preprocess_enabled: bool = True,
    freq_preprocess_keep_bins: int | None = 475,
    freq_preprocess_target_bins: int | None = 512,
    freq_preprocess_mode: str = "triangular",
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
    )
    freq_preprocessor = build_frequency_preprocessor(
        full_n_freq,
        enabled=freq_preprocess_enabled,
        keep_bins=freq_preprocess_keep_bins,
        target_bins=freq_preprocess_target_bins,
        mode=freq_preprocess_mode,
    )
    core = build_band_sfc_net_npu_preset(
        preset,
        n_freq=core_n_freq,
        n_src=n_src,
        n_chan=n_chan,
    )
    model = BandSFCNetNPUModel(core)
    if freq_preprocessor is not None:
        model = FrequencyPreprocessedOnlineModel(
            core=core,
            n_src=n_src,
            n_chan=n_chan,
            freq_preprocessor=freq_preprocessor,
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
