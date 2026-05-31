"""Frequency-preprocessed wrapper for Band-SCNet-NPU.

This is the Phase 12.2 follow-up: wraps the BandSCNetNPU core in a
``FrequencyPreprocessedOnlineModel`` that keeps the first ``keep_bins``
low-frequency bins intact and projects the remaining high-frequency bins
into a smaller representation (``target_bins - keep_bins`` slots).

The core model then sees ``target_bins`` instead of the full ``n_fft/2+1``,
which can significantly reduce compute and streaming state if n_fft is
large (e.g. 4096 → 2049 bins). After the core runs, the inverse projection
reconstructs the full-resolution output.

Usage::

    from BandSCNetNPU.freq_preprocessed import build_freq_preprocessed_band_scnet_npu_system

    system = build_freq_preprocessed_band_scnet_npu_system(
        n_fft=4096,
        hop_length=1024,
        fs=44100,
        preset="rt192k_plus",
        freq_preprocess_keep_bins=512,
        freq_preprocess_target_bins=768,
    )
"""
from __future__ import annotations

from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    build_frequency_preprocessor,
    build_pcen_preprocessor,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper

from .presets import build_band_scnet_npu_preset


def build_freq_preprocessed_band_scnet_npu_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    preset: str = "rt192k_plus",
    freq_preprocess_keep_bins: int = 512,
    freq_preprocess_target_bins: int = 768,
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
    scaling: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
) -> OnlineModelWrapper:
    """Build a Band-SCNet-NPU system with frequency pre/post-processing.

    This is the v2 variant described in the spec (Phase 12.2). The core model
    operates on a reduced frequency representation, saving compute and state.

    Args:
        n_fft: STFT window size (n_freq = n_fft//2 + 1)
        hop_length: STFT hop
        fs: sample rate
        n_src: number of output stems
        n_chan: number of input channels
        preset: BandSCNetNPU preset name (edge_small, rt192k, rt192k_plus)
        freq_preprocess_keep_bins: number of low-frequency bins to keep intact
        freq_preprocess_target_bins: total bins the core sees (keep + projected)
        freq_preprocess_mode: projection mode ("triangular" or "avg")
        scaling: global waveform scaling (must be False for strict causal)
        css_segment_size: chunk-wise separation segment size (seconds)
        css_shift_size: chunk-wise separation shift (seconds)
        css_batch_size: chunk-wise separation batch size

    Returns:
        OnlineModelWrapper ready for training via aiaccel + hydra.
    """
    full_n_freq = (n_fft // 2) + 1
    core_n_freq = resolve_preprocessed_n_freq(
        full_n_freq,
        enabled=True,
        keep_bins=freq_preprocess_keep_bins,
        target_bins=freq_preprocess_target_bins,
        dc_bypass_enabled=dc_bypass_enabled,
    )
    freq_preprocessor = build_frequency_preprocessor(
        full_n_freq,
        enabled=True,
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
    core = build_band_scnet_npu_preset(
        preset,
        n_freq=core_n_freq,
        n_src=n_src,
        n_chan=n_chan,
    )
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
