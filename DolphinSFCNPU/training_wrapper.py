"""Training integration wrapper for DolphinSFCNPU.

Provides ``build_dolphin_sfc_npu_system`` which returns an
``OnlineModelWrapper`` compatible with the existing aiaccel + hydra
training pipeline (same interface as ``build_band_scnet_npu_system`` and
``build_online_sfc_system``).

The ``DolphinSFCNPUSeparator`` forward contract is identical to
``BandSCNetNPU`` (packed-real STFT ``[B, 2*n_chan, T, F]`` in, packed-real
STFT ``[B, 2*n_src*n_chan, T, F]`` out), so this wrapper mirrors the
``BandSCNetNPU.training_wrapper`` layout one-for-one.
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

from .dolphin_sfc import build_dolphin_sfc_npu_from_config


class DolphinSFCNPUOnlineModel(nn.Module):
    """Thin adapter: complex STFT in → complex STFT out.

    Matches the ``model`` interface expected by ``OnlineModelWrapper``:
    - Input: complex STFT ``(B, M, F, T)``
    - Output: complex STFT ``(B, N, M, F, T)``

    Internally delegates to ``DolphinSFCNPUSeparator.forward`` which operates
    on packed-real ``[B, 2*M, T, F]`` tensors and returns packed-real
    ``[B, 2*N*M, T, F]``.
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


def build_dolphin_sfc_npu_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    preset: str = "large_6m",
    band_config: str = "musical",
    mask_activation: str = "sigmoid",
    query_variant: str | None = None,
    query_type: str = "adaptive",
    n_bands: int | None = None,
    d_model: int | None = None,
    num_scales: int | None = None,
    widths: tuple[int, ...] | list[int] | None = None,
    blocks_per_scale: tuple[int, ...] | list[int] | None = None,
    time_kernels: tuple[int, ...] | list[int] | None = None,
    freq_kernels: tuple[int, ...] | list[int] | None = None,
    compressor_freq_kernel: int | None = None,
    ffn_expansion: int | None = None,
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
          _target_: DolphinSFCNPU.training_wrapper.build_dolphin_sfc_npu_system
          n_fft: ${n_fft}
          hop_length: ${hop_length}
          fs: ${sr}
          n_src: ${n_src}
          n_chan: ${n_chan}
          preset: large_6m
          # or preset: large_6m_soft_query / large_6m_crossattn_query
          # or query_variant: soft_band_query / crossattn_query
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
    core = build_dolphin_sfc_npu_from_config(
        preset=preset,
        n_freq=core_n_freq,
        n_fft=n_fft,
        sample_rate=fs,
        n_src=n_src,
        n_chan=n_chan,
        band_config=band_config,
        masking=True,
        mask_activation=mask_activation,
        query_variant=query_variant,
        query_type=query_type,
        n_bands=n_bands,
        d_model=d_model,
        num_scales=num_scales,
        widths=widths,
        blocks_per_scale=blocks_per_scale,
        time_kernels=time_kernels,
        freq_kernels=freq_kernels,
        compressor_freq_kernel=compressor_freq_kernel,
        ffn_expansion=ffn_expansion,
    )
    if freq_preprocessor is None and pcen_preprocessor is None and not dc_bypass_enabled:
        model = DolphinSFCNPUOnlineModel(core=core, n_src=n_src, n_chan=n_chan)
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
