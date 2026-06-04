"""Training integration for BandSFCNetNPU."""

from __future__ import annotations

from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    build_frequency_preprocessor,
    build_hybrid_frequency_bin_frequencies,
    build_pcen_preprocessor,
    resolve_frequency_input_n_freq,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper

from .band_sfc_net_npu import BandSFCNetNPUModel
from .presets import build_band_sfc_net_npu_from_config


def build_band_sfc_net_npu_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    core_n_src: int | None = None,
    preset: str = "safe",
    query_type: str = "adaptive",
    n_bands: int | None = None,
    channels: int | None = None,
    num_stages: int | None = None,
    time_kernel: int | None = None,
    freq_kernel: int | None = None,
    dilation_cycle: tuple[int, ...] | list[int] | None = None,
    transport: str | None = None,
    routing_normalization: str | None = None,
    use_attn: bool | None = None,
    attn_window: int | None = None,
    num_heads: int | None = None,
    head_dim: int | None = None,
    pooled_mixer_hidden: int | None = None,
    pooled_mixer_hidden_schedule: tuple[int, ...] | list[int] | None = None,
    encoder_capacity_mixer_hidden: int | None = None,
    encoder_capacity_mixer_layers: int | None = None,
    decoder_capacity_mixer_hidden: int | None = None,
    decoder_capacity_mixer_layers: int | None = None,
    stage_type: str | None = None,
    cnb_kernel: int | None = None,
    cnb_dilation_schedule: tuple[int, ...] | list[int] | None = None,
    band_spec_type: str | None = None,
    low_freq_hz: float | None = None,
    low_freq_band_fraction: float | None = None,
    overlap_factor: float | None = None,
    low_freq_overlap_factor: float | None = None,
    loco_expansion: int | None = None,
    loco_ffn_expansion: int | None = None,
    loco_time_kernel: int | None = None,
    loco_band_kernel: int | None = None,
    loco_time_dilation: int | None = None,
    bin_frequencies_hz=None,
    residual_head: bool | None = None,
    scaling: bool = False,
    freq_preprocess_enabled: bool = True,
    freq_preprocess_keep_bins: int | None = 475,
    freq_preprocess_target_bins: int | None = 512,
    freq_preprocess_mode: str = "triangular",
    band_config: str = "musical",
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
    residual_source_enabled: bool = False,
    residual_source_index: int | None = None,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
) -> OnlineModelWrapper:
    full_n_freq = (n_fft // 2) + 1
    body_n_freq = resolve_frequency_input_n_freq(full_n_freq, dc_bypass_enabled=dc_bypass_enabled)
    core_n_freq = resolve_preprocessed_n_freq(
        full_n_freq,
        enabled=freq_preprocess_enabled,
        keep_bins=freq_preprocess_keep_bins,
        target_bins=freq_preprocess_target_bins,
        dc_bypass_enabled=dc_bypass_enabled,
    )
    core_n_fft = 2 * (core_n_freq - 1)
    core_bin_frequencies_hz = bin_frequencies_hz
    if core_bin_frequencies_hz is None and freq_preprocess_enabled:
        if freq_preprocess_keep_bins is None or freq_preprocess_target_bins is None:
            raise ValueError("freq_preprocess_keep_bins and target_bins are required when preprocessing is enabled")
        core_bin_frequencies_hz = build_hybrid_frequency_bin_frequencies(
            body_n_freq,
            keep_bins=int(freq_preprocess_keep_bins),
            target_bins=int(freq_preprocess_target_bins),
            n_fft=n_fft,
            sample_rate=fs,
            mode=freq_preprocess_mode,
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
    explicit_n_src = (
        int(core_n_src) if core_n_src is not None else (int(n_src) - 1 if residual_source_enabled else int(n_src))
    )
    if residual_source_enabled and explicit_n_src != int(n_src) - 1:
        raise ValueError(f"residual_source_enabled expects core_n_src=n_src-1={int(n_src) - 1}, got {explicit_n_src}")
    if not residual_source_enabled and explicit_n_src != int(n_src):
        raise ValueError(f"core_n_src={explicit_n_src} requires residual_source_enabled=true when n_src={int(n_src)}")
    core = build_band_sfc_net_npu_from_config(
        preset=preset,
        n_freq=core_n_freq,
        n_fft=core_n_fft,
        sample_rate=fs,
        band_config=band_config,
        n_src=explicit_n_src,
        n_chan=n_chan,
        query_type=query_type,
        n_bands=n_bands,
        channels=channels,
        num_stages=num_stages,
        time_kernel=time_kernel,
        freq_kernel=freq_kernel,
        dilation_cycle=dilation_cycle,
        transport=transport,
        routing_normalization=routing_normalization,
        use_attn=use_attn,
        attn_window=attn_window,
        num_heads=num_heads,
        head_dim=head_dim,
        pooled_mixer_hidden=pooled_mixer_hidden,
        pooled_mixer_hidden_schedule=pooled_mixer_hidden_schedule,
        encoder_capacity_mixer_hidden=encoder_capacity_mixer_hidden,
        encoder_capacity_mixer_layers=encoder_capacity_mixer_layers,
        decoder_capacity_mixer_hidden=decoder_capacity_mixer_hidden,
        decoder_capacity_mixer_layers=decoder_capacity_mixer_layers,
        stage_type=stage_type,
        cnb_kernel=cnb_kernel,
        cnb_dilation_schedule=cnb_dilation_schedule,
        band_spec_type=band_spec_type,
        low_freq_hz=low_freq_hz,
        low_freq_band_fraction=low_freq_band_fraction,
        overlap_factor=overlap_factor,
        low_freq_overlap_factor=low_freq_overlap_factor,
        loco_expansion=loco_expansion,
        loco_ffn_expansion=loco_ffn_expansion,
        loco_time_kernel=loco_time_kernel,
        loco_band_kernel=loco_band_kernel,
        loco_time_dilation=loco_time_dilation,
        bin_frequencies_hz=core_bin_frequencies_hz,
        residual_head=residual_head,
    )
    model = BandSFCNetNPUModel(core)
    if freq_preprocessor is not None or pcen_preprocessor is not None or dc_bypass_enabled or residual_source_enabled:
        model = FrequencyPreprocessedOnlineModel(
            core=core,
            n_src=n_src,
            n_chan=n_chan,
            freq_preprocessor=freq_preprocessor,
            pcen_preprocessor=pcen_preprocessor,
            dc_bypass_enabled=dc_bypass_enabled,
            dc_policy=dc_policy,
            residual_source_enabled=residual_source_enabled,
            residual_source_index=residual_source_index,
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
