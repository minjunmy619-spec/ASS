"""Preset factories for BandSFCNetNPU."""

from __future__ import annotations

from copy import deepcopy

from .band_sfc_net_npu import BandSFCNetNPU


def _build_preset(
    *,
    n_freq: int,
    n_fft: int | None,
    sample_rate: int | None,
    band_config: str,
    n_src: int,
    n_chan: int,
    masking: bool,
    n_bands: int,
    channels: int,
    num_stages: int,
    dilation_cycle: tuple[int, ...],
    transport: str,
    query_type: str,
    routing_normalization: str = "softmax",
    use_attn: bool = False,
    attn_window: int = 16,
    num_heads: int = 4,
    head_dim: int = 8,
    pooled_mixer_hidden: int = 0,
    pooled_mixer_hidden_schedule: tuple[int, ...] | None = None,
    time_kernel: int = 3,
    freq_kernel: int = 3,
    stage_type: str = "band_sfc",
    cnb_kernel: int = 5,
    cnb_dilation_schedule: tuple[int, ...] | None = None,
    residual_head: bool = False,
) -> BandSFCNetNPU:
    return BandSFCNetNPU(
        n_freq=n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_bands=n_bands,
        n_src=n_src,
        n_chan=n_chan,
        channels=channels,
        num_stages=num_stages,
        time_kernel=time_kernel,
        freq_kernel=freq_kernel,
        dilation_cycle=dilation_cycle,
        transport=transport,
        query_type=query_type,
        routing_normalization=routing_normalization,
        use_attn=use_attn,
        attn_window=attn_window,
        num_heads=num_heads,
        head_dim=head_dim,
        pooled_mixer_hidden=pooled_mixer_hidden,
        pooled_mixer_hidden_schedule=pooled_mixer_hidden_schedule,
        stage_type=stage_type,
        cnb_kernel=cnb_kernel,
        cnb_dilation_schedule=cnb_dilation_schedule,
        masking=masking,
        residual_head=residual_head,
    )


def _safe_shape(
    n_freq: int,
    *,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    transport: str = "soft",
    query_type: str = "adaptive",
) -> BandSFCNetNPU:
    return _build_preset(
        n_freq=n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_bands=64,
        n_src=n_src,
        n_chan=n_chan,
        channels=32,
        num_stages=4,
        dilation_cycle=(1, 1, 2, 4),
        transport=transport,
        query_type=query_type,
        use_attn=False,
        pooled_mixer_hidden=1024,
        masking=masking,
    )


def _quality_shape(
    n_freq: int,
    *,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    transport: str = "crossattn_query",
    query_type: str = "adaptive",
    residual_head: bool = False,
) -> BandSFCNetNPU:
    return _build_preset(
        n_freq=n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_bands=64,
        n_src=n_src,
        n_chan=n_chan,
        channels=32,
        num_stages=5,
        dilation_cycle=(1, 1, 2, 4, 6),
        transport=transport,
        query_type=query_type,
        use_attn=True,
        pooled_mixer_hidden=4096,
        masking=masking,
        residual_head=residual_head,
    )


def _quality6m_shape(
    n_freq: int,
    *,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    transport: str = "crossattn_query",
    query_type: str = "adaptive",
) -> BandSFCNetNPU:
    return _build_preset(
        n_freq=n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_bands=64,
        n_src=n_src,
        n_chan=n_chan,
        channels=40,
        num_stages=4,
        dilation_cycle=(1, 2, 4, 6),
        transport=transport,
        query_type=query_type,
        use_attn=True,
        pooled_mixer_hidden=18432,
        masking=masking,
    )


def safe(
    n_freq: int,
    *,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    query_type: str = "adaptive",
) -> BandSFCNetNPU:
    """Deployment-first baseline: soft SFC transport plus BandSCNet stages."""
    return _safe_shape(
        n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        transport="soft",
        query_type=query_type,
    )


def safe_soft_query(
    n_freq: int,
    *,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    query_type: str = "adaptive",
) -> BandSFCNetNPU:
    """Safe-shape variant with explicit K-band soft-query SFC transport."""
    return _safe_shape(
        n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        transport="soft_band_query",
        query_type=query_type,
    )


def safe_crossattn_query(
    n_freq: int,
    *,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    query_type: str = "adaptive",
) -> BandSFCNetNPU:
    """Safe-shape variant with explicit cross-attention-query SFC transport."""
    return _safe_shape(
        n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        transport="crossattn_query",
        query_type=query_type,
    )


def quality(
    n_freq: int,
    *,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    query_type: str = "adaptive",
) -> BandSFCNetNPU:
    """Main quality candidate with cross-attention query transport and pooled attention."""
    return _quality_shape(
        n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        transport="crossattn_query",
        query_type=query_type,
    )


def quality_soft_query(
    n_freq: int,
    *,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    query_type: str = "adaptive",
) -> BandSFCNetNPU:
    """Quality-shape variant with explicit K-band soft-query SFC transport."""
    return _quality_shape(
        n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        transport="soft_band_query",
        query_type=query_type,
    )


def quality_crossattn_query(
    n_freq: int,
    *,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    query_type: str = "adaptive",
) -> BandSFCNetNPU:
    """Quality-shape explicit alias for the cross-attention-query SFC transport."""
    return quality(
        n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        query_type=query_type,
    )


def quality6m(
    n_freq: int,
    *,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    query_type: str = "adaptive",
) -> BandSFCNetNPU:
    """Large parameter probe; validate state and compiler budget before deployment."""
    return _quality6m_shape(
        n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        transport="crossattn_query",
        query_type=query_type,
    )


def quality6m_soft_query(
    n_freq: int,
    *,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    query_type: str = "adaptive",
) -> BandSFCNetNPU:
    """Large quality probe with explicit K-band soft-query SFC transport."""
    return _quality6m_shape(
        n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        transport="soft_band_query",
        query_type=query_type,
    )


def quality6m_crossattn_query(
    n_freq: int,
    *,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    query_type: str = "adaptive",
) -> BandSFCNetNPU:
    """Large quality probe explicit alias for cross-attention-query SFC transport."""
    return quality6m(
        n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        query_type=query_type,
    )


def rt_plus(
    n_freq: int,
    *,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    query_type: str = "adaptive",
) -> BandSFCNetNPU:
    """Research proposal B: quality preset plus complex residual correction."""
    return _quality_shape(
        n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        transport="crossattn_query",
        query_type=query_type,
        residual_head=True,
    )


def rt_plus_soft_query(
    n_freq: int,
    *,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    query_type: str = "adaptive",
) -> BandSFCNetNPU:
    """RT+ residual variant with explicit K-band soft-query SFC transport."""
    return _quality_shape(
        n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        transport="soft_band_query",
        query_type=query_type,
        residual_head=True,
    )


def rt_plus_crossattn_query(
    n_freq: int,
    *,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    query_type: str = "adaptive",
) -> BandSFCNetNPU:
    """RT+ residual variant explicit alias for cross-attention-query SFC transport."""
    return rt_plus(
        n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        query_type=query_type,
    )


_PRESETS = {
    "safe": safe,
    "safe_soft_query": safe_soft_query,
    "safe_soft_band_query": safe_soft_query,
    "safe_crossattn_query": safe_crossattn_query,
    "balanced_soft_query": None,
    "balanced_soft_band_query": None,
    "balanced_crossattn_query": None,
    "quality": quality,
    "quality_soft_query": quality_soft_query,
    "quality_soft_band_query": quality_soft_query,
    "quality_crossattn_query": quality_crossattn_query,
    "quality6m": quality6m,
    "quality6m_soft_query": quality6m_soft_query,
    "quality6m_soft_band_query": quality6m_soft_query,
    "quality6m_crossattn_query": quality6m_crossattn_query,
    "rt_plus": rt_plus,
    "rt_plus_soft_query": rt_plus_soft_query,
    "rt_plus_soft_band_query": rt_plus_soft_query,
    "rt_plus_crossattn_query": rt_plus_crossattn_query,
    "causal_cnb_soft_query": None,
    "causal_cnb_soft_band_query": None,
    "causal_cnb_crossattn_query": None,
}

_PRESET_CONFIGS: dict[str, dict[str, object]] = {
    "safe": dict(
        n_bands=64,
        channels=32,
        num_stages=4,
        dilation_cycle=(1, 1, 2, 4),
        transport="soft",
        use_attn=False,
        pooled_mixer_hidden=1024,
        residual_head=False,
    ),
    "safe_soft_query": dict(
        n_bands=64,
        channels=32,
        num_stages=4,
        dilation_cycle=(1, 1, 2, 4),
        transport="soft_band_query",
        use_attn=False,
        pooled_mixer_hidden=1024,
        residual_head=False,
    ),
    "safe_crossattn_query": dict(
        n_bands=64,
        channels=32,
        num_stages=4,
        dilation_cycle=(1, 1, 2, 4),
        transport="crossattn_query",
        use_attn=False,
        pooled_mixer_hidden=1024,
        residual_head=False,
    ),
    # Balanced query presets are intended as the first useful-capacity NPU
    # targets: 40 latent channels improve all local/cross-band blocks, and the
    # large pooled mixers add millions of parameters at frequency width 1, so
    # capacity increases without increasing streaming state.
    "balanced_soft_query": dict(
        n_bands=64,
        channels=40,
        num_stages=4,
        dilation_cycle=(1, 1, 2, 4),
        transport="soft_band_query",
        use_attn=False,
        pooled_mixer_hidden=8192,
        residual_head=False,
    ),
    "balanced_crossattn_query": dict(
        n_bands=64,
        channels=40,
        num_stages=4,
        dilation_cycle=(1, 1, 2, 4),
        transport="crossattn_query",
        use_attn=False,
        pooled_mixer_hidden=8192,
        residual_head=False,
    ),
    "quality": dict(
        n_bands=64,
        channels=32,
        num_stages=5,
        dilation_cycle=(1, 1, 2, 4, 6),
        transport="crossattn_query",
        use_attn=True,
        pooled_mixer_hidden=4096,
        residual_head=False,
    ),
    "quality_soft_query": dict(
        n_bands=64,
        channels=32,
        num_stages=5,
        dilation_cycle=(1, 1, 2, 4, 6),
        transport="soft_band_query",
        use_attn=True,
        pooled_mixer_hidden=4096,
        residual_head=False,
    ),
    "quality6m": dict(
        n_bands=64,
        channels=40,
        num_stages=4,
        dilation_cycle=(1, 2, 4, 6),
        transport="crossattn_query",
        use_attn=True,
        pooled_mixer_hidden=18432,
        residual_head=False,
    ),
    "quality6m_soft_query": dict(
        n_bands=64,
        channels=40,
        num_stages=4,
        dilation_cycle=(1, 2, 4, 6),
        transport="soft_band_query",
        use_attn=True,
        pooled_mixer_hidden=18432,
        residual_head=False,
    ),
    "rt_plus": dict(
        n_bands=64,
        channels=32,
        num_stages=5,
        dilation_cycle=(1, 1, 2, 4, 6),
        transport="crossattn_query",
        use_attn=True,
        pooled_mixer_hidden=4096,
        residual_head=True,
    ),
    "rt_plus_soft_query": dict(
        n_bands=64,
        channels=32,
        num_stages=5,
        dilation_cycle=(1, 1, 2, 4, 6),
        transport="soft_band_query",
        use_attn=True,
        pooled_mixer_hidden=4096,
        residual_head=True,
    ),
    # Proposal B literal CNB-stage variant.  The document sketch asked for
    # kernel_t=5 with dilations (1, 2, 4), but the current NPU validator rejects
    # the last branch because its span is 16.  Use the nearest deployable FSMN
    # schedule, (1, 2, 3), and keep transport routing temporal kernel at 1 to
    # control state/cache growth.
    "causal_cnb_soft_query": dict(
        n_bands=48,
        channels=24,
        num_stages=5,
        dilation_cycle=(1, 1, 1, 1, 1),
        transport="soft_band_query",
        use_attn=False,
        pooled_mixer_hidden=0,
        time_kernel=1,
        freq_kernel=3,
        stage_type="causal_cnb",
        cnb_kernel=5,
        cnb_dilation_schedule=(1, 2, 3),
        residual_head=False,
    ),
    "causal_cnb_crossattn_query": dict(
        n_bands=48,
        channels=24,
        num_stages=5,
        dilation_cycle=(1, 1, 1, 1, 1),
        transport="crossattn_query",
        use_attn=False,
        pooled_mixer_hidden=0,
        time_kernel=1,
        freq_kernel=3,
        stage_type="causal_cnb",
        cnb_kernel=5,
        cnb_dilation_schedule=(1, 2, 3),
        residual_head=False,
    ),
}

_PRESET_CONFIGS["safe_soft_band_query"] = _PRESET_CONFIGS["safe_soft_query"]
_PRESET_CONFIGS["balanced_soft_band_query"] = _PRESET_CONFIGS["balanced_soft_query"]
_PRESET_CONFIGS["quality_crossattn_query"] = _PRESET_CONFIGS["quality"]
_PRESET_CONFIGS["quality_soft_band_query"] = _PRESET_CONFIGS["quality_soft_query"]
_PRESET_CONFIGS["quality6m_crossattn_query"] = _PRESET_CONFIGS["quality6m"]
_PRESET_CONFIGS["quality6m_soft_band_query"] = _PRESET_CONFIGS["quality6m_soft_query"]
_PRESET_CONFIGS["rt_plus_crossattn_query"] = _PRESET_CONFIGS["rt_plus"]
_PRESET_CONFIGS["rt_plus_soft_band_query"] = _PRESET_CONFIGS["rt_plus_soft_query"]
_PRESET_CONFIGS["causal_cnb_soft_band_query"] = _PRESET_CONFIGS["causal_cnb_soft_query"]


def build_band_sfc_net_npu_from_config(
    *,
    preset: str,
    n_freq: int,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
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
    stage_type: str | None = None,
    cnb_kernel: int | None = None,
    cnb_dilation_schedule: tuple[int, ...] | list[int] | None = None,
    residual_head: bool | None = None,
) -> BandSFCNetNPU:
    if preset not in _PRESET_CONFIGS:
        names = ", ".join(sorted(_PRESET_CONFIGS))
        raise ValueError(f"Unknown BandSFCNetNPU preset {preset!r}. Available: {names}")
    cfg = deepcopy(_PRESET_CONFIGS[preset])
    overrides = {
        "n_bands": n_bands,
        "channels": channels,
        "num_stages": num_stages,
        "time_kernel": time_kernel,
        "freq_kernel": freq_kernel,
        "dilation_cycle": tuple(dilation_cycle) if dilation_cycle is not None else None,
        "transport": transport,
        "routing_normalization": routing_normalization,
        "use_attn": use_attn,
        "attn_window": attn_window,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "pooled_mixer_hidden": pooled_mixer_hidden,
        "pooled_mixer_hidden_schedule": (
            tuple(pooled_mixer_hidden_schedule) if pooled_mixer_hidden_schedule is not None else None
        ),
        "stage_type": stage_type,
        "cnb_kernel": cnb_kernel,
        "cnb_dilation_schedule": tuple(cnb_dilation_schedule) if cnb_dilation_schedule is not None else None,
        "residual_head": residual_head,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return _build_preset(
        n_freq=n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        query_type=query_type,
        **cfg,  # type: ignore[arg-type]
    )


def build_band_sfc_net_npu_preset(
    preset: str,
    *,
    n_freq: int,
    n_fft: int | None = None,
    sample_rate: int | None = None,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
    query_type: str = "adaptive",
) -> BandSFCNetNPU:
    return build_band_sfc_net_npu_from_config(
        preset=preset,
        n_freq=n_freq,
        n_fft=n_fft,
        sample_rate=sample_rate,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        masking=masking,
        query_type=query_type,
    )
