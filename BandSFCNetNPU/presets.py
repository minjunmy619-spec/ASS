"""Preset factories for BandSFCNetNPU."""

from __future__ import annotations

from .band_sfc_net_npu import BandSFCNetNPU


def safe(
    n_freq: int,
    *,
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
) -> BandSFCNetNPU:
    """Deployment-first baseline: soft SFC transport plus BandSCNet stages."""
    return BandSFCNetNPU(
        n_freq=n_freq,
        n_bands=64,
        n_src=n_src,
        n_chan=n_chan,
        channels=32,
        num_stages=4,
        time_kernel=3,
        freq_kernel=3,
        dilation_cycle=(1, 1, 2, 4),
        transport="soft",
        use_attn=False,
        pooled_mixer_hidden=1024,
        masking=masking,
    )


def quality(
    n_freq: int,
    *,
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
) -> BandSFCNetNPU:
    """Main quality candidate with cross-attention transport and pooled attention."""
    return BandSFCNetNPU(
        n_freq=n_freq,
        n_bands=64,
        n_src=n_src,
        n_chan=n_chan,
        channels=32,
        num_stages=5,
        time_kernel=3,
        freq_kernel=3,
        dilation_cycle=(1, 1, 2, 4, 6),
        transport="crossattn",
        use_attn=True,
        attn_window=16,
        num_heads=4,
        head_dim=8,
        pooled_mixer_hidden=4096,
        masking=masking,
    )


def quality6m(
    n_freq: int,
    *,
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
) -> BandSFCNetNPU:
    """Large parameter probe; validate state and compiler budget before deployment."""
    return BandSFCNetNPU(
        n_freq=n_freq,
        n_bands=64,
        n_src=n_src,
        n_chan=n_chan,
        channels=40,
        num_stages=4,
        time_kernel=3,
        freq_kernel=3,
        dilation_cycle=(1, 2, 4, 6),
        transport="crossattn",
        use_attn=True,
        attn_window=16,
        num_heads=4,
        head_dim=8,
        pooled_mixer_hidden=18432,
        masking=masking,
    )


_PRESETS = {
    "safe": safe,
    "quality": quality,
    "quality6m": quality6m,
}


def build_band_sfc_net_npu_preset(
    preset: str,
    *,
    n_freq: int,
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
) -> BandSFCNetNPU:
    if preset not in _PRESETS:
        names = ", ".join(sorted(_PRESETS))
        raise ValueError(f"Unknown BandSFCNetNPU preset {preset!r}. Available: {names}")
    return _PRESETS[preset](n_freq=n_freq, n_src=n_src, n_chan=n_chan, masking=masking)
