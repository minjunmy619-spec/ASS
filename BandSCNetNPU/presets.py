"""Preset factories for Band-SCNet-NPU.

See ``.kiro/specs/band-scnet-npu/design.md §3`` for the preset rationale.

Budget accounting (fp16, n_freq=2049):

Both presets use low-band stride chain 4 (/2/2), mid 4 (/2/2), high 16
(/2/2/2/2), which makes the separator F' roughly equal to F/16.
At n_freq=2049 the split rounds to (368, 800, 880), giving
out_widths = (92, 200, 55) and concat_width = 347.

edge_small  (C_s=16, L=2, Kt=5, no attn)
  separator state ~ 2 * 4 * 16 * 347 * 2 = 87 KiB
  pyramid + decoder state ~ 25 KiB
  total                   ~ 112 KiB    <  192 KiB

rt192k      (C_s=32, L=4, Kt=5, attn W=16 h=4 d=8)
  separator state ~ 4 * 4 * 32 * 347 * 2 = 173 KiB
  attn KV states  ~ 2 KiB
  pyramid + decoder state ~ 25 KiB  (C_p=8, shallow)
  total                   ~ 200 KiB
With Kt=3 for the separator this drops to ~100 KiB sep + 25 KiB pyramid
= ~130 KiB, leaving headroom for the attn KV cache and iSTFT scratch.
"""
from __future__ import annotations

from .band_scnet_npu import BandSCNetNPU


def edge_small(
    n_freq: int,
    *,
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
) -> BandSCNetNPU:
    """Small preset used for MLIR / ONNX smoke testing.

    Separator width C_s=16, L=2, Kt=5, no attention. Pyramid kept shallow/narrow
    to stay well under the 192 KiB DSP state quota at n_freq=2049.
    """
    return BandSCNetNPU(
        n_freq=n_freq,
        n_src=n_src,
        n_chan=n_chan,
        channels=16,
        pyramid_channels=8,
        num_stages=2,
        time_kernel=5,
        freq_kernel=3,
        pyramid_time_kernel=3,
        pyramid_freq_kernel=3,
        pyramid_conv_blocks=(1, 1, 1),
        pyramid_strides=(2, 2, 4),
        use_attn=False,
        masking=masking,
    )


def rt192k(
    n_freq: int,
    *,
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
) -> BandSCNetNPU:
    """Deployment preset targeting the Band-SCNet parameter class.

    Separator width C_s=32, L=4, Kt=3 with bounded causal attention in every
    NarrowBandBlock (W=16, heads=4, head_dim=8). Pyramid kept narrow (C_p=8)
    with aggressive striding so the bulk of the state budget is spent on
    the separator.
    """
    return BandSCNetNPU(
        n_freq=n_freq,
        n_src=n_src,
        n_chan=n_chan,
        channels=40,
        pyramid_channels=8,
        num_stages=3,
        time_kernel=3,
        freq_kernel=3,
        pyramid_time_kernel=3,
        pyramid_freq_kernel=3,
        pyramid_conv_blocks=(1, 1, 1),
        pyramid_strides=(2, 2, 4),
        use_attn=True,
        attn_window=16,
        num_heads=4,
        head_dim=8,
        masking=masking,
    )


_PRESETS = {
    "edge_small": edge_small,
    "rt192k": rt192k,
}


def build_band_scnet_npu_preset(
    preset: str,
    *,
    n_freq: int,
    n_src: int = 3,
    n_chan: int = 1,
    masking: bool = True,
) -> BandSCNetNPU:
    if preset not in _PRESETS:
        names = ", ".join(sorted(_PRESETS))
        raise ValueError(f"Unknown Band-SCNet-NPU preset {preset!r}. Available: {names}")
    return _PRESETS[preset](n_freq=n_freq, n_src=n_src, n_chan=n_chan, masking=masking)
