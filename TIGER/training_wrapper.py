"""Training integration for online TIGER variants.

The deployment TIGER modules operate on causal real/imaginary STFT frames.
This wrapper exposes them as waveform-in, waveform-out systems for the
existing ``SupTask`` training pipeline while reusing ``forward_sequence`` from
the exportable streaming models.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .streaming_io import build_causal_ri_sequence, invert_causal_ri_sequence
from .tiger_npu_edge import TIGERNPUEdgeV1
from .tiger_npu_edge_v2 import TIGERNPUEdgeV2
from .tiger_online import (
    TIGERCtxDeployable,
    TIGERCtxTigerLikeApprox,
    TIGERDeployable,
    TIGERNPULargeDeployable,
    TIGERTigerLikeApprox,
)


def _normalize_variant_name(variant: str) -> str:
    aliases = {
        "deployable": "deployable",
        "tiger-like": "tiger-like",
        "tiger_like": "tiger-like",
        "ctx-deployable": "ctx-deployable",
        "ctx_deployable": "ctx-deployable",
        "ctx-tiger-like": "ctx-tiger-like",
        "ctx_tiger_like": "ctx-tiger-like",
        "npu-large": "npu-large",
        "npu_large": "npu-large",
        "npu-edge": "npu-edge-v1",
        "npu_edge": "npu-edge-v1",
        "npu-edge-v1": "npu-edge-v1",
        "npu_edge_v1": "npu-edge-v1",
        "tiger-edge": "npu-edge-v1",
        "tiger_edge": "npu-edge-v1",
        "npu-edge-v2": "npu-edge-v2",
        "npu_edge_v2": "npu-edge-v2",
        "tiger-edge-v2": "npu-edge-v2",
        "tiger_edge_v2": "npu-edge-v2",
        # TF-MLPNet-style TIGEREdgeMLP (separator replaced by EdgeTFMLPSeparator,
        # encoder/decoder inherited from stock TIGER so this stays inside the
        # build_tiger_system / TIGERWaveformSeparator training path).
        "tf-mlpnet": "tf-mlpnet-edge",
        "tf_mlpnet": "tf-mlpnet-edge",
        "tfmlpnet": "tf-mlpnet-edge",
        "tf-mlpnet-edge": "tf-mlpnet-edge",
        "tf_mlpnet_edge": "tf-mlpnet-edge",
        "tf-mlpnet-balance": "tf-mlpnet-balance",
        "tf_mlpnet_balance": "tf-mlpnet-balance",
        "tf-mlpnet-large": "tf-mlpnet-large",
        "tf_mlpnet_large": "tf-mlpnet-large",
    }
    key = variant.strip().lower()
    if key not in aliases:
        raise ValueError(f"Unknown TIGER variant: {variant}")
    return aliases[key]


# Preset tables for TF-MLPNet v3 (TIGEREdgeMLPV3 with GLU gating + EMA global
# state). The three presets target roughly 3M / 6M / 9M total parameters and
# are all verified to fit the 192 KiB fp16 DSP streaming-state budget by
# ``TF-MLPNet/tests/test_tiger_edge_mlp_smoke.py``.
#
# NPU constraints enforced at construction time:
#   (time_kernel - 1) * dilation < 14  for each dilation
#   (freq_kernel - 1) < 14             (freq dilation is fixed at 1)
#
# ``out_channels`` is the TIGER feature_dim, which must be divisible by
# ``num_sources`` (MaskBlock depthwise groups). 48 / 72 / 96 all work for
# DnR 3-stem.
#
# ``pre_calc_bands`` overrides the stock TIGER band-split (which would
# produce ~67 bands at 44.1 kHz / n_fft=2048 and blow the 192 KiB DSP state
# quota regardless of channel count). The 8-band split below sums to
# enc_dim = n_fft//2 + 1 = 1025 and keeps state well under the quota.
_DNR_8_BANDS: tuple[int, ...] = (10, 28, 56, 93, 186, 186, 279, 187)
assert sum(_DNR_8_BANDS) == 1025, "DnR 8-band split must cover enc_dim = 1025"
_TF_MLPNET_PRESETS: dict[str, dict[str, Any]] = {
    "tf-mlpnet-edge": {
        "out_channels": 48,
        "in_channels": 192,
        "num_blocks": 8,
        "upsampling_depth": 2,
        "pre_calc_bands": _DNR_8_BANDS,
        "edge_hidden_channels": 160,
        "edge_num_blocks": 8,
        "edge_expansion": 2,
        "edge_freq_kernel_size": 5,
        "edge_time_kernel_size": 3,
        "edge_time_dilations": (1, 2, 4),
    },
    "tf-mlpnet-balance": {
        "out_channels": 72,
        "in_channels": 288,
        "num_blocks": 9,
        "upsampling_depth": 2,
        "pre_calc_bands": _DNR_8_BANDS,
        "edge_hidden_channels": 208,
        "edge_num_blocks": 9,
        "edge_expansion": 2,
        "edge_freq_kernel_size": 7,
        "edge_time_kernel_size": 3,
        "edge_time_dilations": (1, 2, 4),
    },
    "tf-mlpnet-large": {
        "out_channels": 96,
        "in_channels": 384,
        "num_blocks": 8,
        "upsampling_depth": 2,
        "pre_calc_bands": _DNR_8_BANDS,
        "edge_hidden_channels": 272,
        "edge_num_blocks": 8,
        "edge_expansion": 2,
        "edge_freq_kernel_size": 7,
        "edge_time_kernel_size": 3,
        "edge_time_dilations": (1, 2, 4),
    },
}


def _import_tiger_edge_mlp_v3():
    """Import ``TIGEREdgeMLPV3`` from the hyphenated ``TF-MLPNet`` sibling.

    ``TF-MLPNet/`` cannot be imported as a top-level Python module because of
    the hyphen, so we add the directory to ``sys.path`` on first use.
    """
    import sys as _sys
    from pathlib import Path as _Path

    _tf_root = _Path(__file__).resolve().parent.parent / "TF-MLPNet"
    if _tf_root.is_dir() and str(_tf_root) not in _sys.path:
        _sys.path.insert(0, str(_tf_root))
    from tf_mlpnet import TIGEREdgeMLPV3  # noqa: E402 - deferred import by design

    return TIGEREdgeMLPV3


def _build_analysis_window(name: str, win: int) -> torch.Tensor | None:
    name = name.strip().lower()
    if name in {"none", "rect", "rectangular"}:
        return None
    if name == "hann":
        return torch.hann_window(win)
    if name == "sqrt_hann":
        return torch.hann_window(win).clamp_min(0.0).sqrt()
    raise ValueError(f"Unknown TIGER analysis window: {name}")


def build_tiger_core(
    *,
    variant: str = "npu-edge-v2",
    n_fft: int = 2048,
    hop_length: int = 512,
    fs: int = 44100,
    n_src: int = 3,
    model_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """Build a TIGER streaming core by recipe variant name."""
    variant = _normalize_variant_name(variant)
    kwargs = dict(model_kwargs or {})

    if variant == "npu-edge-v2":
        return TIGERNPUEdgeV2(
            sample_rate=fs,
            num_sources=n_src,
            win=n_fft,
            stride=hop_length,
            **kwargs,
        )

    if variant in _TF_MLPNET_PRESETS:
        tiger_edge_mlp_cls = _import_tiger_edge_mlp_v3()
        defaults = dict(_TF_MLPNET_PRESETS[variant])
        # Keep explicit control flags off the preset dict so users can't
        # accidentally override them via `model_kwargs` and break streaming.
        defaults.update(kwargs)
        return tiger_edge_mlp_cls(
            sample_rate=fs,
            num_sources=n_src,
            win=n_fft,
            stride=hop_length,
            need_streaming=True,
            **defaults,
        )

    common = {
        "sample_rate": fs,
        "num_sources": n_src,
        "win": n_fft,
        "stride": hop_length,
        "need_streaming": True,
    }

    if variant == "npu-edge-v1":
        return TIGERNPUEdgeV1(**common, **kwargs)
    if variant == "npu-large":
        defaults = {
            "out_channels": 192,
            "in_channels": 1024,
            "upsampling_depth": 5,
            "att_n_head": 4,
            "att_hid_chan": 8,
            "num_stages": 2,
        }
        defaults.update(kwargs)
        return TIGERNPULargeDeployable(**common, **defaults)

    compact_defaults = {
        "out_channels": 132,
        "in_channels": 256,
        "num_blocks": 4,
        "upsampling_depth": 5,
        "att_n_head": 4,
        "att_hid_chan": 4,
        "att_kernel_size": 8,
        "att_stride": 1,
    }
    compact_defaults.update(kwargs)
    model_cls = {
        "deployable": TIGERDeployable,
        "tiger-like": TIGERTigerLikeApprox,
        "ctx-deployable": TIGERCtxDeployable,
        "ctx-tiger-like": TIGERCtxTigerLikeApprox,
    }[variant]
    return model_cls(**common, **compact_defaults)


class TIGERWaveformSeparator(nn.Module):
    """Waveform adapter for streaming TIGER cores.

    Input is a mono waveform ``[B, 1, samples]``. Output matches the existing
    separation contract ``[B, n_src, 1, samples]``.
    """

    def __init__(
        self,
        core: nn.Module,
        *,
        n_src: int = 3,
        n_chan: int = 1,
        win: int = 2048,
        hop: int = 512,
        startup_packet: int = 256,
        analysis_window: str = "sqrt_hann",
        detach_state: bool = False,
        chunk_size: int = 8,
        css_segment_size: int = 12,
        css_shift_size: int = 6,
        css_batch_size: int = 1,
        fs: int = 44100,
    ):
        super().__init__()
        if n_chan != 1:
            raise ValueError("TIGERWaveformSeparator currently supports mono DnR training only (n_chan=1).")
        self.core = core
        self.n_src = n_src
        self.n_chan = n_chan
        self.win = win
        self.hop = hop
        self.startup_packet = startup_packet
        self.detach_state = detach_state
        self.chunk_size = chunk_size
        self.css_segment_size = css_segment_size
        self.css_shift_size = css_shift_size
        self.css_batch_size = css_batch_size
        self.fs = fs

        window = _build_analysis_window(analysis_window, win)
        if window is None:
            self.analysis_window_name = "none"
            self.register_buffer("analysis_window", torch.empty(0), persistent=False)
        else:
            self.analysis_window_name = analysis_window
            self.register_buffer("analysis_window", window, persistent=False)

    def _window_or_none(self) -> torch.Tensor | None:
        if self.analysis_window.numel() == 0:
            return None
        return self.analysis_window

    def _pad_to_frame_boundary(self, wav: torch.Tensor) -> tuple[torch.Tensor, int]:
        num_samples = wav.shape[-1]
        if num_samples <= self.startup_packet:
            target_samples = self.startup_packet
        else:
            steps = math.ceil((num_samples - self.startup_packet) / self.hop)
            target_samples = self.startup_packet + steps * self.hop
        pad_right = target_samples - num_samples
        if pad_right > 0:
            wav = F.pad(wav, (0, pad_right))
        return wav, num_samples

    def _no_autocast(self, tensor: torch.Tensor):
        if tensor.is_cuda:
            return torch.autocast(device_type="cuda", enabled=False)
        return nullcontext()

    def _apply_masks(self, mask_logits: torch.Tensor, mixture_ri: torch.Tensor) -> torch.Tensor:
        batch, _, ri_bins, frames = mixture_ri.shape
        enc_dim = ri_bins // 2
        if mask_logits.shape[1] != 4 * self.n_src:
            raise RuntimeError(f"Expected {4 * self.n_src} TIGER mask channels, got {mask_logits.shape[1]}.")

        mix_real = mixture_ri[:, 0, :enc_dim, :]
        mix_imag = mixture_ri[:, 0, enc_dim:, :]
        mask_logits = mask_logits.reshape(batch, 2, 2, self.n_src, enc_dim, frames)
        complex_mask = mask_logits[:, 0] * torch.sigmoid(mask_logits[:, 1])
        mask_real = complex_mask[:, 0]
        mask_imag = complex_mask[:, 1]

        mask_real = mask_real - (mask_real.sum(1, keepdim=True) - 1.0) / self.n_src
        mask_imag = mask_imag - mask_imag.sum(1, keepdim=True) / self.n_src

        est_real = mix_real.unsqueeze(1) * mask_real - mix_imag.unsqueeze(1) * mask_imag
        est_imag = mix_real.unsqueeze(1) * mask_imag + mix_imag.unsqueeze(1) * mask_real
        return torch.cat([est_real, est_imag], dim=2)

    def forward(self, wav: torch.Tensor, **kwargs) -> torch.Tensor:
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        if wav.dim() != 3 or wav.shape[1] != 1:
            raise ValueError(f"Expected mono waveform [B, 1, samples], got {tuple(wav.shape)}.")

        wav_pad, original_samples = self._pad_to_frame_boundary(wav)
        window = self._window_or_none()
        with self._no_autocast(wav_pad):
            subband_ri = build_causal_ri_sequence(
                wav_pad.float(),
                win=self.win,
                hop=self.hop,
                startup_packet=self.startup_packet,
                analysis_window=window,
            )
        mask_logits = self.core.forward_sequence(
            subband_ri,
            detach_state=self.detach_state,
            chunk_size=self.chunk_size,
        )[0]
        est_ri = self._apply_masks(mask_logits, subband_ri)
        with self._no_autocast(est_ri):
            est_wav = invert_causal_ri_sequence(
                est_ri.float(),
                win=self.win,
                hop=self.hop,
                startup_packet=self.startup_packet,
                num_samples=wav_pad.shape[-1],
                analysis_window=window,
                synthesis_window=window,
            )
        return est_wav[..., :original_samples].unsqueeze(2)

    def css(self, speech_mix: torch.Tensor, **kwargs) -> torch.Tensor:
        speech_length = speech_mix.shape[-1]
        if speech_length <= self.css_segment_size * self.fs:
            return self(speech_mix, **kwargs)

        overlap_length = int(np.round(self.fs * (self.css_segment_size - self.css_shift_size)))
        num_segments = int(np.ceil((speech_length - overlap_length) / (self.css_shift_size * self.fs)))
        total_length = int(self.css_segment_size * self.fs)
        pad_shape = speech_mix[..., :total_length].shape

        segments = []
        is_silent = []
        for i in range(num_segments):
            start = int(i * self.css_shift_size * self.fs)
            end = min(start + total_length, speech_length)
            if end >= speech_length:
                seg = speech_mix.new_zeros(pad_shape)
                valid = end - start
                seg[..., :valid] = speech_mix[..., start:end].clone()
            else:
                seg = speech_mix[..., start:end].clone()
                valid = total_length
            segments.append(seg)
            is_silent.append(abs(seg).sum().item() == 0.0)

        valid_indices = [i for i, silent in enumerate(is_silent) if not silent]
        if not valid_indices:
            return speech_mix.new_zeros(speech_mix.shape[0], self.n_src, self.n_chan, speech_length)

        enhanced = [None] * num_segments
        for batch_start in range(0, len(valid_indices), self.css_batch_size):
            batch_indices = valid_indices[batch_start:batch_start + self.css_batch_size]
            seg_batch = torch.cat([segments[i] for i in batch_indices], dim=0)
            processed = self(seg_batch, **kwargs)[..., :total_length]
            for local_idx, seg_idx in enumerate(batch_indices):
                enhanced[seg_idx] = processed[[local_idx]]

        for i in range(num_segments):
            if enhanced[i] is None:
                enhanced[i] = torch.zeros_like(enhanced[valid_indices[0]])

        waves = enhanced[0]
        for i in range(1, num_segments):
            if i == num_segments - 1:
                enhanced[i][..., valid:] = 0
                tail = enhanced[i][..., overlap_length:valid]
            else:
                tail = enhanced[i][..., overlap_length:]

            if overlap_length > 0:
                waves[..., -overlap_length:] = (waves[..., -overlap_length:] + enhanced[i][..., :overlap_length]) / 2
            waves = torch.cat([waves, tail], dim=-1)
        return waves[..., :speech_length]


def build_tiger_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    variant: str = "npu-edge-v2",
    startup_packet: int = 256,
    analysis_window: str = "sqrt_hann",
    detach_state: bool = False,
    chunk_size: int = 8,
    model_kwargs: dict[str, Any] | None = None,
    scaling: bool = False,
    css_segment_size: int = 12,
    css_shift_size: int = 6,
    css_batch_size: int = 1,
) -> TIGERWaveformSeparator:
    if scaling:
        raise ValueError("TIGER streaming training does not support global scaling.")
    core = build_tiger_core(
        variant=variant,
        n_fft=n_fft,
        hop_length=hop_length,
        fs=fs,
        n_src=n_src,
        model_kwargs=model_kwargs,
    )
    return TIGERWaveformSeparator(
        core=core,
        n_src=n_src,
        n_chan=n_chan,
        win=n_fft,
        hop=hop_length,
        startup_packet=startup_packet,
        analysis_window=analysis_window,
        detach_state=detach_state,
        chunk_size=chunk_size,
        css_segment_size=css_segment_size,
        css_shift_size=css_shift_size,
        css_batch_size=css_batch_size,
        fs=fs,
    )
