"""Training integration for online TIGER variants.

The deployment TIGER modules operate on causal real/imaginary STFT frames.
This wrapper exposes them as waveform-in, waveform-out systems for the
existing ``SupTask`` training pipeline while reusing ``forward_sequence`` from
the exportable streaming models.
"""
from __future__ import annotations

from typing import Any

from contextlib import nullcontext
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_feature_compression.core.model.frequency_preprocessing import (
    HybridFrequencyProjector2d,
    PCENGainNormalizer2d,
    build_frequency_preprocessor,
    build_pcen_preprocessor,
    resolve_frequency_input_n_freq,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.online_sfc_2d import (
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)

from .npu_edge_utils import sanitize_for_npu_edge
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


# Preset tables for TF-MLPNet (TIGEREdgeMLP). Sizes follow context.md:
# - edge:    hidden=96,  num_blocks=6  -> ~TIGER npu-edge-v2 footprint
# - balance: hidden=128, num_blocks=8
# - large:   hidden=160, num_blocks=8
#
# NPU constraint: (time_kernel-1)*dilation < 14. With kt=3 and dilations
# (1,2,4) the worst case is 8, which is safe. (1,2,4,8) would hit 16 and
# must not be used here.
#
# ``out_channels`` must be divisible by ``num_sources`` (MaskBlock depthwise
# groups). For DnR 3-stem we use 24 / 48 / 72.
_TF_MLPNET_PRESETS: dict[str, dict[str, Any]] = {
    "tf-mlpnet-edge": {
        "out_channels": 24,
        "in_channels": 96,
        "num_blocks": 6,
        "upsampling_depth": 2,
        "edge_hidden_channels": 96,
        "edge_num_blocks": 6,
        "edge_expansion": 2,
        "edge_freq_kernel_size": 3,
        "edge_time_kernel_size": 3,
        "edge_time_dilations": (1, 2, 4),
    },
    "tf-mlpnet-balance": {
        "out_channels": 48,
        "in_channels": 192,
        "num_blocks": 8,
        "upsampling_depth": 2,
        "edge_hidden_channels": 128,
        "edge_num_blocks": 8,
        "edge_expansion": 2,
        "edge_freq_kernel_size": 3,
        "edge_time_kernel_size": 3,
        "edge_time_dilations": (1, 2, 4),
    },
    "tf-mlpnet-large": {
        "out_channels": 72,
        "in_channels": 240,
        "num_blocks": 8,
        "upsampling_depth": 2,
        "edge_hidden_channels": 160,
        "edge_num_blocks": 8,
        "edge_expansion": 2,
        "edge_freq_kernel_size": 3,
        "edge_time_kernel_size": 3,
        "edge_time_dilations": (1, 2, 4),
    },
}


def _import_tiger_edge_mlp():
    """Import ``TIGEREdgeMLP`` from the hyphenated ``TF-MLPNet`` sibling.

    ``TF-MLPNet/`` cannot be imported as a top-level Python module because of
    the hyphen, so we add the directory to ``sys.path`` on first use.
    """
    from pathlib import Path as _Path
    import sys as _sys

    _tf_root = _Path(__file__).resolve().parent.parent / "TF-MLPNet"
    if _tf_root.is_dir() and str(_tf_root) not in _sys.path:
        _sys.path.insert(0, str(_tf_root))
    from tf_mlpnet import TIGEREdgeMLP  # noqa: E402 - deferred import by design

    return TIGEREdgeMLP


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
        tiger_edge_mlp_cls = _import_tiger_edge_mlp()
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
        return sanitize_for_npu_edge(TIGERNPULargeDeployable(**common, **defaults))

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
    return sanitize_for_npu_edge(model_cls(**common, **compact_defaults))


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
        freq_preprocessor: HybridFrequencyProjector2d | None = None,
        pcen_preprocessor: PCENGainNormalizer2d | None = None,
        dc_bypass_enabled: bool = False,
        dc_policy: str = "zero",
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
        self.freq_preprocessor = freq_preprocessor
        self.pcen_preprocessor = pcen_preprocessor
        self.dc_bypass_enabled = bool(dc_bypass_enabled)
        if dc_policy not in {"zero", "mixture_equal"}:
            raise ValueError(f"Unsupported dc_policy={dc_policy!r}; expected 'zero' or 'mixture_equal'.")
        self.dc_policy = dc_policy
        self.input_n_freq = (win // 2) + 1
        self.body_input_n_freq = resolve_frequency_input_n_freq(
            self.input_n_freq,
            dc_bypass_enabled=self.dc_bypass_enabled,
        )
        self.core_n_freq = freq_preprocessor.n_freq_out if freq_preprocessor is not None else self.body_input_n_freq
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

    @staticmethod
    def _ri_sequence_to_2d(x_ri: torch.Tensor) -> torch.Tensor:
        batch, channels, ri_bins, frames = x_ri.shape
        freq = ri_bins // 2
        real = x_ri[:, :, :freq, :]
        imag = x_ri[:, :, freq:, :]
        complex_stft = torch.complex(real.to(torch.float32), imag.to(torch.float32))
        return pack_complex_stft_as_2d(complex_stft)

    @staticmethod
    def _two_d_to_ri_sequence(x2d: torch.Tensor, *, n_src: int, n_chan: int) -> torch.Tensor:
        complex_stft = unpack_2d_to_complex_stft(x2d, n_src=n_src, n_chan=n_chan)
        batch, n_src_out, n_chan_out, freq, frames = complex_stft.shape
        flat = complex_stft.reshape(batch, n_src_out * n_chan_out, freq, frames)
        return torch.cat([flat.real, flat.imag], dim=2)

    def _split_dc_2d(self, x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.dc_bypass_enabled:
            return x2d, None
        if x2d.shape[-1] != self.input_n_freq:
            raise ValueError(f"Expected {self.input_n_freq} TIGER input bins with DC bypass, got {x2d.shape[-1]}")
        return x2d[..., 1:], x2d[..., :1]

    def _restore_dc_2d(self, y2d: torch.Tensor, dc2d: torch.Tensor | None) -> torch.Tensor:
        if not self.dc_bypass_enabled:
            return y2d
        if dc2d is None:
            raise ValueError("dc2d must be provided when TIGER DC bypass is enabled.")
        batch, channels, frames, _ = y2d.shape
        expected_channels = 2 * self.n_src * self.n_chan
        if channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} TIGER output channels, got {channels}")

        if self.dc_policy == "zero":
            dc_out = y2d.new_zeros(batch, channels, frames, 1)
        else:
            dc = dc2d.reshape(batch, self.n_chan, 2, frames, 1)
            dc = dc.unsqueeze(1).expand(batch, self.n_src, self.n_chan, 2, frames, 1)
            dc_out = dc.reshape(batch, channels, frames, 1) / float(self.n_src)
        return torch.cat([dc_out, y2d], dim=-1)

    def _preprocess_ri_sequence(self, x_ri: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        x2d = self._ri_sequence_to_2d(x_ri)
        x2d, dc2d = self._split_dc_2d(x2d)
        if self.freq_preprocessor is not None:
            x2d = self.freq_preprocessor.analysis(x2d)
        gain = None
        if self.pcen_preprocessor is not None:
            x2d, gain, _state = self.pcen_preprocessor.forward_with_gain(x2d)
        return self._two_d_to_ri_sequence(x2d, n_src=1, n_chan=self.n_chan), {"dc2d": dc2d, "gain": gain}

    def _postprocess_ri_sequence(
        self,
        y_ri: torch.Tensor,
        context: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        y2d = self._ri_sequence_to_2d(y_ri)
        gain = context.get("gain")
        if self.pcen_preprocessor is not None and gain is not None:
            y2d = self.pcen_preprocessor.invert_output_gain(y2d, gain, n_src=self.n_src)
        if self.freq_preprocessor is not None:
            y2d = self.freq_preprocessor.synthesis(y2d)
        y2d = self._restore_dc_2d(y2d, context.get("dc2d"))
        return self._two_d_to_ri_sequence(y2d, n_src=self.n_src, n_chan=self.n_chan)

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
        subband_ri_core, preprocess_context = self._preprocess_ri_sequence(subband_ri)
        mask_logits = self.core.forward_sequence(
            subband_ri_core,
            detach_state=self.detach_state,
            chunk_size=self.chunk_size,
        )[0]
        est_ri = self._apply_masks(mask_logits, subband_ri_core)
        est_ri = self._postprocess_ri_sequence(est_ri, preprocess_context)
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

    def frequency_preprocess_manifest(self) -> dict[str, object] | None:
        if self.freq_preprocessor is None:
            return None
        return self.freq_preprocessor.manifest()

    def pcen_preprocess_manifest(self) -> dict[str, object] | None:
        if self.pcen_preprocessor is None:
            return None
        return self.pcen_preprocessor.manifest()

    def dc_bypass_manifest(self) -> dict[str, object] | None:
        if not self.dc_bypass_enabled:
            return None
        return {
            "enabled": True,
            "policy": self.dc_policy,
            "input_n_freq": self.input_n_freq,
            "body_input_n_freq": self.body_input_n_freq,
        }


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
) -> TIGERWaveformSeparator:
    if scaling:
        raise ValueError("TIGER streaming training does not support global scaling.")
    full_n_freq = (n_fft // 2) + 1
    core_n_freq = resolve_preprocessed_n_freq(
        full_n_freq,
        enabled=freq_preprocess_enabled,
        keep_bins=freq_preprocess_keep_bins,
        target_bins=freq_preprocess_target_bins,
        dc_bypass_enabled=dc_bypass_enabled,
    )
    core_n_fft = 2 * (core_n_freq - 1)
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
    core = build_tiger_core(
        variant=variant,
        n_fft=core_n_fft,
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
        freq_preprocessor=freq_preprocessor,
        pcen_preprocessor=pcen_preprocessor,
        dc_bypass_enabled=dc_bypass_enabled,
        dc_policy=dc_policy,
        detach_state=detach_state,
        chunk_size=chunk_size,
        css_segment_size=css_segment_size,
        css_shift_size=css_shift_size,
        css_batch_size=css_batch_size,
        fs=fs,
    )
