"""
NPU-friendly student for the source-aware MelBand RoFormer teacher.

The teacher in :mod:`source_aware_melband_roformer` uses non-causal axial
attention, rotary embeddings, explicit 5D source tensors, residual waveform
correction, and mixture consistency.  This module keeps the same separation
biases but lowers them to strict online/NPU primitives:

* adaptive overlapped mel-band SFC compression mirrors the teacher frontend;
* causal dilated time blocks plus local band depthwise mixing approximate axial
  time/band modeling without attention kernels;
* source streams are packed into channels, seeded with learned source biases,
  and repeatedly fused with other-source and mixture context;
* reconstruction is source-shared through the SFC query expander;
* a narrow full-band mask correction head and mixture-consistency projection
  approximate the teacher's residual reconstruction stage.

Runtime tensors remain 4D and the graph uses Conv2d, bmm, reductions, and
basic elementwise operations so it can be exported through the existing online
ONNX/NPU path.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from spectral_feature_compression.core.model.adaptive_mel_sfc_2d import (
    AdaptiveMelBandSpec2d,
    _apply_packed_complex_mask_no_repeat,
)
from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    build_frequency_preprocessor,
    build_hybrid_frequency_bin_frequencies,
    build_pcen_preprocessor,
    resolve_frequency_input_n_freq,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_sfc_2d import (
    RMSNorm2d,
    _runtime_assert,
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)
from spectral_feature_compression.core.model.online_soft_band_dilated_sfc_2d import (
    DilatedBandMixBlock2d,
    _normalize_dilation_schedule,
)
from spectral_feature_compression.core.model.online_soft_band_query_sfc_2d import (
    SoftBandQueryCompressor2d,
    SoftBandQueryExpander2d,
)
from spectral_feature_compression.core.model.residual_refinement_sfc_2d import Mamba2LiteTemporalBranch2d
from spectral_feature_compression.core.model.source_aware_residual_sfc_2d import LowRankResidualCorrectionHead2d


def _as_pair(value: Sequence[int] | int, *, name: str) -> tuple[int, int]:
    pair = (value, value) if isinstance(value, int) else tuple(int(v) for v in value)
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value}.")
    return pair


def _as_dilation_cycle(value: Sequence[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    cycle = tuple(int(v) for v in value)
    if len(cycle) == 0:
        raise ValueError("dilation_cycle must not be empty")
    if any(v <= 0 for v in cycle):
        raise ValueError(f"dilation values must be positive, got {cycle}")
    return cycle


def _packed_complex_features(
    x: torch.Tensor,
    *,
    n_chan: int,
    include_magnitude: bool,
    include_logmag: bool,
) -> torch.Tensor:
    """Return NPU-friendly packed RI plus optional magnitude features."""

    _runtime_assert(x.shape[1] == 2 * n_chan, f"Expected {2 * n_chan} packed channels, got {x.shape}")
    if not include_magnitude and not include_logmag:
        return x

    feats: list[torch.Tensor] = [x]
    mags: list[torch.Tensor] = []
    ri_channels = torch.split(x, 1, dim=1)
    for chan_idx in range(n_chan):
        real = ri_channels[2 * chan_idx]
        imag = ri_channels[2 * chan_idx + 1]
        mags.append(torch.sqrt(real * real + imag * imag + 1e-8))
    mag = torch.cat(mags, dim=1)
    if include_magnitude:
        feats.append(mag)
    if include_logmag:
        feats.append(torch.log1p(mag))
    return torch.cat(feats, dim=1)


def _source_chunks(x: torch.Tensor, *, n_src: int, channels: int) -> list[torch.Tensor]:
    _runtime_assert(x.shape[1] == n_src * channels, f"Expected {n_src * channels} channels, got {x.shape}")
    return list(torch.split(x, channels, dim=1))


def _sum_chunks(chunks: list[torch.Tensor]) -> torch.Tensor:
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    return total


class SourceSeedSplitter2d(nn.Module):
    """Split shared mixture tokens into source-packed streams with learned seeds."""

    def __init__(self, channels: int, n_src: int):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if n_src <= 0:
            raise ValueError(f"n_src must be positive, got {n_src}")
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.pre = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.source_proj = nn.Conv2d(channels, n_src * channels, kernel_size=1, bias=True)
        self.shared_skip = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.source_bias = nn.Parameter(torch.zeros(1, n_src * channels, 1, 1))
        self.split_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, mixture_tokens: torch.Tensor) -> torch.Tensor:
        source_delta = self.source_proj(self.pre(mixture_tokens)) * self.split_scale + self.source_bias
        shared = self.shared_skip(mixture_tokens)
        chunks = _source_chunks(source_delta, n_src=self.n_src, channels=self.channels)
        return torch.cat([chunk + shared for chunk in chunks], dim=1)


class SourceCompetitionFusion2d(nn.Module):
    """Teacher-style mixture/source/other-source fusion using only 1x1 convs."""

    def __init__(self, channels: int, n_src: int, hidden_channels: int | None = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = channels
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.hidden_channels = int(hidden_channels)
        self.norm = RMSNorm2d(3 * channels)
        self.in_proj = nn.Conv2d(3 * channels, 2 * hidden_channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, source_tokens: torch.Tensor, mixture_tokens: torch.Tensor) -> torch.Tensor:
        chunks = _source_chunks(source_tokens, n_src=self.n_src, channels=self.channels)
        source_mean = _sum_chunks(chunks) / float(self.n_src)
        fused = []
        for chunk in chunks:
            if self.n_src > 1:
                other_mean = (source_mean * float(self.n_src) - chunk) / float(self.n_src - 1)
            else:
                other_mean = source_mean
            y = self.norm(torch.cat([chunk, other_mean, mixture_tokens], dim=1))
            value, gate = torch.split(self.in_proj(y), self.hidden_channels, dim=1)
            y = value * torch.sigmoid(gate)
            fused.append(chunk + self.out_proj(y) * self.scale)
        return torch.cat(fused, dim=1)


class SourceCompetitiveDecoderBlock2d(nn.Module):
    """Shared per-source local modeling followed by source competition."""

    def __init__(
        self,
        *,
        channels: int,
        n_src: int,
        time_kernel_size: int,
        band_kernel_size: int,
        time_dilation: int,
        causal: bool,
        expansion: int = 2,
        fusion_hidden_channels: int | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.causal = bool(causal)
        self.refiner = DilatedBandMixBlock2d(
            channels=channels,
            expansion=expansion,
            time_kernel_size=time_kernel_size,
            band_kernel_size=band_kernel_size,
            time_dilation=time_dilation,
            causal=causal,
        )
        self.fusion = SourceCompetitionFusion2d(
            channels=channels,
            n_src=n_src,
            hidden_channels=fusion_hidden_channels,
        )

    def forward(self, source_tokens: torch.Tensor, mixture_tokens: torch.Tensor) -> torch.Tensor:
        chunks = _source_chunks(source_tokens, n_src=self.n_src, channels=self.channels)
        chunks = [self.refiner(chunk) for chunk in chunks]
        return self.fusion(torch.cat(chunks, dim=1), mixture_tokens)

    def stream_context_frames(self) -> int:
        return self.refiner.stream_context_frames()

    def state_tensor_count(self) -> int:
        return self.n_src if self.stream_context_frames() > 0 else 0

    def init_stream_state(
        self, batch_size: int = 1, *, freq_bins: int, device=None, dtype=None
    ) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        if self.state_tensor_count() == 0:
            return ()
        return tuple(
            self.refiner.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)
            for _ in range(self.n_src)
        )

    def forward_stream(
        self,
        source_tokens: torch.Tensor,
        mixture_tokens: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not self.causal:
            raise RuntimeError("forward_stream is only supported when causal=True.")
        expected_states = self.state_tensor_count()
        _runtime_assert(len(states) == expected_states, f"Expected {expected_states} source states, got {len(states)}")
        chunks = _source_chunks(source_tokens, n_src=self.n_src, channels=self.channels)
        new_chunks = []
        new_states = []
        if expected_states == 0:
            new_chunks = [self.refiner(chunk) for chunk in chunks]
        else:
            for chunk, state in zip(chunks, states):
                chunk, state = self.refiner.forward_stream(chunk, state)
                new_chunks.append(chunk)
                new_states.append(state)
        return self.fusion(torch.cat(new_chunks, dim=1), mixture_tokens), tuple(new_states)


class SourceCompetitiveDecoder2d(nn.Module):
    """Repeated source-aware decoder blocks, packed as [B, N*C, T, K]."""

    def __init__(
        self,
        *,
        channels: int,
        n_src: int,
        n_bands: int,
        n_layers: int,
        kernel_size: tuple[int, int] = (3, 3),
        dilation_cycle: Sequence[int] | None = (1, 2, 4),
        causal: bool = True,
        expansion: int = 2,
        fusion_hidden_channels: int | None = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.n_bands = int(n_bands)
        self.causal = bool(causal)
        self.dilation_schedule = _normalize_dilation_schedule(n_layers, _as_dilation_cycle(dilation_cycle))
        self.blocks = nn.ModuleList(
            [
                SourceCompetitiveDecoderBlock2d(
                    channels=channels,
                    n_src=n_src,
                    time_kernel_size=kernel_size[0],
                    band_kernel_size=kernel_size[1],
                    time_dilation=dilation,
                    causal=causal,
                    expansion=expansion,
                    fusion_hidden_channels=fusion_hidden_channels,
                )
                for dilation in self.dilation_schedule
            ]
        )

    def forward(self, source_tokens: torch.Tensor, mixture_tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            source_tokens = block(source_tokens, mixture_tokens)
        return source_tokens

    def stream_context_frames(self) -> int:
        return sum(block.stream_context_frames() for block in self.blocks)

    def state_tensor_count(self) -> int:
        return sum(block.state_tensor_count() for block in self.blocks)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        states: list[torch.Tensor] = []
        for block in self.blocks:
            states.extend(block.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype))
        return tuple(states)

    def forward_stream(
        self,
        source_tokens: torch.Tensor,
        mixture_tokens: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not self.causal:
            raise RuntimeError("forward_stream is only supported when causal=True.")
        expected_states = self.state_tensor_count()
        _runtime_assert(len(states) == expected_states, f"Expected {expected_states} decoder states, got {len(states)}")
        new_states = []
        state_idx = 0
        for block in self.blocks:
            block_end = state_idx + block.state_tensor_count()
            source_tokens, block_states = block.forward_stream(
                source_tokens,
                mixture_tokens,
                states[state_idx:block_end],
            )
            state_idx = block_end
            new_states.extend(block_states)
        return source_tokens, tuple(new_states)


class SourceSharedMelBandReconstructionDecoder2d(nn.Module):
    """Weight-shared SFC K->F reconstruction for all source streams."""

    def __init__(self, *, channels: int, n_src: int, n_chan: int, band_spec: AdaptiveMelBandSpec2d):
        super().__init__()
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.expander = SoftBandQueryExpander2d(channels=channels, band_spec=band_spec)
        self.mask_head = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, 2 * channels, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(2 * channels, 2 * n_chan, kernel_size=1, bias=True),
        )

    def forward(self, source_tokens: torch.Tensor, query_tokens: torch.Tensor) -> torch.Tensor:
        chunks = _source_chunks(source_tokens, n_src=self.n_src, channels=self.channels)
        masks = [self.mask_head(self.expander(chunk, query_tokens)) for chunk in chunks]
        return torch.cat(masks, dim=1)


class OnlineSourceAwareMelBandStudentSFC2D(nn.Module):
    """Strict-online student distilled from the source-aware MelBand RoFormer teacher."""

    def __init__(
        self,
        n_freq: int,
        *,
        n_fft: int | None = None,
        sample_rate: int = 44100,
        n_src: int = 3,
        n_chan: int = 1,
        n_bands: int = 80,
        d_model: int = 40,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 3,
        long_branch_layers: int = 1,
        correction_layers: int = 1,
        correction_channels: int = 16,
        encoder_expansion: int = 2,
        decoder_expansion: int = 2,
        decoder_fusion_hidden: int | None = None,
        kernel_size: Sequence[int] | int = (3, 3),
        decoder_kernel_size: Sequence[int] | int | None = None,
        routing_kernel_size: Sequence[int] | int = (1, 3),
        encoder_dilation_cycle: Sequence[int] | None = (1, 2, 4),
        decoder_dilation_cycle: Sequence[int] | None = (1, 2, 4),
        long_branch_dilation_cycle: Sequence[int] | None = (1, 2, 4),
        low_freq_hz: float = 1000.0,
        low_freq_band_fraction: float = 0.45,
        overlap_factor: float = 1.5,
        low_freq_overlap_factor: float = 2.0,
        bin_frequencies_hz: torch.Tensor | Sequence[float] | None = None,
        include_magnitude_features: bool = True,
        include_logmag_features: bool = False,
        causal: bool = True,
        masking: bool = True,
        mixture_consistency: bool = True,
        routing_normalization: str = "softmax",
    ):
        super().__init__()
        del n_fft
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if n_encoder_layers < 0 or n_decoder_layers < 0 or long_branch_layers < 0 or correction_layers < 0:
            raise ValueError("layer counts must be non-negative")
        kernel_size = _as_pair(kernel_size, name="kernel_size")
        decoder_kernel_size = (
            (1, kernel_size[1])
            if decoder_kernel_size is None
            else _as_pair(
                decoder_kernel_size,
                name="decoder_kernel_size",
            )
        )
        routing_kernel_size = _as_pair(routing_kernel_size, name="routing_kernel_size")

        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.d_model = int(d_model)
        self.n_encoder_layers = int(n_encoder_layers)
        self.n_decoder_layers = int(n_decoder_layers)
        self.long_branch_layers = int(long_branch_layers)
        self.correction_layers = int(correction_layers)
        self.correction_channels = int(correction_channels)
        self.causal = bool(causal)
        self.masking = bool(masking)
        self.mixture_consistency = bool(mixture_consistency)
        self.include_magnitude_features = bool(include_magnitude_features)
        self.include_logmag_features = bool(include_logmag_features)
        self.encoder_dilation_schedule = _normalize_dilation_schedule(
            n_encoder_layers,
            _as_dilation_cycle(encoder_dilation_cycle),
        )

        feature_channels = 2 * n_chan
        if include_magnitude_features:
            feature_channels += n_chan
        if include_logmag_features:
            feature_channels += n_chan

        band_spec = AdaptiveMelBandSpec2d(
            n_freq=n_freq,
            n_bands=n_bands,
            sample_rate=sample_rate,
            low_freq_hz=low_freq_hz,
            low_freq_band_fraction=low_freq_band_fraction,
            overlap_factor=overlap_factor,
            low_freq_overlap_factor=low_freq_overlap_factor,
            bin_frequencies_hz=bin_frequencies_hz,
        )
        self.band_spec = band_spec
        self.input_frontend = nn.Sequential(
            nn.Conv2d(feature_channels, d_model, kernel_size=1, bias=True),
            RMSNorm2d(d_model),
        )
        self.compressor = SoftBandQueryCompressor2d(
            channels=d_model,
            band_spec=band_spec,
            kernel_size=routing_kernel_size,
            causal=causal,
            normalization=routing_normalization,
        )
        self.encoder = nn.ModuleList(
            [
                DilatedBandMixBlock2d(
                    channels=d_model,
                    expansion=encoder_expansion,
                    time_kernel_size=kernel_size[0],
                    band_kernel_size=kernel_size[1],
                    time_dilation=dilation,
                    causal=causal,
                )
                for dilation in self.encoder_dilation_schedule
            ]
        )
        self.source_splitter = SourceSeedSplitter2d(channels=d_model, n_src=n_src)
        self.source_decoder = SourceCompetitiveDecoder2d(
            channels=d_model,
            n_src=n_src,
            n_bands=n_bands,
            n_layers=n_decoder_layers,
            kernel_size=decoder_kernel_size,
            dilation_cycle=decoder_dilation_cycle,
            causal=causal,
            expansion=decoder_expansion,
            fusion_hidden_channels=decoder_fusion_hidden,
        )
        self.long_temporal_refiner = Mamba2LiteTemporalBranch2d(
            channels=d_model,
            n_bands=n_bands,
            n_layers=long_branch_layers,
            kernel_size=kernel_size,
            dilation_cycle=long_branch_dilation_cycle,
            causal=causal,
        )
        self.reconstruction = SourceSharedMelBandReconstructionDecoder2d(
            channels=d_model,
            n_src=n_src,
            n_chan=n_chan,
            band_spec=band_spec,
        )
        self.residual_expander = SoftBandQueryExpander2d(channels=d_model, band_spec=band_spec)
        self.correction_head = LowRankResidualCorrectionHead2d(
            context_channels=d_model,
            correction_channels=correction_channels,
            n_freq=n_freq,
            n_src=n_src,
            n_chan=n_chan,
            n_layers=correction_layers,
            kernel_size=kernel_size,
            causal=causal,
        )

    @property
    def _has_compressor_state(self) -> bool:
        return self.compressor.stream_context_frames() > 0

    def _apply_masks(self, x: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return _apply_packed_complex_mask_no_repeat(
            x=x,
            y=masks,
            n_src=self.n_src,
            n_chan=self.n_chan,
        )

    def _apply_mixture_consistency(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if not self.mixture_consistency:
            return estimates
        chunks = _source_chunks(estimates, n_src=self.n_src, channels=2 * self.n_chan)
        correction = (mixture - _sum_chunks(chunks)) / float(self.n_src)
        return torch.cat([chunk + correction for chunk in chunks], dim=1)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = _packed_complex_features(
            x,
            n_chan=self.n_chan,
            include_magnitude=self.include_magnitude_features,
            include_logmag=self.include_logmag_features,
        )
        h = self.input_frontend(features)
        z, query_tokens = self.compressor(h)
        for block in self.encoder:
            z = block(z)
        return z, query_tokens

    def _decode(self, x: torch.Tensor, z: torch.Tensor, query_tokens: torch.Tensor) -> torch.Tensor:
        z_long = self.long_temporal_refiner(z)
        source_tokens = self.source_splitter(z)
        source_tokens = self.source_decoder(source_tokens, z)
        primary_masks = self.reconstruction(source_tokens, query_tokens)
        residual_context = self.residual_expander(z_long, query_tokens)
        mask_delta = self.correction_head(x, primary_masks, residual_context)
        final_masks = primary_masks + mask_delta
        if not self.masking:
            return final_masks
        estimates = self._apply_masks(x, final_masks)
        return self._apply_mixture_consistency(estimates, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected [B,2M,T,F], got {x.shape}")
        _runtime_assert(x.shape[1] == 2 * self.n_chan, f"Expected {2 * self.n_chan} packed channels, got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"Expected F={self.n_freq}, got {x.shape}")
        z, query_tokens = self._encode(x)
        return self._decode(x, z, query_tokens)

    def stream_context_frames(self) -> int:
        if not self.causal:
            return 0
        compressor_ctx = self.compressor.stream_context_frames()
        encoder_ctx = sum(block.stream_context_frames() for block in self.encoder)
        decoder_ctx = self.source_decoder.stream_context_frames()
        long_ctx = self.long_temporal_refiner.stream_context_frames()
        correction_ctx = self.correction_head.stream_context_frames()
        return compressor_ctx + encoder_ctx + max(decoder_ctx, long_ctx) + correction_ctx

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        states: list[torch.Tensor] = []
        if self._has_compressor_state:
            states.append(
                self.compressor.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype)
            )
        states.extend(
            block.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype)
            for block in self.encoder
        )
        states.extend(self.source_decoder.init_stream_state(batch_size=batch_size, device=device, dtype=dtype))
        states.extend(self.long_temporal_refiner.init_stream_state(batch_size=batch_size, device=device, dtype=dtype))
        states.extend(self.correction_head.init_stream_state(batch_size=batch_size, device=device, dtype=dtype))
        return tuple(states)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not self.causal:
            raise RuntimeError("forward_stream is only supported when causal=True.")
        _runtime_assert(x.ndim == 4, f"Expected [B,2M,T,F], got {x.shape}")
        _runtime_assert(x.shape[1] == 2 * self.n_chan, f"Expected {2 * self.n_chan} packed channels, got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"Expected F={self.n_freq}, got {x.shape}")
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        comp_count = 1 if self._has_compressor_state else 0
        encoder_count = len(self.encoder)
        decoder_count = self.source_decoder.state_tensor_count()
        long_count = len(self.long_temporal_refiner.blocks)
        correction_count = len(self.correction_head.blocks)
        expected_states = comp_count + encoder_count + decoder_count + long_count + correction_count
        _runtime_assert(len(state) == expected_states, f"Expected {expected_states} states, got {len(state)}")

        features = _packed_complex_features(
            x,
            n_chan=self.n_chan,
            include_magnitude=self.include_magnitude_features,
            include_logmag=self.include_logmag_features,
        )
        h = self.input_frontend(features)
        state_idx = 0
        if self._has_compressor_state:
            (z, query_tokens), new_comp_state = self.compressor.forward_stream(h, state[state_idx])
            state_idx += 1
        else:
            (z, query_tokens), new_comp_state = self.compressor.forward_stream(h, None)

        new_states: list[torch.Tensor] = []
        if self._has_compressor_state:
            new_states.append(new_comp_state)

        new_encoder_states = []
        encoder_end = state_idx + encoder_count
        for block, block_state in zip(self.encoder, state[state_idx:encoder_end]):
            z, block_state = block.forward_stream(z, block_state)
            new_encoder_states.append(block_state)
        state_idx = encoder_end

        decoder_end = state_idx + decoder_count
        long_end = decoder_end + long_count
        correction_end = long_end + correction_count

        source_tokens = self.source_splitter(z)
        source_tokens, new_decoder_states = self.source_decoder.forward_stream(
            source_tokens,
            z,
            state[state_idx:decoder_end],
        )
        z_long, new_long_states = self.long_temporal_refiner.forward_stream(z, state[decoder_end:long_end])
        primary_masks = self.reconstruction(source_tokens, query_tokens)
        residual_context = self.residual_expander(z_long, query_tokens)
        mask_delta, new_correction_states = self.correction_head.forward_stream(
            x,
            primary_masks,
            residual_context,
            state[long_end:correction_end],
        )
        final_masks = primary_masks + mask_delta
        y = self._apply_mixture_consistency(self._apply_masks(x, final_masks), x) if self.masking else final_masks
        _runtime_assert(correction_end == len(state), f"Unused stream states: {len(state) - correction_end}")
        return y, (
            *new_states,
            *new_encoder_states,
            *new_decoder_states,
            *new_long_states,
            *new_correction_states,
        )

    def init_input_history(self, batch_size: int = 1, *, device=None, dtype=None) -> torch.Tensor:
        history_frames = self.stream_context_frames()
        return torch.zeros(batch_size, 2 * self.n_chan, history_frames, self.n_freq, device=device, dtype=dtype)

    def forward_stream_recompute(
        self,
        x: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "Exact low-memory recomputation from raw input history is not implemented "
            "for OnlineSourceAwareMelBandStudentSFC2D. Use forward_stream with layer caches for strict equivalence."
        )

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        states = self.init_stream_state(
            batch_size=batch_size,
            device=next(self.parameters()).device,
            dtype=next(self.parameters()).dtype,
        )
        return sum(int(state.numel()) for state in states)

    def input_history_numel(self, batch_size: int = 1) -> int:
        return batch_size * 2 * self.n_chan * self.stream_context_frames() * self.n_freq

    def state_size_bytes(
        self,
        *,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float16,
        mode: str = "layer_cache",
    ) -> int:
        element_size = torch.tensor([], dtype=dtype).element_size()
        if mode == "layer_cache":
            return self.layer_cache_numel(batch_size=batch_size) * element_size
        if mode == "input_history":
            return self.input_history_numel(batch_size=batch_size) * element_size
        raise ValueError(f"Unsupported state mode: {mode}")


class OnlineSourceAwareMelBandStudentSFCModel(nn.Module):
    """Complex-STFT wrapper around OnlineSourceAwareMelBandStudentSFC2D."""

    def __init__(self, *, n_freq: int, n_src: int = 3, n_chan: int = 1, **kwargs):
        super().__init__()
        self.core = OnlineSourceAwareMelBandStudentSFC2D(n_freq=n_freq, n_src=n_src, n_chan=n_chan, **kwargs)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        x2d = pack_complex_stft_as_2d(x)
        y2d = self.core(x2d)
        return unpack_2d_to_complex_stft(y2d, n_src=self.n_src, n_chan=self.n_chan)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None):
        return self.core.init_stream_state(batch_size=batch_size, device=device, dtype=dtype)

    def forward_stream(self, x2d: torch.Tensor, state=None):
        return self.core.forward_stream(x2d, state)

    def init_input_history(self, batch_size: int = 1, *, device=None, dtype=None):
        return self.core.init_input_history(batch_size=batch_size, device=device, dtype=dtype)

    def forward_stream_recompute(self, x2d: torch.Tensor, history=None):
        return self.core.forward_stream_recompute(x2d, history)


def _core_bin_frequencies_hz(
    *,
    n_fft: int,
    sample_rate: int,
    core_n_freq: int,
    freq_preprocess_enabled: bool,
    freq_preprocess_keep_bins: int | None,
    freq_preprocess_target_bins: int | None,
    freq_preprocess_mode: str,
    dc_bypass_enabled: bool,
) -> torch.Tensor | None:
    full_n_freq = (int(n_fft) // 2) + 1
    body_n_freq = resolve_frequency_input_n_freq(full_n_freq, dc_bypass_enabled=dc_bypass_enabled)
    if freq_preprocess_enabled:
        if freq_preprocess_keep_bins is None or freq_preprocess_target_bins is None:
            raise ValueError("keep_bins and target_bins are required for frequency preprocessing")
        return build_hybrid_frequency_bin_frequencies(
            body_n_freq,
            keep_bins=int(freq_preprocess_keep_bins),
            target_bins=int(freq_preprocess_target_bins),
            n_fft=n_fft,
            sample_rate=sample_rate,
            mode=freq_preprocess_mode,
            dc_bypass_enabled=dc_bypass_enabled,
        )
    if dc_bypass_enabled:
        first_bin = 1
        bin_indices = torch.arange(first_bin, first_bin + core_n_freq, dtype=torch.float32)
        return bin_indices * (float(sample_rate) / float(n_fft))
    return None


def build_source_aware_melband_student_sfc_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    n_bands: int = 80,
    d_model: int = 40,
    n_encoder_layers: int = 3,
    n_decoder_layers: int = 3,
    long_branch_layers: int = 1,
    correction_layers: int = 1,
    correction_channels: int = 16,
    encoder_expansion: int = 2,
    decoder_expansion: int = 2,
    decoder_fusion_hidden: int | None = None,
    kernel_size: Sequence[int] | int = (3, 3),
    decoder_kernel_size: Sequence[int] | int | None = None,
    routing_kernel_size: Sequence[int] | int = (1, 3),
    encoder_dilation_cycle: Sequence[int] | None = (1, 2, 4),
    decoder_dilation_cycle: Sequence[int] | None = (1, 2, 4),
    long_branch_dilation_cycle: Sequence[int] | None = (1, 2, 4),
    low_freq_hz: float = 1000.0,
    low_freq_band_fraction: float = 0.45,
    overlap_factor: float = 1.5,
    low_freq_overlap_factor: float = 2.0,
    include_magnitude_features: bool = True,
    include_logmag_features: bool = False,
    causal: bool = True,
    masking: bool = True,
    mixture_consistency: bool = True,
    routing_normalization: str = "softmax",
    freq_preprocess_enabled: bool = True,
    freq_preprocess_keep_bins: int | None = 475,
    freq_preprocess_target_bins: int | None = 512,
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
    full_n_freq = (n_fft // 2) + 1
    core_n_freq = resolve_preprocessed_n_freq(
        full_n_freq,
        enabled=freq_preprocess_enabled,
        keep_bins=freq_preprocess_keep_bins,
        target_bins=freq_preprocess_target_bins,
        dc_bypass_enabled=dc_bypass_enabled,
    )
    core_n_fft = 2 * (core_n_freq - 1)
    bin_frequencies_hz = _core_bin_frequencies_hz(
        n_fft=n_fft,
        sample_rate=fs,
        core_n_freq=core_n_freq,
        freq_preprocess_enabled=freq_preprocess_enabled,
        freq_preprocess_keep_bins=freq_preprocess_keep_bins,
        freq_preprocess_target_bins=freq_preprocess_target_bins,
        freq_preprocess_mode=freq_preprocess_mode,
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
    core = OnlineSourceAwareMelBandStudentSFC2D(
        n_freq=core_n_freq,
        n_fft=core_n_fft,
        sample_rate=fs,
        n_src=n_src,
        n_chan=n_chan,
        n_bands=n_bands,
        d_model=d_model,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        long_branch_layers=long_branch_layers,
        correction_layers=correction_layers,
        correction_channels=correction_channels,
        encoder_expansion=encoder_expansion,
        decoder_expansion=decoder_expansion,
        decoder_fusion_hidden=decoder_fusion_hidden,
        kernel_size=kernel_size,
        decoder_kernel_size=decoder_kernel_size,
        routing_kernel_size=routing_kernel_size,
        encoder_dilation_cycle=encoder_dilation_cycle,
        decoder_dilation_cycle=decoder_dilation_cycle,
        long_branch_dilation_cycle=long_branch_dilation_cycle,
        low_freq_hz=low_freq_hz,
        low_freq_band_fraction=low_freq_band_fraction,
        overlap_factor=overlap_factor,
        low_freq_overlap_factor=low_freq_overlap_factor,
        bin_frequencies_hz=bin_frequencies_hz,
        include_magnitude_features=include_magnitude_features,
        include_logmag_features=include_logmag_features,
        causal=causal,
        masking=masking,
        mixture_consistency=mixture_consistency,
        routing_normalization=routing_normalization,
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


def estimate_source_aware_melband_student_sfc_params(
    *,
    n_fft: int = 2048,
    fs: int = 44100,
    n_src: int = 3,
    n_chan: int = 1,
    n_bands: int = 80,
    d_model: int = 40,
    n_encoder_layers: int = 3,
    n_decoder_layers: int = 3,
    long_branch_layers: int = 1,
    correction_layers: int = 1,
    correction_channels: int = 16,
    decoder_kernel_size: Sequence[int] | int | None = None,
) -> int:
    model = OnlineSourceAwareMelBandStudentSFC2D(
        n_freq=(n_fft // 2) + 1,
        sample_rate=fs,
        n_src=n_src,
        n_chan=n_chan,
        n_bands=n_bands,
        d_model=d_model,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        long_branch_layers=long_branch_layers,
        correction_layers=correction_layers,
        correction_channels=correction_channels,
        decoder_kernel_size=decoder_kernel_size,
    )
    return sum(p.numel() for p in model.parameters())
