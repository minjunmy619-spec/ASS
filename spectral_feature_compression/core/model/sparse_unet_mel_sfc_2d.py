"""
Sparse U-Net Mel-SFC separator for music-first quality probes.

This model is intentionally opt-in.  It keeps the online 2D tensor contract used
by the NPU candidates, but spends more capacity on a sparse low/mid/high mel
band U-Net than the strict RT+ student path.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    build_frequency_preprocessor,
    build_pcen_preprocessor,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.npu_capacity_blocks_2d import build_capacity_mixers
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_sfc_2d import (
    OnlineConvBlock,
    RMSNorm2d,
    _runtime_assert,
    apply_packed_complex_mask,
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import (
    SoftBandCompressor2d,
    SoftBandExpander2d,
)


def _as_pair(value: Sequence[int] | int, *, name: str) -> tuple[int, int]:
    pair = (value, value) if isinstance(value, int) else tuple(int(v) for v in value)
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value}.")
    return pair


def _as_triplet(value: Sequence[int], *, name: str) -> tuple[int, int, int]:
    triplet = tuple(int(v) for v in value)
    if len(triplet) != 3:
        raise ValueError(f"{name} must contain exactly three values, got {value}.")
    if min(triplet) <= 0:
        raise ValueError(f"{name} values must be positive, got {value}.")
    return triplet


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _stream_state_size(blocks: nn.ModuleList) -> int:
    return len(blocks)


def _init_stream_states(
    blocks: nn.ModuleList,
    *,
    batch_size: int,
    freq_bins: int,
    device: torch.device | None,
    dtype: torch.dtype | None,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        block.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype) for block in blocks
    )


def _forward_stream_blocks(
    x: torch.Tensor,
    blocks: nn.ModuleList,
    states: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    _runtime_assert(len(states) == len(blocks), f"Expected {len(blocks)} states, got {len(states)}")
    new_states = []
    for block, state in zip(blocks, states):
        x, state = block.forward_stream(x, state)
        new_states.append(state)
    return x, tuple(new_states)


class RegionalMelBandSpec2d(nn.Module):
    """
    Mel-like branch prior restricted to one overlapped frequency region.
    """

    def __init__(
        self,
        *,
        n_freq: int,
        n_bands: int,
        sample_rate: int,
        f_min_hz: float,
        f_max_hz: float,
    ):
        super().__init__()
        if n_freq <= 0:
            raise ValueError(f"n_freq must be positive, got {n_freq}.")
        if n_bands <= 0:
            raise ValueError(f"n_bands must be positive, got {n_bands}.")
        nyquist = 0.5 * float(sample_rate)
        if nyquist <= 0.0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
        f_min = max(0.0, min(float(f_min_hz), nyquist))
        f_max = max(f_min + 1.0, min(float(f_max_hz), nyquist))

        basis = self._build_mel_basis(
            n_freq=n_freq,
            n_bands=n_bands,
            nyquist=nyquist,
            f_min_hz=f_min,
            f_max_hz=f_max,
        )
        starts, ends = self._basis_bounds(basis)

        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.f_min_hz = f_min
        self.f_max_hz = f_max
        self.register_buffer("starts", starts)
        self.register_buffer("ends", ends)
        self.register_buffer("basis", basis.view(1, n_bands, 1, n_freq))
        self.register_buffer("frequency_gate", basis.amax(dim=0).clamp(0.0, 1.0).view(1, 1, 1, n_freq))

    @staticmethod
    def _build_mel_basis(
        *,
        n_freq: int,
        n_bands: int,
        nyquist: float,
        f_min_hz: float,
        f_max_hz: float,
    ) -> torch.Tensor:
        freqs = torch.linspace(0.0, nyquist, steps=n_freq, dtype=torch.float32)
        mel_freqs = _hz_to_mel(freqs)
        mel_edges = torch.linspace(
            float(_hz_to_mel(torch.tensor(f_min_hz)).item()),
            float(_hz_to_mel(torch.tensor(f_max_hz)).item()),
            steps=n_bands + 2,
            dtype=torch.float32,
        )

        basis = torch.zeros(n_bands, n_freq, dtype=torch.float32)
        for band_idx in range(n_bands):
            left = mel_edges[band_idx]
            center = mel_edges[band_idx + 1]
            right = mel_edges[band_idx + 2]
            rising = (mel_freqs - left) / (center - left).clamp_min(1e-6)
            falling = (right - mel_freqs) / (right - center).clamp_min(1e-6)
            tri = torch.minimum(rising, falling).clamp(min=0.0)
            tri = torch.where((freqs >= f_min_hz) & (freqs <= f_max_hz), tri, torch.zeros_like(tri))
            if float(tri.max().item()) <= 0.0:
                center_hz = _mel_to_hz(center).clamp(0.0, nyquist)
                nearest = int(torch.argmin(torch.abs(freqs - center_hz)).item())
                tri[nearest] = 1.0
            basis[band_idx] = tri
        return basis

    @staticmethod
    def _basis_bounds(basis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        starts = []
        ends = []
        n_freq = basis.shape[1]
        for band in basis:
            active = torch.nonzero(band > 0.0, as_tuple=False).flatten()
            if active.numel() == 0:
                starts.append(0)
                ends.append(1)
            else:
                starts.append(int(active[0].item()))
                ends.append(min(int(active[-1].item()) + 1, n_freq))
        return torch.tensor(starts, dtype=torch.long), torch.tensor(ends, dtype=torch.long)

    def routing_bias(self) -> torch.Tensor:
        peak = self.basis.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        normalized = self.basis / peak
        active = (self.basis > 0.0).to(dtype=self.basis.dtype)
        return (4.0 * normalized) - (8.0 * (1.0 - active))

    def expansion_basis(self) -> torch.Tensor:
        return self.basis / self.basis.sum(dim=1, keepdim=True).clamp_min(1e-6)


class SparseBandUNetEncoder(nn.Module):
    """Band-axis encoder with a single sparse downsampling stage."""

    def __init__(
        self,
        *,
        channels: int,
        n_bands: int,
        layers: Sequence[int] = (1, 1),
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
    ):
        super().__init__()
        layer_pair = _as_pair(layers, name="encoder layers")
        if n_bands % 2 != 0:
            raise ValueError(f"SparseBandUNetEncoder requires even n_bands, got {n_bands}.")
        self.n_bands = int(n_bands)
        self.down_bands = int(n_bands // 2)
        self.causal = causal
        self.full_blocks = nn.ModuleList(
            [
                OnlineConvBlock(channels, expansion=2, kernel_size=kernel_size, causal=causal)
                for _ in range(layer_pair[0])
            ]
        )
        self.down = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1), bias=True),
            nn.SiLU(),
        )
        self.down_blocks = nn.ModuleList(
            [
                OnlineConvBlock(channels, expansion=2, kernel_size=kernel_size, causal=causal)
                for _ in range(layer_pair[1])
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.full_blocks:
            x = block(x)
        skip = x
        x = self.down(x)
        _runtime_assert(x.shape[-1] == self.down_bands, f"Expected {self.down_bands} bands, got {x.shape}")
        for block in self.down_blocks:
            x = block(x)
        return x, skip

    def stream_context_frames(self) -> int:
        return sum(block.stream_context_frames() for block in self.full_blocks) + sum(
            block.stream_context_frames() for block in self.down_blocks
        )

    def state_count(self) -> int:
        return _stream_state_size(self.full_blocks) + _stream_state_size(self.down_blocks)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        return (
            *_init_stream_states(
                self.full_blocks,
                batch_size=batch_size,
                freq_bins=self.n_bands,
                device=device,
                dtype=dtype,
            ),
            *_init_stream_states(
                self.down_blocks,
                batch_size=batch_size,
                freq_bins=self.down_bands,
                device=device,
                dtype=dtype,
            ),
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        full_count = len(self.full_blocks)
        down_count = len(self.down_blocks)
        _runtime_assert(len(states) == full_count + down_count, f"Unexpected encoder state count: {len(states)}")
        x, full_states = _forward_stream_blocks(x, self.full_blocks, states[:full_count])
        skip = x
        x = self.down(x)
        x, down_states = _forward_stream_blocks(x, self.down_blocks, states[full_count:])
        return x, skip, (*full_states, *down_states)


class SparseBandUNetDecoder(nn.Module):
    """Band-axis decoder with an additive U-Net skip path."""

    def __init__(
        self,
        *,
        channels: int,
        n_bands: int,
        layers: int = 1,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
    ):
        super().__init__()
        if n_bands % 2 != 0:
            raise ValueError(f"SparseBandUNetDecoder requires even n_bands, got {n_bands}.")
        self.n_bands = int(n_bands)
        self.down_bands = int(n_bands // 2)
        self.causal = causal
        self.up = nn.Sequential(
            RMSNorm2d(channels),
            nn.Upsample(scale_factor=(1, 2), mode="nearest"),
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), bias=True),
            nn.SiLU(),
        )
        self.skip_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.skip_scale = nn.Parameter(torch.tensor(1.0))
        self.blocks = nn.ModuleList(
            [OnlineConvBlock(channels, expansion=2, kernel_size=kernel_size, causal=causal) for _ in range(int(layers))]
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        _runtime_assert(x.shape[-1] == skip.shape[-1], f"Decoder upsample shape mismatch: {x.shape} vs {skip.shape}")
        x = x + self.skip_proj(skip) * self.skip_scale
        for block in self.blocks:
            x = block(x)
        return x

    def stream_context_frames(self) -> int:
        return sum(block.stream_context_frames() for block in self.blocks)

    def state_count(self) -> int:
        return _stream_state_size(self.blocks)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        return _init_stream_states(
            self.blocks,
            batch_size=batch_size,
            freq_bins=self.n_bands,
            device=device,
            dtype=dtype,
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        x = self.up(x)
        x = x + self.skip_proj(skip) * self.skip_scale
        return _forward_stream_blocks(x, self.blocks, states)


class SparseBandUNetBranch(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        n_bands: int,
        encoder_layers: Sequence[int] = (1, 1),
        bottleneck_layers: int = 2,
        decoder_layers: int = 1,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
    ):
        super().__init__()
        self.encoder = SparseBandUNetEncoder(
            channels=channels,
            n_bands=n_bands,
            layers=encoder_layers,
            kernel_size=kernel_size,
            causal=causal,
        )
        self.bottleneck = nn.ModuleList(
            [
                OnlineConvBlock(channels, expansion=2, kernel_size=kernel_size, causal=causal)
                for _ in range(int(bottleneck_layers))
            ]
        )
        self.decoder = SparseBandUNetDecoder(
            channels=channels,
            n_bands=n_bands,
            layers=decoder_layers,
            kernel_size=kernel_size,
            causal=causal,
        )
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip = self.encoder(x)
        for block in self.bottleneck:
            x = block(x)
        return self.decoder(x, skip)

    def stream_context_frames(self) -> int:
        return (
            self.encoder.stream_context_frames()
            + sum(block.stream_context_frames() for block in self.bottleneck)
            + self.decoder.stream_context_frames()
        )

    def state_count(self) -> int:
        return self.encoder.state_count() + _stream_state_size(self.bottleneck) + self.decoder.state_count()

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        return (
            *self.encoder.init_stream_state(batch_size=batch_size, device=device, dtype=dtype),
            *_init_stream_states(
                self.bottleneck,
                batch_size=batch_size,
                freq_bins=self.encoder.down_bands,
                device=device,
                dtype=dtype,
            ),
            *self.decoder.init_stream_state(batch_size=batch_size, device=device, dtype=dtype),
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        enc_count = self.encoder.state_count()
        bottleneck_count = len(self.bottleneck)
        dec_count = self.decoder.state_count()
        _runtime_assert(
            len(states) == enc_count + bottleneck_count + dec_count,
            f"Unexpected branch state count: {len(states)}",
        )
        x, skip, enc_states = self.encoder.forward_stream(x, states[:enc_count])
        bottleneck_end = enc_count + bottleneck_count
        x, bottleneck_states = _forward_stream_blocks(x, self.bottleneck, states[enc_count:bottleneck_end])
        x, dec_states = self.decoder.forward_stream(x, skip, states[bottleneck_end:])
        return x, (*enc_states, *bottleneck_states, *dec_states)


class SparseUNetMelSFC2D(nn.Module):
    """
    Sparse low/mid/high Mel-SFC U-Net core on packed complex STFT tensors.
    """

    def __init__(
        self,
        n_freq: int,
        n_fft: int | None = None,
        sample_rate: int = 44100,
        n_src: int = 4,
        n_chan: int = 2,
        d_model: int = 64,
        branch_bands: Sequence[int] = (24, 32, 24),
        encoder_layers: Sequence[int] = (1, 1),
        bottleneck_layers: int = 2,
        decoder_layers: int = 1,
        fullband_capacity_hidden: int = 0,
        fullband_capacity_layers: int = 0,
        kernel_size: Sequence[int] | tuple[int, int] = (3, 3),
        routing_kernel_size: Sequence[int] | tuple[int, int] = (1, 3),
        low_cutoff_hz: float = 250.0,
        mid_cutoff_hz: float = 4000.0,
        region_overlap_hz: float = 250.0,
        routing_normalization: str = "softmax",
        causal: bool = True,
        masking: bool = True,
    ):
        super().__init__()
        del n_fft
        branch_bands = _as_triplet(branch_bands, name="branch_bands")
        kernel_size = _as_pair(kernel_size, name="kernel_size")
        routing_kernel_size = _as_pair(routing_kernel_size, name="routing_kernel_size")
        if any(n_bands % 2 != 0 for n_bands in branch_bands):
            raise ValueError(f"All branch band counts must be even for the U-Net downsample, got {branch_bands}.")
        if not (0.0 < low_cutoff_hz < mid_cutoff_hz):
            raise ValueError(f"Expected 0 < low_cutoff_hz < mid_cutoff_hz, got {low_cutoff_hz}, {mid_cutoff_hz}.")

        self.n_freq = int(n_freq)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.d_model = int(d_model)
        self.branch_bands = branch_bands
        self.causal = causal
        self.masking = masking

        in_ch = 2 * n_chan
        out_ch = 2 * n_src * n_chan
        nyquist = 0.5 * float(sample_rate)
        overlap = max(0.0, float(region_overlap_hz))
        region_bounds = (
            (0.0, min(float(low_cutoff_hz) + overlap, nyquist)),
            (max(0.0, float(low_cutoff_hz) - overlap), min(float(mid_cutoff_hz) + overlap, nyquist)),
            (max(0.0, float(mid_cutoff_hz) - overlap), nyquist),
        )

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, d_model, kernel_size=1, bias=True),
            RMSNorm2d(d_model),
        )
        specs = [
            RegionalMelBandSpec2d(
                n_freq=n_freq,
                n_bands=n_bands,
                sample_rate=sample_rate,
                f_min_hz=f_min,
                f_max_hz=f_max,
            )
            for n_bands, (f_min, f_max) in zip(branch_bands, region_bounds)
        ]
        self.compressors = nn.ModuleList(
            [
                SoftBandCompressor2d(
                    channels=d_model,
                    band_spec=spec,
                    kernel_size=routing_kernel_size,
                    causal=causal,
                    normalization=routing_normalization,
                )
                for spec in specs
            ]
        )
        self.branches = nn.ModuleList(
            [
                SparseBandUNetBranch(
                    channels=d_model,
                    n_bands=n_bands,
                    encoder_layers=encoder_layers,
                    bottleneck_layers=bottleneck_layers,
                    decoder_layers=decoder_layers,
                    kernel_size=kernel_size,
                    causal=causal,
                )
                for n_bands in branch_bands
            ]
        )
        self.expanders = nn.ModuleList([SoftBandExpander2d(channels=d_model, band_spec=spec) for spec in specs])
        self.input_skip = nn.Conv2d(d_model, d_model, kernel_size=1, bias=True)
        self.merge = nn.Sequential(
            RMSNorm2d(d_model),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.fullband_capacity_mixers = build_capacity_mixers(
            channels=d_model,
            hidden_channels=fullband_capacity_hidden,
            n_layers=fullband_capacity_layers,
        )
        self.out_proj = nn.Conv2d(d_model, out_ch, kernel_size=1, bias=True)

    def _forward_branches(self, h: torch.Tensor) -> torch.Tensor:
        expanded = []
        for compressor, branch, expander in zip(self.compressors, self.branches, self.expanders):
            gate = compressor.band_spec.frequency_gate.to(device=h.device, dtype=h.dtype)
            z, _ = compressor(h * gate)
            z = branch(z)
            expanded.append(expander(z))
        merged = expanded[0]
        for value in expanded[1:]:
            merged = merged + value
        return merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")
        h = self.in_proj(x)
        h = self.merge(self._forward_branches(h) + self.input_skip(h))
        for mixer in self.fullband_capacity_mixers:
            h = mixer(h)
        y = self.out_proj(h)
        if self.masking:
            return apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y

    def stream_context_frames(self) -> int:
        if not self.causal:
            return 0
        compressor_ctx = max(compressor.stream_context_frames() for compressor in self.compressors)
        branch_ctx = max(branch.stream_context_frames() for branch in self.branches)
        return compressor_ctx + branch_ctx

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        states = []
        for compressor, branch in zip(self.compressors, self.branches):
            states.append(
                compressor.init_stream_state(
                    batch_size,
                    freq_bins=self.n_freq,
                    device=device,
                    dtype=dtype,
                )
            )
            states.extend(branch.init_stream_state(batch_size=batch_size, device=device, dtype=dtype))
        return tuple(states)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not self.causal:
            raise RuntimeError("forward_stream is only supported when causal=True.")
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        h = self.in_proj(x)
        expanded = []
        new_states = []
        state_idx = 0
        for compressor, branch, expander in zip(self.compressors, self.branches, self.expanders):
            branch_state_count = branch.state_count()
            comp_state = state[state_idx]
            branch_state = state[state_idx + 1 : state_idx + 1 + branch_state_count]
            state_idx += 1 + branch_state_count

            gate = compressor.band_spec.frequency_gate.to(device=h.device, dtype=h.dtype)
            z, new_comp_state = compressor.forward_stream(h * gate, comp_state)
            z, new_branch_state = branch.forward_stream(z, branch_state)
            expanded.append(expander(z))
            new_states.append(new_comp_state)
            new_states.extend(new_branch_state)

        _runtime_assert(state_idx == len(state), f"Unused stream states: {len(state) - state_idx}")
        merged = expanded[0]
        for value in expanded[1:]:
            merged = merged + value
        h = self.merge(merged + self.input_skip(h))
        for mixer in self.fullband_capacity_mixers:
            h = mixer(h)
        y = self.out_proj(h)
        if self.masking:
            y = apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y, tuple(new_states)

    def init_input_history(self, batch_size: int = 1, *, device=None, dtype=None) -> torch.Tensor:
        history_frames = self.stream_context_frames()
        return torch.zeros(batch_size, 2 * self.n_chan, history_frames, self.n_freq, device=device, dtype=dtype)

    def forward_stream_recompute(
        self,
        x: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "Exact low-memory recomputation from raw input history is not implemented for SparseUNetMelSFC2D. "
            "Use forward_stream with layer caches for strict realtime equivalence."
        )

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        states = self.init_stream_state(
            batch_size=batch_size,
            device=self.out_proj.weight.device,
            dtype=self.out_proj.weight.dtype,
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


class SparseUNetMelSFCModel(nn.Module):
    """Complex-STFT wrapper around SparseUNetMelSFC2D."""

    def __init__(self, *, n_freq: int, n_src: int = 4, n_chan: int = 2, **kwargs):
        super().__init__()
        self.core = SparseUNetMelSFC2D(n_freq=n_freq, n_src=n_src, n_chan=n_chan, **kwargs)
        self.n_src = n_src
        self.n_chan = n_chan

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


def build_sparse_unet_mel_sfc_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 4,
    n_chan: int = 2,
    d_model: int = 64,
    branch_bands: Sequence[int] = (24, 32, 24),
    encoder_layers: Sequence[int] = (1, 1),
    bottleneck_layers: int = 2,
    decoder_layers: int = 1,
    fullband_capacity_hidden: int = 8192,
    fullband_capacity_layers: int = 1,
    kernel_size: Sequence[int] | tuple[int, int] = (3, 3),
    routing_kernel_size: Sequence[int] | tuple[int, int] = (1, 3),
    low_cutoff_hz: float = 250.0,
    mid_cutoff_hz: float = 4000.0,
    region_overlap_hz: float = 250.0,
    routing_normalization: str = "softmax",
    causal: bool = True,
    masking: bool = True,
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
    scaling: bool = False,
    css_segment_size: int = 6,
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
    core = SparseUNetMelSFC2D(
        n_freq=core_n_freq,
        n_fft=core_n_fft,
        sample_rate=fs,
        n_src=n_src,
        n_chan=n_chan,
        d_model=d_model,
        branch_bands=branch_bands,
        encoder_layers=encoder_layers,
        bottleneck_layers=bottleneck_layers,
        decoder_layers=decoder_layers,
        fullband_capacity_hidden=fullband_capacity_hidden,
        fullband_capacity_layers=fullband_capacity_layers,
        kernel_size=kernel_size,
        routing_kernel_size=routing_kernel_size,
        low_cutoff_hz=low_cutoff_hz,
        mid_cutoff_hz=mid_cutoff_hz,
        region_overlap_hz=region_overlap_hz,
        routing_normalization=routing_normalization,
        causal=causal,
        masking=masking,
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
