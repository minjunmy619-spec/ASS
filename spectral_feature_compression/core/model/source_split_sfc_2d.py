"""
SepReformer-style early source split over compressed SFC tokens.

The research sketch uses a conceptual 5D tensor ``[B, N, D, T, K]``.  This
implementation keeps the deployed tensor contract 4D by packing the fixed source
axis into channels: ``[B, N * D, T, K]``.  Static Python loops over ``N`` are
used only to apply shared modules per source and are unrolled during export.
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
from spectral_feature_compression.core.model.online_soft_band_query_sfc_2d import (
    SoftBandQueryCompressor2d,
    SoftBandQueryExpander2d,
)
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandSpec2d


def _as_pair(value: Sequence[int] | int, *, name: str) -> tuple[int, int]:
    pair = (value, value) if isinstance(value, int) else tuple(int(v) for v in value)
    if len(pair) != 2:
        raise ValueError(f"{name} must contain exactly two values, got {value}.")
    return pair


def _source_chunks(x: torch.Tensor, *, n_src: int, channels: int) -> list[torch.Tensor]:
    _runtime_assert(x.shape[1] == n_src * channels, f"Expected {n_src * channels} channels, got {x.shape}")
    return [x[:, idx * channels : (idx + 1) * channels, :, :] for idx in range(n_src)]


def _sum_chunks(chunks: list[torch.Tensor]) -> torch.Tensor:
    total = chunks[0]
    for chunk in chunks[1:]:
        total = total + chunk
    return total


class SourceTokenSplitter2d(nn.Module):
    """Split shared compressed SFC tokens into source-packed token channels."""

    def __init__(self, channels: int, n_src: int):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if n_src <= 0:
            raise ValueError(f"n_src must be positive, got {n_src}.")
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.pre = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.source_proj = nn.Conv2d(channels, n_src * channels, kernel_size=1, bias=True)
        self.shared_skip = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.split_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        source_delta = self.source_proj(self.pre(z)) * self.split_scale
        shared = self.shared_skip(z)
        chunks = _source_chunks(source_delta, n_src=self.n_src, channels=self.channels)
        return torch.cat([chunk + shared for chunk in chunks], dim=1)


class SharedSourceRefiner2d(nn.Module):
    """Apply the same refiner blocks to each source token stream."""

    def __init__(
        self,
        *,
        channels: int,
        n_src: int,
        n_bands: int,
        n_layers: int,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.n_bands = int(n_bands)
        self.causal = causal
        self.blocks = nn.ModuleList(
            [OnlineConvBlock(channels, expansion=2, kernel_size=kernel_size, causal=causal) for _ in range(n_layers)]
        )

    def forward(self, source_tokens: torch.Tensor) -> torch.Tensor:
        chunks = _source_chunks(source_tokens, n_src=self.n_src, channels=self.channels)
        for block in self.blocks:
            chunks = [block(chunk) for chunk in chunks]
        return torch.cat(chunks, dim=1)

    def stream_context_frames(self) -> int:
        return sum(block.stream_context_frames() for block in self.blocks)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        states = []
        for block in self.blocks:
            for _ in range(self.n_src):
                states.append(
                    block.init_stream_state(
                        batch_size,
                        freq_bins=self.n_bands,
                        device=device,
                        dtype=dtype,
                    )
                )
        return tuple(states)

    def forward_stream(
        self,
        source_tokens: torch.Tensor,
        states: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        expected_states = len(self.blocks) * self.n_src
        _runtime_assert(len(states) == expected_states, f"Expected {expected_states} refiner states, got {len(states)}")
        chunks = _source_chunks(source_tokens, n_src=self.n_src, channels=self.channels)
        new_states = []
        state_idx = 0
        for block in self.blocks:
            next_chunks = []
            for chunk in chunks:
                chunk, state = block.forward_stream(chunk, states[state_idx])
                state_idx += 1
                next_chunks.append(chunk)
                new_states.append(state)
            chunks = next_chunks
        return torch.cat(chunks, dim=1), tuple(new_states)


class CrossSourceReconstructionMixer2d(nn.Module):
    """Mix per-source tokens with other-source and mixture-token context."""

    def __init__(self, channels: int, n_src: int):
        super().__init__()
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.mix = nn.Sequential(
            RMSNorm2d(3 * channels),
            nn.Conv2d(3 * channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.mix_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, source_tokens: torch.Tensor, mixture_tokens: torch.Tensor) -> torch.Tensor:
        chunks = _source_chunks(source_tokens, n_src=self.n_src, channels=self.channels)
        source_mean = _sum_chunks(chunks) / float(self.n_src)
        mixed = []
        for chunk in chunks:
            if self.n_src > 1:
                other_mean = (source_mean * float(self.n_src) - chunk) / float(self.n_src - 1)
            else:
                other_mean = source_mean
            residual = self.mix(torch.cat([chunk, other_mean, mixture_tokens], dim=1))
            mixed.append(chunk + residual * self.mix_scale)
        return torch.cat(mixed, dim=1)


class SourceSharedReconstructionDecoder2d(nn.Module):
    """Weight-shared K->F decoder and mask head for all split sources."""

    def __init__(self, *, channels: int, n_src: int, n_chan: int, band_spec: SoftBandSpec2d):
        super().__init__()
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.cross_source = CrossSourceReconstructionMixer2d(channels=channels, n_src=n_src)
        self.expander = SoftBandQueryExpander2d(channels=channels, band_spec=band_spec)
        self.out_proj = nn.Conv2d(channels, 2 * n_chan, kernel_size=1, bias=True)

    def forward(
        self,
        source_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        mixture_tokens: torch.Tensor,
    ) -> torch.Tensor:
        source_tokens = self.cross_source(source_tokens, mixture_tokens)
        chunks = _source_chunks(source_tokens, n_src=self.n_src, channels=self.channels)
        masks = [self.out_proj(self.expander(chunk, query_tokens)) for chunk in chunks]
        return torch.cat(masks, dim=1)


class OnlineSourceSplitSFC2D(nn.Module):
    """Online SFC core with SepReformer-style early source disentanglement."""

    def __init__(
        self,
        n_freq: int,
        n_bands: int = 64,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
        n_src: int = 3,
        n_chan: int = 1,
        d_model: int = 32,
        n_shared_layers: int = 1,
        n_source_layers: int = 2,
        shared_capacity_hidden: int = 0,
        shared_capacity_layers: int = 0,
        kernel_size: Sequence[int] | tuple[int, int] = (3, 3),
        routing_kernel_size: Sequence[int] | tuple[int, int] = (1, 3),
        causal: bool = True,
        masking: bool = True,
        routing_normalization: str = "softmax",
    ):
        super().__init__()
        kernel_size = _as_pair(kernel_size, name="kernel_size")
        routing_kernel_size = _as_pair(routing_kernel_size, name="routing_kernel_size")
        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.d_model = int(d_model)
        self.causal = causal
        self.masking = masking

        band_spec = SoftBandSpec2d(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        self.in_proj = nn.Sequential(
            nn.Conv2d(2 * n_chan, d_model, kernel_size=1, bias=True),
            RMSNorm2d(d_model),
        )
        self.compressor = SoftBandQueryCompressor2d(
            channels=d_model,
            band_spec=band_spec,
            kernel_size=routing_kernel_size,
            causal=causal,
            normalization=routing_normalization,
        )
        self.shared_analysis = nn.ModuleList(
            [
                OnlineConvBlock(d_model, expansion=2, kernel_size=kernel_size, causal=causal)
                for _ in range(n_shared_layers)
            ]
        )
        self.shared_capacity_mixers = build_capacity_mixers(
            channels=d_model,
            hidden_channels=shared_capacity_hidden,
            n_layers=shared_capacity_layers,
        )
        self.source_splitter = SourceTokenSplitter2d(channels=d_model, n_src=n_src)
        self.source_refiner = SharedSourceRefiner2d(
            channels=d_model,
            n_src=n_src,
            n_bands=n_bands,
            n_layers=n_source_layers,
            kernel_size=kernel_size,
            causal=causal,
        )
        self.reconstructor = SourceSharedReconstructionDecoder2d(
            channels=d_model,
            n_src=n_src,
            n_chan=n_chan,
            band_spec=band_spec,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")
        h = self.in_proj(x)
        z, query_tokens = self.compressor(h)
        for block_idx, block in enumerate(self.shared_analysis):
            z = block(z)
            if block_idx < len(self.shared_capacity_mixers):
                z = self.shared_capacity_mixers[block_idx](z)
        for block_idx in range(len(self.shared_analysis), len(self.shared_capacity_mixers)):
            z = self.shared_capacity_mixers[block_idx](z)
        source_tokens = self.source_splitter(z)
        source_tokens = self.source_refiner(source_tokens)
        y = self.reconstructor(source_tokens, query_tokens, z)
        if self.masking:
            return apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y

    def stream_context_frames(self) -> int:
        if not self.causal:
            return 0
        return (
            self.compressor.stream_context_frames()
            + sum(block.stream_context_frames() for block in self.shared_analysis)
            + self.source_refiner.stream_context_frames()
        )

    def init_stream_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        comp = self.compressor.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype)
        shared = tuple(
            block.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype)
            for block in self.shared_analysis
        )
        source = self.source_refiner.init_stream_state(batch_size=batch_size, device=device, dtype=dtype)
        return (comp, *shared, *source)

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

        shared_count = len(self.shared_analysis)
        expected_states = 1 + shared_count + len(self.source_refiner.blocks) * self.n_src
        _runtime_assert(len(state) == expected_states, f"Expected {expected_states} stream states, got {len(state)}")

        h = self.in_proj(x)
        (z, query_tokens), new_comp_state = self.compressor.forward_stream(h, state[0])
        new_shared_states = []
        for block_idx, (block, block_state) in enumerate(zip(self.shared_analysis, state[1 : 1 + shared_count])):
            z, block_state = block.forward_stream(z, block_state)
            if block_idx < len(self.shared_capacity_mixers):
                z = self.shared_capacity_mixers[block_idx](z)
            new_shared_states.append(block_state)
        for block_idx in range(len(self.shared_analysis), len(self.shared_capacity_mixers)):
            z = self.shared_capacity_mixers[block_idx](z)

        source_tokens = self.source_splitter(z)
        source_tokens, new_source_states = self.source_refiner.forward_stream(source_tokens, state[1 + shared_count :])
        y = self.reconstructor(source_tokens, query_tokens, z)
        if self.masking:
            y = apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y, (new_comp_state, *new_shared_states, *new_source_states)

    def init_input_history(self, batch_size: int = 1, *, device=None, dtype=None) -> torch.Tensor:
        history_frames = self.stream_context_frames()
        return torch.zeros(batch_size, 2 * self.n_chan, history_frames, self.n_freq, device=device, dtype=dtype)

    def forward_stream_recompute(
        self,
        x: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "Exact low-memory recomputation from raw input history is not implemented for OnlineSourceSplitSFC2D. "
            "Use forward_stream with layer caches for strict realtime equivalence."
        )

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        states = self.init_stream_state(
            batch_size=batch_size,
            device=self.reconstructor.out_proj.weight.device,
            dtype=self.reconstructor.out_proj.weight.dtype,
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


class OnlineSourceSplitSFCModel(nn.Module):
    """Complex-STFT wrapper around OnlineSourceSplitSFC2D."""

    def __init__(self, *, n_freq: int, n_src: int = 3, n_chan: int = 1, **kwargs):
        super().__init__()
        self.core = OnlineSourceSplitSFC2D(n_freq=n_freq, n_src=n_src, n_chan=n_chan, **kwargs)
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


def build_source_split_sfc_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_bands: int = 64,
    band_config: str = "musical",
    n_src: int = 3,
    n_chan: int = 1,
    d_model: int = 32,
    n_shared_layers: int = 1,
    n_source_layers: int = 2,
    shared_capacity_hidden: int = 6144,
    shared_capacity_layers: int = 4,
    kernel_size: Sequence[int] | tuple[int, int] = (3, 3),
    routing_kernel_size: Sequence[int] | tuple[int, int] = (1, 3),
    causal: bool = True,
    masking: bool = True,
    routing_normalization: str = "softmax",
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
    core = OnlineSourceSplitSFC2D(
        n_freq=core_n_freq,
        n_bands=n_bands,
        n_fft=core_n_fft,
        sample_rate=fs,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        d_model=d_model,
        n_shared_layers=n_shared_layers,
        n_source_layers=n_source_layers,
        shared_capacity_hidden=shared_capacity_hidden,
        shared_capacity_layers=shared_capacity_layers,
        kernel_size=kernel_size,
        routing_kernel_size=routing_kernel_size,
        causal=causal,
        masking=masking,
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
