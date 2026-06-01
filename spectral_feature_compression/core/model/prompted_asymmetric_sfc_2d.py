"""
Online Prompted Asymmetric SFC.

This module implements Proposal D as a strict packed-2D online core.  The NPU
export path uses a fixed prompt set, so the exported graph keeps the same single
input contract as the other online cores.  The prompt-conditioned decoder is
weight-shared across stems, and the encoder side is intentionally deeper than
the decoder side.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from spectral_feature_compression.core.model.adaptive_mel_sfc_2d import _apply_packed_complex_mask_no_repeat
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
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)
from spectral_feature_compression.core.model.online_soft_band_query_sfc_2d import (
    SoftBandQueryCompressor2d,
    SoftBandQueryExpander2d,
)
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandSpec2d

DEFAULT_PROMPT_LABELS = (
    "speech",
    "music",
    "effects",
    "vocals",
    "drums",
    "bass",
    "other",
    "dialogue",
)


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


class PromptConditioner2d(nn.Module):
    """Prompt FiLM block for 4D band-token tensors."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.prompt_norm = RMSNorm2d(channels)
        self.scale = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.bias = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.scale_strength = nn.Parameter(torch.tensor(0.1))
        self.bias_strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        _runtime_assert(prompt.ndim == 4, f"Expected 4D prompt tensor, got {prompt.shape}")
        _runtime_assert(prompt.shape[1] == self.channels, f"{prompt.shape} vs channels={self.channels}")
        prompt = self.prompt_norm(prompt)
        scale = torch.sigmoid(self.scale(prompt)) * self.scale_strength
        bias = self.bias(prompt) * self.bias_strength
        return x * (1.0 + scale) + bias


class PromptedTokenSplitter2d(nn.Module):
    """Build one source-token stream per prompt using shared parameters."""

    def __init__(self, *, channels: int, n_src: int):
        super().__init__()
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.conditioner = PromptConditioner2d(channels)
        self.pre = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.delta = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.shared_skip = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.split_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, z: torch.Tensor, prompts: tuple[torch.Tensor, ...]) -> torch.Tensor:
        _runtime_assert(len(prompts) == self.n_src, f"Expected {self.n_src} prompts, got {len(prompts)}")
        shared = self.shared_skip(z)
        chunks = []
        for prompt in prompts:
            conditioned = self.conditioner(z, prompt)
            chunks.append(shared + self.delta(self.pre(conditioned)) * self.split_scale)
        return torch.cat(chunks, dim=1)


class PromptedSharedRefiner2d(nn.Module):
    """Shared conditional decoder/refiner applied independently per prompt."""

    def __init__(
        self,
        *,
        channels: int,
        n_src: int,
        n_bands: int,
        n_layers: int,
        kernel_size: tuple[int, int],
        causal: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.n_bands = int(n_bands)
        self.causal = causal
        self.conditioner = PromptConditioner2d(channels)
        self.blocks = nn.ModuleList(
            [OnlineConvBlock(channels, expansion=2, kernel_size=kernel_size, causal=causal) for _ in range(n_layers)]
        )

    def forward(self, source_tokens: torch.Tensor, prompts: tuple[torch.Tensor, ...]) -> torch.Tensor:
        chunks = _source_chunks(source_tokens, n_src=self.n_src, channels=self.channels)
        chunks = [self.conditioner(chunk, prompt) for chunk, prompt in zip(chunks, prompts)]
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
        prompts: tuple[torch.Tensor, ...],
        states: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        expected_states = len(self.blocks) * self.n_src
        _runtime_assert(len(states) == expected_states, f"Expected {expected_states} states, got {len(states)}")
        chunks = _source_chunks(source_tokens, n_src=self.n_src, channels=self.channels)
        chunks = [self.conditioner(chunk, prompt) for chunk, prompt in zip(chunks, prompts)]

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


class PromptedCrossSourceMixer2d(nn.Module):
    """Mix each prompted source token with other-source and mixture context."""

    def __init__(self, *, channels: int, n_src: int):
        super().__init__()
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.conditioner = PromptConditioner2d(channels)
        self.mix = nn.Sequential(
            RMSNorm2d(3 * channels),
            nn.Conv2d(3 * channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
        )
        self.mix_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        source_tokens: torch.Tensor,
        mixture_tokens: torch.Tensor,
        prompts: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        chunks = _source_chunks(source_tokens, n_src=self.n_src, channels=self.channels)
        source_mean = _sum_chunks(chunks) / float(self.n_src)
        mixed = []
        for chunk, prompt in zip(chunks, prompts):
            if self.n_src > 1:
                other_mean = (source_mean * float(self.n_src) - chunk) / float(self.n_src - 1)
            else:
                other_mean = source_mean
            chunk = self.conditioner(chunk, prompt)
            residual = self.mix(torch.cat([chunk, other_mean, mixture_tokens], dim=1))
            mixed.append(chunk + residual * self.mix_scale)
        return torch.cat(mixed, dim=1)


class PromptedSharedDecoder2d(nn.Module):
    """Shared prompt-conditioned K->F decoder and packed complex mask head."""

    def __init__(self, *, channels: int, n_src: int, n_chan: int, band_spec: SoftBandSpec2d):
        super().__init__()
        self.channels = int(channels)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.cross_source = PromptedCrossSourceMixer2d(channels=channels, n_src=n_src)
        self.conditioner = PromptConditioner2d(channels)
        self.expander = SoftBandQueryExpander2d(channels=channels, band_spec=band_spec)
        self.out_proj = nn.Conv2d(channels, 2 * n_chan, kernel_size=1, bias=True)

    def forward(
        self,
        source_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        mixture_tokens: torch.Tensor,
        prompts: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        source_tokens = self.cross_source(source_tokens, mixture_tokens, prompts)
        chunks = _source_chunks(source_tokens, n_src=self.n_src, channels=self.channels)
        masks = []
        for chunk, prompt in zip(chunks, prompts):
            chunk = self.conditioner(chunk, prompt)
            masks.append(self.out_proj(self.expander(chunk, query_tokens)))
        return torch.cat(masks, dim=1)


class OnlinePromptedAsymmetricSFC2D(nn.Module):
    """Strict online SFC core with fixed prompts and shared asymmetric decoder."""

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
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 1,
        encoder_capacity_hidden: int = 0,
        encoder_capacity_layers: int = 0,
        source_capacity_hidden: int = 0,
        source_capacity_layers: int = 0,
        kernel_size: Sequence[int] | tuple[int, int] = (3, 3),
        routing_kernel_size: Sequence[int] | tuple[int, int] = (1, 3),
        prompt_labels: Sequence[str] | None = None,
        causal: bool = True,
        masking: bool = True,
        routing_normalization: str = "softmax",
    ):
        super().__init__()
        if n_encoder_layers < 1:
            raise ValueError(f"n_encoder_layers must be positive, got {n_encoder_layers}")
        if n_decoder_layers < 0:
            raise ValueError(f"n_decoder_layers must be non-negative, got {n_decoder_layers}")
        kernel_size = _as_pair(kernel_size, name="kernel_size")
        routing_kernel_size = _as_pair(routing_kernel_size, name="routing_kernel_size")
        self.n_freq = int(n_freq)
        self.n_bands = int(n_bands)
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.d_model = int(d_model)
        self.causal = bool(causal)
        self.masking = bool(masking)

        labels = tuple(prompt_labels) if prompt_labels is not None else DEFAULT_PROMPT_LABELS[:n_src]
        if len(labels) != n_src:
            raise ValueError(f"Expected {n_src} prompt labels, got {len(labels)}")
        self.prompt_labels = labels
        self.prompt_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(1, d_model, 1, 1) * 0.02) for _ in range(n_src)]
        )

        band_spec = SoftBandSpec2d(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        self.band_spec = band_spec
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
        self.cross_prompt = PromptConditioner2d(d_model)
        self.encoder = nn.ModuleList(
            [
                OnlineConvBlock(d_model, expansion=2, kernel_size=kernel_size, causal=causal)
                for _ in range(n_encoder_layers)
            ]
        )
        self.encoder_capacity_mixers = build_capacity_mixers(
            channels=d_model,
            hidden_channels=encoder_capacity_hidden,
            n_layers=encoder_capacity_layers,
        )
        self.prompt_splitter = PromptedTokenSplitter2d(channels=d_model, n_src=n_src)
        self.prompt_refiner = PromptedSharedRefiner2d(
            channels=d_model,
            n_src=n_src,
            n_bands=n_bands,
            n_layers=n_decoder_layers,
            kernel_size=kernel_size,
            causal=causal,
        )
        self.source_capacity_mixers = build_capacity_mixers(
            channels=n_src * d_model,
            hidden_channels=source_capacity_hidden,
            n_layers=source_capacity_layers,
        )
        self.decoder = PromptedSharedDecoder2d(
            channels=d_model,
            n_src=n_src,
            n_chan=n_chan,
            band_spec=band_spec,
        )

    def _default_prompts(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.prompt_embeddings)

    def _external_prompts(self, prompt_embeddings: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if prompt_embeddings.ndim == 2:
            _runtime_assert(prompt_embeddings.shape == (self.n_src, self.d_model), str(prompt_embeddings.shape))
            return tuple(
                prompt_embeddings[idx : idx + 1, :].reshape(1, self.d_model, 1, 1)
                for idx in range(self.n_src)
            )
        if prompt_embeddings.ndim == 3:
            _runtime_assert(prompt_embeddings.shape[1:] == (self.n_src, self.d_model), str(prompt_embeddings.shape))
            return tuple(
                prompt_embeddings[:, idx, :].reshape(prompt_embeddings.shape[0], self.d_model, 1, 1)
                for idx in range(self.n_src)
            )
        raise ValueError(f"prompt_embeddings must be 2D or 3D, got {prompt_embeddings.shape}")

    def _resolve_prompts(self, prompt_embeddings: torch.Tensor | None = None) -> tuple[torch.Tensor, ...]:
        if prompt_embeddings is None:
            return self._default_prompts()
        return self._external_prompts(prompt_embeddings)

    @staticmethod
    def _prompt_summary(prompts: tuple[torch.Tensor, ...]) -> torch.Tensor:
        summary = prompts[0]
        for prompt in prompts[1:]:
            summary = summary + prompt
        return summary / float(len(prompts))

    def prompt_manifest(self) -> dict[str, object]:
        return {
            "type": "fixed_prompt_conditioned_shared_decoder",
            "labels": list(self.prompt_labels),
            "n_prompts": self.n_src,
            "prompt_dim": self.d_model,
            "static_export_prompts": True,
        }

    def forward(self, x: torch.Tensor, prompt_embeddings: torch.Tensor | None = None) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")

        prompts = self._resolve_prompts(prompt_embeddings)
        h = self.in_proj(x)
        z, query_tokens = self.compressor(h)
        z = self.cross_prompt(z, self._prompt_summary(prompts))
        for block_idx, block in enumerate(self.encoder):
            z = block(z)
            if block_idx < len(self.encoder_capacity_mixers):
                z = self.encoder_capacity_mixers[block_idx](z)
        for block_idx in range(len(self.encoder), len(self.encoder_capacity_mixers)):
            z = self.encoder_capacity_mixers[block_idx](z)
        source_tokens = self.prompt_splitter(z, prompts)
        source_tokens = self.prompt_refiner(source_tokens, prompts)
        for mixer in self.source_capacity_mixers:
            source_tokens = mixer(source_tokens)
        y = self.decoder(source_tokens, query_tokens, z, prompts)
        if self.masking:
            return _apply_packed_complex_mask_no_repeat(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y

    def stream_context_frames(self) -> int:
        if not self.causal:
            return 0
        return (
            self.compressor.stream_context_frames()
            + sum(block.stream_context_frames() for block in self.encoder)
            + self.prompt_refiner.stream_context_frames()
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
        enc = tuple(
            block.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype)
            for block in self.encoder
        )
        dec = self.prompt_refiner.init_stream_state(batch_size=batch_size, device=device, dtype=dtype)
        return (comp, *enc, *dec)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
        prompt_embeddings: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if not self.causal:
            raise RuntimeError("forward_stream is only supported when causal=True.")
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")
        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        enc_count = len(self.encoder)
        expected_states = 1 + enc_count + len(self.prompt_refiner.blocks) * self.n_src
        _runtime_assert(len(state) == expected_states, f"Expected {expected_states} stream states, got {len(state)}")

        prompts = self._resolve_prompts(prompt_embeddings)
        h = self.in_proj(x)
        (z, query_tokens), new_comp_state = self.compressor.forward_stream(h, state[0])
        z = self.cross_prompt(z, self._prompt_summary(prompts))
        new_enc_states = []
        for block_idx, (block, block_state) in enumerate(zip(self.encoder, state[1 : 1 + enc_count])):
            z, block_state = block.forward_stream(z, block_state)
            if block_idx < len(self.encoder_capacity_mixers):
                z = self.encoder_capacity_mixers[block_idx](z)
            new_enc_states.append(block_state)
        for block_idx in range(len(self.encoder), len(self.encoder_capacity_mixers)):
            z = self.encoder_capacity_mixers[block_idx](z)

        source_tokens = self.prompt_splitter(z, prompts)
        source_tokens, new_dec_states = self.prompt_refiner.forward_stream(
            source_tokens,
            prompts,
            state[1 + enc_count :],
        )
        for mixer in self.source_capacity_mixers:
            source_tokens = mixer(source_tokens)
        y = self.decoder(source_tokens, query_tokens, z, prompts)
        if self.masking:
            y = _apply_packed_complex_mask_no_repeat(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y, (new_comp_state, *new_enc_states, *new_dec_states)

    def init_input_history(self, batch_size: int = 1, *, device=None, dtype=None) -> torch.Tensor:
        history_frames = self.stream_context_frames()
        return torch.zeros(batch_size, 2 * self.n_chan, history_frames, self.n_freq, device=device, dtype=dtype)

    def forward_stream_recompute(
        self,
        x: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "Exact low-memory recomputation from raw input history is not implemented for "
            "OnlinePromptedAsymmetricSFC2D. Use forward_stream with layer caches for strict equivalence."
        )

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        states = self.init_stream_state(
            batch_size=batch_size,
            device=self.decoder.out_proj.weight.device,
            dtype=self.decoder.out_proj.weight.dtype,
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


class OnlinePromptedAsymmetricSFCModel(nn.Module):
    """Complex-STFT wrapper around OnlinePromptedAsymmetricSFC2D."""

    def __init__(self, *, n_freq: int, n_src: int = 3, n_chan: int = 1, **kwargs):
        super().__init__()
        self.core = OnlinePromptedAsymmetricSFC2D(n_freq=n_freq, n_src=n_src, n_chan=n_chan, **kwargs)
        self.n_src = n_src
        self.n_chan = n_chan

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x2d = pack_complex_stft_as_2d(x)
        y2d = self.core(x2d, **kwargs)
        return unpack_2d_to_complex_stft(y2d, n_src=self.n_src, n_chan=self.n_chan)


def build_prompted_asymmetric_sfc_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_src: int = 3,
    n_chan: int = 1,
    n_bands: int = 64,
    band_config: str = "musical",
    d_model: int = 32,
    n_encoder_layers: int = 3,
    n_decoder_layers: int = 1,
    encoder_capacity_hidden: int = 6144,
    encoder_capacity_layers: int = 3,
    source_capacity_hidden: int = 2048,
    source_capacity_layers: int = 1,
    kernel_size: Sequence[int] | tuple[int, int] = (3, 3),
    routing_kernel_size: Sequence[int] | tuple[int, int] = (1, 3),
    prompt_labels: Sequence[str] | None = None,
    causal: bool = True,
    masking: bool = True,
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
    core = OnlinePromptedAsymmetricSFC2D(
        n_freq=core_n_freq,
        n_bands=n_bands,
        n_fft=core_n_fft,
        sample_rate=fs,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        d_model=d_model,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        encoder_capacity_hidden=encoder_capacity_hidden,
        encoder_capacity_layers=encoder_capacity_layers,
        source_capacity_hidden=source_capacity_hidden,
        source_capacity_layers=source_capacity_layers,
        kernel_size=kernel_size,
        routing_kernel_size=routing_kernel_size,
        prompt_labels=prompt_labels,
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
