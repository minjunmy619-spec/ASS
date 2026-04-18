"""
Online / realtime SFC variant with NPU-friendly cross-attention encoder/decoder.

This family keeps the paper-style encoder/decoder contract more faithfully than
the soft-band-query variant while respecting the edge deployment constraints:
- encoder compresses full-resolution F bins into K latent bands with adaptive or
  learnable queries,
- encoder also returns a full-resolution side-path embedding for the decoder,
- decoder reconstructs F bins from latent K bands by cross-attending from
  full-resolution queries to latent keys/values,
- all runtime ops stay within Conv2d / bmm / softmax / basic tensor ops,
- streaming state remains limited to causal conv caches.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    build_frequency_preprocessor,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_sfc_2d import (
    CausalConv2d,
    OnlineConvBlock,
    RMSNorm2d,
    SameConv2d,
    _runtime_assert,
    apply_packed_complex_mask,
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandSpec2d, normalize_band_scores


class PointwiseSwiGLU2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.in_proj = nn.Conv2d(channels, 2 * channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.in_proj(x).chunk(2, dim=1)
        return self.out_proj(a * F.silu(b))


class NPUSafeCrossAttnEncoder2d(nn.Module):
    """
    Encoder-side F -> K cross-attention.

    Returns:
    - latent tokens over K bands for the separator,
    - full-resolution side embedding that the decoder later consumes as its
      adaptive full-frequency query source.
    """

    def __init__(
        self,
        channels: int,
        band_spec: SoftBandSpec2d,
        *,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
        query_type: str = "adaptive",
        routing_normalization: str = "softmax",
    ):
        super().__init__()
        if query_type not in {"adaptive", "learnable"}:
            raise ValueError(f"Unsupported query_type: {query_type}")

        Conv = CausalConv2d if causal else SameConv2d
        self.channels = channels
        self.n_bands = band_spec.n_bands
        self.n_freq = band_spec.n_freq
        self.query_type = query_type
        self.routing_normalization = routing_normalization
        self.causal = causal

        self.pre_norm = RMSNorm2d(channels)
        self.pre_pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.pre_dw = Conv(channels, channels, kernel_size=kernel_size, groups=channels, bias=True)

        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.ffn_norm = RMSNorm2d(channels)
        self.ffn = PointwiseSwiGLU2d(channels)

        self.score_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(max(channels, 1)), dtype=torch.float32))
        self.prior_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        query_basis = band_spec.expansion_basis()
        self.register_buffer("routing_bias", band_spec.routing_bias())
        self.register_buffer("query_basis", query_basis)

        if query_type == "learnable":
            self.query = nn.Parameter(torch.randn(1, channels, 1, self.n_bands) * 0.02)
        else:
            self.query = None

    def _pool_query_tokens(self, emb: torch.Tensor) -> torch.Tensor:
        batch, channels, n_frames, n_freq = emb.shape
        emb_btfc = emb.permute(0, 2, 3, 1).reshape(batch * n_frames, n_freq, channels)
        basis = self.query_basis.expand(batch, -1, n_frames, -1)
        basis_btkf = basis.permute(0, 2, 1, 3).reshape(batch * n_frames, self.n_bands, n_freq)
        pooled = torch.bmm(basis_btkf, emb_btfc)
        return pooled.reshape(batch, n_frames, self.n_bands, channels).permute(0, 3, 1, 2)

    def _prepare_query(self, emb: torch.Tensor) -> torch.Tensor:
        if self.query_type == "learnable":
            return self.query.expand(emb.shape[0], -1, emb.shape[2], -1)
        return self._pool_query_tokens(emb)

    def _cross_attend(self, emb: torch.Tensor, query_seed: torch.Tensor) -> torch.Tensor:
        batch, channels, n_frames, n_freq = emb.shape
        keys = self.k_proj(emb)
        values = self.v_proj(emb)
        queries = self.q_proj(query_seed)

        q = queries.permute(0, 2, 3, 1).reshape(batch * n_frames, self.n_bands, channels)
        k = keys.permute(0, 2, 3, 1).reshape(batch * n_frames, n_freq, channels)
        v = values.permute(0, 2, 3, 1).reshape(batch * n_frames, n_freq, channels)

        scores = torch.bmm(q, k.transpose(1, 2))
        scores = scores * self.score_scale.to(dtype=scores.dtype)
        bias = self.routing_bias.expand(batch, -1, n_frames, -1).permute(0, 2, 1, 3).reshape(
            batch * n_frames, self.n_bands, n_freq
        )
        scores = scores + bias.to(dtype=scores.dtype) * self.prior_scale.to(dtype=scores.dtype)
        weights = normalize_band_scores(scores, mode=self.routing_normalization)
        attended = torch.bmm(weights, v)
        attended = attended.reshape(batch, n_frames, self.n_bands, channels).permute(0, 3, 1, 2)

        hidden = self.out_proj(attended)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")
        emb = F.silu(self.pre_dw(self.pre_pw(self.pre_norm(x))))
        latent = self._cross_attend(emb, self._prepare_query(emb))
        return latent, emb

    def stream_context_frames(self) -> int:
        if isinstance(self.pre_dw, CausalConv2d):
            return self.pre_dw.stream_context_frames()
        return 0

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not isinstance(self.pre_dw, CausalConv2d):
            raise RuntimeError("Streaming state is only supported when causal=True.")
        return self.pre_dw.init_stream_state(batch_size, freq_bins=freq_bins, device=device, dtype=dtype)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if not isinstance(self.pre_dw, CausalConv2d):
            raise RuntimeError("forward_stream is only supported when causal=True.")

        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")
        emb = self.pre_norm(x)
        emb = self.pre_pw(emb)
        emb, new_state = self.pre_dw.forward_stream(emb, state)
        emb = F.silu(emb)
        latent = self._cross_attend(emb, self._prepare_query(emb))
        return (latent, emb), new_state


class NPUSafeCrossAttnDecoder2d(nn.Module):
    """
    Decoder-side K -> F cross-attention.

    The adaptive query is produced from the full-resolution encoder side-path,
    closely matching the paper's decoder contract.
    """

    def __init__(
        self,
        channels: int,
        band_spec: SoftBandSpec2d,
        *,
        query_type: str = "adaptive",
        routing_normalization: str = "softmax",
    ):
        super().__init__()
        if query_type not in {"adaptive", "learnable"}:
            raise ValueError(f"Unsupported query_type: {query_type}")

        self.channels = channels
        self.n_bands = band_spec.n_bands
        self.n_freq = band_spec.n_freq
        self.query_type = query_type
        self.routing_normalization = routing_normalization

        self.latent_pre = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.query_norm = RMSNorm2d(channels)
        self.query_mlp = PointwiseSwiGLU2d(channels)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.ffn_norm = RMSNorm2d(channels)
        self.ffn = PointwiseSwiGLU2d(channels)

        self.score_scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(max(channels, 1)), dtype=torch.float32))
        self.prior_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.query_skip_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.register_buffer("expansion_basis", band_spec.expansion_basis())
        if query_type == "learnable":
            self.query = nn.Parameter(torch.randn(1, channels, 1, self.n_freq) * 0.02)
        else:
            self.query = None

    def _prepare_query(self, side: torch.Tensor) -> torch.Tensor:
        if self.query_type == "learnable":
            return self.query.expand(side.shape[0], -1, side.shape[2], -1)
        side = self.query_norm(side)
        return self.query_mlp(side)

    def forward(self, latent: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        _runtime_assert(latent.shape[-1] == self.n_bands, f"{latent.shape} vs {self.n_bands}")
        _runtime_assert(side.shape[-1] == self.n_freq, f"{side.shape} vs {self.n_freq}")

        latent_h = self.latent_pre(latent)
        query_h = self._prepare_query(side)

        batch, channels, n_frames, _ = side.shape
        q = self.q_proj(query_h).permute(0, 2, 3, 1).reshape(batch * n_frames, self.n_freq, channels)
        k = self.k_proj(latent_h).permute(0, 2, 3, 1).reshape(batch * n_frames, self.n_bands, channels)
        v = self.v_proj(latent_h).permute(0, 2, 3, 1).reshape(batch * n_frames, self.n_bands, channels)

        scores = torch.bmm(q, k.transpose(1, 2))
        scores = scores * self.score_scale.to(dtype=scores.dtype)
        bias = self.expansion_basis.squeeze(0).squeeze(1).transpose(0, 1)
        bias = bias.unsqueeze(0).unsqueeze(0).expand(batch, n_frames, self.n_freq, self.n_bands).reshape(
            batch * n_frames, self.n_freq, self.n_bands
        )
        scores = scores + bias.to(dtype=scores.dtype) * self.prior_scale.to(dtype=scores.dtype)
        weights = normalize_band_scores(scores, mode=self.routing_normalization)

        attended = torch.bmm(weights, v)
        attended = attended.reshape(batch, n_frames, self.n_freq, channels).permute(0, 3, 1, 2)
        hidden = self.out_proj(attended)
        hidden = hidden + query_h * self.query_skip_scale
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden


class OnlineCrossAttnQuerySFC2D(nn.Module):
    """
    Online SFC variant with an NPU-friendly cross-attention encoder/decoder.

    This is the closest repo-local approximation to the paper's adaptive
    encoder/decoder query contract while keeping the graph exportable and
    streamable on the constrained edge target.
    """

    def __init__(
        self,
        n_freq: int,
        n_bands: int = 64,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
        n_src: int = 2,
        n_chan: int = 1,
        d_model: int = 96,
        n_layers: int = 12,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
        masking: bool = True,
        query_type: str = "adaptive",
        routing_normalization: str = "softmax",
    ):
        super().__init__()
        self.n_freq = n_freq
        self.n_bands = n_bands
        self.n_src = n_src
        self.n_chan = n_chan
        self.masking = masking
        self.causal = causal

        in_ch = 2 * n_chan
        out_ch = 2 * n_src * n_chan

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, d_model, kernel_size=1, bias=True),
            RMSNorm2d(d_model),
        )
        band_spec = SoftBandSpec2d(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        self.encoder = NPUSafeCrossAttnEncoder2d(
            channels=d_model,
            band_spec=band_spec,
            kernel_size=kernel_size,
            causal=causal,
            query_type=query_type,
            routing_normalization=routing_normalization,
        )
        self.separator = nn.ModuleList(
            [OnlineConvBlock(d_model, expansion=2, kernel_size=kernel_size, causal=causal) for _ in range(n_layers)]
        )
        self.decoder = NPUSafeCrossAttnDecoder2d(
            channels=d_model,
            band_spec=band_spec,
            query_type=query_type,
            routing_normalization=routing_normalization,
        )
        self.out_proj = nn.Conv2d(d_model, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")

        h = self.in_proj(x)
        z, side = self.encoder(h)
        for block in self.separator:
            z = block(z)
        h = self.decoder(z, side)
        y = self.out_proj(h)
        if self.masking:
            return apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y

    def stream_context_frames(self) -> int:
        if not self.causal:
            return 0
        return self.encoder.stream_context_frames() + sum(block.stream_context_frames() for block in self.separator)

    def init_stream_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if not self.causal:
            raise RuntimeError("Streaming state is only supported when causal=True.")
        enc = self.encoder.init_stream_state(batch_size, freq_bins=self.n_freq, device=device, dtype=dtype)
        sep = tuple(
            block.init_stream_state(batch_size, freq_bins=self.n_bands, device=device, dtype=dtype)
            for block in self.separator
        )
        return (enc, *sep)

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

        _runtime_assert(len(state) == 1 + len(self.separator), f"Unexpected state tuple: {len(state)}")
        enc_state = state[0]
        sep_state = state[1:]

        h = self.in_proj(x)
        (z, side), new_enc_state = self.encoder.forward_stream(h, enc_state)
        new_sep_states = []
        for block, block_state in zip(self.separator, sep_state):
            z, block_state = block.forward_stream(z, block_state)
            new_sep_states.append(block_state)
        h = self.decoder(z, side)
        y = self.out_proj(h)
        if self.masking:
            y = apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y, (new_enc_state, *new_sep_states)

    def init_input_history(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        history_frames = self.stream_context_frames()
        return torch.zeros(batch_size, 2 * self.n_chan, history_frames, self.n_freq, device=device, dtype=dtype)

    def forward_stream_recompute(
        self,
        x: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "Exact low-memory recomputation from raw input history is not implemented for this model. "
            "Use forward_stream with layer caches for strict realtime equivalence."
        )

    def layer_cache_numel(self, batch_size: int = 1) -> int:
        states = self.init_stream_state(
            batch_size=batch_size,
            device=self.out_proj.weight.device,
            dtype=self.out_proj.weight.dtype,
        )
        return sum(int(s.numel()) for s in states)

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


class OnlineCrossAttnQuerySFCModel(nn.Module):
    """
    Torch-friendly complex-STFT wrapper for the NPU-friendly cross-attn family.
    """

    def __init__(
        self,
        n_freq: int,
        n_bands: int = 64,
        n_fft: int | None = None,
        sample_rate: int | None = None,
        band_config: str = "musical",
        n_src: int = 2,
        n_chan: int = 1,
        d_model: int = 96,
        n_layers: int = 12,
        kernel_size: tuple[int, int] = (3, 3),
        causal: bool = True,
        masking: bool = True,
        query_type: str = "adaptive",
        routing_normalization: str = "softmax",
    ):
        super().__init__()
        self.core = OnlineCrossAttnQuerySFC2D(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
            n_src=n_src,
            n_chan=n_chan,
            d_model=d_model,
            n_layers=n_layers,
            kernel_size=kernel_size,
            causal=causal,
            masking=masking,
            query_type=query_type,
            routing_normalization=routing_normalization,
        )
        self.n_src = n_src
        self.n_chan = n_chan

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
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


def build_online_crossattn_query_sfc_system(
    *,
    n_fft: int,
    hop_length: int,
    fs: int,
    n_bands: int = 64,
    band_config: str = "musical",
    n_src: int = 2,
    n_chan: int = 1,
    d_model: int = 96,
    n_layers: int = 12,
    kernel_size: tuple[int, int] = (3, 3),
    causal: bool = True,
    masking: bool = True,
    query_type: str = "adaptive",
    routing_normalization: str = "softmax",
    freq_preprocess_enabled: bool = False,
    freq_preprocess_keep_bins: int | None = None,
    freq_preprocess_target_bins: int | None = None,
    freq_preprocess_mode: str = "triangular",
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
    )
    freq_preprocessor = build_frequency_preprocessor(
        full_n_freq,
        enabled=freq_preprocess_enabled,
        keep_bins=freq_preprocess_keep_bins,
        target_bins=freq_preprocess_target_bins,
        mode=freq_preprocess_mode,
    )
    core = OnlineCrossAttnQuerySFC2D(
        n_freq=core_n_freq,
        n_bands=n_bands,
        n_fft=n_fft,
        sample_rate=fs,
        band_config=band_config,
        n_src=n_src,
        n_chan=n_chan,
        d_model=d_model,
        n_layers=n_layers,
        kernel_size=kernel_size,
        causal=causal,
        masking=masking,
        query_type=query_type,
        routing_normalization=routing_normalization,
    )
    model = FrequencyPreprocessedOnlineModel(
        core=core,
        n_src=n_src,
        n_chan=n_chan,
        freq_preprocessor=freq_preprocessor,
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
