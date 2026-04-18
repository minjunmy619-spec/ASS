"""
Online / realtime baseline with hard band compression.

This baseline keeps the same 2D-only separator structure as the soft-band
variant, but removes input-adaptive routing. Frequency compression and
expansion are driven only by a fixed band basis.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectral_feature_compression.core.model.frequency_preprocessing import (
    FrequencyPreprocessedOnlineModel,
    build_frequency_preprocessor,
    resolve_preprocessed_n_freq,
)
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_sfc_2d import (
    OnlineConvBlock,
    RMSNorm2d,
    _runtime_assert,
    apply_packed_complex_mask,
    pack_complex_stft_as_2d,
    unpack_2d_to_complex_stft,
)
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import SoftBandSpec2d


class HardBandCompressor2d(nn.Module):
    def __init__(self, channels: int, band_spec: SoftBandSpec2d):
        super().__init__()
        self.channels = channels
        self.band_spec = band_spec
        self.n_bands = band_spec.n_bands

        self.pre = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.register_buffer("weights", band_spec.expansion_basis())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)
        batch, channels, n_frames, n_freq = h.shape
        h_btfc = h.permute(0, 2, 3, 1).reshape(batch * n_frames, n_freq, channels)
        weights = self.weights.expand(batch, -1, n_frames, -1)
        weights_btkf = weights.permute(0, 2, 1, 3).reshape(batch * n_frames, self.n_bands, n_freq)
        latent_btkc = torch.bmm(weights_btkf, h_btfc)
        return latent_btkc.reshape(batch, n_frames, self.n_bands, channels).permute(0, 3, 1, 2)

    def stream_context_frames(self) -> int:
        return 0

    def forward_stream(self, x: torch.Tensor, state: None = None) -> tuple[torch.Tensor, None]:
        return self.forward(x), None


class HardBandExpander2d(nn.Module):
    def __init__(self, channels: int, band_spec: SoftBandSpec2d):
        super().__init__()
        self.channels = channels
        self.n_bands = band_spec.n_bands
        self.n_freq = band_spec.n_freq

        self.pre = nn.Sequential(
            RMSNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.register_buffer("basis", band_spec.expansion_basis())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.pre(z)
        batch, channels, n_frames, n_bands = h.shape
        h_btck = h.permute(0, 2, 1, 3).reshape(batch * n_frames, channels, n_bands)
        basis = self.basis.expand(batch, -1, n_frames, -1)
        basis_btkf = basis.permute(0, 2, 1, 3).reshape(batch * n_frames, n_bands, self.n_freq)
        expanded_btcf = torch.bmm(h_btck, basis_btkf)
        return expanded_btcf.reshape(batch, n_frames, channels, self.n_freq).permute(0, 2, 1, 3)


class OnlineHardBandSFC2D(nn.Module):
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
    ):
        super().__init__()
        self.n_freq = n_freq
        self.n_src = n_src
        self.n_chan = n_chan
        self.masking = masking

        in_ch = 2 * n_chan
        out_ch = 2 * n_src * n_chan
        band_spec = SoftBandSpec2d(
            n_freq=n_freq,
            n_bands=n_bands,
            n_fft=n_fft,
            sample_rate=sample_rate,
            band_config=band_config,
        )
        self.in_proj = nn.Sequential(
            nn.Conv2d(in_ch, d_model, kernel_size=1, bias=True),
            RMSNorm2d(d_model),
        )
        self.compressor = HardBandCompressor2d(channels=d_model, band_spec=band_spec)
        self.separator = nn.ModuleList(
            [OnlineConvBlock(d_model, expansion=2, kernel_size=kernel_size, causal=causal) for _ in range(n_layers)]
        )
        self.expander = HardBandExpander2d(channels=d_model, band_spec=band_spec)
        self.out_proj = nn.Conv2d(d_model, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")

        h = self.in_proj(x)
        z = self.compressor(h)
        for block in self.separator:
            z = block(z)
        h = self.expander(z)
        y = self.out_proj(h)
        if self.masking:
            return apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y

    def stream_context_frames(self) -> int:
        return sum(block.stream_context_frames() for block in self.separator)

    def init_stream_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            block.init_stream_state(batch_size, freq_bins=self.compressor.n_bands, device=device, dtype=dtype)
            for block in self.separator
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        _runtime_assert(x.ndim == 4, f"Expected 4D input (B,C,T,F), got {x.shape}")
        _runtime_assert(x.shape[-1] == self.n_freq, f"{x.shape} vs {self.n_freq}")

        if state is None:
            state = self.init_stream_state(batch_size=x.shape[0], device=x.device, dtype=x.dtype)

        _runtime_assert(len(state) == len(self.separator), f"Unexpected state tuple: {len(state)}")

        h = self.in_proj(x)
        z, _ = self.compressor.forward_stream(h)
        new_sep_states = []
        for block, block_state in zip(self.separator, state):
            z, block_state = block.forward_stream(z, block_state)
            new_sep_states.append(block_state)
        h = self.expander(z)
        y = self.out_proj(h)
        if self.masking:
            y = apply_packed_complex_mask(x=x, y=y, n_src=self.n_src, n_chan=self.n_chan)
        return y, tuple(new_sep_states)

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
        states = self.init_stream_state(batch_size=batch_size, device=self.out_proj.weight.device, dtype=self.out_proj.weight.dtype)
        return sum(int(s.numel()) for s in states)

    def input_history_numel(self, batch_size: int = 1) -> int:
        return batch_size * 2 * self.n_chan * self.stream_context_frames() * self.n_freq

    def state_size_bytes(self, *, batch_size: int = 1, dtype: torch.dtype = torch.float16, mode: str = "layer_cache") -> int:
        element_size = torch.tensor([], dtype=dtype).element_size()
        if mode == "layer_cache":
            return self.layer_cache_numel(batch_size=batch_size) * element_size
        if mode == "input_history":
            return self.input_history_numel(batch_size=batch_size) * element_size
        raise ValueError(f"Unsupported state mode: {mode}")


class OnlineHardBandSFCModel(nn.Module):
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
    ):
        super().__init__()
        self.core = OnlineHardBandSFC2D(
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


def build_online_hard_band_sfc_system(
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
    core = OnlineHardBandSFC2D(
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
