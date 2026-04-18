from __future__ import annotations

import torch
import torch.nn as nn

from spectral_feature_compression.core.model.online_sfc_2d import pack_complex_stft_as_2d, unpack_2d_to_complex_stft


def resolve_preprocessed_n_freq(
    n_freq: int,
    *,
    enabled: bool = False,
    keep_bins: int | None = None,
    target_bins: int | None = None,
) -> int:
    if not enabled:
        return int(n_freq)
    if target_bins is None:
        raise ValueError("target_bins must be set when frequency preprocessing is enabled.")
    if keep_bins is None:
        raise ValueError("keep_bins must be set when frequency preprocessing is enabled.")
    if not (0 < keep_bins < target_bins <= n_freq):
        raise ValueError(
            "Expected 0 < keep_bins < target_bins <= n_freq, "
            f"got keep_bins={keep_bins}, target_bins={target_bins}, n_freq={n_freq}"
        )
    return int(target_bins)


def _build_avg_high_basis(high_in: int, high_out: int) -> torch.Tensor:
    basis = torch.zeros(high_out, high_in, dtype=torch.float32)
    edges = torch.linspace(0, high_in, steps=high_out + 1, dtype=torch.float32)
    for idx in range(high_out):
        start = int(torch.floor(edges[idx]).item())
        end = int(torch.ceil(edges[idx + 1]).item())
        end = min(max(end, start + 1), high_in)
        basis[idx, start:end] = 1.0
    return basis


def _build_triangular_high_basis(high_in: int, high_out: int) -> torch.Tensor:
    if high_out == 1:
        return torch.ones(1, high_in, dtype=torch.float32)

    basis = torch.zeros(high_out, high_in, dtype=torch.float32)
    positions = torch.arange(high_in, dtype=torch.float32)
    centers = torch.linspace(0.0, float(max(high_in - 1, 0)), steps=high_out, dtype=torch.float32)
    for idx in range(high_out):
        center = float(centers[idx].item())
        left = 0.0 if idx == 0 else 0.5 * float(centers[idx - 1].item() + center)
        right = float(max(high_in - 1, 0)) if idx == high_out - 1 else 0.5 * float(center + centers[idx + 1].item())
        left_width = max(center - left, 1e-6)
        right_width = max(right - center, 1e-6)

        values = torch.zeros_like(positions)
        left_mask = (positions >= left) & (positions <= center)
        right_mask = (positions >= center) & (positions <= right)
        values[left_mask] = (positions[left_mask] - left) / left_width
        values[right_mask] = (right - positions[right_mask]) / right_width
        values[int(round(center))] = 1.0
        basis[idx] = torch.maximum(values, torch.zeros_like(values))
    return basis


def build_hybrid_frequency_matrices(
    n_freq_in: int,
    *,
    keep_bins: int,
    target_bins: int,
    mode: str = "triangular",
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (0 < keep_bins < target_bins <= n_freq_in):
        raise ValueError(
            "Expected 0 < keep_bins < target_bins <= n_freq_in, "
            f"got keep_bins={keep_bins}, target_bins={target_bins}, n_freq_in={n_freq_in}"
        )

    high_in = n_freq_in - keep_bins
    high_out = target_bins - keep_bins
    if high_out <= 0:
        raise ValueError(f"Expected target_bins > keep_bins, got {target_bins} vs {keep_bins}")

    if mode == "avg":
        high_basis = _build_avg_high_basis(high_in, high_out)
    elif mode == "triangular":
        high_basis = _build_triangular_high_basis(high_in, high_out)
    else:
        raise ValueError(f"Unsupported frequency preprocessing mode: {mode}")

    analysis = torch.zeros(target_bins, n_freq_in, dtype=torch.float32)
    synthesis = torch.zeros(n_freq_in, target_bins, dtype=torch.float32)

    analysis[:keep_bins, :keep_bins] = torch.eye(keep_bins, dtype=torch.float32)
    synthesis[:keep_bins, :keep_bins] = torch.eye(keep_bins, dtype=torch.float32)

    high_analysis = high_basis / high_basis.sum(dim=1, keepdim=True).clamp_min(1e-6)
    high_synthesis = (high_basis / high_basis.sum(dim=0, keepdim=True).clamp_min(1e-6)).transpose(0, 1)

    analysis[keep_bins:, keep_bins:] = high_analysis
    synthesis[keep_bins:, keep_bins:] = high_synthesis
    return analysis, synthesis


class HybridFrequencyProjector2d(nn.Module):
    """
    Stateless frequency-axis preprocessing/postprocessing for online models.

    Low-frequency bins are kept exactly while the remaining high-frequency bins
    are projected onto fewer slots using a fixed basis.
    """

    def __init__(
        self,
        n_freq_in: int,
        *,
        keep_bins: int,
        target_bins: int,
        mode: str = "triangular",
    ):
        super().__init__()
        analysis, synthesis = build_hybrid_frequency_matrices(
            n_freq_in=n_freq_in,
            keep_bins=keep_bins,
            target_bins=target_bins,
            mode=mode,
        )
        self.n_freq_in = int(n_freq_in)
        self.keep_bins = int(keep_bins)
        self.target_bins = int(target_bins)
        self.mode = mode
        self.register_buffer("analysis_matrix", analysis)
        self.register_buffer("synthesis_matrix", synthesis)

    @property
    def n_freq_out(self) -> int:
        return self.target_bins

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, n_freq = x.shape
        if n_freq != self.n_freq_in:
            raise ValueError(f"Expected {self.n_freq_in} input bins, got {n_freq}")
        flat = x.reshape(batch * channels * frames, n_freq)
        y = flat @ self.analysis_matrix.transpose(0, 1)
        return y.reshape(batch, channels, frames, self.n_freq_out)

    def synthesis(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, n_freq = x.shape
        if n_freq != self.n_freq_out:
            raise ValueError(f"Expected {self.n_freq_out} projected bins, got {n_freq}")
        flat = x.reshape(batch * channels * frames, n_freq)
        y = flat @ self.synthesis_matrix.transpose(0, 1)
        return y.reshape(batch, channels, frames, self.n_freq_in)

    def manifest(self) -> dict[str, object]:
        return {
            "enabled": True,
            "type": "hybrid_keep_plus_high_project",
            "n_freq_in": self.n_freq_in,
            "n_freq_out": self.n_freq_out,
            "keep_bins": self.keep_bins,
            "mode": self.mode,
        }


def build_frequency_preprocessor(
    n_freq_in: int,
    *,
    enabled: bool = False,
    keep_bins: int | None = None,
    target_bins: int | None = None,
    mode: str = "triangular",
) -> HybridFrequencyProjector2d | None:
    if not enabled:
        return None
    if keep_bins is None or target_bins is None:
        raise ValueError("keep_bins and target_bins must be provided when frequency preprocessing is enabled.")
    return HybridFrequencyProjector2d(
        n_freq_in=n_freq_in,
        keep_bins=int(keep_bins),
        target_bins=int(target_bins),
        mode=mode,
    )


class FrequencyPreprocessedOnlineModel(nn.Module):
    """
    Shared complex-STFT wrapper that applies fixed frequency preprocessing
    before the online core and the matching synthesis afterwards.
    """

    def __init__(
        self,
        *,
        core: nn.Module,
        n_src: int,
        n_chan: int,
        freq_preprocessor: HybridFrequencyProjector2d | None = None,
    ):
        super().__init__()
        self.core = core
        self.n_src = n_src
        self.n_chan = n_chan
        self.freq_preprocessor = freq_preprocessor
        self.input_n_freq = (
            freq_preprocessor.n_freq_in if freq_preprocessor is not None else int(getattr(core, "n_freq"))
        )
        self.core_n_freq = int(getattr(core, "n_freq"))

    def preprocess_2d(self, x2d: torch.Tensor) -> torch.Tensor:
        if self.freq_preprocessor is None:
            return x2d
        return self.freq_preprocessor.analysis(x2d)

    def postprocess_2d(self, y2d: torch.Tensor) -> torch.Tensor:
        if self.freq_preprocessor is None:
            return y2d
        return self.freq_preprocessor.synthesis(y2d)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x2d = pack_complex_stft_as_2d(x)
        y2d = self.core(self.preprocess_2d(x2d), **kwargs)
        return unpack_2d_to_complex_stft(self.postprocess_2d(y2d), n_src=self.n_src, n_chan=self.n_chan)

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None):
        return self.core.init_stream_state(batch_size=batch_size, device=device, dtype=dtype)

    def forward_stream(self, x2d: torch.Tensor, state=None):
        y2d, new_state = self.core.forward_stream(self.preprocess_2d(x2d), state)
        return self.postprocess_2d(y2d), new_state

    def stream_context_frames(self) -> int:
        return self.core.stream_context_frames()

    def init_input_history(self, batch_size: int = 1, *, device=None, dtype=None):
        history_frames = self.stream_context_frames()
        return torch.zeros(batch_size, 2 * self.n_chan, history_frames, self.input_n_freq, device=device, dtype=dtype)

    def forward_stream_recompute(self, x2d: torch.Tensor, history=None):
        if history is None:
            history = self.init_input_history(batch_size=x2d.shape[0], device=x2d.device, dtype=x2d.dtype)

        ctx = self.stream_context_frames()
        x2d_reduced = self.preprocess_2d(x2d)
        history_reduced = self.preprocess_2d(history)
        y2d_reduced, _ = self.core.forward_stream_recompute(x2d_reduced, history_reduced)
        full_history = torch.cat([history, x2d], dim=2)
        new_history = full_history[:, :, -ctx:, :] if ctx > 0 else full_history[:, :, :0, :]
        return self.postprocess_2d(y2d_reduced), new_history

    def frequency_preprocess_manifest(self) -> dict[str, object] | None:
        if self.freq_preprocessor is None:
            return None
        return self.freq_preprocessor.manifest()
