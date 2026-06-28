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
    dc_bypass_enabled: bool = False,
) -> int:
    n_freq = resolve_frequency_input_n_freq(n_freq, dc_bypass_enabled=dc_bypass_enabled)
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


def resolve_frequency_input_n_freq(n_freq: int, *, dc_bypass_enabled: bool = False) -> int:
    n_freq = int(n_freq)
    if not dc_bypass_enabled:
        return n_freq
    if n_freq <= 1:
        raise ValueError(f"DC bypass requires at least two frequency bins, got {n_freq}")
    return n_freq - 1


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


def _build_triangular_high_basis_from_centers(high_in: int, centers: torch.Tensor) -> torch.Tensor:
    high_out = int(centers.numel())
    if high_out == 1:
        return torch.ones(1, high_in, dtype=torch.float32)

    centers = centers.to(dtype=torch.float32)
    centers = centers.clone()
    centers[0] = 0.0
    centers[-1] = float(max(high_in - 1, 0))
    if torch.any(centers[1:] <= centers[:-1]):
        raise ValueError(f"Expected strictly increasing high-band centers, got {centers.tolist()}")

    basis = torch.zeros(high_out, high_in, dtype=torch.float32)
    positions = torch.arange(high_in, dtype=torch.float32)
    for idx in range(high_out):
        center = centers[idx]
        left = centers[idx - 1] if idx > 0 else centers[idx]
        right = centers[idx + 1] if idx < high_out - 1 else centers[idx]

        values = torch.zeros_like(positions)
        if center > left:
            left_mask = (positions >= left) & (positions <= center)
            values[left_mask] = (positions[left_mask] - left) / (center - left).clamp_min(1e-6)
        if right > center:
            right_mask = (positions >= center) & (positions <= right)
            values[right_mask] = torch.maximum(
                values[right_mask],
                (right - positions[right_mask]) / (right - center).clamp_min(1e-6),
            )
        values[int(round(float(center.item())))] = 1.0
        basis[idx] = values.clamp_min(0.0)
    return basis


def _build_log_high_basis(high_in: int, high_out: int) -> torch.Tensor:
    if high_out == 1:
        return torch.ones(1, high_in, dtype=torch.float32)
    log_span = 3.0
    u = torch.linspace(0.0, 1.0, steps=high_out, dtype=torch.float32)
    centers = torch.expm1(u * log_span) / torch.expm1(torch.tensor(log_span, dtype=torch.float32))
    centers = centers * float(max(high_in - 1, 0))
    return _build_triangular_high_basis_from_centers(high_in, centers)


def _build_piecewise_high_basis(high_in: int, high_out: int) -> torch.Tensor:
    if high_out == 1:
        return torch.ones(1, high_in, dtype=torch.float32)

    lower_input_fraction = 0.55
    lower_output_fraction = 0.70
    lower_count = int(round(float(high_out) * lower_output_fraction))
    lower_count = min(max(lower_count, 1), high_out - 1)
    upper_count = high_out - lower_count

    split = float(max(high_in - 1, 0)) * lower_input_fraction
    lower = torch.linspace(0.0, split, steps=lower_count, dtype=torch.float32)
    upper = torch.linspace(split, float(max(high_in - 1, 0)), steps=upper_count + 1, dtype=torch.float32)[1:]
    centers = torch.cat([lower, upper], dim=0)
    return _build_triangular_high_basis_from_centers(high_in, centers)


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

    if mode in {"learnable_query", "sfclite_query"}:
        mode = "triangular"

    if mode == "avg":
        high_basis = _build_avg_high_basis(high_in, high_out)
    elif mode == "triangular":
        high_basis = _build_triangular_high_basis(high_in, high_out)
    elif mode == "hybrid_log_high":
        high_basis = _build_log_high_basis(high_in, high_out)
    elif mode == "hybrid_piecewise_high":
        high_basis = _build_piecewise_high_basis(high_in, high_out)
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


def build_hybrid_frequency_bin_frequencies(
    n_freq_in: int,
    *,
    keep_bins: int,
    target_bins: int,
    n_fft: int,
    sample_rate: int,
    mode: str = "triangular",
    dc_bypass_enabled: bool = False,
) -> torch.Tensor:
    """Return physical bin-center frequencies after hybrid preprocessing.

    The hybrid projector keeps the low-frequency bins exactly and projects the
    high-frequency tail into fewer slots.  A plain ``torch.linspace`` over the
    projected axis is therefore wrong: the first ``keep_bins`` positions still
    correspond to original FFT bins, while each high projected slot corresponds
    to a weighted average of original high-bin centers.  Adaptive/mel priors
    should use this vector when the core sees preprocessed frequency bins.
    """

    n_freq_in = int(n_freq_in)
    n_fft = int(n_fft)
    sample_rate = int(sample_rate)
    if n_freq_in <= 0:
        raise ValueError(f"n_freq_in must be positive, got {n_freq_in}")
    if n_fft <= 0 or sample_rate <= 0:
        raise ValueError(f"n_fft and sample_rate must be positive, got {n_fft}, {sample_rate}")
    full_n_freq = (n_fft // 2) + 1
    if n_freq_in not in {full_n_freq, full_n_freq - 1}:
        raise ValueError(f"n_freq_in={n_freq_in} is not compatible with n_fft={n_fft}")
    first_bin = 1 if dc_bypass_enabled else 0
    original_bin_indices = torch.arange(first_bin, first_bin + n_freq_in, dtype=torch.float32)
    original_freqs = original_bin_indices * (float(sample_rate) / float(n_fft))
    analysis, _synthesis = build_hybrid_frequency_matrices(
        n_freq_in=n_freq_in,
        keep_bins=keep_bins,
        target_bins=target_bins,
        mode=mode,
    )
    weights = analysis / analysis.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return weights @ original_freqs


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

    def _matrix_for(self, matrix: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return matrix.to(device=ref.device, dtype=ref.dtype)

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, n_freq = x.shape
        if n_freq != self.n_freq_in:
            raise ValueError(f"Expected {self.n_freq_in} input bins, got {n_freq}")
        flat = x.reshape(batch * channels * frames, n_freq)
        analysis_matrix = self._matrix_for(self.analysis_matrix, flat)
        y = flat @ analysis_matrix.transpose(0, 1)
        return y.reshape(batch, channels, frames, self.n_freq_out)

    def synthesis(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, n_freq = x.shape
        if n_freq != self.n_freq_out:
            raise ValueError(f"Expected {self.n_freq_out} projected bins, got {n_freq}")
        flat = x.reshape(batch * channels * frames, n_freq)
        synthesis_matrix = self._matrix_for(self.synthesis_matrix, flat)
        y = flat @ synthesis_matrix.transpose(0, 1)
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


class LearnableQueryFrequencyProjector2d(nn.Module):
    """
    SFC-lite frequency encoder/decoder with a tied learnable query.

    The encoder projects full-frequency packed STFT bins into ``target_bins``
    using one learnable query matrix.  The decoder uses the same query matrix
    directly, so there is no independent synthesis table to drift away from the
    encoder transport learned during training.
    """

    def __init__(
        self,
        n_freq_in: int,
        *,
        keep_bins: int,
        target_bins: int,
        init_mode: str = "triangular",
    ):
        super().__init__()
        analysis, _synthesis = build_hybrid_frequency_matrices(
            n_freq_in=n_freq_in,
            keep_bins=keep_bins,
            target_bins=target_bins,
            mode=init_mode,
        )
        self.n_freq_in = int(n_freq_in)
        self.keep_bins = int(keep_bins)
        self.target_bins = int(target_bins)
        self.init_mode = init_mode
        self.frequency_query = nn.Parameter(analysis)

    @property
    def n_freq_out(self) -> int:
        return self.target_bins

    def encoder_query(self) -> torch.Tensor:
        return self.frequency_query

    def analysis(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, n_freq = x.shape
        if n_freq != self.n_freq_in:
            raise ValueError(f"Expected {self.n_freq_in} input bins, got {n_freq}")
        flat = x.reshape(batch * channels * frames, n_freq)
        query = self.encoder_query().to(device=flat.device, dtype=flat.dtype)
        y = flat @ query.transpose(0, 1)
        return y.reshape(batch, channels, frames, self.n_freq_out)

    def synthesis_with_query(self, x: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, n_freq = x.shape
        if n_freq != self.n_freq_out:
            raise ValueError(f"Expected {self.n_freq_out} projected bins, got {n_freq}")
        flat = x.reshape(batch * channels * frames, n_freq)
        query = query.to(device=flat.device, dtype=flat.dtype)
        y = flat @ query
        return y.reshape(batch, channels, frames, self.n_freq_in)

    def synthesis(self, x: torch.Tensor) -> torch.Tensor:
        return self.synthesis_with_query(x, self.encoder_query())

    def manifest(self) -> dict[str, object]:
        return {
            "enabled": True,
            "type": "sfc_lite_learnable_query",
            "n_freq_in": self.n_freq_in,
            "n_freq_out": self.n_freq_out,
            "keep_bins": self.keep_bins,
            "init_mode": self.init_mode,
            "tied_synthesis_query": True,
        }


class PCENGainNormalizer2d(nn.Module):
    """
    Causal PCEN-style gain normalization for packed complex STFT tensors.

    The normalizer is intentionally wrapper-side only: it multiplies the complex
    input by a magnitude-derived gain before the core and the wrapper divides the
    core output by the matching source-expanded gain afterwards.  For masking
    cores this keeps the effective mask applied to the original mixture STFT.
    """

    def __init__(
        self,
        *,
        n_chan: int,
        smooth_coef: float = 0.98,
        alpha: float = 0.5,
        delta: float = 2.0,
        root: float = 0.5,
        eps: float = 1e-6,
        gain_floor: float = 0.05,
        gain_ceiling: float = 20.0,
    ):
        super().__init__()
        if n_chan <= 0:
            raise ValueError(f"n_chan must be positive, got {n_chan}")
        if not (0.0 <= smooth_coef < 1.0):
            raise ValueError(f"smooth_coef must be in [0, 1), got {smooth_coef}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if delta < 0.0:
            raise ValueError(f"delta must be non-negative, got {delta}")
        if root <= 0.0:
            raise ValueError(f"root must be positive, got {root}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        if not (0.0 < gain_floor <= gain_ceiling):
            raise ValueError(f"Expected 0 < gain_floor <= gain_ceiling, got {gain_floor} <= {gain_ceiling}")

        self.n_chan = int(n_chan)
        self.smooth_coef = float(smooth_coef)
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.root = float(root)
        self.eps = float(eps)
        self.gain_floor = float(gain_floor)
        self.gain_ceiling = float(gain_ceiling)

    def init_stream_state(self, batch_size: int, *, n_freq: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.n_chan, 1, int(n_freq), device=device, dtype=dtype)

    def _magnitude(self, x2d: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, n_freq = x2d.shape
        if channels != 2 * self.n_chan:
            raise ValueError(f"Expected {2 * self.n_chan} packed channels, got {channels}")
        x = x2d.reshape(batch, self.n_chan, 2, frames, n_freq)
        return torch.sqrt(x[:, :, 0].square() + x[:, :, 1].square()).clamp_min(self.eps)

    def _smooth_magnitude(
        self,
        mag: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            prev = self.init_stream_state(
                mag.shape[0],
                n_freq=mag.shape[-1],
                device=mag.device,
                dtype=mag.dtype,
            )
        else:
            prev = state.to(device=mag.device, dtype=mag.dtype)
        expected_state_shape = (mag.shape[0], self.n_chan, 1, mag.shape[-1])
        if tuple(prev.shape) != expected_state_shape:
            raise ValueError(f"Expected PCEN state shape {expected_state_shape}, got {tuple(prev.shape)}")

        if mag.shape[2] == 0:
            return mag, prev

        coeff = self.smooth_coef
        update = 1.0 - coeff
        frames = []
        for frame_idx in range(mag.shape[2]):
            current = mag[:, :, frame_idx : frame_idx + 1, :]
            prev = coeff * prev + update * current
            frames.append(prev)
        return torch.cat(frames, dim=2), prev

    def forward_with_gain(
        self,
        x2d: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mag = self._magnitude(x2d)
        smooth, new_state = self._smooth_magnitude(mag, state)
        normalized_mag = mag / smooth.clamp_min(self.eps).pow(self.alpha)
        pcen_mag = (normalized_mag + self.delta).pow(self.root) - (self.delta**self.root)
        gain = (pcen_mag / mag).clamp(min=self.gain_floor, max=self.gain_ceiling)
        packed_gain = gain.repeat_interleave(2, dim=1)
        return x2d * packed_gain, gain, new_state

    def forward(self, x2d: torch.Tensor) -> torch.Tensor:
        y, _gain, _state = self.forward_with_gain(x2d)
        return y

    def invert_output_gain(self, y2d: torch.Tensor, gain: torch.Tensor, *, n_src: int) -> torch.Tensor:
        batch, channels, frames, n_freq = y2d.shape
        if channels != 2 * int(n_src) * self.n_chan:
            raise ValueError(f"Expected {2 * int(n_src) * self.n_chan} output channels, got {channels}")
        if gain.shape != (batch, self.n_chan, frames, n_freq):
            raise ValueError(f"Expected gain shape {(batch, self.n_chan, frames, n_freq)}, got {tuple(gain.shape)}")

        src_gain = gain.unsqueeze(1).expand(batch, int(n_src), self.n_chan, frames, n_freq)
        src_gain = src_gain.reshape(batch, int(n_src) * self.n_chan, frames, n_freq).repeat_interleave(2, dim=1)
        return y2d / src_gain.clamp_min(self.eps)

    def manifest(self) -> dict[str, object]:
        return {
            "enabled": True,
            "type": "pcen_gain_normalizer_2d",
            "n_chan": self.n_chan,
            "smooth_coef": self.smooth_coef,
            "alpha": self.alpha,
            "delta": self.delta,
            "root": self.root,
            "eps": self.eps,
            "gain_floor": self.gain_floor,
            "gain_ceiling": self.gain_ceiling,
        }


def build_frequency_preprocessor(
    n_freq_in: int,
    *,
    enabled: bool = False,
    keep_bins: int | None = None,
    target_bins: int | None = None,
    mode: str = "triangular",
    dc_bypass_enabled: bool = False,
) -> HybridFrequencyProjector2d | LearnableQueryFrequencyProjector2d | None:
    if not enabled:
        return None
    n_freq_in = resolve_frequency_input_n_freq(n_freq_in, dc_bypass_enabled=dc_bypass_enabled)
    if keep_bins is None or target_bins is None:
        raise ValueError("keep_bins and target_bins must be provided when frequency preprocessing is enabled.")
    if mode in {"learnable_query", "sfclite_query"}:
        return LearnableQueryFrequencyProjector2d(
            n_freq_in=n_freq_in,
            keep_bins=int(keep_bins),
            target_bins=int(target_bins),
            init_mode="triangular",
        )
    return HybridFrequencyProjector2d(
        n_freq_in=n_freq_in,
        keep_bins=int(keep_bins),
        target_bins=int(target_bins),
        mode=mode,
    )


def build_pcen_preprocessor(
    *,
    n_chan: int,
    enabled: bool = False,
    smooth_coef: float = 0.98,
    alpha: float = 0.5,
    delta: float = 2.0,
    root: float = 0.5,
    eps: float = 1e-6,
    gain_floor: float = 0.05,
    gain_ceiling: float = 20.0,
) -> PCENGainNormalizer2d | None:
    if not enabled:
        return None
    return PCENGainNormalizer2d(
        n_chan=n_chan,
        smooth_coef=smooth_coef,
        alpha=alpha,
        delta=delta,
        root=root,
        eps=eps,
        gain_floor=gain_floor,
        gain_ceiling=gain_ceiling,
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
        freq_preprocessor: HybridFrequencyProjector2d | LearnableQueryFrequencyProjector2d | None = None,
        pcen_preprocessor: PCENGainNormalizer2d | None = None,
        dc_bypass_enabled: bool = False,
        dc_policy: str = "zero",
        residual_source_enabled: bool = False,
        residual_source_index: int | None = None,
    ):
        super().__init__()
        self.core = core
        self.n_src = int(n_src)
        self.n_chan = int(n_chan)
        self.residual_source_enabled = bool(residual_source_enabled)
        self.residual_source_index = self.n_src - 1 if residual_source_index is None else int(residual_source_index)
        if self.residual_source_enabled:
            if self.n_src <= 1:
                raise ValueError("Residual source reconstruction requires at least two output sources.")
            if self.residual_source_index != self.n_src - 1:
                raise ValueError("Residual source reconstruction currently supports the final output source only.")
        self.core_n_src = int(getattr(core, "n_src", self.n_src - 1 if self.residual_source_enabled else self.n_src))
        if self.residual_source_enabled and self.core_n_src != self.n_src - 1:
            raise ValueError(
                f"Residual source reconstruction expects core_n_src=n_src-1={self.n_src - 1}, got {self.core_n_src}."
            )
        if not self.residual_source_enabled and self.core_n_src != self.n_src:
            raise ValueError(f"Core n_src={self.core_n_src} does not match wrapper n_src={self.n_src}.")
        self.freq_preprocessor = freq_preprocessor
        self.pcen_preprocessor = pcen_preprocessor
        self.dc_bypass_enabled = bool(dc_bypass_enabled)
        if dc_policy not in {"zero", "mixture_equal"}:
            raise ValueError(f"Unsupported dc_policy={dc_policy!r}; expected 'zero' or 'mixture_equal'.")
        self.dc_policy = dc_policy
        self.body_input_n_freq = freq_preprocessor.n_freq_in if freq_preprocessor is not None else int(core.n_freq)
        self.input_n_freq = self.body_input_n_freq + 1 if self.dc_bypass_enabled else self.body_input_n_freq
        self.core_n_freq = int(core.n_freq)
        expected_core_n_freq = freq_preprocessor.n_freq_out if freq_preprocessor is not None else self.body_input_n_freq
        if self.core_n_freq != int(expected_core_n_freq):
            raise ValueError(
                f"Core n_freq={self.core_n_freq} does not match wrapper core input bins {int(expected_core_n_freq)}."
            )
        if hasattr(core, "n_chan") and int(core.n_chan) != self.n_chan:
            raise ValueError(f"Core n_chan={int(core.n_chan)} does not match wrapper n_chan={self.n_chan}.")

    def append_residual_source_2d(self, y2d: torch.Tensor, mixture2d: torch.Tensor) -> torch.Tensor:
        if not self.residual_source_enabled:
            return y2d
        batch, channels, frames, n_freq = y2d.shape
        expected_channels = 2 * self.core_n_src * self.n_chan
        if channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} explicit output channels, got {channels}")
        if mixture2d.shape != (batch, 2 * self.n_chan, frames, n_freq):
            raise ValueError(
                f"Expected mixture shape {(batch, 2 * self.n_chan, frames, n_freq)}, got {tuple(mixture2d.shape)}"
            )

        explicit = y2d.reshape(batch, self.core_n_src, self.n_chan, 2, frames, n_freq)
        mixture = mixture2d.reshape(batch, self.n_chan, 2, frames, n_freq)
        residual = mixture - explicit.sum(dim=1)
        out = torch.cat([explicit, residual.unsqueeze(1)], dim=1)
        return out.reshape(batch, 2 * self.n_src * self.n_chan, frames, n_freq)

    def preprocess_2d(self, x2d: torch.Tensor) -> torch.Tensor:
        if self.freq_preprocessor is None:
            return x2d
        return self.freq_preprocessor.analysis(x2d)

    def postprocess_2d(self, y2d: torch.Tensor) -> torch.Tensor:
        if self.freq_preprocessor is None:
            return y2d
        return self.freq_preprocessor.synthesis(y2d)

    def split_dc_2d(self, x2d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.dc_bypass_enabled:
            return x2d, None
        if x2d.shape[-1] != self.input_n_freq:
            raise ValueError(f"Expected {self.input_n_freq} input bins with DC bypass, got {x2d.shape[-1]}")
        return x2d[..., 1:], x2d[..., :1]

    def restore_dc_2d(self, y2d: torch.Tensor, dc2d: torch.Tensor | None) -> torch.Tensor:
        if not self.dc_bypass_enabled:
            return y2d
        if dc2d is None:
            raise ValueError("dc2d must be provided when DC bypass is enabled.")

        batch, channels, frames, _n_freq = y2d.shape
        source_count = self.core_n_src if self.residual_source_enabled else self.n_src
        expected_channels = 2 * source_count * self.n_chan
        if channels != expected_channels:
            raise ValueError(f"Expected {expected_channels} output channels, got {channels}")
        if dc2d.shape != (batch, 2 * self.n_chan, frames, 1):
            raise ValueError(f"Expected DC shape {(batch, 2 * self.n_chan, frames, 1)}, got {tuple(dc2d.shape)}")

        if self.dc_policy == "zero":
            dc_out = y2d.new_zeros(batch, channels, frames, 1)
        else:
            dc = dc2d.reshape(batch, self.n_chan, 2, frames, 1)
            dc = dc.unsqueeze(1).expand(batch, source_count, self.n_chan, 2, frames, 1)
            dc_out = dc.reshape(batch, channels, frames, 1) / float(self.n_src)
        return torch.cat([dc_out, y2d], dim=-1)

    def preprocess_core_input_2d(
        self,
        x2d: torch.Tensor,
        pcen_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.pcen_preprocessor is None:
            return x2d, None, pcen_state
        return self.pcen_preprocessor.forward_with_gain(x2d, pcen_state)

    def invert_core_output_gain(self, y2d: torch.Tensor, gain: torch.Tensor | None) -> torch.Tensor:
        if self.pcen_preprocessor is None or gain is None:
            return y2d
        return self.pcen_preprocessor.invert_output_gain(y2d, gain, n_src=self.core_n_src)

    def forward(self, x: torch.Tensor, **kwargs):
        kwargs.pop("ref", None)
        return_aux = bool(kwargs.pop("return_aux", False))
        mixture2d = pack_complex_stft_as_2d(x)
        x2d, dc2d = self.split_dc_2d(mixture2d)
        core_ref = self.preprocess_2d(x2d)
        core_in, gain, _pcen_state = self.preprocess_core_input_2d(core_ref)
        core_output = self.core(core_in, return_aux=True, **kwargs) if return_aux else self.core(core_in, **kwargs)
        if isinstance(core_output, tuple):
            y2d, aux = core_output
        else:
            y2d = core_output
            aux = {}
        y2d = self.invert_core_output_gain(y2d, gain)
        y2d = self.restore_dc_2d(self.postprocess_2d(y2d), dc2d)
        y2d = self.append_residual_source_2d(y2d, mixture2d)
        estimate = unpack_2d_to_complex_stft(y2d, n_src=self.n_src, n_chan=self.n_chan)
        if return_aux:
            return estimate, aux
        return estimate

    def _split_stream_state(self, state, *, batch_size: int, device=None, dtype=None):
        if self.pcen_preprocessor is None:
            return None, state
        if state is None:
            pcen_state = self.pcen_preprocessor.init_stream_state(
                batch_size,
                n_freq=self.core_n_freq,
                device=device,
                dtype=dtype,
            )
            core_state = self.core.init_stream_state(batch_size=batch_size, device=device, dtype=dtype)
            return pcen_state, core_state
        if not (isinstance(state, tuple) and len(state) == 2):
            raise ValueError("PCEN-enabled streaming state must be a (pcen_state, core_state) tuple.")
        return state[0], state[1]

    def init_stream_state(self, batch_size: int = 1, *, device=None, dtype=None):
        core_state = self.core.init_stream_state(batch_size=batch_size, device=device, dtype=dtype)
        if self.pcen_preprocessor is None:
            return core_state
        pcen_state = self.pcen_preprocessor.init_stream_state(
            batch_size,
            n_freq=self.core_n_freq,
            device=device,
            dtype=dtype,
        )
        return pcen_state, core_state

    def forward_stream(self, x2d: torch.Tensor, state=None):
        mixture2d = x2d
        pcen_state, core_state = self._split_stream_state(
            state,
            batch_size=x2d.shape[0],
            device=x2d.device,
            dtype=x2d.dtype,
        )
        x2d, dc2d = self.split_dc_2d(x2d)
        core_ref = self.preprocess_2d(x2d)
        core_in, gain, new_pcen_state = self.preprocess_core_input_2d(core_ref, pcen_state)
        y2d, new_core_state = self.core.forward_stream(core_in, core_state)
        y2d = self.invert_core_output_gain(y2d, gain)
        new_state = new_core_state if self.pcen_preprocessor is None else (new_pcen_state, new_core_state)
        y2d = self.restore_dc_2d(self.postprocess_2d(y2d), dc2d)
        y2d = self.append_residual_source_2d(y2d, mixture2d)
        return y2d, new_state

    def stream_context_frames(self) -> int:
        return self.core.stream_context_frames()

    def init_input_history(self, batch_size: int = 1, *, device=None, dtype=None):
        history_frames = self.stream_context_frames()
        return torch.zeros(batch_size, 2 * self.n_chan, history_frames, self.input_n_freq, device=device, dtype=dtype)

    def _init_core_history(self, batch_size: int = 1, *, device=None, dtype=None):
        history_frames = self.stream_context_frames()
        return torch.zeros(batch_size, 2 * self.n_chan, history_frames, self.core_n_freq, device=device, dtype=dtype)

    def _split_recompute_history(self, history, *, batch_size: int, device=None, dtype=None):
        if self.pcen_preprocessor is None:
            if history is None:
                history = self.init_input_history(batch_size=batch_size, device=device, dtype=dtype)
            return history, None, None

        if history is None:
            input_history = self.init_input_history(batch_size=batch_size, device=device, dtype=dtype)
            core_history = self._init_core_history(batch_size=batch_size, device=device, dtype=dtype)
            pcen_state = self.pcen_preprocessor.init_stream_state(
                batch_size,
                n_freq=self.core_n_freq,
                device=device,
                dtype=dtype,
            )
            return input_history, core_history, pcen_state

        if isinstance(history, tuple) and len(history) == 3:
            return history[0], history[1], history[2]

        input_history = history
        core_history = self.preprocess_2d(self.split_dc_2d(input_history)[0])
        pcen_state = self.pcen_preprocessor.init_stream_state(
            batch_size,
            n_freq=self.core_n_freq,
            device=device,
            dtype=dtype,
        )
        core_history, _gain, pcen_state = self.pcen_preprocessor.forward_with_gain(core_history, pcen_state)
        return input_history, core_history, pcen_state

    def forward_stream_recompute(self, x2d: torch.Tensor, history=None):
        mixture2d = x2d
        history, core_history, pcen_state = self._split_recompute_history(
            history,
            batch_size=x2d.shape[0],
            device=x2d.device,
            dtype=x2d.dtype,
        )

        ctx = self.stream_context_frames()
        full_history = torch.cat([history, x2d], dim=2)
        new_history = full_history[:, :, -ctx:, :] if ctx > 0 else full_history[:, :, :0, :]
        x2d, dc2d = self.split_dc_2d(x2d)
        history, _history_dc = self.split_dc_2d(history)

        if self.pcen_preprocessor is None:
            x2d_reduced = self.preprocess_2d(x2d)
            history_reduced = self.preprocess_2d(history)
            y2d_reduced, _ = self.core.forward_stream_recompute(x2d_reduced, history_reduced)
            y2d = self.restore_dc_2d(self.postprocess_2d(y2d_reduced), dc2d)
            y2d = self.append_residual_source_2d(y2d, mixture2d)
            return y2d, new_history

        x2d_reduced = self.preprocess_2d(x2d)
        x_core, gain, new_pcen_state = self.preprocess_core_input_2d(x2d_reduced, pcen_state)
        y2d_reduced, _ = self.core.forward_stream_recompute(x_core, core_history)
        y2d_reduced = self.invert_core_output_gain(y2d_reduced, gain)
        full_core_history = torch.cat([core_history, x_core], dim=2)
        new_core_history = full_core_history[:, :, -ctx:, :] if ctx > 0 else full_core_history[:, :, :0, :]
        new_state = (new_history, new_core_history, new_pcen_state)
        y2d = self.restore_dc_2d(self.postprocess_2d(y2d_reduced), dc2d)
        y2d = self.append_residual_source_2d(y2d, mixture2d)
        return y2d, new_state

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

    def residual_source_manifest(self) -> dict[str, object] | None:
        if not self.residual_source_enabled:
            return None
        return {
            "enabled": True,
            "mode": "mixture_minus_explicit_sources",
            "explicit_n_src": self.core_n_src,
            "output_n_src": self.n_src,
            "residual_source_index": self.residual_source_index,
        }

    def prompt_conditioning_manifest(self) -> dict[str, object] | None:
        if not hasattr(self.core, "prompt_manifest"):
            return None
        return self.core.prompt_manifest()
