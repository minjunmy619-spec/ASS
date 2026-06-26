from __future__ import annotations

import math
from numbers import Integral

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_finite_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value}")
    return result


def _as_int_index(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer or None, got {value}")
    return int(value)


class SourceSeparationPostProcessor(nn.Module):
    """Optional STFT-domain post-processing for separated source estimates."""

    def __init__(
        self,
        *,
        mixture_consistency: str = "none",
        final_mixture_consistency: str = "none",
        power_beta: float = 1.0,
        power_smoothing: float = 0.0,
        wiener_blend: float = 0.0,
        wiener_alpha: float = 1.0,
        leakage_gate_enabled: bool = False,
        leakage_gate_threshold_db: float = 12.0,
        leakage_gate_attenuation_db: float = 6.0,
        residual_source_index: int | None = None,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        power_beta = _as_finite_float(power_beta, name="power_beta")
        power_smoothing = _as_finite_float(power_smoothing, name="power_smoothing")
        wiener_blend = _as_finite_float(wiener_blend, name="wiener_blend")
        wiener_alpha = _as_finite_float(wiener_alpha, name="wiener_alpha")
        leakage_gate_threshold_db = _as_finite_float(
            leakage_gate_threshold_db,
            name="leakage_gate_threshold_db",
        )
        leakage_gate_attenuation_db = _as_finite_float(
            leakage_gate_attenuation_db,
            name="leakage_gate_attenuation_db",
        )
        eps = _as_finite_float(eps, name="eps")
        residual_source_index = _as_int_index(residual_source_index, name="residual_source_index")

        valid_consistency = {"none", "uniform", "power"}
        if mixture_consistency not in valid_consistency:
            raise ValueError(
                f"mixture_consistency must be one of {sorted(valid_consistency)}, got {mixture_consistency}"
            )
        if final_mixture_consistency not in valid_consistency:
            raise ValueError(
                f"final_mixture_consistency must be one of {sorted(valid_consistency)}, got {final_mixture_consistency}"
            )
        if power_beta <= 0.0:
            raise ValueError(f"power_beta must be positive, got {power_beta}")
        if not 0.0 <= power_smoothing < 1.0:
            raise ValueError(f"power_smoothing must be in [0, 1), got {power_smoothing}")
        if not 0.0 <= wiener_blend <= 1.0:
            raise ValueError(f"wiener_blend must be in [0, 1], got {wiener_blend}")
        if wiener_alpha <= 0.0:
            raise ValueError(f"wiener_alpha must be positive, got {wiener_alpha}")
        if leakage_gate_threshold_db < 0.0:
            raise ValueError(f"leakage_gate_threshold_db must be non-negative, got {leakage_gate_threshold_db}")
        if leakage_gate_attenuation_db < 0.0:
            raise ValueError(f"leakage_gate_attenuation_db must be non-negative, got {leakage_gate_attenuation_db}")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.mixture_consistency = str(mixture_consistency)
        self.final_mixture_consistency = str(final_mixture_consistency)
        self.power_beta = float(power_beta)
        self.power_smoothing = float(power_smoothing)
        self.wiener_blend = float(wiener_blend)
        self.wiener_alpha = float(wiener_alpha)
        self.leakage_gate_enabled = bool(leakage_gate_enabled)
        self.leakage_gate_threshold_db = float(leakage_gate_threshold_db)
        self.leakage_gate_attenuation_db = float(leakage_gate_attenuation_db)
        self.residual_source_index = residual_source_index
        self.eps = float(eps)

    def _validate_shapes(self, estimates: torch.Tensor, mixture: torch.Tensor) -> None:
        if not torch.is_complex(estimates) or not torch.is_complex(mixture):
            raise TypeError("SourceSeparationPostProcessor expects complex STFT tensors.")
        if estimates.ndim != 5:
            raise ValueError(f"Expected estimates with shape [B,N,M,F,T], got {tuple(estimates.shape)}")
        if mixture.ndim != 4:
            raise ValueError(f"Expected mixture with shape [B,M,F,T], got {tuple(mixture.shape)}")
        if estimates.shape[0] != mixture.shape[0] or estimates.shape[2:] != mixture.shape[1:]:
            raise ValueError(
                f"Estimate/mixture shapes are incompatible: {tuple(estimates.shape)} vs {tuple(mixture.shape)}"
            )
        if estimates.shape[1] <= 0:
            raise ValueError("Expected at least one source estimate.")
        if self.residual_source_index is not None and not 0 <= self.residual_source_index < estimates.shape[1]:
            raise ValueError(
                f"residual_source_index={self.residual_source_index} is outside the source axis "
                f"with size {estimates.shape[1]}"
            )

    def _smooth_power(self, power: torch.Tensor) -> torch.Tensor:
        if self.power_smoothing <= 0.0 or power.shape[-1] <= 1:
            return power
        frames = [power[..., 0]]
        prev = frames[0]
        alpha = self.power_smoothing
        for frame_idx in range(1, power.shape[-1]):
            prev = alpha * prev + (1.0 - alpha) * power[..., frame_idx]
            frames.append(prev)
        return torch.stack(frames, dim=-1)

    def _power_scores(self, estimates: torch.Tensor, *, exponent: float) -> torch.Tensor:
        power = estimates.abs().square().clamp_min(self.eps)
        power = self._smooth_power(power)
        return power.pow(exponent)

    def _apply_mixture_consistency(self, estimates: torch.Tensor, mixture: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "none":
            return estimates
        residual = mixture.unsqueeze(1) - estimates.sum(dim=1, keepdim=True)
        if mode == "uniform":
            return estimates + residual / float(estimates.shape[1])

        weights = self._power_scores(estimates, exponent=self.power_beta)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return estimates + weights.to(dtype=estimates.real.dtype) * residual

    def _apply_wiener(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if self.wiener_blend <= 0.0:
            return estimates
        scores = self._power_scores(estimates, exponent=self.wiener_alpha)
        gains = scores / scores.sum(dim=1, keepdim=True).clamp_min(self.eps)
        wiener_estimates = gains.to(dtype=estimates.real.dtype) * mixture.unsqueeze(1)
        return (1.0 - self.wiener_blend) * estimates + self.wiener_blend * wiener_estimates

    def _apply_leakage_gate(self, estimates: torch.Tensor) -> torch.Tensor:
        if (
            not self.leakage_gate_enabled
            or self.leakage_gate_attenuation_db <= 0.0
            or estimates.shape[1] <= 1
        ):
            return estimates

        power = self._smooth_power(estimates.abs().square().clamp_min(self.eps))
        max_power = power.max(dim=1, keepdim=True).values
        dominance_ratio = 10.0 ** (self.leakage_gate_threshold_db / 10.0)
        weak = power * dominance_ratio < max_power
        gate_gain = 10.0 ** (-self.leakage_gate_attenuation_db / 20.0)
        return torch.where(weak, estimates * gate_gain, estimates)

    def _apply_residual_source(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        if self.residual_source_index is None:
            return estimates
        chunks = list(estimates.unbind(dim=1))
        residual = mixture - sum(chunk for idx, chunk in enumerate(chunks) if idx != self.residual_source_index)
        chunks[self.residual_source_index] = residual
        return torch.stack(chunks, dim=1)

    def forward(self, estimates: torch.Tensor, mixture: torch.Tensor) -> torch.Tensor:
        self._validate_shapes(estimates, mixture)
        estimates = self._apply_mixture_consistency(estimates, mixture, self.mixture_consistency)
        estimates = self._apply_wiener(estimates, mixture)
        estimates = self._apply_leakage_gate(estimates)
        estimates = self._apply_residual_source(estimates, mixture)
        return self._apply_mixture_consistency(estimates, mixture, self.final_mixture_consistency)


def build_source_separation_postprocessor(
    *,
    enabled: bool = False,
    mixture_consistency: str = "none",
    final_mixture_consistency: str = "none",
    power_beta: float = 1.0,
    power_smoothing: float = 0.0,
    wiener_blend: float = 0.0,
    wiener_alpha: float = 1.0,
    leakage_gate_enabled: bool = False,
    leakage_gate_threshold_db: float = 12.0,
    leakage_gate_attenuation_db: float = 6.0,
    residual_source_index: int | None = None,
    eps: float = 1.0e-8,
) -> SourceSeparationPostProcessor | None:
    if not enabled:
        return None
    return SourceSeparationPostProcessor(
        mixture_consistency=mixture_consistency,
        final_mixture_consistency=final_mixture_consistency,
        power_beta=power_beta,
        power_smoothing=power_smoothing,
        wiener_blend=wiener_blend,
        wiener_alpha=wiener_alpha,
        leakage_gate_enabled=leakage_gate_enabled,
        leakage_gate_threshold_db=leakage_gate_threshold_db,
        leakage_gate_attenuation_db=leakage_gate_attenuation_db,
        residual_source_index=residual_source_index,
        eps=eps,
    )


class MISIPhaseConsistency(nn.Module):
    """Optional MISI-style phase/mixing consistency projection."""

    def __init__(self, *, iterations: int = 0, eps: float = 1.0e-8) -> None:
        super().__init__()
        if isinstance(iterations, bool) or not isinstance(iterations, Integral):
            raise TypeError(f"iterations must be an integer, got {iterations}")
        if iterations < 0:
            raise ValueError(f"iterations must be non-negative, got {iterations}")
        eps = _as_finite_float(eps, name="eps")
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.iterations = int(iterations)
        self.eps = float(eps)

    @staticmethod
    def _match_frames(value: torch.Tensor, target_frames: int) -> torch.Tensor:
        frames = int(value.shape[-1])
        target_frames = int(target_frames)
        if frames == target_frames:
            return value
        if frames > target_frames:
            return value[..., :target_frames]
        return F.pad(value, (0, target_frames - frames))

    def forward(
        self,
        estimates: torch.Tensor,
        mixture_wave: torch.Tensor,
        *,
        stft: nn.Module,
        istft: nn.Module,
        length: int,
        target_frames: int,
    ) -> torch.Tensor:
        if self.iterations == 0:
            return estimates
        if not torch.is_complex(estimates):
            raise TypeError("MISIPhaseConsistency expects complex STFT estimates.")
        if estimates.ndim != 5:
            raise ValueError(f"Expected estimates with shape [B,N,M,F,T], got {tuple(estimates.shape)}")
        if mixture_wave.ndim != 3:
            raise ValueError(f"Expected mixture waveform with shape [B,M,L], got {tuple(mixture_wave.shape)}")
        if estimates.shape[0] != mixture_wave.shape[0] or estimates.shape[2] != mixture_wave.shape[1]:
            raise ValueError(
                f"Estimate/mixture shapes are incompatible: {tuple(estimates.shape)} vs {tuple(mixture_wave.shape)}"
            )
        if estimates.shape[1] <= 0:
            raise ValueError("Expected at least one source estimate.")
        target_frames = int(target_frames)
        if target_frames <= 0:
            raise ValueError(f"target_frames must be positive, got {target_frames}")
        if estimates.shape[-1] != target_frames:
            raise ValueError(f"Expected target_frames={estimates.shape[-1]}, got {target_frames}")
        length = int(length)
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if mixture_wave.shape[-1] != length:
            raise ValueError(f"Expected length={mixture_wave.shape[-1]} for mixture_wave, got {length}")

        target_magnitude = estimates.abs()
        current = estimates
        projected_wave = None
        for _ in range(self.iterations):
            source_wave = istft(current, length)
            residual = mixture_wave.unsqueeze(1) - source_wave.sum(dim=1, keepdim=True)
            projected_wave = source_wave + residual / float(estimates.shape[1])
            projected = stft(projected_wave)
            projected = self._match_frames(projected, target_frames)
            current = target_magnitude * projected / projected.abs().clamp_min(self.eps)

        if projected_wave is None:
            return current
        return self._match_frames(stft(projected_wave), target_frames)


def build_misi_phase_consistency(*, iterations: int = 0, eps: float = 1.0e-8) -> MISIPhaseConsistency | None:
    if iterations == 0:
        return None
    return MISIPhaseConsistency(iterations=iterations, eps=eps)
