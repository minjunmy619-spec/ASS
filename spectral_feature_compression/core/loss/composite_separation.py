from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn
import torch.nn.functional as F

Resolution = tuple[int, int]


class CompositeSeparationSpectralLoss(nn.Module):
    """Opt-in spectral loss stack for waveform source-separation training.

    Inputs use the repo's separated waveform layout: ``[B, N, C, samples]``.
    The module is training-only and intentionally independent from the exported
    NPU graph.
    """

    def __init__(
        self,
        *,
        n_fft: int,
        hop_length: int,
        complex_ri_weight: float = 0.0,
        log_magnitude_weight: float = 0.0,
        multi_resolution_stft_weight: float = 0.0,
        multi_resolution_stft_resolutions: Sequence[Sequence[int] | Resolution] | None = None,
        transient_weight: float = 0.0,
        source_order: Sequence[str] = ("speech", "music", "effects"),
        speech_leakage_weight: float = 0.0,
        speech_leakage_source: str = "speech",
        speech_leakage_target_sources: Sequence[str] | None = None,
        speech_leakage_n_fft: int | None = None,
        speech_leakage_hop_length: int | None = None,
        speech_leakage_speech_active_db: float = -45.0,
        speech_leakage_target_relative_db: float = 12.0,
        speech_leakage_mask_softness_db: float = 3.0,
        speech_leakage_tolerance_ratio: float = 0.0,
        eps: float = 1.0e-7,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.complex_ri_weight = float(complex_ri_weight)
        self.log_magnitude_weight = float(log_magnitude_weight)
        self.multi_resolution_stft_weight = float(multi_resolution_stft_weight)
        self.transient_weight = float(transient_weight)
        self.source_order = tuple(str(source) for source in source_order)
        self.speech_leakage_weight = float(speech_leakage_weight)
        self.speech_leakage_source = str(speech_leakage_source)
        self.speech_leakage_target_sources = (
            None
            if speech_leakage_target_sources is None
            else tuple(str(source) for source in speech_leakage_target_sources)
        )
        self.speech_leakage_n_fft = int(n_fft if speech_leakage_n_fft is None else speech_leakage_n_fft)
        self.speech_leakage_hop_length = int(
            hop_length if speech_leakage_hop_length is None else speech_leakage_hop_length
        )
        self.speech_leakage_speech_active_db = float(speech_leakage_speech_active_db)
        self.speech_leakage_target_relative_db = float(speech_leakage_target_relative_db)
        self.speech_leakage_mask_softness_db = float(speech_leakage_mask_softness_db)
        self.speech_leakage_tolerance_ratio = float(speech_leakage_tolerance_ratio)
        self.eps = float(eps)

        if self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("n_fft and hop_length must be positive")
        if self.speech_leakage_n_fft <= 0 or self.speech_leakage_hop_length <= 0:
            raise ValueError("speech_leakage_n_fft and speech_leakage_hop_length must be positive")
        if not self.source_order or len(set(self.source_order)) != len(self.source_order):
            raise ValueError(f"source_order must contain unique source names, got {self.source_order}")
        for name, value in {
            "complex_ri_weight": self.complex_ri_weight,
            "log_magnitude_weight": self.log_magnitude_weight,
            "multi_resolution_stft_weight": self.multi_resolution_stft_weight,
            "transient_weight": self.transient_weight,
            "speech_leakage_weight": self.speech_leakage_weight,
            "speech_leakage_speech_active_db": self.speech_leakage_speech_active_db,
            "speech_leakage_target_relative_db": self.speech_leakage_target_relative_db,
            "speech_leakage_mask_softness_db": self.speech_leakage_mask_softness_db,
            "speech_leakage_tolerance_ratio": self.speech_leakage_tolerance_ratio,
            "eps": self.eps,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if any(
            value < 0.0
            for value in (
                self.complex_ri_weight,
                self.log_magnitude_weight,
                self.multi_resolution_stft_weight,
                self.transient_weight,
                self.speech_leakage_weight,
                self.speech_leakage_tolerance_ratio,
            )
        ):
            raise ValueError("Composite loss weights and speech_leakage_tolerance_ratio must be non-negative")
        if self.speech_leakage_mask_softness_db <= 0.0:
            raise ValueError("speech_leakage_mask_softness_db must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

        if multi_resolution_stft_resolutions is None:
            multi_resolution_stft_resolutions = ((self.n_fft, self.hop_length),)
        self.multi_resolution_stft_resolutions = tuple(
            (int(resolution[0]), int(resolution[1])) for resolution in multi_resolution_stft_resolutions
        )

    @property
    def enabled(self) -> bool:
        return any(
            weight > 0.0
            for weight in (
                self.complex_ri_weight,
                self.log_magnitude_weight,
                self.multi_resolution_stft_weight,
                self.transient_weight,
                self.speech_leakage_weight,
            )
        )

    def _stft(self, wav: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
        wav = wav.float().reshape(-1, wav.shape[-1])
        if wav.shape[-1] < n_fft:
            wav = F.pad(wav, (0, n_fft - wav.shape[-1]))
        window = torch.hann_window(n_fft, device=wav.device, dtype=wav.dtype)
        return torch.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )

    def _complex_ri_l1(self, est: torch.Tensor, ref: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
        est_spec = self._stft(est, n_fft=n_fft, hop_length=hop_length)
        ref_spec = self._stft(ref, n_fft=n_fft, hop_length=hop_length)
        return F.l1_loss(torch.view_as_real(est_spec), torch.view_as_real(ref_spec))

    def _log_magnitude_l1(self, est: torch.Tensor, ref: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
        est_spec = self._stft(est, n_fft=n_fft, hop_length=hop_length)
        ref_spec = self._stft(ref, n_fft=n_fft, hop_length=hop_length)
        return F.l1_loss(torch.log(est_spec.abs() + self.eps), torch.log(ref_spec.abs() + self.eps))

    def _multi_resolution_stft(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        losses = []
        for n_fft, hop_length in self.multi_resolution_stft_resolutions:
            est_spec = self._stft(est, n_fft=n_fft, hop_length=hop_length)
            ref_spec = self._stft(ref, n_fft=n_fft, hop_length=hop_length)
            complex_loss = F.l1_loss(torch.view_as_real(est_spec), torch.view_as_real(ref_spec))
            log_mag_loss = F.l1_loss(torch.log(est_spec.abs() + self.eps), torch.log(ref_spec.abs() + self.eps))
            losses.append(complex_loss + log_mag_loss)
        return torch.stack(losses).mean()

    def _transient_l1(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if est.shape[-1] < 2:
            return est.new_zeros(())
        est_diff = est[..., 1:] - est[..., :-1]
        ref_diff = ref[..., 1:] - ref[..., :-1]
        return F.l1_loss(est_diff, ref_diff)

    def _leakage_source_indices(self, n_sources: int) -> tuple[int, tuple[int, ...]]:
        if n_sources != len(self.source_order):
            raise ValueError(
                f"speech leakage loss expected {len(self.source_order)} sources from source_order, got {n_sources}"
            )
        if self.speech_leakage_source not in self.source_order:
            raise ValueError(
                f"speech_leakage_source={self.speech_leakage_source!r} is not in source_order={self.source_order}"
            )
        targets = (
            tuple(source for source in self.source_order if source != self.speech_leakage_source)
            if self.speech_leakage_target_sources is None
            else self.speech_leakage_target_sources
        )
        if not targets:
            raise ValueError("speech_leakage_target_sources must contain at least one non-speech source")
        unknown = sorted(set(targets) - set(self.source_order))
        if unknown:
            raise ValueError(f"speech_leakage_target_sources contains unknown sources: {unknown}")
        if self.speech_leakage_source in targets:
            raise ValueError("speech_leakage_target_sources must not contain speech_leakage_source")
        return (
            self.source_order.index(self.speech_leakage_source),
            tuple(self.source_order.index(source) for source in targets),
        )

    def _source_magnitudes(self, wav: torch.Tensor, *, n_fft: int, hop_length: int) -> torch.Tensor:
        if wav.ndim != 4:
            raise ValueError(f"speech leakage loss expects [batch, source, channel, samples], got {tuple(wav.shape)}")
        spec = self._stft(wav, n_fft=n_fft, hop_length=hop_length)
        reshaped = spec.reshape(*wav.shape[:-1], *spec.shape[-2:]).abs()
        return reshaped.mean(dim=2)

    def _speech_leakage_tf(self, est: torch.Tensor, ref: torch.Tensor) -> dict[str, torch.Tensor]:
        speech_index, target_indices = self._leakage_source_indices(est.shape[1])
        est_mag = self._source_magnitudes(
            est,
            n_fft=self.speech_leakage_n_fft,
            hop_length=self.speech_leakage_hop_length,
        )
        with torch.no_grad():
            ref_mag = self._source_magnitudes(
                ref,
                n_fft=self.speech_leakage_n_fft,
                hop_length=self.speech_leakage_hop_length,
            )

        speech_mag = ref_mag[:, speech_index]
        speech_db = 20.0 * torch.log10(speech_mag.clamp_min(self.eps))
        speech_active = torch.sigmoid(
            (speech_db - self.speech_leakage_speech_active_db) / self.speech_leakage_mask_softness_db
        )
        speech_scale = speech_mag.detach().clamp_min(self.eps)
        components: dict[str, torch.Tensor] = {}
        target_values = []
        for target_index in target_indices:
            target_mag = ref_mag[:, target_index]
            target_db = 20.0 * torch.log10(target_mag.clamp_min(self.eps))
            target_quiet = torch.sigmoid(
                (speech_db - target_db - self.speech_leakage_target_relative_db)
                / self.speech_leakage_mask_softness_db
            )
            mask = speech_active * target_quiet
            excess = F.relu(
                (est_mag[:, target_index] - target_mag) / speech_scale - self.speech_leakage_tolerance_ratio
            )
            denominator = mask.sum()
            value = excess.sum() * 0.0 if not bool(denominator > 0) else (mask * excess).sum() / denominator
            target_name = self.source_order[target_index]
            components[f"speech_leakage_tf_{target_name}"] = value
            target_values.append(value)
        components["speech_leakage_tf"] = torch.stack(target_values).mean()
        return components

    def speech_leakage_components(self, est: torch.Tensor, ref: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return GT-gated Speech-to-target leakage diagnostics.

        This method intentionally works regardless of ``speech_leakage_weight``
        so paired-reference evaluators can report the same quantity without
        changing the configured training objective.
        """

        if est.shape != ref.shape:
            raise ValueError(f"est/ref shape mismatch: {tuple(est.shape)} vs {tuple(ref.shape)}")
        return self._speech_leakage_tf(est, ref)

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if est.shape != ref.shape:
            raise ValueError(f"est/ref shape mismatch: {tuple(est.shape)} vs {tuple(ref.shape)}")

        total = est.new_zeros(())
        components: dict[str, torch.Tensor] = {}

        if self.complex_ri_weight > 0.0:
            value = self._complex_ri_l1(est, ref, self.n_fft, self.hop_length)
            components["complex_ri"] = value
            total = total + self.complex_ri_weight * value

        if self.log_magnitude_weight > 0.0:
            value = self._log_magnitude_l1(est, ref, self.n_fft, self.hop_length)
            components["log_magnitude"] = value
            total = total + self.log_magnitude_weight * value

        if self.multi_resolution_stft_weight > 0.0:
            value = self._multi_resolution_stft(est, ref)
            components["multi_resolution_stft"] = value
            total = total + self.multi_resolution_stft_weight * value

        if self.transient_weight > 0.0:
            value = self._transient_l1(est, ref)
            components["transient"] = value
            total = total + self.transient_weight * value

        if self.speech_leakage_weight > 0.0:
            leakage_components = self.speech_leakage_components(est, ref)
            components.update(leakage_components)
            total = total + self.speech_leakage_weight * leakage_components["speech_leakage_tf"]

        return total, components
