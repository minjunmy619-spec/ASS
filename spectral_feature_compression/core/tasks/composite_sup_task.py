from __future__ import annotations

import torch
import torch.nn.functional as F

from spectral_feature_compression.core.loss.composite_separation import CompositeSeparationSpectralLoss
from spectral_feature_compression.core.tasks.sup_task import SupTask


class CompositeSupTask(SupTask):
    """Supervised separation task with optional spectral and consistency losses.

    ``SupTask`` is intentionally minimal and optimizes one waveform loss.  This
    task keeps that behavior as the base objective, then adds quality-teacher
    losses that are useful for performance-first source separation:

    - mixture consistency in waveform domain;
    - low-frequency magnitude matching;
    - multi-resolution complex/log-magnitude/transient spectral losses.
    """

    def __init__(
        self,
        *args,
        fs: int | None = None,
        mixture_consistency_weight: float = 0.0,
        low_frequency_weight: float = 0.0,
        low_frequency_hz: float = 300.0,
        complex_ri_weight: float = 0.0,
        log_magnitude_weight: float = 0.0,
        multi_resolution_stft_weight: float = 0.0,
        multi_resolution_stft_resolutions=None,
        transient_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fs = fs
        self.mixture_consistency_weight = float(mixture_consistency_weight)
        self.low_frequency_weight = float(low_frequency_weight)
        self.low_frequency_hz = float(low_frequency_hz)
        self.composite_loss = CompositeSeparationSpectralLoss(
            n_fft=self.stft[0].n_fft,
            hop_length=self.stft[0].hop_length,
            complex_ri_weight=complex_ri_weight,
            log_magnitude_weight=log_magnitude_weight,
            multi_resolution_stft_weight=multi_resolution_stft_weight,
            multi_resolution_stft_resolutions=multi_resolution_stft_resolutions,
            transient_weight=transient_weight,
        )

    def _low_frequency_l1(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if self.fs is None:
            raise ValueError("low_frequency_weight requires fs so the cutoff can be mapped to STFT bins")
        est_spec = self.stft(est.float()).abs()
        ref_spec = self.stft(ref.float()).abs()
        max_bin = int(self.low_frequency_hz * self.stft[0].n_fft / self.fs) + 1
        max_bin = max(1, min(max_bin, est_spec.shape[-2]))
        return F.l1_loss(est_spec[..., :max_bin, :], ref_spec[..., :max_bin, :])

    def _step(self, wav: torch.Tensor, ref: torch.Tensor, log_prefix: str):
        model = self.ema_model.module if self.use_ema_model and log_prefix != "training" else self.model
        est = model.css(wav, ref=ref) if log_prefix != "training" and self.css_validation else model(wav)

        supervised_loss = self.loss(est.transpose(1, 2), ref.transpose(1, 2)).mean()
        loss = supervised_loss
        log_dict = {
            "step": float(self.trainer.current_epoch),
            f"{log_prefix}/loss_supervised": supervised_loss,
        }

        if self.mixture_consistency_weight > 0.0:
            mixture_loss = F.l1_loss(est.sum(dim=1), wav)
            loss = loss + self.mixture_consistency_weight * mixture_loss
            log_dict[f"{log_prefix}/loss_mixture_consistency"] = mixture_loss

        if self.low_frequency_weight > 0.0:
            low_freq_loss = self._low_frequency_l1(est, ref)
            loss = loss + self.low_frequency_weight * low_freq_loss
            log_dict[f"{log_prefix}/loss_low_frequency"] = low_freq_loss

        if self.composite_loss.enabled:
            composite_loss, component_losses = self.composite_loss(est, ref)
            loss = loss + composite_loss
            log_dict[f"{log_prefix}/loss_composite_spectral"] = composite_loss
            for name, value in component_losses.items():
                log_dict[f"{log_prefix}/loss_{name}"] = value

        log_dict[f"{log_prefix}/loss"] = loss
        if log_prefix == "validation":
            snr_score = self.snr(est.transpose(1, 2), ref.transpose(1, 2)).mean()
            log_dict[f"{log_prefix}/snr"] = snr_score

        self.log_dict(log_dict, prog_bar=False, on_epoch=True, on_step=False, batch_size=wav.shape[0], sync_dist=True)
        return loss
