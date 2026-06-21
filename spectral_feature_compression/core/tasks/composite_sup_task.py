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
        normalize_active_sources_for_aux_loss: bool = False,
        aux_activity_threshold_db: float = -60.0,
        normalize_mixture_consistency: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fs = fs
        self.mixture_consistency_weight = float(mixture_consistency_weight)
        self.low_frequency_weight = float(low_frequency_weight)
        self.low_frequency_hz = float(low_frequency_hz)
        self.normalize_active_sources_for_aux_loss = bool(normalize_active_sources_for_aux_loss)
        self.aux_activity_threshold_db = float(aux_activity_threshold_db)
        self.normalize_mixture_consistency = bool(normalize_mixture_consistency)
        self.composite_loss = CompositeSeparationSpectralLoss(
            n_fft=self.stft[0].n_fft,
            hop_length=self.stft[0].hop_length,
            complex_ri_weight=complex_ri_weight,
            log_magnitude_weight=log_magnitude_weight,
            multi_resolution_stft_weight=multi_resolution_stft_weight,
            multi_resolution_stft_resolutions=multi_resolution_stft_resolutions,
            transient_weight=transient_weight,
        )

    def _prepare_auxiliary_sources(
        self,
        wav: torch.Tensor,
        est: torch.Tensor,
        ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.normalize_active_sources_for_aux_loss:
            return est, ref

        est = est.float()
        ref = ref.float()
        wav = wav.float()
        eps = 1.0e-8
        ref_rms = ref.square().mean(dim=(-1, -2), keepdim=True).sqrt()
        mix_rms = wav.square().mean(dim=(-1, -2), keepdim=True).sqrt().unsqueeze(1)
        activity_ratio = ref_rms / mix_rms.clamp_min(eps)
        activity_threshold = 10.0 ** (self.aux_activity_threshold_db / 20.0)
        active = (mix_rms > eps) & activity_ratio.ge(activity_threshold)
        scale = ref_rms.detach().clamp_min(eps)
        active_mask = active.squeeze(-1).squeeze(-1)
        return (est / scale)[active_mask], (ref / scale)[active_mask]

    def _mixture_consistency_l1(self, wav: torch.Tensor, est: torch.Tensor) -> torch.Tensor:
        est_mix = est.sum(dim=1)
        if not self.normalize_mixture_consistency:
            return F.l1_loss(est_mix, wav)
        scale = wav.float().square().mean(dim=(-1, -2), keepdim=True).sqrt().detach().clamp_min(1.0e-8)
        return F.l1_loss(est_mix.float() / scale, wav.float() / scale)

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

        aux_est, aux_ref = self._prepare_auxiliary_sources(wav, est, ref)

        if self.mixture_consistency_weight > 0.0:
            mixture_loss = self._mixture_consistency_l1(wav, est)
            loss = loss + self.mixture_consistency_weight * mixture_loss
            log_dict[f"{log_prefix}/loss_mixture_consistency"] = mixture_loss

        if self.low_frequency_weight > 0.0:
            low_freq_loss = self._low_frequency_l1(aux_est, aux_ref) if aux_est.shape[0] > 0 else est.sum() * 0.0
            loss = loss + self.low_frequency_weight * low_freq_loss
            log_dict[f"{log_prefix}/loss_low_frequency"] = low_freq_loss

        if self.composite_loss.enabled:
            if aux_est.shape[0] > 0:
                composite_loss, component_losses = self.composite_loss(aux_est, aux_ref)
            else:
                composite_loss = est.sum() * 0.0
                component_losses = {}
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
