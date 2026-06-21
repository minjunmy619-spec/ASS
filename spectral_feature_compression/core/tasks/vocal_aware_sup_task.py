from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from spectral_feature_compression.core.tasks.composite_sup_task import CompositeSupTask


class VocalAwareCompositeSupTask(CompositeSupTask):
    """Composite supervised teacher task with speech/vocal-aware robust losses.

    This task is intended for quality-teacher training when the ``speech`` stem
    includes singing vocals and some references may be pseudo-separated/noisy.
    The additional losses are training-only and do not affect exported models.

    Added terms:

    - soft-truncated speech log-magnitude MR-STFT loss, robust to noisy labels;
    - soft-truncated temporal/frequency log-magnitude gradient losses, for
      smoother harmonic/formant continuity;
    - inactive speech leakage penalty, not truncated, to keep vocal gaps clean.
    """

    _BAND_DEPENDENT_MODEL_KEYS = (
        "model.encoder.query",
        "model.encoder.block.mixer.pos_bias.pos_bias",
        "model.encoder.block.mixer.pos_bias.attention_mask",
        "model.decoder.block.mixer.pos_bias.pos_bias",
        "model.decoder.block.mixer.pos_bias.attention_mask",
    )

    def load_pretrained_weight(self) -> None:
        """Warm-start shared weights while retaining the configured vocal band map."""

        if self.pretrained_model_path is None:
            return
        if not self.preserve_initialized_band_layout_on_pretrained_load:
            super().load_pretrained_weight()
            return
        current_state = self.model.state_dict()
        preserved = {
            key: current_state[key].detach().clone()
            for key in self._BAND_DEPENDENT_MODEL_KEYS
            if key in current_state
        }
        super().load_pretrained_weight()
        loaded_state = self.model.state_dict()
        with torch.no_grad():
            for key, value in preserved.items():
                loaded_state[key].copy_(value)

    def __init__(
        self,
        *args,
        speech_source_index: int = 0,
        speech_robust_logmag_weight: float = 0.0,
        speech_robust_logmag_tau: float = 1.0,
        speech_robust_logmag_resolutions: Sequence[Sequence[int]] | None = None,
        vocal_active_frame_weight: float = 0.0,
        speech_temporal_logmag_gradient_weight: float = 0.0,
        speech_frequency_logmag_gradient_weight: float = 0.0,
        speech_gradient_tau: float = 1.0,
        speech_inactive_leakage_weight: float = 0.0,
        speech_inactive_threshold_db: float = -45.0,
        speech_inactive_softness_db: float = 6.0,
        preserve_initialized_band_layout_on_pretrained_load: bool = False,
        **kwargs,
    ):
        self.preserve_initialized_band_layout_on_pretrained_load = bool(
            preserve_initialized_band_layout_on_pretrained_load
        )
        super().__init__(*args, **kwargs)
        self.speech_source_index = int(speech_source_index)
        self.speech_robust_logmag_weight = float(speech_robust_logmag_weight)
        self.speech_robust_logmag_tau = float(speech_robust_logmag_tau)
        self.vocal_active_frame_weight = float(vocal_active_frame_weight)
        self.speech_temporal_logmag_gradient_weight = float(speech_temporal_logmag_gradient_weight)
        self.speech_frequency_logmag_gradient_weight = float(speech_frequency_logmag_gradient_weight)
        self.speech_gradient_tau = float(speech_gradient_tau)
        self.speech_inactive_leakage_weight = float(speech_inactive_leakage_weight)
        self.speech_inactive_threshold_db = float(speech_inactive_threshold_db)
        self.speech_inactive_softness_db = float(speech_inactive_softness_db)
        if speech_robust_logmag_resolutions is None:
            speech_robust_logmag_resolutions = ((512, 128), (1024, 256), (2048, 512))
        self.speech_robust_logmag_resolutions = tuple(
            (int(resolution[0]), int(resolution[1])) for resolution in speech_robust_logmag_resolutions
        )

    @property
    def vocal_losses_enabled(self) -> bool:
        return any(
            weight > 0.0
            for weight in (
                self.speech_robust_logmag_weight,
                self.speech_temporal_logmag_gradient_weight,
                self.speech_frequency_logmag_gradient_weight,
                self.speech_inactive_leakage_weight,
            )
        )

    def _soft_truncated_l1(self, error: torch.Tensor, tau: float) -> torch.Tensor:
        tau = max(float(tau), 1.0e-6)
        return tau * (1.0 - torch.exp(-error.abs() / tau))

    def _stft_logmag(self, wav: torch.Tensor, n_fft: int, hop_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        # wav: [B, C, T]
        if wav.ndim != 3:
            raise ValueError(f"Expected [B, C, T] speech wav, got {tuple(wav.shape)}")
        bsz, n_chan, n_samples = wav.shape
        flat = wav.float().reshape(bsz * n_chan, n_samples)
        if flat.shape[-1] < n_fft:
            flat = F.pad(flat, (0, n_fft - flat.shape[-1]))
        window = torch.hann_window(n_fft, device=flat.device, dtype=flat.dtype)
        spec = torch.stft(
            flat,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        mag = spec.abs().reshape(bsz, n_chan, spec.shape[-2], spec.shape[-1])
        return torch.log(mag + 1.0e-7), mag

    def _vocal_activity_weight(self, ref_mag: torch.Tensor) -> torch.Tensor:
        # ref_mag: [B, C, F, T].  Return [B, 1, 1, T].
        if self.vocal_active_frame_weight <= 0.0:
            return ref_mag.new_ones(ref_mag.shape[0], 1, 1, ref_mag.shape[-1])
        frame_energy = ref_mag.float().mean(dim=(1, 2), keepdim=True)  # [B, 1, 1, T]
        max_energy = frame_energy.amax(dim=-1, keepdim=True).clamp_min(1.0e-8)
        activity = (frame_energy / max_energy).clamp(0.0, 1.0)  # [B, 1, 1, T]
        return 1.0 + self.vocal_active_frame_weight * activity

    def _speech_robust_logmag_losses(
        self, est_speech: torch.Tensor, ref_speech: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total = est_speech.new_zeros(())
        components: dict[str, torch.Tensor] = {}
        logmag_losses = []
        temporal_losses = []
        frequency_losses = []

        for n_fft, hop_length in self.speech_robust_logmag_resolutions:
            est_log, _ = self._stft_logmag(est_speech, n_fft=n_fft, hop_length=hop_length)
            ref_log, ref_mag = self._stft_logmag(ref_speech, n_fft=n_fft, hop_length=hop_length)
            frame_weight = self._vocal_activity_weight(ref_mag).to(dtype=est_log.dtype)

            if self.speech_robust_logmag_weight > 0.0:
                err = (est_log - ref_log).abs()
                robust_error = self._soft_truncated_l1(err, self.speech_robust_logmag_tau)
                logmag_losses.append((robust_error * frame_weight).mean())

            if self.speech_temporal_logmag_gradient_weight > 0.0 and est_log.shape[-1] > 1:
                est_dt = est_log[..., 1:] - est_log[..., :-1]
                ref_dt = ref_log[..., 1:] - ref_log[..., :-1]
                dt_weight = frame_weight[..., 1:]
                err = (est_dt - ref_dt).abs()
                temporal_losses.append((self._soft_truncated_l1(err, self.speech_gradient_tau) * dt_weight).mean())

            if self.speech_frequency_logmag_gradient_weight > 0.0 and est_log.shape[-2] > 1:
                est_df = est_log[..., 1:, :] - est_log[..., :-1, :]
                ref_df = ref_log[..., 1:, :] - ref_log[..., :-1, :]
                err = (est_df - ref_df).abs()
                frequency_losses.append((self._soft_truncated_l1(err, self.speech_gradient_tau) * frame_weight).mean())

        if logmag_losses:
            value = torch.stack(logmag_losses).mean()
            components["speech_robust_logmag"] = value
            total = total + self.speech_robust_logmag_weight * value
        if temporal_losses:
            value = torch.stack(temporal_losses).mean()
            components["speech_temporal_logmag_gradient"] = value
            total = total + self.speech_temporal_logmag_gradient_weight * value
        if frequency_losses:
            value = torch.stack(frequency_losses).mean()
            components["speech_frequency_logmag_gradient"] = value
            total = total + self.speech_frequency_logmag_gradient_weight * value
        return total, components

    def _speech_inactive_leakage_loss(self, est_speech: torch.Tensor, ref_speech: torch.Tensor) -> torch.Tensor:
        n_fft = self.stft[0].n_fft
        hop_length = self.stft[0].hop_length
        _, est_mag = self._stft_logmag(est_speech, n_fft=n_fft, hop_length=hop_length)
        _, ref_mag = self._stft_logmag(ref_speech, n_fft=n_fft, hop_length=hop_length)
        ref_frame = ref_mag.float().mean(dim=(1, 2), keepdim=True)  # [B, 1, 1, T]
        ref_db = 20.0 * torch.log10(ref_frame.clamp_min(1.0e-8))
        inactive = torch.sigmoid(
            (self.speech_inactive_threshold_db - ref_db) / max(self.speech_inactive_softness_db, 1.0e-6)
        ).to(dtype=est_mag.dtype)  # [B, 1, 1, T]
        return (est_mag * inactive).mean()

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

        if self.vocal_losses_enabled:
            est_speech = est[:, self.speech_source_index]
            ref_speech = ref[:, self.speech_source_index]
            vocal_loss, vocal_components = self._speech_robust_logmag_losses(est_speech, ref_speech)
            if self.speech_inactive_leakage_weight > 0.0:
                leakage_loss = self._speech_inactive_leakage_loss(est_speech, ref_speech)
                vocal_loss = vocal_loss + self.speech_inactive_leakage_weight * leakage_loss
                vocal_components["speech_inactive_leakage"] = leakage_loss
            loss = loss + vocal_loss
            log_dict[f"{log_prefix}/loss_vocal_aware"] = vocal_loss
            for name, value in vocal_components.items():
                log_dict[f"{log_prefix}/loss_{name}"] = value

        log_dict[f"{log_prefix}/loss"] = loss
        if log_prefix == "validation":
            snr_score = self.snr(est.transpose(1, 2), ref.transpose(1, 2)).mean()
            log_dict[f"{log_prefix}/snr"] = snr_score

        self.log_dict(log_dict, prog_bar=False, on_epoch=True, on_step=False, batch_size=wav.shape[0], sync_dist=True)
        return loss
