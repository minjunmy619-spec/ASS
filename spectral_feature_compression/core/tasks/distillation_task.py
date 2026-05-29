"""Teacher-student training task for edge source-separation students."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

_LOCAL_AIACCEL = Path(__file__).resolve().parents[3] / "aiaccel"
if _LOCAL_AIACCEL.is_dir() and str(_LOCAL_AIACCEL) not in sys.path:
    sys.path.insert(0, str(_LOCAL_AIACCEL))

from aiaccel.torch.lightning import OptimizerConfig  # noqa: E402

from spectral_feature_compression.core.loss.composite_separation import CompositeSeparationSpectralLoss  # noqa: E402
from spectral_feature_compression.core.tasks.sup_task import SupTask  # noqa: E402


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}


def _load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    candidate_state_dicts = [
        _strip_prefix(state_dict, "model."),
        _strip_prefix(state_dict, "ema_model.module."),
        state_dict,
    ]
    last_error: RuntimeError | None = None
    for candidate in candidate_state_dicts:
        if not candidate:
            continue
        candidate = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in candidate.items()
            if key != "n_averaged"
        }
        try:
            model.load_state_dict(candidate, strict=strict)
            return
        except RuntimeError as exc:
            last_error = exc
    if last_error is None:
        raise RuntimeError(f"No loadable state dict entries found in {checkpoint_path}")
    raise last_error


class TeacherStudentDistillationTask(SupTask):
    """Supervised separation task with opt-in teacher distillation terms.

    The base supervised loss remains the primary objective.  Distillation,
    mixture consistency, low-frequency preservation, and silent-source penalties
    are additive and controlled by explicit weights.
    """

    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        n_fft: int,
        hop_length: int,
        optimizer_config: OptimizerConfig,
        *,
        fs: int | None = None,
        teacher_model: nn.Module | None = None,
        teacher_checkpoint_path: str | None = None,
        require_teacher_checkpoint: bool = False,
        teacher_strict: bool = True,
        teacher_loss_weight: float = 0.0,
        mixture_consistency_weight: float = 0.0,
        low_frequency_weight: float = 0.0,
        low_frequency_hz: float = 300.0,
        silent_source_weight: float = 0.0,
        silent_source_db: float = -60.0,
        complex_ri_weight: float = 0.0,
        log_magnitude_weight: float = 0.0,
        multi_resolution_stft_weight: float = 0.0,
        multi_resolution_stft_resolutions: Sequence[Sequence[int]] | None = None,
        transient_weight: float = 0.0,
        pretrained_model_path: str | None = None,
        css_validation: bool = False,
        teacher_css_validation: bool = False,
        ema_weight: float | None = None,
        ema_update_freq: int | None = None,
    ):
        super().__init__(
            model=model,
            loss=loss,
            n_fft=n_fft,
            hop_length=hop_length,
            optimizer_config=optimizer_config,
            pretrained_model_path=pretrained_model_path,
            css_validation=css_validation,
            ema_weight=ema_weight,
            ema_update_freq=ema_update_freq,
        )
        if require_teacher_checkpoint and teacher_checkpoint_path is None:
            raise ValueError("require_teacher_checkpoint=True but teacher_checkpoint_path is not set")
        if teacher_loss_weight > 0.0 and teacher_model is None:
            raise ValueError("teacher_loss_weight requires teacher_model")

        self.fs = fs
        self.n_fft = n_fft
        self.teacher_model = teacher_model
        self.teacher_checkpoint_path = teacher_checkpoint_path
        self.teacher_loss_weight = teacher_loss_weight
        self.mixture_consistency_weight = mixture_consistency_weight
        self.low_frequency_weight = low_frequency_weight
        self.low_frequency_hz = low_frequency_hz
        self.silent_source_weight = silent_source_weight
        self.silent_source_db = silent_source_db
        self.teacher_css_validation = teacher_css_validation
        self.composite_loss = CompositeSeparationSpectralLoss(
            n_fft=n_fft,
            hop_length=hop_length,
            complex_ri_weight=complex_ri_weight,
            log_magnitude_weight=log_magnitude_weight,
            multi_resolution_stft_weight=multi_resolution_stft_weight,
            multi_resolution_stft_resolutions=multi_resolution_stft_resolutions,
            transient_weight=transient_weight,
        )

        if self.teacher_model is not None:
            if teacher_checkpoint_path is not None:
                _load_model_checkpoint(self.teacher_model, teacher_checkpoint_path, strict=teacher_strict)
            self.teacher_model.eval()
            for parameter in self.teacher_model.parameters():
                parameter.requires_grad_(False)

    def _teacher_forward(self, wav: torch.Tensor, ref: torch.Tensor | None, log_prefix: str) -> torch.Tensor:
        if self.teacher_model is None:
            raise RuntimeError("teacher_model is not configured")
        self.teacher_model.eval()
        with torch.no_grad():
            if log_prefix != "training" and self.teacher_css_validation:
                return self.teacher_model.css(wav, ref=ref)
            return self.teacher_model(wav)

    def _low_frequency_l1(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if self.fs is None:
            raise ValueError("low_frequency_weight requires fs so the cutoff can be mapped to STFT bins")
        est_spec = self.stft(est.float()).abs()
        ref_spec = self.stft(ref.float()).abs()
        max_bin = int(self.low_frequency_hz * self.n_fft / self.fs) + 1
        max_bin = max(1, min(max_bin, est_spec.shape[-2]))
        return F.l1_loss(est_spec[..., :max_bin, :], ref_spec[..., :max_bin, :])

    def _silent_source_penalty(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        ref_power = ref.float().square().mean(dim=(-1, -2))
        est_power = est.float().square().mean(dim=(-1, -2))
        inactive = ref_power <= 10 ** (self.silent_source_db / 10.0)
        if not inactive.any():
            return est.new_zeros(())
        return est_power[inactive].mean()

    def _step(self, wav: torch.Tensor, ref: torch.Tensor, log_prefix: str):
        model = self.ema_model.module if self.use_ema_model and log_prefix != "training" else self.model
        est = model.css(wav, ref=ref) if log_prefix != "training" and self.css_validation else model(wav)

        supervised_loss = self.loss(est.transpose(1, 2), ref.transpose(1, 2)).mean()
        loss = supervised_loss
        log_dict = {
            "step": float(self.trainer.current_epoch),
            f"{log_prefix}/loss_supervised": supervised_loss,
        }

        if self.teacher_loss_weight > 0.0:
            teacher_est = self._teacher_forward(wav, ref=ref, log_prefix=log_prefix)
            teacher_loss = F.l1_loss(est, teacher_est.detach())
            loss = loss + self.teacher_loss_weight * teacher_loss
            log_dict[f"{log_prefix}/loss_teacher"] = teacher_loss

        if self.mixture_consistency_weight > 0.0:
            mixture_loss = F.l1_loss(est.sum(dim=1), wav)
            loss = loss + self.mixture_consistency_weight * mixture_loss
            log_dict[f"{log_prefix}/loss_mixture_consistency"] = mixture_loss

        if self.low_frequency_weight > 0.0:
            low_frequency_loss = self._low_frequency_l1(est, ref)
            loss = loss + self.low_frequency_weight * low_frequency_loss
            log_dict[f"{log_prefix}/loss_low_frequency"] = low_frequency_loss

        if self.silent_source_weight > 0.0:
            silent_loss = self._silent_source_penalty(est, ref)
            loss = loss + self.silent_source_weight * silent_loss
            log_dict[f"{log_prefix}/loss_silent_source"] = silent_loss

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
