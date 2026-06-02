"""Teacher-student training task for edge source-separation students."""

from __future__ import annotations

from typing import Any

from collections.abc import Mapping, Sequence
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


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _split_model_output(output: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    if isinstance(output, torch.Tensor):
        return output, {}

    if isinstance(output, Mapping):
        aux = dict(output)
        est = None
        for key in ("estimate", "est", "waveform", "wav", "sources", "output", "y"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                est = value
                aux.pop(key, None)
                break
        if est is None:
            est = _first_tensor(output)
        nested_aux = aux.pop("aux", None)
        if isinstance(nested_aux, Mapping):
            aux.update(nested_aux)
        if est is None:
            raise TypeError("Could not find tensor estimate in mapping model output")
        return est, aux

    if isinstance(output, (tuple, list)) and output:
        est = _first_tensor(output[0])
        if est is None:
            raise TypeError("First tuple/list model output does not contain a tensor estimate")
        aux: dict[str, Any] = {}
        for idx, item in enumerate(output[1:], start=1):
            if isinstance(item, Mapping):
                aux.update(item)
            else:
                aux[f"aux_{idx}"] = item
        return est, aux

    raise TypeError(f"Unsupported model output type: {type(output).__name__}")


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
        source_activity_loss_weight: float = 0.0,
        source_activity_db: float = -50.0,
        source_activity_active_weight: float = 1.0,
        source_activity_inactive_weight: float = 0.25,
        complex_ri_weight: float = 0.0,
        log_magnitude_weight: float = 0.0,
        multi_resolution_stft_weight: float = 0.0,
        multi_resolution_stft_resolutions: Sequence[Sequence[int]] | None = None,
        transient_weight: float = 0.0,
        teacher_mask_loss_weight: float = 0.0,
        teacher_logit_loss_weight: float = 0.0,
        mask_loss_eps: float = 1.0e-4,
        mask_loss_max: float = 4.0,
        latent_distillation_weight: float = 0.0,
        latent_distillation_loss: str = "l2",
        student_latent_modules: Sequence[str] | None = None,
        teacher_latent_modules: Sequence[str] | None = None,
        latent_allow_missing: bool = False,
        latent_allow_shape_mismatch: bool = False,
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
        teacher_required_weight = teacher_loss_weight + teacher_mask_loss_weight + teacher_logit_loss_weight
        teacher_required_weight += latent_distillation_weight
        if teacher_required_weight > 0.0 and teacher_model is None:
            raise ValueError("teacher distillation weights require teacher_model")
        self.teacher_metric_enabled = teacher_model is not None and (
            teacher_checkpoint_path is not None or teacher_required_weight > 0.0
        )
        if latent_distillation_loss not in {"l1", "l2"}:
            raise ValueError("latent_distillation_loss must be 'l1' or 'l2'")

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
        self.source_activity_loss_weight = source_activity_loss_weight
        self.source_activity_db = source_activity_db
        self.source_activity_active_weight = source_activity_active_weight
        self.source_activity_inactive_weight = source_activity_inactive_weight
        self.teacher_mask_loss_weight = teacher_mask_loss_weight
        self.teacher_logit_loss_weight = teacher_logit_loss_weight
        self.mask_loss_eps = mask_loss_eps
        self.mask_loss_max = mask_loss_max
        self.latent_distillation_weight = latent_distillation_weight
        self.latent_distillation_loss = latent_distillation_loss
        self.student_latent_module_names = tuple(student_latent_modules or ())
        self.teacher_latent_module_names = tuple(teacher_latent_modules or self.student_latent_module_names)
        self.latent_allow_missing = latent_allow_missing
        self.latent_allow_shape_mismatch = latent_allow_shape_mismatch
        self.teacher_css_validation = teacher_css_validation
        self._student_latents: dict[str, torch.Tensor] = {}
        self._teacher_latents: dict[str, torch.Tensor] = {}
        self._latent_hook_handles: list[Any] = []
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

        self._register_latent_hooks()

    def _register_latent_hooks(self) -> None:
        if self.latent_distillation_weight <= 0.0:
            return
        if not self.student_latent_module_names:
            return
        self._latent_hook_handles.extend(
            self._register_hooks_for(self.model, self.student_latent_module_names, self._student_latents, "student")
        )
        if self.teacher_model is not None:
            self._latent_hook_handles.extend(
                self._register_hooks_for(
                    self.teacher_model,
                    self.teacher_latent_module_names,
                    self._teacher_latents,
                    "teacher",
                )
            )

    def _register_hooks_for(
        self,
        root: nn.Module,
        module_names: Sequence[str],
        store: dict[str, torch.Tensor],
        role: str,
    ) -> list[Any]:
        modules = dict(root.named_modules())
        handles = []
        for name in module_names:
            if name not in modules:
                if self.latent_allow_missing:
                    continue
                raise ValueError(f"{role} latent module not found: {name}")

            def hook(_module, _inputs, output, *, capture_name=name):
                tensor = _first_tensor(output)
                if tensor is not None:
                    store[capture_name] = tensor

            handles.append(modules[name].register_forward_hook(hook))
        return handles

    def _forward_model(
        self,
        model: nn.Module,
        wav: torch.Tensor,
        ref: torch.Tensor | None,
        log_prefix: str,
        *,
        css_validation: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        output = model.css(wav, ref=ref) if log_prefix != "training" and css_validation else model(wav)
        return _split_model_output(output)

    def _teacher_forward(
        self,
        wav: torch.Tensor,
        ref: torch.Tensor | None,
        log_prefix: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.teacher_model is None:
            raise RuntimeError("teacher_model is not configured")
        self.teacher_model.eval()
        self._teacher_latents.clear()
        with torch.no_grad():
            est, aux = self._forward_model(
                self.teacher_model,
                wav,
                ref,
                log_prefix,
                css_validation=self.teacher_css_validation,
            )
        if self._teacher_latents:
            aux = dict(aux)
            aux["latents"] = dict(self._teacher_latents)
        return est, aux

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

    def _source_activity_l1(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        per_source_l1 = (est - ref).abs().mean(dim=(-1, -2))
        ref_power = ref.float().square().mean(dim=(-1, -2))
        active = ref_power > 10 ** (self.source_activity_db / 10.0)
        active_weight = per_source_l1.new_tensor(self.source_activity_active_weight)
        inactive_weight = per_source_l1.new_tensor(self.source_activity_inactive_weight)
        weights = torch.where(active, active_weight, inactive_weight)
        return (per_source_l1 * weights).sum() / weights.sum().clamp_min(1.0)

    def _spectral_mask(self, est: torch.Tensor, wav: torch.Tensor) -> torch.Tensor:
        est_spec = self.stft(est.float()).abs()
        mix_spec = self.stft(wav.float()).abs().unsqueeze(1)
        mask = est_spec / mix_spec.clamp_min(self.mask_loss_eps)
        return mask.clamp(0.0, self.mask_loss_max)

    def _aux_tensor(self, aux: Mapping[str, Any], keys: Sequence[str]) -> torch.Tensor | None:
        for key in keys:
            value = aux.get(key)
            tensor = _first_tensor(value)
            if tensor is not None:
                return tensor
        return None

    def _mask_or_logit_loss(
        self,
        est: torch.Tensor,
        teacher_est: torch.Tensor,
        wav: torch.Tensor,
        student_aux: Mapping[str, Any],
        teacher_aux: Mapping[str, Any],
        *,
        logit: bool,
    ) -> torch.Tensor:
        if logit:
            student_value = self._aux_tensor(student_aux, ("mask_logits", "logits"))
            teacher_value = self._aux_tensor(teacher_aux, ("mask_logits", "logits"))
            if student_value is None or teacher_value is None:
                student_mask = self._spectral_mask(est, wav).clamp(self.mask_loss_eps, 1.0 - self.mask_loss_eps)
                teacher_mask = self._spectral_mask(teacher_est, wav).clamp(self.mask_loss_eps, 1.0 - self.mask_loss_eps)
                student_value = torch.logit(student_mask)
                teacher_value = torch.logit(teacher_mask)
        else:
            student_value = self._aux_tensor(student_aux, ("mask", "masks"))
            teacher_value = self._aux_tensor(teacher_aux, ("mask", "masks"))
            if student_value is None or teacher_value is None:
                student_value = self._spectral_mask(est, wav)
                teacher_value = self._spectral_mask(teacher_est, wav)
        if student_value.shape != teacher_value.shape:
            raise ValueError(f"mask/logit distillation shape mismatch: {student_value.shape} vs {teacher_value.shape}")
        return F.l1_loss(student_value.float(), teacher_value.detach().float())

    def _latent_dict(self, aux: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        raw = aux.get("latents")
        if isinstance(raw, Mapping):
            return {str(key): value for key, value in raw.items() if isinstance(value, torch.Tensor)}
        out = {}
        for key in ("latent", "features", "feature"):
            tensor = _first_tensor(aux.get(key))
            if tensor is not None:
                out[key] = tensor
        return out

    def _latent_distillation_loss(self, student_aux: Mapping[str, Any], teacher_aux: Mapping[str, Any]) -> torch.Tensor:
        student_latents = self._latent_dict(student_aux)
        teacher_latents = self._latent_dict(teacher_aux)
        if self.student_latent_module_names:
            pairs = tuple(zip(self.student_latent_module_names, self.teacher_latent_module_names, strict=False))
        else:
            common = sorted(set(student_latents) & set(teacher_latents))
            pairs = tuple((name, name) for name in common)

        losses = []
        for student_name, teacher_name in pairs:
            student_value = student_latents.get(student_name)
            teacher_value = teacher_latents.get(teacher_name)
            if student_value is None or teacher_value is None:
                if self.latent_allow_missing:
                    continue
                raise ValueError(f"Missing latent pair: student={student_name}, teacher={teacher_name}")
            if student_value.shape != teacher_value.shape:
                if self.latent_allow_shape_mismatch:
                    continue
                raise ValueError(
                    f"Latent shape mismatch for {student_name}/{teacher_name}: "
                    f"{student_value.shape} vs {teacher_value.shape}"
                )
            if self.latent_distillation_loss == "l1":
                losses.append(F.l1_loss(student_value.float(), teacher_value.detach().float()))
            else:
                losses.append(F.mse_loss(student_value.float(), teacher_value.detach().float()))

        if not losses:
            if self.latent_allow_missing:
                return next(self.model.parameters()).new_zeros(())
            raise ValueError("latent_distillation_weight is enabled but no latent pairs were available")
        return torch.stack(losses).mean()

    def _step(self, wav: torch.Tensor, ref: torch.Tensor, log_prefix: str):
        model = self.ema_model.module if self.use_ema_model and log_prefix != "training" else self.model
        self._student_latents.clear()
        est, student_aux = self._forward_model(model, wav, ref, log_prefix, css_validation=self.css_validation)
        if self._student_latents:
            student_aux = dict(student_aux)
            student_aux["latents"] = dict(self._student_latents)

        teacher_est: torch.Tensor | None = None
        teacher_aux: dict[str, Any] = {}

        def get_teacher() -> tuple[torch.Tensor, dict[str, Any]]:
            nonlocal teacher_est, teacher_aux
            if teacher_est is None:
                teacher_est, teacher_aux = self._teacher_forward(wav, ref=ref, log_prefix=log_prefix)
            return teacher_est, teacher_aux

        supervised_loss = self.loss(est.transpose(1, 2), ref.transpose(1, 2)).mean()
        loss = supervised_loss
        log_dict = {
            "step": float(self.trainer.current_epoch),
            f"{log_prefix}/loss_supervised": supervised_loss,
        }

        if self.teacher_loss_weight > 0.0:
            teacher_est_value, _ = get_teacher()
            teacher_loss = F.l1_loss(est, teacher_est_value.detach())
            loss = loss + self.teacher_loss_weight * teacher_loss
            log_dict[f"{log_prefix}/loss_teacher"] = teacher_loss

        if self.teacher_mask_loss_weight > 0.0:
            teacher_est_value, teacher_aux_value = get_teacher()
            mask_loss = self._mask_or_logit_loss(
                est,
                teacher_est_value,
                wav,
                student_aux,
                teacher_aux_value,
                logit=False,
            )
            loss = loss + self.teacher_mask_loss_weight * mask_loss
            log_dict[f"{log_prefix}/loss_teacher_mask"] = mask_loss

        if self.teacher_logit_loss_weight > 0.0:
            teacher_est_value, teacher_aux_value = get_teacher()
            logit_loss = self._mask_or_logit_loss(
                est,
                teacher_est_value,
                wav,
                student_aux,
                teacher_aux_value,
                logit=True,
            )
            loss = loss + self.teacher_logit_loss_weight * logit_loss
            log_dict[f"{log_prefix}/loss_teacher_logit"] = logit_loss

        if self.latent_distillation_weight > 0.0:
            _, teacher_aux_value = get_teacher()
            latent_loss = self._latent_distillation_loss(student_aux, teacher_aux_value)
            loss = loss + self.latent_distillation_weight * latent_loss
            log_dict[f"{log_prefix}/loss_latent_distillation"] = latent_loss

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

        if self.source_activity_loss_weight > 0.0:
            activity_loss = self._source_activity_l1(est, ref)
            loss = loss + self.source_activity_loss_weight * activity_loss
            log_dict[f"{log_prefix}/loss_source_activity"] = activity_loss

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
            if self.teacher_metric_enabled:
                teacher_est_value, _ = get_teacher()
                teacher_snr_score = self.snr(teacher_est_value.transpose(1, 2), ref.transpose(1, 2)).mean()
                student_teacher_snr_score = self.snr(
                    est.transpose(1, 2),
                    teacher_est_value.detach().transpose(1, 2),
                ).mean()
                log_dict[f"{log_prefix}/teacher_snr"] = teacher_snr_score
                log_dict[f"{log_prefix}/student_teacher_snr"] = student_teacher_snr_score

        self.log_dict(log_dict, prog_bar=False, on_epoch=True, on_step=False, batch_size=wav.shape[0], sync_dist=True)
        return loss
