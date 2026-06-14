"""Teacher-student training task for edge source-separation students."""

from __future__ import annotations

from typing import Any, cast

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
        _strip_prefix(state_dict, "ema_model.module."),
        _strip_prefix(state_dict, "model."),
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


def _find_band_spec(root: nn.Module | None) -> nn.Module | None:
    if root is None:
        return None
    for module in root.modules():
        centers = getattr(module, "centers_hz", None)
        n_bands = getattr(module, "n_bands", None)
        if not isinstance(centers, torch.Tensor) or n_bands is None:
            continue
        centers_tensor = cast(torch.Tensor, centers)
        if int(centers_tensor.numel()) == int(n_bands):
            return module
    return None


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
        distillation_band_mapping: str | None = "none",
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
        band_mapping = "none" if distillation_band_mapping is None else str(distillation_band_mapping).lower()
        band_mapping_aliases = {
            "off": "none",
            "false": "none",
            "disabled": "none",
            "mel": "mel_centers",
            "mel_center": "mel_centers",
            "center": "mel_centers",
            "centers": "mel_centers",
        }
        band_mapping = band_mapping_aliases.get(band_mapping, band_mapping)
        if band_mapping not in {"none", "linear", "mel_centers", "auto"}:
            raise ValueError("distillation_band_mapping must be one of 'none', 'linear', 'mel_centers', or 'auto'")

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
        self.distillation_band_mapping = band_mapping
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

        self._student_band_spec = _find_band_spec(self.model) if self.distillation_band_mapping != "none" else None
        self._teacher_band_spec = (
            _find_band_spec(self.teacher_model) if self.distillation_band_mapping != "none" else None
        )

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

    def _linear_map_last_dim(self, value: torch.Tensor, target_size: int) -> torch.Tensor:
        source_size = int(value.shape[-1])
        target_size = int(target_size)
        if source_size == target_size:
            return value
        if source_size <= 0 or target_size <= 0:
            raise ValueError(f"Cannot map empty band axis: {source_size} -> {target_size}")
        flat = value.reshape(-1, 1, source_size).float()
        if source_size == 1:
            mapped = flat.expand(-1, -1, target_size)
        else:
            mapped = F.interpolate(flat, size=target_size, mode="linear", align_corners=True)
        return mapped.reshape(*value.shape[:-1], target_size).to(dtype=value.dtype)

    def _center_band_mapping(self, source_size: int, target_size: int, ref: torch.Tensor) -> torch.Tensor | None:
        if self._teacher_band_spec is None or self._student_band_spec is None:
            if self.distillation_band_mapping == "auto":
                return None
            raise ValueError("mel-center band mapping requires band_spec modules on both student and teacher")

        teacher_centers = getattr(self._teacher_band_spec, "centers_hz", None)
        student_centers = getattr(self._student_band_spec, "centers_hz", None)
        if not isinstance(teacher_centers, torch.Tensor):
            if self.distillation_band_mapping == "auto":
                return None
            raise ValueError("mel-center band mapping requires a centers_hz buffer on the teacher band spec")
        if not isinstance(student_centers, torch.Tensor):
            if self.distillation_band_mapping == "auto":
                return None
            raise ValueError("mel-center band mapping requires a centers_hz buffer on the student band spec")
        teacher_centers_tensor = cast(torch.Tensor, teacher_centers)
        student_centers_tensor = cast(torch.Tensor, student_centers)
        if int(teacher_centers_tensor.numel()) != int(source_size) or int(student_centers_tensor.numel()) != int(
            target_size
        ):
            if self.distillation_band_mapping == "auto":
                return None
            raise ValueError(
                "Band tensor sizes do not match discovered band specs: "
                f"teacher tensor K={source_size}, teacher spec K={int(teacher_centers_tensor.numel())}; "
                f"student tensor K={target_size}, student spec K={int(student_centers_tensor.numel())}"
            )

        source = teacher_centers_tensor.to(device=ref.device, dtype=torch.float32).flatten()
        target = student_centers_tensor.to(device=ref.device, dtype=torch.float32).flatten()
        if source.numel() == 1:
            return torch.ones(target_size, source_size, device=ref.device, dtype=torch.float32)

        right = torch.searchsorted(source.contiguous(), target.contiguous()).clamp(max=source.numel() - 1)
        left = (right - 1).clamp(min=0)
        left = torch.where(target <= source[0], torch.zeros_like(left), left)
        right = torch.where(target <= source[0], torch.zeros_like(right), right)
        left = torch.where(target >= source[-1], torch.full_like(left, source.numel() - 1), left)
        right = torch.where(target >= source[-1], torch.full_like(right, source.numel() - 1), right)

        left_hz = source[left]
        right_hz = source[right]
        denom = (right_hz - left_hz).clamp_min(1.0e-6)
        right_weight = torch.where(left == right, torch.zeros_like(target), (target - left_hz) / denom)
        right_weight = right_weight.clamp(0.0, 1.0)
        left_weight = 1.0 - right_weight

        mapping = torch.zeros(target_size, source_size, device=ref.device, dtype=torch.float32)
        mapping.scatter_add_(1, left.unsqueeze(1), left_weight.unsqueeze(1))
        mapping.scatter_add_(1, right.unsqueeze(1), right_weight.unsqueeze(1))
        mapping = mapping / mapping.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return mapping

    def _map_teacher_to_student_bands(self, teacher_value: torch.Tensor, target_size: int) -> torch.Tensor:
        source_size = int(teacher_value.shape[-1])
        target_size = int(target_size)
        if source_size == target_size:
            return teacher_value
        if self.distillation_band_mapping == "none":
            raise ValueError(
                "Band-domain distillation shape mismatch requires distillation_band_mapping; "
                f"got teacher K={source_size}, student K={target_size}"
            )
        if self.distillation_band_mapping in {"mel_centers", "auto"}:
            mapping = self._center_band_mapping(source_size, target_size, teacher_value)
            if mapping is not None:
                mapped = torch.matmul(teacher_value.float(), mapping.transpose(0, 1))
                return mapped.to(dtype=teacher_value.dtype)
        return self._linear_map_last_dim(teacher_value, target_size)

    def _align_distillation_tensors(
        self,
        student_value: torch.Tensor,
        teacher_value: torch.Tensor,
        *,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if student_value.shape == teacher_value.shape:
            return student_value, teacher_value
        if student_value.ndim != teacher_value.ndim:
            raise ValueError(f"{name} distillation rank mismatch: {student_value.shape} vs {teacher_value.shape}")
        if student_value.shape[:-1] == teacher_value.shape[:-1]:
            teacher_value = self._map_teacher_to_student_bands(teacher_value, int(student_value.shape[-1]))
            if student_value.shape == teacher_value.shape:
                return student_value, teacher_value
        raise ValueError(f"{name} distillation shape mismatch: {student_value.shape} vs {teacher_value.shape}")

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
                # Most waveform separators in this repo return estimates only. In
                # that case, logit distillation intentionally falls back to
                # waveform-derived spectral pseudo-masks instead of internal mask
                # logits.
                student_mask = self._spectral_mask(est, wav).clamp(self.mask_loss_eps, 1.0 - self.mask_loss_eps)
                teacher_mask = self._spectral_mask(teacher_est, wav).clamp(self.mask_loss_eps, 1.0 - self.mask_loss_eps)
                student_value = torch.logit(student_mask)
                teacher_value = torch.logit(teacher_mask)
        else:
            student_value = self._aux_tensor(student_aux, ("mask", "masks"))
            teacher_value = self._aux_tensor(teacher_aux, ("mask", "masks"))
            if student_value is None or teacher_value is None:
                # Match spectral pseudo-masks when true model masks are not
                # exposed by the model output contract.
                student_value = self._spectral_mask(est, wav)
                teacher_value = self._spectral_mask(teacher_est, wav)
        student_value, teacher_value = self._align_distillation_tensors(
            student_value,
            teacher_value,
            name="mask/logit",
        )
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
            try:
                student_value, teacher_value = self._align_distillation_tensors(
                    student_value,
                    teacher_value,
                    name=f"latent {student_name}/{teacher_name}",
                )
            except ValueError:
                if self.latent_allow_shape_mismatch:
                    continue
                raise
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
