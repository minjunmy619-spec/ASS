"""Teacher-student training task for edge source-separation students."""

from __future__ import annotations

from typing import Any, cast

from collections.abc import Mapping, Sequence
import math
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


def _is_unexpected_keyword_type_error(exc: TypeError, keyword: str) -> bool:
    message = str(exc)
    patterns = (
        f"unexpected keyword argument '{keyword}'",
        f'unexpected keyword argument "{keyword}"',
        f"got an unexpected keyword argument '{keyword}'",
        f'got an unexpected keyword argument "{keyword}"',
    )
    return any(pattern in message for pattern in patterns)


def _find_band_spec(root: nn.Module | None) -> nn.Module | None:
    if root is None:
        return None
    for module in root.modules():
        n_bands = getattr(module, "n_bands", None)
        if n_bands is None:
            continue
        basis = _band_spec_basis(module)
        if basis is not None and int(basis.shape[0]) == int(n_bands):
            return module
        for center_name in ("centers_hz", "centers"):
            centers = getattr(module, center_name, None)
            if not isinstance(centers, torch.Tensor):
                continue
            centers_tensor = cast(torch.Tensor, centers)
            if int(centers_tensor.numel()) == int(n_bands):
                return module
    return None


def _band_spec_basis(module: nn.Module | None) -> torch.Tensor | None:
    if module is None:
        return None
    basis = getattr(module, "basis", None)
    if not isinstance(basis, torch.Tensor):
        return None
    basis_tensor = cast(torch.Tensor, basis)
    if basis_tensor.ndim == 4 and int(basis_tensor.shape[0]) == 1 and int(basis_tensor.shape[2]) == 1:
        return basis_tensor[0, :, 0, :]
    if basis_tensor.ndim == 2:
        return basis_tensor
    return None


def _find_frequency_projectors(root: nn.Module | None) -> tuple[nn.Module, ...]:
    if root is None:
        return ()
    projectors = []
    for module in root.modules():
        analysis = getattr(module, "analysis_matrix", None)
        synthesis = getattr(module, "synthesis_matrix", None)
        if isinstance(analysis, torch.Tensor) or isinstance(synthesis, torch.Tensor):
            projectors.append(module)
    return tuple(projectors)


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
        frame_silent_source_weight: float = 0.0,
        frame_silent_source_db: float = -50.0,
        frame_silent_window_ms: float = 80.0,
        frame_silent_hop_ms: float = 40.0,
        clap_semantic_loss: nn.Module | None = None,
        clap_semantic_loss_weight: float = 0.0,
        whisper_feature_loss: nn.Module | None = None,
        whisper_feature_loss_weight: float = 0.0,
        whisper_source: str = "speech",
        perceptual_loss_start_step: int = 0,
        perceptual_loss_every_n_steps: int = 1,
        perceptual_loss_compensate_cadence: bool = False,
        source_activity_loss_weight: float = 0.0,
        source_activity_db: float = -50.0,
        source_activity_active_weight: float = 1.0,
        source_activity_inactive_weight: float = 0.25,
        source_order: Sequence[str] | None = None,
        source_loss_weights: Sequence[float] | Mapping[str, float] | None = None,
        source_loss_weight_normalization: str = "subset_mean",
        source_weighted_snr_loss_weight: float = 0.0,
        explicit_source_loss_weight: float = 0.0,
        residual_source_loss_weight: float = 0.0,
        residual_source_index: int | None = None,
        robust_label_loss_weight: float = 0.0,
        robust_label_loss: str = "charbonnier",
        robust_label_eps: float = 1.0e-3,
        complex_ri_weight: float = 0.0,
        log_magnitude_weight: float = 0.0,
        multi_resolution_stft_weight: float = 0.0,
        multi_resolution_stft_resolutions: Sequence[Sequence[int]] | None = None,
        transient_weight: float = 0.0,
        teacher_mask_loss_weight: float = 0.0,
        teacher_logit_loss_weight: float = 0.0,
        request_model_aux: bool = False,
        require_model_aux: bool = False,
        mask_aux_alignment: str = "strict",
        mask_aux_max_frame_mismatch: int = 0,
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
        for name, value in {
            "teacher_loss_weight": teacher_loss_weight,
            "mixture_consistency_weight": mixture_consistency_weight,
            "low_frequency_weight": low_frequency_weight,
            "silent_source_weight": silent_source_weight,
            "frame_silent_source_weight": frame_silent_source_weight,
            "clap_semantic_loss_weight": clap_semantic_loss_weight,
            "whisper_feature_loss_weight": whisper_feature_loss_weight,
            "source_activity_loss_weight": source_activity_loss_weight,
            "robust_label_loss_weight": robust_label_loss_weight,
            "complex_ri_weight": complex_ri_weight,
            "log_magnitude_weight": log_magnitude_weight,
            "multi_resolution_stft_weight": multi_resolution_stft_weight,
            "transient_weight": transient_weight,
            "teacher_mask_loss_weight": teacher_mask_loss_weight,
            "teacher_logit_loss_weight": teacher_logit_loss_weight,
            "latent_distillation_weight": latent_distillation_weight,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if frame_silent_source_weight > 0.0 and fs is None:
            raise ValueError("frame_silent_source_weight requires fs")
        if frame_silent_window_ms <= 0.0 or frame_silent_hop_ms <= 0.0:
            raise ValueError("frame_silent_window_ms and frame_silent_hop_ms must be positive")
        if clap_semantic_loss_weight > 0.0 and clap_semantic_loss is None:
            raise ValueError("clap_semantic_loss_weight requires clap_semantic_loss")
        if whisper_feature_loss_weight > 0.0 and whisper_feature_loss is None:
            raise ValueError("whisper_feature_loss_weight requires whisper_feature_loss")
        if perceptual_loss_start_step < 0:
            raise ValueError(f"perceptual_loss_start_step must be non-negative, got {perceptual_loss_start_step}")
        if perceptual_loss_every_n_steps <= 0:
            raise ValueError(
                f"perceptual_loss_every_n_steps must be positive, got {perceptual_loss_every_n_steps}"
            )
        robust_label_loss = str(robust_label_loss).lower()
        if robust_label_loss not in {"charbonnier", "l1"}:
            raise ValueError("robust_label_loss must be 'charbonnier' or 'l1'")
        if robust_label_eps <= 0.0:
            raise ValueError(f"robust_label_eps must be positive, got {robust_label_eps}")
        if mask_loss_eps <= 0.0:
            raise ValueError(f"mask_loss_eps must be positive, got {mask_loss_eps}")
        if mask_loss_max <= 0.0:
            raise ValueError(f"mask_loss_max must be positive, got {mask_loss_max}")
        mask_aux_alignment = str(mask_aux_alignment).lower()
        if mask_aux_alignment not in {"strict", "shared_prefix"}:
            raise ValueError("mask_aux_alignment must be 'strict' or 'shared_prefix'")
        if mask_aux_max_frame_mismatch < 0:
            raise ValueError(f"mask_aux_max_frame_mismatch must be non-negative, got {mask_aux_max_frame_mismatch}")
        for name, value in {
            "source_weighted_snr_loss_weight": source_weighted_snr_loss_weight,
            "explicit_source_loss_weight": explicit_source_loss_weight,
            "residual_source_loss_weight": residual_source_loss_weight,
        }.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if residual_source_index is not None and residual_source_index < 0:
            raise ValueError(f"residual_source_index must be non-negative, got {residual_source_index}")
        source_loss_weight_normalization = str(source_loss_weight_normalization).lower()
        if source_loss_weight_normalization not in {"subset_mean", "full_mean", "none"}:
            raise ValueError("source_loss_weight_normalization must be one of 'subset_mean', 'full_mean', or 'none'")
        band_mapping = "none" if distillation_band_mapping is None else str(distillation_band_mapping).lower()
        band_mapping_aliases = {
            "off": "none",
            "false": "none",
            "disabled": "none",
            "mel": "mel_centers",
            "mel_center": "mel_centers",
            "center": "mel_centers",
            "centers": "mel_centers",
            "overlap": "basis",
            "overlap_basis": "basis",
            "basis_overlap": "basis",
        }
        band_mapping = band_mapping_aliases.get(band_mapping, band_mapping)
        if band_mapping not in {"none", "linear", "basis", "mel_centers", "auto"}:
            raise ValueError(
                "distillation_band_mapping must be one of 'none', 'linear', 'basis', 'mel_centers', or 'auto'"
            )

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
        self.frame_silent_source_weight = float(frame_silent_source_weight)
        self.frame_silent_source_db = float(frame_silent_source_db)
        self.frame_silent_window_ms = float(frame_silent_window_ms)
        self.frame_silent_hop_ms = float(frame_silent_hop_ms)
        self.clap_semantic_loss = clap_semantic_loss
        self.clap_semantic_loss_weight = float(clap_semantic_loss_weight)
        self.whisper_feature_loss = whisper_feature_loss
        self.whisper_feature_loss_weight = float(whisper_feature_loss_weight)
        self.whisper_source = str(whisper_source)
        self.perceptual_loss_start_step = int(perceptual_loss_start_step)
        self.perceptual_loss_every_n_steps = int(perceptual_loss_every_n_steps)
        self.perceptual_loss_compensate_cadence = bool(perceptual_loss_compensate_cadence)
        self.source_activity_loss_weight = source_activity_loss_weight
        self.source_activity_db = source_activity_db
        self.source_activity_active_weight = source_activity_active_weight
        self.source_activity_inactive_weight = source_activity_inactive_weight
        self.source_order = tuple(source_order or ("speech", "music", "effects"))
        if self.whisper_feature_loss_weight > 0.0 and self.whisper_source not in self.source_order:
            raise ValueError(
                f"whisper_source={self.whisper_source!r} is not in source_order={self.source_order}"
            )
        self.whisper_source_index = (
            self.source_order.index(self.whisper_source) if self.whisper_source in self.source_order else 0
        )
        self.source_loss_weight_normalization = source_loss_weight_normalization
        self.source_weighted_snr_loss_weight = float(source_weighted_snr_loss_weight)
        self.explicit_source_loss_weight = float(explicit_source_loss_weight)
        self.residual_source_loss_weight = float(residual_source_loss_weight)
        self.residual_source_index = residual_source_index
        self.robust_label_loss_weight = robust_label_loss_weight
        self.robust_label_loss = robust_label_loss
        self.robust_label_eps = robust_label_eps
        self.teacher_mask_loss_weight = teacher_mask_loss_weight
        self.teacher_logit_loss_weight = teacher_logit_loss_weight
        self.request_model_aux = bool(request_model_aux)
        self.require_model_aux = bool(require_model_aux)
        self.mask_aux_alignment = mask_aux_alignment
        self.mask_aux_max_frame_mismatch = int(mask_aux_max_frame_mismatch)
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
        self.register_buffer(
            "_source_loss_weights",
            self._build_source_loss_weights(source_loss_weights),
            persistent=False,
        )
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
        self._student_frequency_projectors = _find_frequency_projectors(self.model)
        self._teacher_frequency_projectors = _find_frequency_projectors(self.teacher_model)

        self._register_latent_hooks()

    def _build_source_loss_weights(
        self,
        source_loss_weights: Sequence[float] | Mapping[str, float] | None,
    ) -> torch.Tensor | None:
        if source_loss_weights is None:
            return None
        if isinstance(source_loss_weights, Mapping):
            unknown = sorted(set(source_loss_weights) - set(self.source_order))
            if unknown:
                raise ValueError(
                    f"source_loss_weights contains unknown sources: {unknown}. "
                    f"Valid sources are: {list(self.source_order)}"
                )
            weights = [float(source_loss_weights.get(source_name, 1.0)) for source_name in self.source_order]
        else:
            weights = [float(weight) for weight in source_loss_weights]
        if not weights:
            raise ValueError("source_loss_weights must not be empty")
        if len(weights) != len(self.source_order):
            raise ValueError(
                f"source_loss_weights has {len(weights)} entries, but source_order has {len(self.source_order)}"
            )
        if any(weight <= 0.0 for weight in weights):
            raise ValueError(f"source_loss_weights values must be positive, got {weights}")
        return torch.tensor(weights, dtype=torch.float32)

    def _source_weights_for(
        self,
        value: torch.Tensor,
        source_indices: Sequence[int] | None = None,
    ) -> torch.Tensor | None:
        weights = self._source_loss_weights
        if weights is None:
            return None
        full_mean = weights.mean().clamp_min(1.0e-8)
        if source_indices is not None:
            indices = torch.tensor(tuple(int(idx) for idx in source_indices), device=weights.device, dtype=torch.long)
            weights = weights.index_select(0, indices)
        if weights.numel() != value.shape[1]:
            raise ValueError(
                f"source_loss_weights has {weights.numel()} entries, but estimate has {value.shape[1]} sources"
            )
        weights = weights.to(device=value.device, dtype=value.dtype)
        if self.source_loss_weight_normalization == "subset_mean":
            return weights / weights.mean().clamp_min(1.0e-8)
        if self.source_loss_weight_normalization == "full_mean":
            return weights / full_mean.to(device=value.device, dtype=value.dtype)
        return weights

    def _source_weighted_reduce(
        self,
        per_source_loss: torch.Tensor,
        source_indices: Sequence[int] | None = None,
    ) -> torch.Tensor:
        weights = self._source_weights_for(per_source_loss, source_indices=source_indices)
        if weights is None:
            return per_source_loss.mean()
        return (per_source_loss * weights.view(1, -1)).mean()

    def _source_weighted_l1(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        per_source_loss = (est - ref).abs().mean(dim=(-1, -2))
        return self._source_weighted_reduce(per_source_loss)

    def _robust_label_loss(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        diff = est - ref
        if self.robust_label_loss == "l1":
            per_element = diff.abs()
        else:
            eps = float(self.robust_label_eps)
            per_element = torch.sqrt(diff.square() + eps * eps) - eps
        per_source_loss = per_element.mean(dim=(-1, -2))
        return self._source_weighted_reduce(per_source_loss)

    def _robust_source_subset_loss(
        self,
        est: torch.Tensor,
        ref: torch.Tensor,
        source_indices: Sequence[int],
    ) -> torch.Tensor:
        indices = tuple(int(idx) for idx in source_indices)
        if not indices:
            return est.new_zeros(())
        subset_est = est[:, indices]
        subset_ref = ref[:, indices]
        diff = subset_est - subset_ref
        if self.robust_label_loss == "l1":
            per_element = diff.abs()
        else:
            eps = float(self.robust_label_eps)
            per_element = torch.sqrt(diff.square() + eps * eps) - eps
        per_source_loss = per_element.mean(dim=(-1, -2))
        return self._source_weighted_reduce(per_source_loss, source_indices=indices)

    def _source_weighted_snr_loss(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        err_power = (est.float() - ref.float()).square().mean(dim=(-1, -2))
        ref_power = ref.float().square().mean(dim=(-1, -2))
        est_power = est.float().square().mean(dim=(-1, -2))
        active = ref_power > 10 ** (self.silent_source_db / 10.0)
        eps = 1.0e-8
        active_loss = 10.0 * torch.log10(err_power + eps) - 10.0 * torch.log10(ref_power + eps)
        inactive_loss = 10.0 * torch.log10(est_power + eps)
        per_source_loss = torch.where(active, active_loss, inactive_loss)
        return self._source_weighted_reduce(per_source_loss.to(dtype=est.dtype))

    def _residual_source_index(self, n_src: int) -> int:
        index = int(n_src - 1 if self.residual_source_index is None else self.residual_source_index)
        if not 0 <= index < n_src:
            raise ValueError(f"residual_source_index={index} is out of range for {n_src} sources")
        return index

    def _explicit_source_indices(self, n_src: int) -> tuple[int, ...]:
        residual_index = self._residual_source_index(n_src)
        return tuple(idx for idx in range(n_src) if idx != residual_index)

    def _new_scalar_zero(self) -> torch.Tensor:
        for tensor in self.parameters():
            return tensor.new_zeros(())
        for tensor in self.buffers():
            return tensor.new_zeros(())
        return torch.zeros((), device=self.device)

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
        use_css = log_prefix != "training" and css_validation
        if use_css:
            output = model.css(wav, ref=ref)
        elif self.request_model_aux:
            try:
                output = model(wav, return_aux=True)
            except TypeError as exc:
                if not _is_unexpected_keyword_type_error(exc, "return_aux"):
                    raise
                if self.require_model_aux:
                    raise RuntimeError(
                        "request_model_aux=True and require_model_aux=True, but the model does not accept return_aux"
                    ) from exc
                output = model(wav)
        else:
            output = model(wav)
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

    def _frame_silent_source_penalty(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if self.fs is None:
            raise ValueError("frame-local silent-source loss requires fs")
        batch, n_src, n_chan, n_samples = ref.shape
        window = max(1, int(round(self.frame_silent_window_ms * self.fs / 1000.0)))
        hop = max(1, int(round(self.frame_silent_hop_ms * self.fs / 1000.0)))
        n_frames = max(1, math.ceil(max(0, n_samples - window) / hop) + 1)
        padded_samples = (n_frames - 1) * hop + window

        def frame_power(value: torch.Tensor) -> torch.Tensor:
            flattened = value.float().square().reshape(batch * n_src * n_chan, 1, n_samples)
            if padded_samples > n_samples:
                flattened = F.pad(flattened, (0, padded_samples - n_samples))
            pooled = F.avg_pool1d(flattened, kernel_size=window, stride=hop)
            return pooled.reshape(batch, n_src, n_chan, n_frames).mean(dim=2)

        ref_power = frame_power(ref)
        est_power = frame_power(est)
        inactive = ref_power <= 10 ** (self.frame_silent_source_db / 10.0)
        if not inactive.any():
            return est.new_zeros(())
        weights = torch.ones_like(est_power)
        source_weights = self._source_weights_for(est_power[:, :, 0])
        if source_weights is not None:
            weights *= source_weights.view(1, -1, 1)
        inactive_weights = weights * inactive
        return (est_power * inactive_weights).sum() / inactive_weights.sum().clamp_min(1.0)

    def _compute_frozen_perceptual_losses(
        self,
        est: torch.Tensor,
        ref: torch.Tensor,
        *,
        log_prefix: str,
    ) -> dict[str, torch.Tensor]:
        step = int(self.global_step)
        if step < self.perceptual_loss_start_step:
            return {}
        if log_prefix == "training" and step % self.perceptual_loss_every_n_steps != 0:
            return {}

        losses: dict[str, torch.Tensor] = {}
        if self.clap_semantic_loss_weight > 0.0:
            if self.clap_semantic_loss is None:
                raise RuntimeError("clap_semantic_loss is unexpectedly missing")
            forward_with_components = getattr(self.clap_semantic_loss, "forward_with_components", None)
            if callable(forward_with_components):
                clap_total, clap_components = forward_with_components(est, ref)
                if not isinstance(clap_total, torch.Tensor) or clap_total.ndim != 0:
                    raise TypeError("CLAP forward_with_components() must return a scalar tensor total")
                if not isinstance(clap_components, Mapping):
                    raise TypeError("CLAP forward_with_components() must return a component mapping")
                losses["clap_semantic"] = clap_total
                for name, value in clap_components.items():
                    if not isinstance(value, torch.Tensor) or value.ndim != 0:
                        raise TypeError(f"CLAP component {name!r} must be a scalar tensor")
                    losses[f"clap_semantic_{name}"] = value
            else:
                losses["clap_semantic"] = self.clap_semantic_loss(est, ref)
        if self.whisper_feature_loss_weight > 0.0:
            if self.whisper_feature_loss is None:
                raise RuntimeError("whisper_feature_loss is unexpectedly missing")
            source_index = self.whisper_source_index
            losses["whisper_feature"] = self.whisper_feature_loss(
                est[:, source_index],
                ref[:, source_index],
            )
        return losses

    def _source_activity_l1(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        per_source_l1 = (est - ref).abs().mean(dim=(-1, -2))
        ref_power = ref.float().square().mean(dim=(-1, -2))
        active = ref_power > 10 ** (self.source_activity_db / 10.0)
        active_weight = per_source_l1.new_tensor(self.source_activity_active_weight)
        inactive_weight = per_source_l1.new_tensor(self.source_activity_inactive_weight)
        weights = torch.where(active, active_weight, inactive_weight)
        source_weights = self._source_weights_for(per_source_l1)
        if source_weights is not None:
            weights = weights * source_weights.view(1, -1)
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

    def _aux_tensor_and_domain(
        self,
        aux: Mapping[str, Any],
        keys: Sequence[str],
    ) -> tuple[torch.Tensor | None, str | None]:
        for key in keys:
            tensor = _first_tensor(aux.get(key))
            if tensor is None:
                continue

            domain = aux.get(f"{key}_domain")
            if not isinstance(domain, str):
                domains = aux.get("aux_domains")
                if isinstance(domains, Mapping):
                    domain = domains.get(key)
            return tensor, domain if isinstance(domain, str) else None
        return None, None

    def _mask_to_logit_domain(
        self,
        mask: torch.Tensor,
        *,
        target_domain: str,
        target_aux: Mapping[str, Any],
    ) -> torch.Tensor | None:
        transform = target_aux.get("mask_logits_transform")
        if not isinstance(transform, str):
            return None
        if target_domain != target_aux.get("mask_logits_domain"):
            return None
        if transform != "sigmoid_tanh_complex_mask":
            return None
        if mask.ndim != 4 or mask.shape[1] % 2 != 0:
            return None

        real_scale = float(target_aux.get("mask_logits_real_scale", 1.0))
        imag_scale = float(target_aux.get("mask_logits_imag_scale", 1.0))
        if real_scale <= 0.0 or imag_scale <= 0.0:
            return None

        eps = float(self.mask_loss_eps)
        real = (mask[:, 0::2, :, :] / real_scale).clamp(eps, 1.0 - eps)
        imag = (mask[:, 1::2, :, :] / imag_scale).clamp(-1.0 + eps, 1.0 - eps)
        real_logits = torch.logit(real)
        imag_logits = 0.5 * (torch.log1p(imag) - torch.log1p(-imag))
        logits = mask.new_empty(mask.shape)
        logits[:, 0::2, :, :] = real_logits
        logits[:, 1::2, :, :] = imag_logits
        return logits

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

    def _projector_matrix(
        self,
        projector: nn.Module,
        attr_name: str,
        *,
        source_size: int,
        target_size: int,
        ref: torch.Tensor,
    ) -> torch.Tensor | None:
        matrix = getattr(projector, attr_name, None)
        if not isinstance(matrix, torch.Tensor):
            return None
        matrix_tensor = cast(torch.Tensor, matrix)
        if matrix_tensor.ndim != 2:
            return None
        if tuple(matrix_tensor.shape) != (target_size, source_size):
            return None
        return matrix_tensor.to(device=ref.device, dtype=torch.float32)

    def _frequency_projector_mapping(
        self,
        source_size: int,
        target_size: int,
        ref: torch.Tensor,
    ) -> torch.Tensor | None:
        for projector in self._student_frequency_projectors:
            mapping = self._projector_matrix(
                projector,
                "analysis_matrix",
                source_size=source_size,
                target_size=target_size,
                ref=ref,
            )
            if mapping is not None:
                return mapping

        for projector in self._teacher_frequency_projectors:
            mapping = self._projector_matrix(
                projector,
                "synthesis_matrix",
                source_size=source_size,
                target_size=target_size,
                ref=ref,
            )
            if mapping is not None:
                return mapping

        for teacher_projector in self._teacher_frequency_projectors:
            teacher_to_full = getattr(teacher_projector, "synthesis_matrix", None)
            if not isinstance(teacher_to_full, torch.Tensor):
                continue
            teacher_to_full = cast(torch.Tensor, teacher_to_full)
            if teacher_to_full.ndim != 2 or int(teacher_to_full.shape[1]) != source_size:
                continue
            teacher_to_full = teacher_to_full.to(device=ref.device, dtype=torch.float32)
            for student_projector in self._student_frequency_projectors:
                full_to_student = getattr(student_projector, "analysis_matrix", None)
                if not isinstance(full_to_student, torch.Tensor):
                    continue
                full_to_student = cast(torch.Tensor, full_to_student)
                if full_to_student.ndim != 2:
                    continue
                if int(full_to_student.shape[0]) != target_size:
                    continue
                if int(full_to_student.shape[1]) != int(teacher_to_full.shape[0]):
                    continue
                full_to_student = full_to_student.to(device=ref.device, dtype=torch.float32)
                return torch.matmul(full_to_student, teacher_to_full)
        return None

    def _band_frequency_axis(self, band_spec: nn.Module | None, size: int, ref: torch.Tensor) -> torch.Tensor | None:
        if band_spec is None:
            return None
        axis = getattr(band_spec, "bin_frequencies_hz", None)
        if not isinstance(axis, torch.Tensor):
            return None
        axis_tensor = cast(torch.Tensor, axis).flatten()
        if int(axis_tensor.numel()) != int(size):
            return None
        return axis_tensor.to(device=ref.device, dtype=torch.float32)

    def _resample_band_rows(
        self,
        value: torch.Tensor,
        *,
        source_axis: torch.Tensor | None,
        target_axis: torch.Tensor | None,
        target_size: int,
    ) -> torch.Tensor:
        source_size = int(value.shape[-1])
        target_size = int(target_size)
        if source_size == target_size:
            return value
        if source_axis is None or target_axis is None:
            return F.interpolate(value.unsqueeze(0), size=target_size, mode="linear", align_corners=True).squeeze(0)

        right = torch.searchsorted(source_axis.contiguous(), target_axis.contiguous()).clamp(max=source_size - 1)
        left = (right - 1).clamp(min=0)
        left = torch.where(target_axis <= source_axis[0], torch.zeros_like(left), left)
        right = torch.where(target_axis <= source_axis[0], torch.zeros_like(right), right)
        left = torch.where(target_axis >= source_axis[-1], torch.full_like(left, source_size - 1), left)
        right = torch.where(target_axis >= source_axis[-1], torch.full_like(right, source_size - 1), right)

        left_hz = source_axis[left]
        right_hz = source_axis[right]
        denom = (right_hz - left_hz).clamp_min(1.0e-6)
        right_weight = torch.where(left == right, torch.zeros_like(target_axis), (target_axis - left_hz) / denom)
        right_weight = right_weight.clamp(0.0, 1.0)
        left_weight = 1.0 - right_weight
        return value[:, left] * left_weight.unsqueeze(0) + value[:, right] * right_weight.unsqueeze(0)

    def _basis_band_mapping(self, source_size: int, target_size: int, ref: torch.Tensor) -> torch.Tensor | None:
        teacher_basis = _band_spec_basis(self._teacher_band_spec)
        student_basis = _band_spec_basis(self._student_band_spec)
        if teacher_basis is None or student_basis is None:
            if self.distillation_band_mapping == "basis":
                raise ValueError("basis band mapping requires a basis buffer on both student and teacher band specs")
            return None
        if int(teacher_basis.shape[0]) != int(source_size) or int(student_basis.shape[0]) != int(target_size):
            return None

        teacher_basis = teacher_basis.to(device=ref.device, dtype=torch.float32)
        student_basis = student_basis.to(device=ref.device, dtype=torch.float32)
        teacher_f = int(teacher_basis.shape[-1])
        student_f = int(student_basis.shape[-1])
        if teacher_f <= 0 or student_f <= 0:
            return None

        teacher_expand = teacher_basis / teacher_basis.sum(dim=0, keepdim=True).clamp_min(1.0e-6)
        teacher_axis = self._band_frequency_axis(self._teacher_band_spec, teacher_f, ref)
        student_axis = self._band_frequency_axis(self._student_band_spec, student_f, ref)
        teacher_expand = self._resample_band_rows(
            teacher_expand,
            source_axis=teacher_axis,
            target_axis=student_axis,
            target_size=student_f,
        )
        student_compress = student_basis / student_basis.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        mapping = torch.matmul(student_compress, teacher_expand.transpose(0, 1))
        mapping = mapping / mapping.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        return mapping

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
            # Mask/logit aux tensors are usually expanded back to the dense
            # frequency grid before they are exposed.  In that case the model
            # still has a band_spec, but the tensor width is no longer the
            # band-token count, so mel-center mapping is the wrong domain.
            return None

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
        if self.distillation_band_mapping in {"basis", "mel_centers", "auto"}:
            mapping = self._basis_band_mapping(source_size, target_size, teacher_value)
            if mapping is not None:
                mapped = torch.matmul(teacher_value.float(), mapping.transpose(0, 1))
                return mapped.to(dtype=teacher_value.dtype)
        if self.distillation_band_mapping in {"mel_centers", "auto"}:
            mapping = self._center_band_mapping(source_size, target_size, teacher_value)
            if mapping is not None:
                mapped = torch.matmul(teacher_value.float(), mapping.transpose(0, 1))
                return mapped.to(dtype=teacher_value.dtype)
        mapping = self._frequency_projector_mapping(source_size, target_size, teacher_value)
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

    def _align_mask_aux_tensors(
        self,
        student_value: torch.Tensor,
        teacher_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mask_aux_alignment == "strict":
            return student_value, teacher_value
        if student_value.ndim != 4 or teacher_value.ndim != 4:
            return student_value, teacher_value
        if student_value.shape[0] != teacher_value.shape[0]:
            return student_value, teacher_value

        shared_channels = min(int(student_value.shape[1]), int(teacher_value.shape[1]))
        shared_frames = min(int(student_value.shape[2]), int(teacher_value.shape[2]))
        if shared_channels <= 0 or shared_frames <= 0:
            return student_value, teacher_value
        frame_mismatch = abs(int(student_value.shape[2]) - int(teacher_value.shape[2]))
        if frame_mismatch > self.mask_aux_max_frame_mismatch:
            raise ValueError(
                "mask/logit aux frame mismatch exceeds configured tolerance: "
                f"{student_value.shape[2]} vs {teacher_value.shape[2]} "
                f"(max {self.mask_aux_max_frame_mismatch})"
            )
        if student_value.shape[1] != shared_channels or student_value.shape[2] != shared_frames:
            student_value = student_value[:, :shared_channels, :shared_frames, :]
        if teacher_value.shape[1] != shared_channels or teacher_value.shape[2] != shared_frames:
            teacher_value = teacher_value[:, :shared_channels, :shared_frames, :]
        return student_value, teacher_value

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
            student_value, student_domain = self._aux_tensor_and_domain(student_aux, ("mask_logits", "logits"))
            teacher_value, teacher_domain = self._aux_tensor_and_domain(teacher_aux, ("mask_logits", "logits"))
            if (
                student_value is not None
                and student_domain is not None
                and (teacher_value is None or teacher_domain != student_domain)
            ):
                teacher_mask, teacher_mask_domain = self._aux_tensor_and_domain(teacher_aux, ("mask", "masks"))
                if teacher_mask is not None and teacher_mask_domain == "packed_complex_mask":
                    converted_teacher = self._mask_to_logit_domain(
                        teacher_mask,
                        target_domain=student_domain,
                        target_aux=student_aux,
                    )
                    if converted_teacher is not None:
                        teacher_value = converted_teacher
                        teacher_domain = student_domain
            if (
                teacher_value is not None
                and teacher_domain is not None
                and (student_value is None or student_domain != teacher_domain)
            ):
                student_mask, student_mask_domain = self._aux_tensor_and_domain(student_aux, ("mask", "masks"))
                if student_mask is not None and student_mask_domain == "packed_complex_mask":
                    converted_student = self._mask_to_logit_domain(
                        student_mask,
                        target_domain=teacher_domain,
                        target_aux=teacher_aux,
                    )
                    if converted_student is not None:
                        student_value = converted_student
                        student_domain = teacher_domain
            if (
                student_value is None
                or teacher_value is None
                or student_domain is None
                or teacher_domain is None
                or student_domain != teacher_domain
            ):
                # Most waveform separators in this repo return estimates only. In
                # that case, logit distillation intentionally falls back to
                # waveform-derived spectral pseudo-masks instead of internal mask
                # logits. Raw aux logits are compared only when both models
                # declare the same logit domain.
                student_mask = self._spectral_mask(est, wav).clamp(self.mask_loss_eps, 1.0 - self.mask_loss_eps)
                teacher_mask = self._spectral_mask(teacher_est, wav).clamp(self.mask_loss_eps, 1.0 - self.mask_loss_eps)
                student_value = torch.logit(student_mask)
                teacher_value = torch.logit(teacher_mask)
        else:
            student_value, student_domain = self._aux_tensor_and_domain(student_aux, ("mask", "masks"))
            teacher_value, teacher_domain = self._aux_tensor_and_domain(teacher_aux, ("mask", "masks"))
            if (
                student_value is None
                or teacher_value is None
                or (student_domain is not None and teacher_domain is not None and student_domain != teacher_domain)
            ):
                # Match spectral pseudo-masks when true model masks are not
                # exposed by the model output contract or declare incompatible
                # domains.
                student_value = self._spectral_mask(est, wav)
                teacher_value = self._spectral_mask(teacher_est, wav)
        student_value, teacher_value = self._align_mask_aux_tensors(student_value, teacher_value)
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
                return self._new_scalar_zero()
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
            teacher_loss = self._source_weighted_l1(est, teacher_est_value.detach())
            loss = loss + self.teacher_loss_weight * teacher_loss
            log_dict[f"{log_prefix}/loss_teacher"] = teacher_loss

        if self.robust_label_loss_weight > 0.0:
            robust_label_loss = self._robust_label_loss(est, ref)
            loss = loss + self.robust_label_loss_weight * robust_label_loss
            log_dict[f"{log_prefix}/loss_robust_label"] = robust_label_loss

        if self.source_weighted_snr_loss_weight > 0.0:
            source_weighted_snr_loss = self._source_weighted_snr_loss(est, ref)
            loss = loss + self.source_weighted_snr_loss_weight * source_weighted_snr_loss
            log_dict[f"{log_prefix}/loss_source_weighted_snr"] = source_weighted_snr_loss

        if self.explicit_source_loss_weight > 0.0:
            explicit_source_loss = self._robust_source_subset_loss(
                est,
                ref,
                self._explicit_source_indices(ref.shape[1]),
            )
            loss = loss + self.explicit_source_loss_weight * explicit_source_loss
            log_dict[f"{log_prefix}/loss_explicit_sources"] = explicit_source_loss

        if self.residual_source_loss_weight > 0.0:
            residual_source_loss = self._robust_source_subset_loss(
                est,
                ref,
                (self._residual_source_index(ref.shape[1]),),
            )
            loss = loss + self.residual_source_loss_weight * residual_source_loss
            log_dict[f"{log_prefix}/loss_residual_source"] = residual_source_loss

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

        if self.frame_silent_source_weight > 0.0:
            frame_silent_loss = self._frame_silent_source_penalty(est, ref)
            loss = loss + self.frame_silent_source_weight * frame_silent_loss
            log_dict[f"{log_prefix}/loss_frame_silent_source"] = frame_silent_loss

        perceptual_losses = self._compute_frozen_perceptual_losses(est, ref, log_prefix=log_prefix)
        perceptual_scale = (
            float(self.perceptual_loss_every_n_steps)
            if log_prefix == "training" and self.perceptual_loss_compensate_cadence
            else 1.0
        )
        if "clap_semantic" in perceptual_losses:
            clap_loss = perceptual_losses["clap_semantic"]
            loss = loss + perceptual_scale * self.clap_semantic_loss_weight * clap_loss
            log_dict[f"{log_prefix}/loss_clap_semantic"] = clap_loss
        if "whisper_feature" in perceptual_losses:
            whisper_loss = perceptual_losses["whisper_feature"]
            loss = loss + perceptual_scale * self.whisper_feature_loss_weight * whisper_loss
            log_dict[f"{log_prefix}/loss_whisper_feature"] = whisper_loss
        if perceptual_losses:
            log_dict[f"{log_prefix}/perceptual_cadence_scale"] = loss.new_tensor(perceptual_scale)
        for name, component_loss in perceptual_losses.items():
            if name.startswith("clap_semantic_"):
                log_dict[f"{log_prefix}/loss_{name}"] = component_loss

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
