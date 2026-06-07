from __future__ import annotations

from typing import Any, cast

from copy import deepcopy
import itertools
import math

import torch
import torch.nn.functional as F

import lightning.pytorch as pl

from src.temporal import SILENCE_SPAN_SEC
from src.tools.estimated_source_matching import (
    pairwise_match_score,
    quality_and_weight,
    second_best_and_margin,
)
from src.training.lightningmodule.online_teacher_tse import (
    _TSE_EXTRA_CONDITION_KEYS,
    _load_model_checkpoint,
)
from src.training.loss.class_aware_pit import infer_active_mask_from_label
from src.utils import initialize_config


def _grad_scale(value, scale: float):
    """Keep forward value unchanged while scaling gradient into its producer."""

    if not torch.is_tensor(value):
        return value
    scale = float(scale)
    if scale <= 0.0:
        return value.detach()
    if scale == 1.0:
        return value
    detached = value.detach()
    return detached + scale * (value - detached)


def _energy_db_by_slot(waveform, eps=1e-12):
    flat = waveform.flatten(start_dim=2).float()
    rms = torch.sqrt(flat.pow(2).mean(dim=-1) + eps)
    return 20.0 * torch.log10(rms.clamp_min(1e-8))


def _active_indices(is_silence_row):
    return [idx for idx, is_sil in enumerate(is_silence_row.tolist()) if not bool(is_sil)]


def _warmup_cosine_lambda(warmup_steps: int, max_steps: int, min_lr_scale: float = 0.1, start_lr_scale: float = 0.0):
    warmup_steps = max(0, int(warmup_steps))
    max_steps = max(1, int(max_steps))
    min_lr_scale = float(min_lr_scale)
    start_lr_scale = float(start_lr_scale)

    def lr_lambda(step):
        step = int(step)
        if warmup_steps > 0 and step < warmup_steps:
            progress = float(step) / float(warmup_steps)
            return start_lr_scale + progress * (1.0 - start_lr_scale)
        denom = max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
        return min_lr_scale + 0.5 * (1.0 - min_lr_scale) * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def _best_est_to_ref(scores, active_refs, n_est):
    """Return estimated-slot -> reference-slot assignment maximizing score."""

    if not active_refs:
        return {}
    n_match = min(len(active_refs), n_est)
    refs = list(active_refs[:n_match])
    best_score = None
    best_perm = None
    for perm in itertools.permutations(range(n_est), n_match):
        vals = torch.stack([scores[ref_idx, est_idx] for ref_idx, est_idx in zip(refs, perm)])
        score = vals.mean()
        if best_score is None or score > best_score:
            best_score = score
            best_perm = perm
    if best_perm is None:
        return {}
    return {int(est_idx): int(ref_idx) for ref_idx, est_idx in zip(refs, best_perm)}


def _best_ref_to_est(scores, active_refs, n_est):
    return {ref_idx: est_idx for est_idx, ref_idx in _best_est_to_ref(scores, active_refs, n_est).items()}


class USSScTSEJointModelParallelLightning(pl.LightningModule):
    """Single-process model-parallel joint fine-tuning for USS, SC, and TSE.

    The module trains USS, SC, and TSE with explicit supervised anchor losses
    and controlled gradient routing.  By default each model is trainable but
    downstream losses do not push gradients into upstream models unless the
    corresponding gradient scale is enabled in the config.
    """

    def __init__(
        self,
        uss_model: dict,
        sc_model: dict,
        tse_model: dict,
        uss_loss: dict,
        sc_loss: dict,
        tse_loss: dict,
        optimizer_uss: dict | None = None,
        optimizer_sc: dict | None = None,
        optimizer_tse: dict | None = None,
        uss_lr_scheduler: dict | None = None,
        sc_lr_scheduler: dict | None = None,
        tse_lr_scheduler: dict | None = None,
        uss_pretrained_ckpt: str | None = None,
        sc_pretrained_ckpt: str | None = None,
        tse_pretrained_ckpt: str | None = None,
        uss_pretrained_strict: bool = True,
        sc_pretrained_strict: bool = True,
        tse_pretrained_strict: bool = True,
        uss_device: str = "cuda:0",
        sc_device: str = "cuda:1",
        tse_device: str = "cuda:2",
        loss_device: str = "cuda:0",
        uss_output_key: str = "foreground_waveform",
        lambda_uss: float = 1.0,
        lambda_sc_uss: float = 0.05,
        lambda_tse: float = 1.0,
        lambda_sc_tse: float = 0.0,
        lambda_uss_sc_consistency: float = 0.01,
        consistency_temperature: float = 1.0,
        match_metric: str = "sa_sdr",
        min_match_score: float = -10.0,
        min_match_margin: float = -1.0e9,
        min_energy_db: float = -60.0,
        clean_match_score: float = 0.0,
        clean_match_margin: float = 2.0,
        uncertain_weight: float = 0.35,
        use_uncertain_matches: bool = False,
        bad_match_silence_weight: float = 0.0,
        unmatched_estimated_silence_weight: float = 0.0,
        unmatched_estimated_min_energy_db: float | None = None,
        unmatched_estimated_max_match_score: float | None = None,
        clean_source_mix_prob: float = 0.0,
        clean_source_mix_weight: float = 1.0,
        clean_silence_mix_prob: float = 0.0,
        clean_silence_mix_weight: float = 1.0,
        min_tse_match_score: float | None = None,
        min_estimate_energy_db: float | None = None,
        require_sc_active_for_tse_loss: bool = False,
        require_sc_class_match_for_tse_loss: bool = False,
        tse_use_match_quality_weight: bool = False,
        query_condition_enabled: bool = True,
        query_condition_key: str | None = None,
        temporal_conditioning_source: str = "auto",
        tse_label_mode: str = "soft_detached",
        tse_label_temperature: float = 1.0,
        tse_label_grad_scale: float = 0.0,
        tse_label_silence_gate: str = "none",
        tse_label_silence_temperature: float = 0.5,
        sc_tse_active_sample_weight: float = 1.0,
        sc_tse_silence_sample_weight: float = 0.2,
        sc_uss_to_uss_grad_scale: float = 0.0,
        tse_to_uss_enrollment_grad_scale: float = 0.0,
        tse_to_uss_condition_grad_scale: float = 0.0,
        sc_tse_to_tse_grad_scale: float = 0.0,
        uss_update_every: int = 1,
        sc_update_every: int = 1,
        tse_update_every: int = 1,
        is_validation: bool = True,
    ):
        super().__init__()
        self.automatic_optimization = False

        self.uss_model = initialize_config(deepcopy(uss_model))
        self.sc_model = initialize_config(deepcopy(sc_model))
        self.tse_model = initialize_config(deepcopy(tse_model))
        _load_model_checkpoint(self.uss_model, uss_pretrained_ckpt, strict=uss_pretrained_strict, name="uss")
        _load_model_checkpoint(self.sc_model, sc_pretrained_ckpt, strict=sc_pretrained_strict, name="sc")
        _load_model_checkpoint(
            self.tse_model,
            tse_pretrained_ckpt,
            strict=tse_pretrained_strict,
            name="tse",
            allowed_missing_prefixes=(
                "query_conditioner.",
                "bridge_to_label.",
                "activity_head.",
                "temporal_conditioner.",
            ),
        )

        self.uss_loss_func = initialize_config(deepcopy(uss_loss))
        self.sc_loss_func = initialize_config(deepcopy(sc_loss))
        self.tse_loss_func = initialize_config(deepcopy(tse_loss))
        self.optimizer_uss_config = deepcopy(optimizer_uss)
        self.optimizer_sc_config = deepcopy(optimizer_sc)
        self.optimizer_tse_config = deepcopy(optimizer_tse)
        self.uss_lr_scheduler_config = deepcopy(uss_lr_scheduler)
        self.sc_lr_scheduler_config = deepcopy(sc_lr_scheduler)
        self.tse_lr_scheduler_config = deepcopy(tse_lr_scheduler)
        self._optimizer_names: list[str] = []
        self._scheduler_names: list[str] = []
        self._scheduler_intervals: list[str] = []
        self._scheduler_frequencies: list[int] = []

        self.uss_device_name = uss_device
        self.sc_device_name = sc_device
        self.tse_device_name = tse_device
        self.loss_device_name = loss_device
        self.uss_output_key = uss_output_key
        self.lambda_uss = float(lambda_uss)
        self.lambda_sc_uss = float(lambda_sc_uss)
        self.lambda_tse = float(lambda_tse)
        self.lambda_sc_tse = float(lambda_sc_tse)
        self.lambda_uss_sc_consistency = float(lambda_uss_sc_consistency)
        self.consistency_temperature = float(consistency_temperature)
        self.match_metric = match_metric
        self.min_match_score = float(min_match_score)
        self.min_match_margin = float(min_match_margin)
        self.min_energy_db = float(min_energy_db)
        self.clean_match_score = float(clean_match_score)
        self.clean_match_margin = float(clean_match_margin)
        self.uncertain_weight = float(uncertain_weight)
        self.use_uncertain_matches = bool(use_uncertain_matches)
        self.bad_match_silence_weight = max(0.0, float(bad_match_silence_weight))
        self.unmatched_estimated_silence_weight = max(0.0, float(unmatched_estimated_silence_weight))
        self.unmatched_estimated_min_energy_db = (
            self.min_energy_db
            if unmatched_estimated_min_energy_db is None
            else float(unmatched_estimated_min_energy_db)
        )
        self.unmatched_estimated_max_match_score = (
            None if unmatched_estimated_max_match_score is None else float(unmatched_estimated_max_match_score)
        )
        self.clean_source_mix_prob = min(1.0, max(0.0, float(clean_source_mix_prob)))
        self.clean_source_mix_weight = max(0.0, float(clean_source_mix_weight))
        self.clean_silence_mix_prob = min(1.0, max(0.0, float(clean_silence_mix_prob)))
        self.clean_silence_mix_weight = max(0.0, float(clean_silence_mix_weight))
        self.min_tse_match_score = self.min_match_score if min_tse_match_score is None else float(min_tse_match_score)
        self.min_estimate_energy_db = (
            self.min_energy_db if min_estimate_energy_db is None else float(min_estimate_energy_db)
        )
        self.require_sc_active_for_tse_loss = bool(require_sc_active_for_tse_loss)
        self.require_sc_class_match_for_tse_loss = bool(require_sc_class_match_for_tse_loss)
        self.tse_use_match_quality_weight = bool(tse_use_match_quality_weight)
        self.query_condition_enabled = bool(query_condition_enabled)
        self.query_condition_key = query_condition_key
        self.temporal_conditioning_source = temporal_conditioning_source

        if tse_label_mode not in {"hard_detached", "soft_detached", "soft_grad", "straight_through"}:
            raise ValueError("Unsupported tse_label_mode")
        if tse_label_silence_gate not in {"none", "hard_energy", "soft_energy"}:
            raise ValueError("Unsupported tse_label_silence_gate")
        self.tse_label_mode = tse_label_mode
        self.tse_label_temperature = float(tse_label_temperature)
        self.tse_label_grad_scale = float(tse_label_grad_scale)
        self.tse_label_silence_gate = tse_label_silence_gate
        self.tse_label_silence_temperature = float(tse_label_silence_temperature)
        self.sc_tse_active_sample_weight = max(0.0, float(sc_tse_active_sample_weight))
        self.sc_tse_silence_sample_weight = max(0.0, float(sc_tse_silence_sample_weight))
        self.sc_uss_to_uss_grad_scale = float(sc_uss_to_uss_grad_scale)
        self.tse_to_uss_enrollment_grad_scale = float(tse_to_uss_enrollment_grad_scale)
        self.tse_to_uss_condition_grad_scale = float(tse_to_uss_condition_grad_scale)
        self.sc_tse_to_tse_grad_scale = float(sc_tse_to_tse_grad_scale)
        self.uss_update_every = max(1, int(uss_update_every))
        self.sc_update_every = max(1, int(sc_update_every))
        self.tse_update_every = max(1, int(tse_update_every))
        self.is_validation = bool(is_validation)
        self.label_dim = int(getattr(self.tse_model, "label_dim", getattr(self.sc_model, "num_classes", 18)))

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        return batch

    def _init_devices(self):
        self.uss_device = torch.device(self.uss_device_name)
        self.sc_device = torch.device(self.sc_device_name)
        self.tse_device = torch.device(self.tse_device_name)
        self.loss_device = torch.device(self.loss_device_name)
        for name, device in (
            ("uss_device", self.uss_device),
            ("sc_device", self.sc_device),
            ("tse_device", self.tse_device),
            ("loss_device", self.loss_device),
        ):
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError(f"CUDA is required for {name}={device}")
                if device.index is not None and torch.cuda.device_count() <= device.index:
                    raise RuntimeError(
                        f"Requested {name}={device}, but only {torch.cuda.device_count()} CUDA devices are visible"
                    )

    def _place_submodels(self):
        if not hasattr(self, "uss_device"):
            self._init_devices()
        self.uss_model.to(self.uss_device)
        self.sc_model.to(self.sc_device)
        self.tse_model.to(self.tse_device)

    def _assert_submodel_devices(self):
        for name, model, device in (
            ("USS", self.uss_model, self.uss_device),
            ("SC", self.sc_model, self.sc_device),
            ("TSE", self.tse_model, self.tse_device),
        ):
            param = next(model.parameters(), None)
            if param is not None and param.device != device:
                raise RuntimeError(f"{name} parameters are on {param.device}, expected {device}")

    def setup(self, stage=None):
        self._init_devices()
        self._place_submodels()

    def on_fit_start(self):
        self._place_submodels()
        self._assert_submodel_devices()

    def on_validation_start(self):
        self._place_submodels()
        self._assert_submodel_devices()

    def on_test_start(self):
        self._place_submodels()
        self._assert_submodel_devices()

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "uss_model"):
            self.uss_model.train(mode)
        if hasattr(self, "sc_model"):
            self.sc_model.train(mode)
        if hasattr(self, "tse_model"):
            self.tse_model.train(mode)
        return self

    def _to_uss(self, value):
        return value.to(self.uss_device) if torch.is_tensor(value) else value

    def _to_sc(self, value):
        return value.to(self.sc_device) if torch.is_tensor(value) else value

    def _to_tse(self, value):
        return value.to(self.tse_device) if torch.is_tensor(value) else value

    def _to_loss(self, value):
        return value.to(self.loss_device) if torch.is_tensor(value) else value

    def _uss_input(self, batch):
        input_dict = {"mixture": self._to_uss(batch["mixture"])}
        for key in ("spatial_vector", "spatial_clue", "doa_vector"):
            if key in batch:
                input_dict[key] = self._to_uss(batch[key])
        return input_dict

    def _uss_target(self, batch):
        keys = (
            "mixture",
            "foreground_waveform",
            "interference_waveform",
            "noise_waveform",
            "class_index",
            "is_silence",
            "foreground_span_sec",
            "interference_span_sec",
            "noise_span_sec",
            "spatial_vector",
            "foreground_doa",
            "foreground_doa_mask",
            "class_confidence",
            "soft_class_target",
            "uncertain_slot_mask",
            "bad_slot_mask",
        )
        target = {key: self._to_uss(batch[key]) for key in keys if key in batch}
        if "spatial_vector" not in target and "foreground_doa" in target:
            target["spatial_vector"] = target["foreground_doa"]
        target["current_epoch"] = self.current_epoch
        return target

    def _oracle_label_vector(self, batch, device, dtype):
        class_index = batch["class_index"].to(device=device, dtype=torch.long)
        is_silence = batch["is_silence"].to(device=device, dtype=torch.bool)
        batch_size, n_sources = class_index.shape
        label_vector = batch.get("label_vector")
        if label_vector is not None:
            label_vector = label_vector.to(device=device, dtype=dtype)
            if label_vector.dim() == 2:
                if label_vector.shape[-1] == n_sources * self.label_dim:
                    label_vector = label_vector.view(batch_size, n_sources, self.label_dim)
                elif label_vector.shape[-1] % n_sources == 0:
                    label_vector = label_vector.view(batch_size, n_sources, -1)
                else:
                    label_vector = F.one_hot(class_index.clamp_min(0), num_classes=self.label_dim).to(dtype=dtype)
            elif label_vector.dim() != 3:
                label_vector = F.one_hot(class_index.clamp_min(0), num_classes=self.label_dim).to(dtype=dtype)
        else:
            label_vector = F.one_hot(class_index.clamp_min(0), num_classes=self.label_dim).to(dtype=dtype)
        if label_vector.shape[-1] < self.label_dim:
            pad = label_vector.new_zeros(*label_vector.shape[:-1], self.label_dim - label_vector.shape[-1])
            label_vector = torch.cat([label_vector, pad], dim=-1)
        elif label_vector.shape[-1] > self.label_dim:
            label_vector = label_vector[..., : self.label_dim]
        label_vector = label_vector.clone()
        label_vector[is_silence] = 0.0
        return label_vector

    def _build_slot_targets(self, sep, batch):
        ref = self._to_uss(batch["foreground_waveform"])
        class_index_ref = self._to_uss(batch["class_index"]).long()
        is_silence_ref = self._to_uss(batch["is_silence"]).bool()
        span_ref = self._to_uss(batch["foreground_span_sec"]) if "foreground_span_sec" in batch else None
        bsz, n_est = sep.shape[:2]
        class_idx = torch.zeros(bsz, n_est, dtype=torch.long, device=self.uss_device)
        is_silence = torch.ones(bsz, n_est, dtype=torch.bool, device=self.uss_device)
        sample_weight = torch.zeros(bsz, n_est, dtype=sep.dtype, device=self.uss_device)
        quality_code = torch.zeros(bsz, n_est, dtype=torch.long, device=self.uss_device)
        ref_index = torch.full((bsz, n_est), -1, dtype=torch.long, device=self.uss_device)
        span_sec = None if span_ref is None else span_ref.new_full((bsz, n_est, 2), -1.0)
        match_score = torch.full((bsz, n_est), float("nan"), dtype=sep.dtype, device=self.uss_device)
        second_best_score = torch.full_like(match_score, float("nan"))
        match_margin = torch.full_like(match_score, float("nan"))
        energy_db = _energy_db_by_slot(sep.detach())
        with torch.no_grad():
            scores = pairwise_match_score(sep.detach(), ref.detach(), metric=self.match_metric).to(self.uss_device)
            for b in range(bsz):
                active_refs = _active_indices(is_silence_ref[b])
                est_to_ref = _best_est_to_ref(scores[b], active_refs, n_est)
                matched_est = set(est_to_ref.keys())
                for est_idx, ref_idx in est_to_ref.items():
                    ref_index[b, est_idx] = ref_idx
                    score = float(scores[b, ref_idx, est_idx].item())
                    second, margin = second_best_and_margin(scores[b], ref_idx, est_idx)
                    energy = float(energy_db[b, est_idx].item())
                    match_score[b, est_idx] = score
                    second_best_score[b, est_idx] = second
                    match_margin[b, est_idx] = margin
                    quality, weight, valid = quality_and_weight(
                        score=score,
                        margin=margin,
                        energy_db=energy,
                        min_match_score=self.min_match_score,
                        min_match_margin=self.min_match_margin,
                        min_energy_db=self.min_energy_db,
                        clean_match_score=self.clean_match_score,
                        clean_match_margin=self.clean_match_margin,
                        uncertain_weight=self.uncertain_weight,
                    )
                    if quality == "clean":
                        quality_code[b, est_idx] = 1
                    elif quality == "uncertain":
                        quality_code[b, est_idx] = 2
                    else:
                        quality_code[b, est_idx] = 3
                        if self.bad_match_silence_weight > 0.0 and energy >= self.min_energy_db:
                            sample_weight[b, est_idx] = self.bad_match_silence_weight
                    if not valid or (quality == "uncertain" and not self.use_uncertain_matches):
                        continue
                    class_idx[b, est_idx] = class_index_ref[b, ref_idx]
                    is_silence[b, est_idx] = False
                    sample_weight[b, est_idx] = float(weight)
                    if span_sec is not None and span_ref is not None:
                        span_sec[b, est_idx] = span_ref[b, ref_idx]
                if self.unmatched_estimated_silence_weight > 0.0:
                    for est_idx in range(n_est):
                        if est_idx in matched_est:
                            continue
                        energy = float(energy_db[b, est_idx].item())
                        if energy < self.unmatched_estimated_min_energy_db:
                            continue
                        if active_refs and self.unmatched_estimated_max_match_score is not None:
                            active_tensor = torch.tensor(active_refs, dtype=torch.long, device=self.uss_device)
                            best_active_score = float(scores[b, active_tensor, est_idx].max().item())
                            if best_active_score > self.unmatched_estimated_max_match_score:
                                continue
                        sample_weight[b, est_idx] = max(
                            float(sample_weight[b, est_idx].item()),
                            self.unmatched_estimated_silence_weight,
                        )
                        quality_code[b, est_idx] = 4
        return {
            "class_idx": class_idx,
            "is_silence": is_silence,
            "sample_weight": sample_weight,
            "span_sec": span_sec,
            "quality_code": quality_code,
            "ref_index": ref_index,
            "match_info": {
                "match_score": match_score,
                "second_best_score": second_best_score,
                "match_margin": match_margin,
                "energy_db": energy_db,
            },
        }

    def _maybe_mix_clean_sources(self, sep, batch, targets, is_training):
        ref_index = targets["ref_index"]
        clean_mask = torch.zeros(ref_index.shape, dtype=torch.bool, device=self.uss_device)
        if (not is_training) or self.clean_source_mix_prob <= 0.0:
            return sep, targets, clean_mask
        ref = self._to_uss(batch["foreground_waveform"])
        class_index_ref = self._to_uss(batch["class_index"]).long()
        is_silence_ref = self._to_uss(batch["is_silence"]).bool()
        span_ref = self._to_uss(batch["foreground_span_sec"]) if "foreground_span_sec" in batch else None
        eligible = ref_index.ge(0) & ~torch.gather(is_silence_ref, 1, ref_index.clamp_min(0))
        clean_mask = eligible & (torch.rand(eligible.shape, device=self.uss_device) < self.clean_source_mix_prob)
        if not clean_mask.any():
            return sep, targets, clean_mask
        b_idx, est_idx = clean_mask.nonzero(as_tuple=True)
        ref_idx = ref_index[b_idx, est_idx]
        out = dict(targets)
        sep = sep.clone()
        out["class_idx"] = targets["class_idx"].clone()
        out["is_silence"] = targets["is_silence"].clone()
        out["sample_weight"] = targets["sample_weight"].clone()
        if targets["span_sec"] is not None:
            out["span_sec"] = targets["span_sec"].clone()
        sep[b_idx, est_idx] = ref[b_idx, ref_idx].to(dtype=sep.dtype)
        out["class_idx"][b_idx, est_idx] = class_index_ref[b_idx, ref_idx]
        out["is_silence"][b_idx, est_idx] = False
        out["sample_weight"][b_idx, est_idx] = self.clean_source_mix_weight
        if out["span_sec"] is not None and span_ref is not None:
            out["span_sec"][b_idx, est_idx] = span_ref[b_idx, ref_idx].to(dtype=out["span_sec"].dtype)
        return sep, out, clean_mask

    def _maybe_mix_clean_silence(self, sep, batch, targets, is_training):
        is_silence = targets["is_silence"]
        clean_mask = torch.zeros(is_silence.shape, dtype=torch.bool, device=self.uss_device)
        if (not is_training) or self.clean_silence_mix_prob <= 0.0:
            return sep, targets, clean_mask
        ref = self._to_uss(batch["foreground_waveform"])
        silence_ref = self._to_uss(batch["is_silence"]).bool()
        selected_b, selected_est, selected_ref = [], [], []
        for b in range(sep.shape[0]):
            refs = torch.nonzero(silence_ref[b], as_tuple=False).flatten()
            ests = torch.nonzero(is_silence[b] & targets["sample_weight"][b].le(0.0), as_tuple=False).flatten()
            if refs.numel() == 0 or ests.numel() == 0:
                continue
            ests = ests[torch.rand(ests.shape, device=self.uss_device) < self.clean_silence_mix_prob]
            if ests.numel() == 0:
                continue
            selected_b.append(torch.full_like(ests, b))
            selected_est.append(ests)
            selected_ref.append(refs[torch.arange(ests.numel(), device=self.uss_device) % refs.numel()])
        if not selected_b:
            return sep, targets, clean_mask
        b_idx = torch.cat(selected_b)
        est_idx = torch.cat(selected_est)
        ref_idx = torch.cat(selected_ref)
        out = dict(targets)
        sep = sep.clone()
        out["class_idx"] = targets["class_idx"].clone()
        out["is_silence"] = targets["is_silence"].clone()
        out["sample_weight"] = targets["sample_weight"].clone()
        if targets["span_sec"] is not None:
            out["span_sec"] = targets["span_sec"].clone()
        sep[b_idx, est_idx] = ref[b_idx, ref_idx].to(dtype=sep.dtype)
        out["class_idx"][b_idx, est_idx] = 0
        out["is_silence"][b_idx, est_idx] = True
        out["sample_weight"][b_idx, est_idx] = self.clean_silence_mix_weight
        clean_mask[b_idx, est_idx] = True
        if out["span_sec"] is not None:
            out["span_sec"][b_idx, est_idx] = torch.tensor(
                SILENCE_SPAN_SEC, device=self.uss_device, dtype=out["span_sec"].dtype
            )
        return sep, out, clean_mask

    def _sc_forward_slots(self, sep, class_idx, span_sec=None):
        bsz, n_slots, channels, samples = sep.shape
        waveform = sep.reshape(bsz * n_slots, channels, samples).to(self.sc_device)
        class_flat = class_idx.reshape(bsz * n_slots).to(self.sc_device)
        sc_input = {"waveform": waveform, "class_index": class_flat}
        if span_sec is not None:
            sc_input["span_sec"] = span_sec.reshape(bsz * n_slots, 2).to(self.sc_device)
        return self.sc_model(sc_input)

    def _sc_uss_forward_and_loss(self, uss_out, batch, is_training):
        sep_actual = _grad_scale(uss_out[self.uss_output_key], self.sc_uss_to_uss_grad_scale)
        targets = self._build_slot_targets(sep_actual, batch)
        sep_loss, loss_targets, clean_source_mask = self._maybe_mix_clean_sources(
            sep_actual, batch, targets, is_training
        )
        sep_loss, loss_targets, clean_silence_mask = self._maybe_mix_clean_silence(
            sep_loss, batch, loss_targets, is_training
        )
        sc_out_loss = self._sc_forward_slots(sep_loss, loss_targets["class_idx"], loss_targets["span_sec"])
        bsz, n_slots = loss_targets["class_idx"].shape
        class_flat = loss_targets["class_idx"].reshape(bsz * n_slots).to(self.sc_device)
        silence_flat = loss_targets["is_silence"].reshape(bsz * n_slots).to(self.sc_device)
        weight_flat = loss_targets["sample_weight"].reshape(bsz * n_slots).to(self.sc_device)
        sc_target = {
            "class_index": class_flat,
            "is_silence": silence_flat,
            "sample_weight": weight_flat,
            "current_epoch": self.current_epoch,
            "is_training": is_training,
        }
        if loss_targets["span_sec"] is not None:
            sc_target["span_sec"] = loss_targets["span_sec"].reshape(bsz * n_slots, 2).to(self.sc_device)
        loss_dict = self.sc_loss_func(sc_out_loss, sc_target)
        if clean_source_mask.any() or clean_silence_mask.any():
            sc_out_query = self._sc_forward_slots(sep_actual, targets["class_idx"], targets["span_sec"])
        else:
            sc_out_query = sc_out_loss
        logits = sc_out_loss.get("plain_logits", sc_out_loss.get("logits"))
        active_weight = (~silence_flat).to(dtype=logits.dtype) * weight_flat.to(dtype=logits.dtype)
        top1 = logits.new_zeros(())
        if active_weight.sum() > 0:
            pred = logits.argmax(dim=-1)
            top1 = (
                ((pred == class_flat).to(dtype=logits.dtype) * active_weight).sum()
                / active_weight.sum().clamp_min(1.0)
                * 100.0
            )
        quality_flat = targets["quality_code"].reshape(bsz * n_slots).to(self.sc_device)
        diagnostics = {
            "sc_uss_top1": top1,
            "sc_uss_active_weight_mean": active_weight.mean(),
            "sc_uss_used_match_count": weight_flat.gt(0).to(dtype=logits.dtype).sum(),
            "sc_uss_clean_match_count": (quality_flat == 1).to(dtype=logits.dtype).sum(),
            "sc_uss_uncertain_match_count": (quality_flat == 2).to(dtype=logits.dtype).sum(),
            "sc_uss_bad_match_count": (quality_flat == 3).to(dtype=logits.dtype).sum(),
            "sc_uss_unmatched_silence_count": (quality_flat == 4).to(dtype=logits.dtype).sum(),
            "sc_uss_clean_source_mix_count": clean_source_mask.reshape(-1).to(self.sc_device, dtype=logits.dtype).sum(),
            "sc_uss_clean_silence_mix_count": clean_silence_mask.reshape(-1)
            .to(self.sc_device, dtype=logits.dtype)
            .sum(),
        }
        return {
            "loss": loss_dict["loss"],
            "loss_dict": loss_dict,
            "sc_out": sc_out_query,
            "targets": targets,
            "diagnostics": diagnostics,
        }

    def _threshold_for_indices(self, indices, device, dtype):
        thresholds = getattr(self.sc_model, "energy_thresholds", None) or {}
        values = []
        for idx in indices.detach().reshape(-1).tolist():
            threshold = thresholds.get(str(int(idx)), thresholds.get(int(idx), thresholds.get("default", None)))
            values.append(float("inf") if threshold is None else float(threshold))
        return torch.tensor(values, device=device, dtype=dtype).view(indices.shape)

    def _build_tse_label_from_sc(self, sc_out, batch_size, n_slots):
        logits = sc_out.get("plain_logits", sc_out.get("logits"))
        if logits is None:
            raise KeyError("SC output must contain plain_logits or logits")
        temperature = max(self.tse_label_temperature, 1e-6)
        soft = F.softmax(logits.float() / temperature, dim=-1).to(dtype=logits.dtype)
        hard = F.one_hot(soft.argmax(dim=-1), num_classes=soft.shape[-1]).to(dtype=soft.dtype)
        indices = soft.argmax(dim=-1)
        energy = sc_out.get("energy")
        if energy is None:
            energy = -torch.logsumexp(logits.float(), dim=-1).to(dtype=logits.dtype)
        threshold = self._threshold_for_indices(indices, logits.device, logits.dtype)
        hard_active = torch.where(
            torch.isfinite(threshold),
            energy <= threshold,
            torch.ones_like(energy, dtype=torch.bool),
        )
        gate = None
        if self.tse_label_silence_gate != "none":
            if self.tse_label_silence_gate == "hard_energy":
                gate = hard_active.to(dtype=soft.dtype)
            else:
                gate = torch.sigmoid((threshold - energy) / max(self.tse_label_silence_temperature, 1e-6)).to(
                    dtype=soft.dtype
                )
                gate = torch.where(torch.isfinite(threshold), gate, torch.ones_like(gate))
            soft = soft * gate.unsqueeze(-1)
            hard = hard * gate.detach().unsqueeze(-1)
        if self.tse_label_mode == "hard_detached":
            label = hard.detach()
        elif self.tse_label_mode == "soft_detached":
            label = soft.detach()
        elif self.tse_label_mode == "soft_grad":
            label = _grad_scale(soft, self.tse_label_grad_scale)
        elif self.tse_label_mode == "straight_through":
            soft_grad = _grad_scale(soft, self.tse_label_grad_scale)
            label = hard.detach() + (soft_grad - soft_grad.detach())
        else:
            raise ValueError(f"Unknown tse_label_mode: {self.tse_label_mode}")
        if label.shape[-1] < self.label_dim:
            label = torch.cat([label, label.new_zeros(label.shape[0], self.label_dim - label.shape[-1])], dim=-1)
        elif label.shape[-1] > self.label_dim:
            label = label[:, : self.label_dim]
        sc_active = (gate.detach() > 0.5) if gate is not None else hard_active.detach()
        return (
            label.view(batch_size, n_slots, self.label_dim),
            sc_active.view(batch_size, n_slots),
            soft.view(batch_size, n_slots, -1),
        )

    def _match_condition_slots(self, tensor, n_sources, device, dtype):
        tensor = tensor.to(device=device, dtype=dtype)
        if tensor.dim() == 2:
            if tensor.shape[1] == n_sources:
                tensor = tensor.unsqueeze(-1)
            else:
                tensor = tensor.unsqueeze(1).expand(-1, n_sources, -1)
        if tensor.dim() != 3:
            raise ValueError(f"USS/TSE condition tensor must be [B,S,D], got {tuple(tensor.shape)}")
        if tensor.shape[1] < n_sources:
            pad = tensor.new_zeros(tensor.shape[0], n_sources - tensor.shape[1], tensor.shape[-1])
            tensor = torch.cat([tensor, pad], dim=1)
        elif tensor.shape[1] > n_sources:
            tensor = tensor[:, :n_sources]
        return tensor

    def _build_query_condition(self, uss_out, enrollment):
        if not self.query_condition_enabled:
            return None
        n_sources = enrollment.shape[1]
        device = enrollment.device
        dtype = enrollment.dtype
        keys = (
            (self.query_condition_key,)
            if self.query_condition_key
            else ("tse_condition", "query_condition", "bridge_condition", "proposal_condition")
        )
        for key in keys:
            if key and key in uss_out:
                return self._match_condition_slots(uss_out[key], n_sources, device, dtype)
        parts = []
        if "class_logits" in uss_out:
            parts.append(self._match_condition_slots(uss_out["class_logits"].softmax(dim=-1), n_sources, device, dtype))
        if "silence_logits" in uss_out:
            parts.append(
                self._match_condition_slots(uss_out["silence_logits"].sigmoid().unsqueeze(-1), n_sources, device, dtype)
            )
        if "count_logits" in uss_out:
            count_prob = uss_out["count_logits"].softmax(dim=-1).to(device=device, dtype=dtype)
            parts.append(count_prob.unsqueeze(1).expand(-1, n_sources, -1))
        for key in ("spatial_embedding", "doa_vector", "pred_doa_vector", "used_spatial_vector"):
            if key in uss_out:
                parts.append(self._match_condition_slots(uss_out[key], n_sources, device, dtype))
        if "foreground_activity_logits" in uss_out:
            activity = uss_out["foreground_activity_logits"].sigmoid()
            activity = torch.stack([activity.mean(dim=-1), activity.amax(dim=-1)], dim=-1)
            parts.append(self._match_condition_slots(activity, n_sources, device, dtype))
        slot_rms = enrollment[:, :, 0].float().pow(2).mean(dim=-1).sqrt().to(dtype=dtype).unsqueeze(-1)
        parts.append(slot_rms)
        return torch.cat(parts, dim=-1)

    def _build_tse_extra_conditions(self, uss_out, enrollment):
        out = {}
        for key in _TSE_EXTRA_CONDITION_KEYS:
            if key not in uss_out or not torch.is_tensor(uss_out[key]):
                continue
            value = self._match_condition_slots(uss_out[key], enrollment.shape[1], enrollment.device, enrollment.dtype)
            out[key] = _grad_scale(value, self.tse_to_uss_condition_grad_scale).to(self.tse_device)
        return out

    def _normalize_temporal_conditioning(self, condition, n_sources):
        if condition is None:
            return None
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)
        if condition.dim() != 3:
            raise ValueError(f"temporal conditioning must be [B,T] or [B,S,T], got {tuple(condition.shape)}")
        if condition.shape[1] == 1 and n_sources != 1:
            condition = condition.expand(-1, n_sources, -1)
        if condition.shape[1] < n_sources:
            pad = condition.new_zeros(condition.shape[0], n_sources - condition.shape[1], condition.shape[-1])
            condition = torch.cat([condition, pad], dim=1)
        elif condition.shape[1] > n_sources:
            condition = condition[:, :n_sources]
        return condition

    def _build_temporal_conditioning(self, uss_out, sc_out, batch_size, n_slots):
        source = self.temporal_conditioning_source
        if source in {"auto", "sc"} and "activity_probabilities" in sc_out:
            value = sc_out["activity_probabilities"].view(batch_size, n_slots, -1).to(self.tse_device)
            return self._normalize_temporal_conditioning(value, n_slots)
        if source in {"auto", "uss"} and "foreground_activity_logits" in uss_out:
            value = _grad_scale(uss_out["foreground_activity_logits"].sigmoid(), self.tse_to_uss_condition_grad_scale)
            return self._normalize_temporal_conditioning(value.to(self.tse_device), n_slots)
        return None

    def _align_oracle_to_estimate_slots(self, enrollment, batch):
        ref = self._to_tse(batch["foreground_waveform"]).to(dtype=enrollment.dtype)
        label = self._oracle_label_vector(batch, device=self.tse_device, dtype=enrollment.dtype)
        active = (~batch["is_silence"].to(device=self.tse_device, dtype=torch.bool)) & infer_active_mask_from_label(
            label
        )
        span_sec = batch.get("foreground_span_sec", batch.get("span_sec"))
        if span_sec is not None:
            span_sec = span_sec.to(device=self.tse_device, dtype=enrollment.dtype)
        bsz, n_est = enrollment.shape[:2]
        aligned_waveform = ref.new_zeros(bsz, n_est, *ref.shape[2:])
        aligned_label = label.new_zeros(bsz, n_est, label.shape[-1])
        aligned_span = None if span_sec is None else span_sec.new_full((bsz, n_est, 2), -1.0)
        matched_mask = torch.zeros(bsz, n_est, device=self.tse_device, dtype=torch.bool)
        match_score = enrollment.new_full((bsz, n_est), float("nan"))
        quality_weight = enrollment.new_zeros(bsz, n_est)
        with torch.no_grad():
            scores = pairwise_match_score(enrollment.detach(), ref.detach(), metric=self.match_metric).to(
                self.tse_device
            )
            energy_db = _energy_db_by_slot(enrollment.detach()).to(self.tse_device)
            for b in range(bsz):
                assignment = _best_ref_to_est(
                    scores[b], torch.nonzero(active[b], as_tuple=False).flatten().tolist(), n_est
                )
                for ref_idx, est_idx in assignment.items():
                    score = scores[b, ref_idx, est_idx]
                    energy = float(energy_db[b, est_idx].item())
                    second, margin = second_best_and_margin(scores[b], ref_idx, est_idx)
                    quality, weight, valid = quality_and_weight(
                        score=float(score.item()),
                        margin=margin,
                        energy_db=energy,
                        min_match_score=self.min_tse_match_score,
                        min_match_margin=self.min_match_margin,
                        min_energy_db=self.min_estimate_energy_db,
                        clean_match_score=self.clean_match_score,
                        clean_match_margin=self.clean_match_margin,
                        uncertain_weight=self.uncertain_weight,
                    )
                    use_match = valid and (quality != "uncertain" or self.use_uncertain_matches)
                    if self.tse_use_match_quality_weight and not use_match:
                        continue
                    if score < self.min_tse_match_score or energy_db[b, est_idx] < self.min_estimate_energy_db:
                        continue
                    quality_weight[b, est_idx] = float(weight) if use_match else 1.0
                    aligned_waveform[b, est_idx] = ref[b, ref_idx]
                    aligned_label[b, est_idx] = label[b, ref_idx]
                    if aligned_span is not None and span_sec is not None:
                        aligned_span[b, est_idx] = span_sec[b, ref_idx]
                    matched_mask[b, est_idx] = True
                    match_score[b, est_idx] = score
        return {
            "waveform": aligned_waveform,
            "label_vector": aligned_label,
            "active_mask": matched_mask,
            "span_sec": aligned_span,
            "match_score": match_score,
            "estimate_energy_db": energy_db,
            "quality_weight": quality_weight,
        }

    def _class_match_mask(self, sc_label, oracle_label):
        sc_active = sc_label.abs().sum(dim=-1) > 0
        oracle_active = oracle_label.abs().sum(dim=-1) > 0
        return sc_active & oracle_active & (sc_label.argmax(dim=-1) == oracle_label.argmax(dim=-1))

    def _build_tse_input_and_target(self, uss_out, sc_out, batch):
        enrollment_src = uss_out[self.uss_output_key]
        if enrollment_src.dim() != 4:
            raise ValueError(f"USS enrollment must be [B,S,C,T], got {tuple(enrollment_src.shape)}")
        bsz, n_slots = enrollment_src.shape[:2]
        mixture = self._to_tse(batch["mixture"])
        enrollment = _grad_scale(enrollment_src, self.tse_to_uss_enrollment_grad_scale).to(self.tse_device)
        if enrollment.shape[-1] != mixture.shape[-1]:
            enrollment = F.interpolate(
                enrollment.flatten(0, 1),
                size=mixture.shape[-1],
                mode="linear",
                align_corners=False,
            ).view(bsz, n_slots, enrollment.shape[2], mixture.shape[-1])
        tse_label, sc_active, sc_soft = self._build_tse_label_from_sc(sc_out, bsz, n_slots)
        input_dict = {"mixture": mixture, "enrollment": enrollment, "label_vector": tse_label.to(self.tse_device)}
        query_condition = self._build_query_condition(uss_out, enrollment_src)
        if query_condition is not None:
            input_dict["query_condition"] = _grad_scale(query_condition, self.tse_to_uss_condition_grad_scale).to(
                self.tse_device
            )
        input_dict.update(self._build_tse_extra_conditions(uss_out, enrollment_src))
        temporal_conditioning = self._build_temporal_conditioning(uss_out, sc_out, bsz, n_slots)
        if temporal_conditioning is not None:
            input_dict["temporal_conditioning"] = temporal_conditioning
        aligned = self._align_oracle_to_estimate_slots(enrollment.detach(), batch)
        active_mask = aligned["active_mask"].clone()
        sc_label = tse_label.detach().to(self.tse_device)
        sc_active = sc_active.to(self.tse_device)
        class_match = self._class_match_mask(sc_label, aligned["label_vector"])
        if self.require_sc_active_for_tse_loss:
            active_mask = active_mask & sc_active
        if self.require_sc_class_match_for_tse_loss:
            active_mask = active_mask & class_match
        if self.tse_use_match_quality_weight:
            active_mask = active_mask & aligned["quality_weight"].gt(0)
        target = {
            "waveform": aligned["waveform"].clone(),
            "label_vector": aligned["label_vector"].clone(),
            "active_mask": active_mask,
            "sample_weight": aligned["quality_weight"].clone()
            if self.tse_use_match_quality_weight
            else torch.ones_like(active_mask, dtype=aligned["waveform"].dtype),
        }
        target["waveform"][~active_mask] = 0.0
        target["label_vector"][~active_mask] = 0.0
        if aligned["span_sec"] is not None:
            target["span_sec"] = aligned["span_sec"].clone()
            target["span_sec"][~active_mask] = -1.0
        finite_scores = torch.isfinite(aligned["match_score"])
        sc_soft_tse = sc_soft.to(self.tse_device)
        diagnostics = {
            "tse_matched_slots": active_mask.float().sum(dim=1).mean(),
            "tse_raw_matched_slots": aligned["active_mask"].float().sum(dim=1).mean(),
            "tse_sc_active_rate": sc_active.float().mean(),
            "tse_sc_class_match_rate": class_match[aligned["active_mask"]].float().mean()
            if aligned["active_mask"].any()
            else mixture.new_zeros(()),
            "tse_estimate_energy_db": aligned["estimate_energy_db"].mean(),
            "tse_match_score": aligned["match_score"][finite_scores].mean()
            if finite_scores.any()
            else mixture.new_zeros(()),
            "tse_label_entropy": (-(sc_soft_tse.clamp_min(1e-8).log() * sc_soft_tse).sum(dim=-1)).mean(),
            "tse_match_quality_weight": aligned["quality_weight"][active_mask].mean()
            if active_mask.any()
            else mixture.new_zeros(()),
        }
        return input_dict, target, diagnostics

    def _uss_sc_consistency_loss(self, uss_out, sc_out, sample_weight, is_silence):
        if self.lambda_uss_sc_consistency <= 0.0 or "class_logits" not in uss_out:
            return uss_out[self.uss_output_key].new_zeros(())
        logits = sc_out.get("plain_logits", sc_out.get("logits"))
        if logits is None:
            return uss_out["class_logits"].new_zeros(())
        bsz, n_slots = uss_out["class_logits"].shape[:2]
        t = max(self.consistency_temperature, 1e-6)
        uss_logits = uss_out["class_logits"].reshape(bsz * n_slots, -1).to(self.sc_device)
        sc_probs = F.softmax(logits.detach().float() / t, dim=-1)
        if uss_logits.shape[-1] != sc_probs.shape[-1]:
            dim = min(uss_logits.shape[-1], sc_probs.shape[-1])
            uss_logits = uss_logits[:, :dim]
            sc_probs = sc_probs[:, :dim]
        weights = sample_weight.reshape(-1).to(self.sc_device, dtype=uss_logits.dtype)
        active = (~is_silence).reshape(-1).to(self.sc_device, dtype=torch.bool)
        weights = weights * active.to(dtype=weights.dtype)
        if weights.sum() <= 0:
            return uss_logits.new_zeros(())
        student = F.log_softmax(uss_logits.float() / t, dim=-1)
        kl_each = F.kl_div(student, sc_probs, reduction="none").sum(dim=-1) * (t * t)
        return (kl_each * weights).sum() / weights.sum().clamp_min(1.0)

    def _sc_tse_forward_and_loss(self, tse_out, target_dict, is_training):
        if self.lambda_sc_tse <= 0.0:
            zero = tse_out["waveform"].new_zeros(())
            return zero, {}, {}
        waveform = _grad_scale(tse_out["waveform"], self.sc_tse_to_tse_grad_scale)
        bsz, n_slots, channels, samples = waveform.shape
        flat_waveform = waveform.reshape(bsz * n_slots, channels, samples).to(self.sc_device)
        label_vector = target_dict["label_vector"].to(self.sc_device)
        active_mask = target_dict["active_mask"].to(self.sc_device, dtype=torch.bool) & infer_active_mask_from_label(
            label_vector
        )
        class_index = torch.argmax(label_vector, dim=-1).long()
        class_index = torch.where(active_mask, class_index, torch.zeros_like(class_index))
        is_silence = ~active_mask
        sample_weight = torch.where(
            active_mask,
            torch.full(
                active_mask.shape, self.sc_tse_active_sample_weight, device=self.sc_device, dtype=flat_waveform.dtype
            ),
            torch.full(
                active_mask.shape, self.sc_tse_silence_sample_weight, device=self.sc_device, dtype=flat_waveform.dtype
            ),
        )
        sc_input = {"waveform": flat_waveform, "class_index": class_index.reshape(-1)}
        sc_target = {
            "class_index": class_index.reshape(-1),
            "is_silence": is_silence.reshape(-1),
            "sample_weight": sample_weight.reshape(-1),
            "current_epoch": self.current_epoch,
            "is_training": is_training,
        }
        if "span_sec" in target_dict:
            span = target_dict["span_sec"].to(self.sc_device, dtype=flat_waveform.dtype).reshape(-1, 2)
            sc_input["span_sec"] = span
            sc_target["span_sec"] = span
        sc_out = self.sc_model(sc_input)
        loss_dict = self.sc_loss_func(sc_out, sc_target)
        logits = sc_out.get("plain_logits", sc_out.get("logits"))
        active_weight = (~sc_target["is_silence"]).to(dtype=logits.dtype) * sc_target["sample_weight"].to(
            dtype=logits.dtype
        )
        top1 = logits.new_zeros(())
        if active_weight.sum() > 0:
            pred = logits.argmax(dim=-1)
            top1 = (
                ((pred == sc_target["class_index"]).to(dtype=logits.dtype) * active_weight).sum()
                / active_weight.sum().clamp_min(1.0)
                * 100.0
            )
        diagnostics = {
            "sc_tse_top1": top1,
            "sc_tse_active_slots": active_mask.to(dtype=logits.dtype).sum(),
            "sc_tse_silence_slots": is_silence.to(dtype=logits.dtype).sum(),
            "sc_tse_sample_weight_mean": sample_weight.mean(),
        }
        return loss_dict["loss"], loss_dict, diagnostics

    def _forward_losses(self, batch, stage, batch_idx=None):
        is_training = stage == "train"
        self.uss_model.train(is_training)
        self.sc_model.train(is_training)
        self.tse_model.train(is_training)
        uss_out = self.uss_model(self._uss_input(batch))
        if self.uss_output_key not in uss_out:
            raise KeyError(f"USS output does not contain '{self.uss_output_key}'")
        uss_loss_dict = self.uss_loss_func(uss_out, self._uss_target(batch))
        sc_uss = self._sc_uss_forward_and_loss(uss_out, batch, is_training=is_training)
        tse_input, tse_target, tse_diag = self._build_tse_input_and_target(uss_out, sc_uss["sc_out"], batch)
        tse_out = self.tse_model(tse_input)
        tse_loss_dict = self.tse_loss_func(tse_out, tse_target)
        loss_tse = tse_loss_dict["loss"]
        if self.tse_use_match_quality_weight:
            weights = tse_target["sample_weight"].to(device=loss_tse.device, dtype=loss_tse.dtype)
            active = tse_target["active_mask"].to(device=loss_tse.device, dtype=torch.bool)
            if active.any():
                tse_quality_scale = (weights * active.to(dtype=weights.dtype)).sum() / active.to(
                    dtype=weights.dtype
                ).sum().clamp_min(1.0)
                loss_tse = loss_tse * tse_quality_scale
            else:
                tse_quality_scale = loss_tse.new_tensor(1.0)
        else:
            tse_quality_scale = loss_tse.new_tensor(1.0)
        loss_consistency = self._uss_sc_consistency_loss(
            uss_out,
            sc_uss["sc_out"],
            sc_uss["targets"]["sample_weight"],
            sc_uss["targets"]["is_silence"],
        )
        loss_sc_tse, sc_tse_loss_dict, sc_tse_diag = self._sc_tse_forward_and_loss(tse_out, tse_target, is_training)
        loss = (
            self.lambda_uss * self._to_loss(uss_loss_dict["loss"])
            + self.lambda_sc_uss * self._to_loss(sc_uss["loss"])
            + self.lambda_tse * self._to_loss(loss_tse)
            + self.lambda_sc_tse * self._to_loss(loss_sc_tse)
            + self.lambda_uss_sc_consistency * self._to_loss(loss_consistency)
        )
        logs = {
            "loss": loss,
            "loss_uss": self._to_loss(uss_loss_dict["loss"]),
            "loss_uss_weighted": self.lambda_uss * self._to_loss(uss_loss_dict["loss"]),
            "loss_sc_uss": self._to_loss(sc_uss["loss"]),
            "loss_sc_uss_weighted": self.lambda_sc_uss * self._to_loss(sc_uss["loss"]),
            "loss_tse": self._to_loss(loss_tse),
            "loss_tse_unweighted": self._to_loss(tse_loss_dict["loss"]),
            "loss_tse_quality_scale": self._to_loss(tse_quality_scale),
            "loss_tse_weighted": self.lambda_tse * self._to_loss(loss_tse),
            "loss_sc_tse": self._to_loss(loss_sc_tse),
            "loss_sc_tse_weighted": self.lambda_sc_tse * self._to_loss(loss_sc_tse),
            "loss_uss_sc_consistency": self._to_loss(loss_consistency),
            "loss_uss_sc_consistency_weighted": self.lambda_uss_sc_consistency * self._to_loss(loss_consistency),
        }
        logs.update({f"uss_{k}": self._to_loss(v) for k, v in uss_loss_dict.items() if torch.is_tensor(v)})
        logs.update({f"sc_uss_{k}": self._to_loss(v) for k, v in sc_uss["loss_dict"].items() if torch.is_tensor(v)})
        logs.update({f"tse_{k}": self._to_loss(v) for k, v in tse_loss_dict.items() if torch.is_tensor(v)})
        logs.update({k: self._to_loss(v) for k, v in sc_uss["diagnostics"].items() if torch.is_tensor(v)})
        logs.update({k: self._to_loss(v) for k, v in tse_diag.items() if torch.is_tensor(v)})
        logs.update({k: self._to_loss(v) for k, v in sc_tse_diag.items() if torch.is_tensor(v)})
        logs.update({f"sc_tse_{k}": self._to_loss(v) for k, v in sc_tse_loss_dict.items() if torch.is_tensor(v)})
        return loss, logs

    def _optimizer_map(self):
        opts = self.optimizers()
        if opts is None:
            return {}
        if not isinstance(opts, (list, tuple)):
            opts = [opts]
        return {name: opt for name, opt in zip(self._optimizer_names, opts)}

    def _should_step_optimizer(self, batch_idx, every):
        return int(batch_idx) % max(1, int(every)) == 0

    def _clip_optimizer(self, optimizer):
        clip_val = getattr(self.trainer, "gradient_clip_val", None) if self.trainer is not None else None
        if clip_val is None or float(clip_val) <= 0.0:
            return
        clip_algorithm = getattr(self.trainer, "gradient_clip_algorithm", None) or "norm"
        self.clip_gradients(
            optimizer,
            gradient_clip_val=float(clip_val),
            gradient_clip_algorithm=clip_algorithm,
        )

    def _scheduler_map(self):
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return {}
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        return {name: scheduler for name, scheduler in zip(self._scheduler_names, schedulers)}

    def _step_schedulers(self, interval, index_value):
        scheduler_map = self._scheduler_map()
        for idx, name in enumerate(self._scheduler_names):
            if idx >= len(self._scheduler_intervals) or self._scheduler_intervals[idx] != interval:
                continue
            frequency = self._scheduler_frequencies[idx] if idx < len(self._scheduler_frequencies) else 1
            if int(index_value) % max(1, int(frequency)) != 0:
                continue
            scheduler = scheduler_map.get(name)
            if scheduler is not None:
                cast(Any, scheduler).step()

    def training_step(self, batch, batch_idx):
        opt_map = self._optimizer_map()
        for opt in opt_map.values():
            opt.zero_grad(set_to_none=True)
        loss, logs = self._forward_losses(batch, "train", batch_idx)
        self.manual_backward(loss)
        if "uss" in opt_map and self._should_step_optimizer(batch_idx, self.uss_update_every):
            self._clip_optimizer(opt_map["uss"])
            opt_map["uss"].step()
        if "sc" in opt_map and self._should_step_optimizer(batch_idx, self.sc_update_every):
            self._clip_optimizer(opt_map["sc"])
            opt_map["sc"].step()
        if "tse" in opt_map and self._should_step_optimizer(batch_idx, self.tse_update_every):
            self._clip_optimizer(opt_map["tse"])
            opt_map["tse"].step()
        self._step_schedulers("step", batch_idx)
        batchsize = batch["mixture"].shape[0]
        self.log_dict(
            {f"step_train/{k}": v.detach() for k, v in logs.items() if torch.is_tensor(v)},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            batch_size=batchsize,
            sync_dist=False,
        )
        self.log_dict(
            {f"epoch_train/{k}": v.detach() for k, v in logs.items() if torch.is_tensor(v)},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=batchsize,
            sync_dist=False,
        )
        for name, opt in opt_map.items():
            if opt.param_groups:
                self.log(f"epoch/lr_{name}", opt.param_groups[0]["lr"], logger=True)
        return loss.detach()

    def on_train_epoch_end(self):
        self._step_schedulers("epoch", self.current_epoch)

    def validation_step(self, batch, batch_idx):
        self.uss_model.eval()
        self.sc_model.eval()
        self.tse_model.eval()
        with torch.no_grad():
            loss, logs = self._forward_losses(batch, "val", batch_idx)
        batchsize = batch["mixture"].shape[0]
        self.log_dict(
            {f"step_val/{k}": v.detach() for k, v in logs.items() if torch.is_tensor(v)},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            batch_size=batchsize,
            sync_dist=False,
        )
        self.log_dict(
            {f"epoch_val/{k}": v.detach() for k, v in logs.items() if torch.is_tensor(v)},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=batchsize,
            sync_dist=False,
        )
        return loss.detach()

    def _build_scheduler(self, optimizer, scheduler_config):
        scheduler_config = deepcopy(scheduler_config)
        interval = scheduler_config.get("interval", "epoch")
        frequency = int(scheduler_config.get("frequency", 1))
        scheduler_cfg = deepcopy(scheduler_config.get("scheduler", scheduler_config))
        main = str(scheduler_cfg.get("main", scheduler_cfg.get("type", ""))).lower()
        if main in {"warmup_cosine", "warmupcosine", "warmupcosinelr", "warmup_cosine_lr"}:
            args = scheduler_cfg.get("args", {})
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=_warmup_cosine_lambda(
                    warmup_steps=args.get("warmup_steps", args.get("warmup_epochs", 0)),
                    max_steps=args.get("max_steps", args.get("max_epochs", 1)),
                    min_lr_scale=args.get("min_lr_scale", 0.1),
                    start_lr_scale=args.get("start_lr_scale", 0.0),
                ),
            )
        else:
            scheduler_cfg.setdefault("args", {})
            scheduler_cfg["args"]["optimizer"] = optimizer
            scheduler = initialize_config(scheduler_cfg)
        return scheduler, interval, frequency

    def _configure_one_optimizer(self, name, model, config, scheduler_config, optimizers, schedulers):
        if config is None:
            for param in model.parameters():
                param.requires_grad = False
            print(f"[USS-SC-TSE joint] {name} model has no optimizer and is frozen.")
            return
        params = [param for param in model.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError(f"No trainable parameters found for {name} optimizer")
        config = deepcopy(config)
        config["args"]["params"] = params
        optimizer = initialize_config(config)
        optimizers.append(optimizer)
        self._optimizer_names.append(name)
        if scheduler_config is not None:
            scheduler, interval, frequency = self._build_scheduler(optimizer, scheduler_config)
            schedulers.append({"scheduler": scheduler, "interval": interval, "frequency": frequency})
            self._scheduler_names.append(name)
            self._scheduler_intervals.append(interval)
            self._scheduler_frequencies.append(frequency)

    def configure_optimizers(self):
        self._optimizer_names = []
        self._scheduler_names = []
        self._scheduler_intervals = []
        self._scheduler_frequencies = []
        optimizers = []
        schedulers = []
        self._configure_one_optimizer(
            "uss", self.uss_model, self.optimizer_uss_config, self.uss_lr_scheduler_config, optimizers, schedulers
        )
        self._configure_one_optimizer(
            "sc", self.sc_model, self.optimizer_sc_config, self.sc_lr_scheduler_config, optimizers, schedulers
        )
        self._configure_one_optimizer(
            "tse", self.tse_model, self.optimizer_tse_config, self.tse_lr_scheduler_config, optimizers, schedulers
        )
        if not optimizers:
            raise RuntimeError("At least one of optimizer_uss, optimizer_sc, or optimizer_tse must be configured")
        if schedulers:
            return optimizers, schedulers
        return optimizers
