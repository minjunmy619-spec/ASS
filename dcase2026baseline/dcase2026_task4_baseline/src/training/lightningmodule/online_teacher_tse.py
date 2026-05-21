from __future__ import annotations

from itertools import permutations
from typing import Dict, Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from src.tools.estimated_source_matching import pairwise_match_score
from src.training.loss.class_aware_pit import infer_active_mask_from_label
from src.utils import initialize_config


def _strip_lightning_prefix(state_dict):
    out = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and key.startswith("model."):
            key = key[len("model.") :]
        out[key] = value
    return out


def _select_prefixed_model_state(state_dict, model, preferred_prefixes=()):
    model_state = model.state_dict()
    if set(state_dict.keys()) == set(model_state.keys()):
        return state_dict

    for prefix in preferred_prefixes:
        stripped = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
        if stripped and any(key in model_state for key in stripped):
            return stripped

    exact_matches = {key: value for key, value in state_dict.items() if key in model_state}
    if exact_matches:
        return exact_matches

    one_model_key = next(iter(model_state.keys()))
    suffix_matches = [
        key for key in state_dict
        if isinstance(key, str) and key.endswith(one_model_key)
    ]
    if suffix_matches:
        prefix = suffix_matches[0][:-len(one_model_key)]
        return {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if isinstance(key, str) and key.startswith(prefix)
        }
    return state_dict


def _load_model_checkpoint(model, checkpoint_path, strict=True, name="model", allowed_missing_prefixes=()):
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = _strip_lightning_prefix(state)
    preferred_prefixes = {
        "uss": ("uss_model.",),
        "sc": ("sc_model.",),
        "tse": ("tse_model.", "model."),
    }.get(name, ())
    state = _select_prefixed_model_state(state, model, preferred_prefixes=preferred_prefixes)
    if strict:
        model.load_state_dict(state, strict=True)
        return

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[OnlineTeacherTSE] {name} checkpoint {checkpoint_path}: missing={len(missing)}, unexpected={len(unexpected)}")
    disallowed_missing = [
        key for key in missing
        if not key.startswith(tuple(allowed_missing_prefixes))
    ]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            f"{name} checkpoint/config mismatch: missing={disallowed_missing[:20]}, unexpected={unexpected[:20]}"
        )


def _freeze_eval(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def _energy_db_by_slot(waveform, eps=1e-12):
    flat = waveform.flatten(start_dim=2).float()
    rms = torch.sqrt(flat.pow(2).mean(dim=-1) + eps)
    return 20.0 * torch.log10(rms.clamp_min(1e-8))


def _best_ref_to_est(scores, active_refs, n_est):
    if not active_refs:
        return {}
    n_match = min(len(active_refs), n_est)
    refs = list(active_refs[:n_match])
    best_score = None
    best_perm = None
    for perm in permutations(range(n_est), n_match):
        vals = torch.stack([scores[ref_idx, est_idx] for ref_idx, est_idx in zip(refs, perm)])
        score = vals.mean()
        if best_score is None or score > best_score:
            best_score = score
            best_perm = perm
    return {int(ref_idx): int(est_idx) for ref_idx, est_idx in zip(refs, best_perm)}


class OnlineTeacherTSELightning(pl.LightningModule):
    """Train TSE with frozen online USS and SC teachers.

    Contract:
      dataset batch:
        mixture      [B,C,T]       dynamically synthesized soundscape
        waveform     [B,S,1,T]     oracle clean target sources
        label_vector [B,S,K]       oracle target labels
        active_mask  [B,S]         oracle active slots
        span_sec     [B,S,2]       optional oracle event spans

      frozen teachers:
        USS(mixture) -> foreground_waveform [B,S,1,T] plus optional condition keys
        SC(USS slot waveform).predict(...) -> slot label_vector [B,S,K]

      trainable TSE input:
        mixture, enrollment=USS foreground_waveform, label_vector=SC predicted labels,
        optional query_condition from USS outputs, optional temporal_conditioning.
        With ``tse_refinement_passes=2``, pass 2 uses detached pass-1 TSE output
        as enrollment and re-runs frozen SC for the pass-2 query label.

      loss target:
        oracle waveform/label/span aligned into USS estimate-slot order.  The SC
        label is never used as the PIT target label; it is only the query given
        to TSE.
    """

    def __init__(
        self,
        model: Dict,
        loss: Dict,
        optimizer: Dict,
        uss_model: Dict,
        sc_model: Dict,
        lr_scheduler: Optional[Dict] = None,
        metric: Optional[Dict] = None,
        pretrained_model_ckpt: Optional[str] = None,
        pretrained_model_strict: bool = True,
        uss_pretrained_ckpt: Optional[str] = None,
        sc_pretrained_ckpt: Optional[str] = None,
        uss_pretrained_strict: bool = True,
        sc_pretrained_strict: bool = True,
        uss_output_key: str = "foreground_waveform",
        match_metric: str = "sa_sdr",
        min_match_score: float = -1.0e9,
        min_estimate_energy_db: float = -80.0,
        require_sc_active_for_loss: bool = True,
        require_sc_class_match_for_loss: bool = False,
        label_source: str = "sc",
        query_condition_enabled: bool = True,
        query_condition_key: Optional[str] = None,
        temporal_conditioning_source: str = "auto",
        tse_refinement_passes: int = 1,
        second_pass_loss_weight: float = 1.0,
        second_pass_detach_enrollment: bool = True,
        is_validation: bool = False,
    ):
        super().__init__()
        self.model = initialize_config(model)
        self.uss_model = initialize_config(uss_model)
        self.sc_model = initialize_config(sc_model)
        _load_model_checkpoint(
            self.model,
            pretrained_model_ckpt,
            strict=pretrained_model_strict,
            name="tse",
            allowed_missing_prefixes=(
                "query_conditioner.",
                "bridge_to_label.",
                "activity_head.",
                "temporal_conditioner.",
            ),
        )
        _load_model_checkpoint(self.uss_model, uss_pretrained_ckpt, strict=uss_pretrained_strict, name="uss")
        _load_model_checkpoint(self.sc_model, sc_pretrained_ckpt, strict=sc_pretrained_strict, name="sc")

        _freeze_eval(self.uss_model)
        _freeze_eval(self.sc_model)

        self.loss_func = initialize_config(loss)
        self.metric_func = initialize_config(metric) if metric else None
        self.optimizer_config = optimizer
        self.optimizer_config["args"]["params"] = self.model.parameters()
        self.optimizer = initialize_config(self.optimizer_config)

        self.lr_scheduler_config = lr_scheduler
        if self.lr_scheduler_config:
            self.lr_scheduler_config["scheduler"]["args"]["optimizer"] = self.optimizer
            self.scheduler = initialize_config(self.lr_scheduler_config["scheduler"])

        self.uss_output_key = uss_output_key
        self.match_metric = match_metric
        self.min_match_score = float(min_match_score)
        self.min_estimate_energy_db = float(min_estimate_energy_db)
        self.require_sc_active_for_loss = bool(require_sc_active_for_loss)
        self.require_sc_class_match_for_loss = bool(require_sc_class_match_for_loss)
        self.label_source = label_source
        self.query_condition_enabled = bool(query_condition_enabled)
        self.query_condition_key = query_condition_key
        self.temporal_conditioning_source = temporal_conditioning_source
        self.tse_refinement_passes = self._validate_tse_refinement_passes(tse_refinement_passes)
        self.second_pass_loss_weight = float(second_pass_loss_weight)
        self.second_pass_detach_enrollment = bool(second_pass_detach_enrollment)
        self.is_validation = bool(is_validation)

    def _validate_tse_refinement_passes(self, passes):
        passes = int(passes)
        if passes not in (1, 2):
            raise ValueError("tse_refinement_passes must be 1 or 2")
        return passes

    def train(self, mode=True):
        super().train(mode)
        _freeze_eval(self.uss_model)
        _freeze_eval(self.sc_model)
        return self

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

        condition_keys = (self.query_condition_key,) if self.query_condition_key else (
            "tse_condition",
            "query_condition",
            "bridge_condition",
            "proposal_condition",
        )
        for key in condition_keys:
            if key and key in uss_out:
                return self._match_condition_slots(uss_out[key], n_sources, device, dtype)

        parts = []
        if "class_logits" in uss_out:
            parts.append(self._match_condition_slots(uss_out["class_logits"].softmax(dim=-1), n_sources, device, dtype))
        if "silence_logits" in uss_out:
            parts.append(self._match_condition_slots(uss_out["silence_logits"].sigmoid().unsqueeze(-1), n_sources, device, dtype))
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

    def _normalize_temporal_conditioning(self, condition, n_sources=None):
        if condition is None:
            return None
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)
        if condition.dim() != 3:
            raise ValueError(f"temporal conditioning must be [B,T] or [B,S,T], got {tuple(condition.shape)}")
        if n_sources is not None and condition.shape[1] == 1 and n_sources != 1:
            condition = condition.expand(-1, n_sources, -1)
        if n_sources is not None and condition.shape[1] != n_sources:
            raise ValueError("temporal conditioning source dimension does not match enrollment slots")
        return condition

    def _combine_temporal_conditioning(self, *conditions):
        conditions = [condition for condition in conditions if condition is not None]
        if not conditions:
            return None
        n_sources = None
        for condition in conditions:
            if condition.dim() == 3 and condition.shape[1] != 1:
                n_sources = condition.shape[1]
                break
        out = self._normalize_temporal_conditioning(conditions[0], n_sources=n_sources)
        for condition in conditions[1:]:
            condition = self._normalize_temporal_conditioning(condition, n_sources=out.shape[1])
            if condition.shape[-1] != out.shape[-1]:
                condition = F.interpolate(
                    condition.reshape(-1, 1, condition.shape[-1]),
                    size=out.shape[-1],
                    mode="linear",
                    align_corners=False,
                ).view(condition.shape[0], condition.shape[1], out.shape[-1])
            out = torch.maximum(out, condition)
        return out

    def _mask_target_for_sc_output(self, target_dict, sc_out):
        active_mask = target_dict["active_mask"].clone()
        sc_active = sc_out["label_vector"].abs().sum(dim=-1) > 0
        if self.require_sc_active_for_loss:
            active_mask = active_mask & sc_active
        class_match = self._class_match_mask(sc_out["label_vector"], target_dict["label_vector"])
        if self.require_sc_class_match_for_loss:
            active_mask = active_mask & class_match

        masked_target = {
            "waveform": target_dict["waveform"].clone(),
            "label_vector": target_dict["label_vector"].clone(),
            "active_mask": active_mask,
        }
        masked_target["waveform"][~active_mask] = 0.0
        masked_target["label_vector"][~active_mask] = 0.0
        if "span_sec" in target_dict:
            masked_target["span_sec"] = target_dict["span_sec"].clone()
            masked_target["span_sec"][~active_mask] = -1.0
        diagnostics = {
            "active_rate": sc_active.float().mean(),
            "class_match_rate": class_match[target_dict["active_mask"]].float().mean()
            if target_dict["active_mask"].any()
            else target_dict["waveform"].new_zeros(()),
        }
        return masked_target, diagnostics

    def _build_second_pass_input(self, first_input, first_output, target_dict):
        if "waveform" not in first_output:
            raise KeyError("TSE output does not contain 'waveform'")
        enrollment = first_output["waveform"]
        if getattr(self, "second_pass_detach_enrollment", True):
            enrollment = enrollment.detach()
        sc_enrollment = enrollment.detach()
        with torch.no_grad():
            sc_out = self._teacher_sc_predict(sc_enrollment)

        if self.label_source == "sc":
            tse_label = sc_out["label_vector"].detach()
        elif self.label_source == "oracle":
            tse_label = first_input["label_vector"].detach()
        else:
            raise ValueError("label_source must be 'sc' or 'oracle'")

        input_dict = {
            "mixture": first_input["mixture"],
            "enrollment": enrollment,
            "label_vector": tse_label,
        }
        if "query_condition" in first_input:
            input_dict["query_condition"] = first_input["query_condition"].detach()

        temporal_parts = []
        temporal_conditioning_source = getattr(self, "temporal_conditioning_source", "auto")
        if temporal_conditioning_source in {"auto", "uss"} and "temporal_conditioning" in first_input:
            temporal_parts.append(first_input["temporal_conditioning"].detach())
        if temporal_conditioning_source in {"auto", "tse"} and "activity_logits" in first_output:
            temporal_parts.append(first_output["activity_logits"].sigmoid().detach())
        if temporal_conditioning_source in {"auto", "sc"} and "activity_probabilities" in sc_out:
            temporal_parts.append(sc_out["activity_probabilities"].detach())
        temporal_conditioning = self._combine_temporal_conditioning(*temporal_parts)
        if temporal_conditioning is not None:
            input_dict["temporal_conditioning"] = temporal_conditioning

        second_target_dict, target_diagnostics = self._mask_target_for_sc_output(target_dict, sc_out)
        diagnostics = {
            "teacher_second_pass_sc_active_rate": (sc_out["label_vector"].abs().sum(dim=-1) > 0).float().mean(),
            "teacher_second_pass_sc_class_match_rate": target_diagnostics["class_match_rate"],
            "teacher_second_pass_matched_slots": second_target_dict["active_mask"].float().sum(dim=1).mean(),
        }
        return input_dict, second_target_dict, diagnostics

    def _teacher_sc_predict(self, enrollment):
        batch_size, n_sources, _, samples = enrollment.shape
        flat = enrollment[:, :, 0].reshape(batch_size * n_sources, samples)
        if hasattr(self.sc_model, "predict"):
            out = self.sc_model.predict({"waveform": flat})
        else:
            raw = self.sc_model({"waveform": flat})
            logits = raw.get("plain_logits", raw.get("logits"))
            probs = torch.softmax(logits, dim=-1)
            values, indices = torch.max(probs, dim=-1)
            out = {
                "label_vector": F.one_hot(indices, num_classes=logits.shape[-1]).float(),
                "raw_label_vector": F.one_hot(indices, num_classes=logits.shape[-1]).float(),
                "class_indices": indices,
                "probabilities": values,
                "energy": -torch.logsumexp(logits, dim=-1),
                "silence": torch.zeros_like(indices, dtype=torch.bool),
            }
        out = dict(out)
        out["label_vector"] = out["label_vector"].view(batch_size, n_sources, -1)
        if "raw_label_vector" in out:
            out["raw_label_vector"] = out["raw_label_vector"].view(batch_size, n_sources, -1)
        for key in ("probabilities", "energy", "silence", "class_indices"):
            if key in out:
                out[key] = out[key].view(batch_size, n_sources)
        if "activity_probabilities" in out:
            out["activity_probabilities"] = out["activity_probabilities"].view(batch_size, n_sources, -1)
        return out

    def _align_oracle_to_estimate_slots(self, enrollment, target):
        ref = target["waveform"].to(device=enrollment.device, dtype=enrollment.dtype)
        label = target["label_vector"].to(device=enrollment.device, dtype=enrollment.dtype)
        active = target.get("active_mask", infer_active_mask_from_label(label)).to(device=enrollment.device, dtype=torch.bool)
        span_sec = target.get("span_sec")
        if span_sec is not None:
            span_sec = span_sec.to(device=enrollment.device, dtype=enrollment.dtype)

        batch_size, n_est = enrollment.shape[:2]
        label_dim = label.shape[-1]
        aligned_waveform = ref.new_zeros(batch_size, n_est, *ref.shape[2:])
        aligned_label = label.new_zeros(batch_size, n_est, label_dim)
        aligned_span = None
        if span_sec is not None:
            aligned_span = span_sec.new_full((batch_size, n_est, 2), -1.0)
        matched_mask = torch.zeros(batch_size, n_est, device=enrollment.device, dtype=torch.bool)
        match_score = enrollment.new_full((batch_size, n_est), float("nan"))

        scores = pairwise_match_score(enrollment.detach(), ref.detach(), metric=self.match_metric).to(enrollment.device)
        energy_db = _energy_db_by_slot(enrollment.detach())

        for batch_idx in range(batch_size):
            active_refs = torch.nonzero(active[batch_idx], as_tuple=False).flatten().tolist()
            assignment = _best_ref_to_est(scores[batch_idx], active_refs, n_est)
            for ref_idx, est_idx in assignment.items():
                score = scores[batch_idx, ref_idx, est_idx]
                if score < self.min_match_score:
                    continue
                if energy_db[batch_idx, est_idx] < self.min_estimate_energy_db:
                    continue
                aligned_waveform[batch_idx, est_idx] = ref[batch_idx, ref_idx]
                aligned_label[batch_idx, est_idx] = label[batch_idx, ref_idx]
                if aligned_span is not None:
                    aligned_span[batch_idx, est_idx] = span_sec[batch_idx, ref_idx]
                matched_mask[batch_idx, est_idx] = True
                match_score[batch_idx, est_idx] = score

        return {
            "waveform": aligned_waveform,
            "label_vector": aligned_label,
            "active_mask": matched_mask,
            "span_sec": aligned_span,
            "match_score": match_score,
            "estimate_energy_db": energy_db,
        }

    def _class_match_mask(self, sc_label, oracle_label):
        sc_active = sc_label.abs().sum(dim=-1) > 0
        oracle_active = oracle_label.abs().sum(dim=-1) > 0
        sc_idx = torch.argmax(sc_label, dim=-1)
        oracle_idx = torch.argmax(oracle_label, dim=-1)
        return sc_active & oracle_active & (sc_idx == oracle_idx)

    def _build_teacher_batch(self, batch):
        mixture = batch["mixture"]
        target = {
            "waveform": batch["waveform"],
            "label_vector": batch["label_vector"],
            "active_mask": batch["active_mask"],
        }
        if "span_sec" in batch:
            target["span_sec"] = batch["span_sec"]

        with torch.no_grad():
            uss_out = self.uss_model({"mixture": mixture})
            if self.uss_output_key not in uss_out:
                raise KeyError(f"USS output does not contain '{self.uss_output_key}'")
            enrollment = uss_out[self.uss_output_key].detach()
            if enrollment.dim() != 4:
                raise ValueError(f"USS enrollment must have shape [B,S,C,T], got {tuple(enrollment.shape)}")
            if enrollment.shape[-1] != mixture.shape[-1]:
                enrollment = F.interpolate(
                    enrollment.flatten(0, 1),
                    size=mixture.shape[-1],
                    mode="linear",
                    align_corners=False,
                ).view(enrollment.shape[0], enrollment.shape[1], enrollment.shape[2], mixture.shape[-1])
            sc_out = self._teacher_sc_predict(enrollment)
            query_condition = self._build_query_condition(uss_out, enrollment)

        aligned = self._align_oracle_to_estimate_slots(enrollment, target)
        active_mask = aligned["active_mask"].clone()
        sc_active = sc_out["label_vector"].abs().sum(dim=-1) > 0
        if self.require_sc_active_for_loss:
            active_mask = active_mask & sc_active
        class_match = self._class_match_mask(sc_out["label_vector"], aligned["label_vector"])
        if self.require_sc_class_match_for_loss:
            active_mask = active_mask & class_match

        aligned["waveform"] = aligned["waveform"].clone()
        aligned["label_vector"] = aligned["label_vector"].clone()
        aligned["waveform"][~active_mask] = 0.0
        aligned["label_vector"][~active_mask] = 0.0
        if aligned["span_sec"] is not None:
            aligned["span_sec"] = aligned["span_sec"].clone()
            aligned["span_sec"][~active_mask] = -1.0
        aligned["active_mask"] = active_mask

        if self.label_source == "sc":
            tse_label = sc_out["label_vector"].detach()
        elif self.label_source == "oracle":
            tse_label = aligned["label_vector"].detach()
        else:
            raise ValueError("label_source must be 'sc' or 'oracle'")

        input_dict = {
            "mixture": mixture,
            "enrollment": enrollment.detach(),
            "label_vector": tse_label,
        }
        if query_condition is not None:
            input_dict["query_condition"] = query_condition.detach()

        temporal_conditioning = None
        if self.temporal_conditioning_source in {"auto", "sc"} and "activity_probabilities" in sc_out:
            temporal_conditioning = sc_out["activity_probabilities"]
        elif self.temporal_conditioning_source in {"auto", "uss"} and "foreground_activity_logits" in uss_out:
            temporal_conditioning = uss_out["foreground_activity_logits"].sigmoid()
        if temporal_conditioning is not None:
            input_dict["temporal_conditioning"] = temporal_conditioning.detach()

        target_dict = {
            "waveform": aligned["waveform"],
            "label_vector": aligned["label_vector"],
            "active_mask": aligned["active_mask"],
        }
        if aligned["span_sec"] is not None:
            target_dict["span_sec"] = aligned["span_sec"]

        diagnostics = {
            "teacher_matched_slots": aligned["active_mask"].float().sum(dim=1).mean(),
            "teacher_sc_active_rate": sc_active.float().mean(),
            "teacher_sc_class_match_rate": class_match[aligned["active_mask"]].float().mean()
            if aligned["active_mask"].any()
            else mixture.new_zeros(()),
            "teacher_estimate_energy_db": aligned["estimate_energy_db"].mean(),
        }
        finite_scores = torch.isfinite(aligned["match_score"])
        diagnostics["teacher_match_score"] = (
            aligned["match_score"][finite_scores].mean() if finite_scores.any() else mixture.new_zeros(())
        )
        return input_dict, target_dict, diagnostics

    def _step(self, batch, stage):
        self.model.train(stage == "train")
        _freeze_eval(self.uss_model)
        _freeze_eval(self.sc_model)
        input_dict, target_dict, diagnostics = self._build_teacher_batch(batch)
        output_dict = self.model(input_dict)
        loss_dict = self.loss_func(output_dict, target_dict)
        metric_output_dict = output_dict
        refinement_passes = self._validate_tse_refinement_passes(getattr(self, "tse_refinement_passes", 1))
        if refinement_passes == 2:
            second_input, second_target_dict, second_diagnostics = self._build_second_pass_input(
                input_dict,
                output_dict,
                target_dict,
            )
            second_output = self.model(second_input)
            second_loss_dict = self.loss_func(second_output, second_target_dict)
            metric_output_dict = second_output
            loss_dict["first_pass_loss"] = loss_dict["loss"]
            for key, value in second_loss_dict.items():
                loss_dict[f"second_pass_{key}"] = value
            loss_dict["loss"] = (
                loss_dict["loss"]
                + getattr(self, "second_pass_loss_weight", 1.0) * second_loss_dict["loss"]
            )
            diagnostics = {
                **diagnostics,
                **second_diagnostics,
                "teacher_second_pass_enabled": batch["mixture"].new_tensor(1.0),
            }
        else:
            diagnostics = {
                **diagnostics,
                "teacher_second_pass_enabled": batch["mixture"].new_tensor(0.0),
            }
        loss_dict = {**loss_dict, **diagnostics}
        if stage == "val" and self.metric_func:
            metric = self.metric_func(metric_output_dict, target_dict)
            for key, value in metric.items():
                loss_dict[key] = value.mean()
        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self._step(batch, "train")
        batchsize = batch["mixture"].shape[0]
        self.log_dict(
            {f"step_train/{key}": value.detach() for key, value in loss_dict.items() if torch.is_tensor(value)},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            batch_size=batchsize,
            sync_dist=True,
        )
        self.log_dict(
            {f"epoch_train/{key}": value.detach() for key, value in loss_dict.items() if torch.is_tensor(value)},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=batchsize,
            sync_dist=True,
        )
        self.log_dict({"epoch/lr": self.optimizer.param_groups[0]["lr"]})
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self._step(batch, "val")
        batchsize = batch["mixture"].shape[0]
        self.log_dict(
            {f"step_val/{key}": value.detach() for key, value in loss_dict.items() if torch.is_tensor(value)},
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
            batch_size=batchsize,
            sync_dist=True,
        )
        self.log_dict(
            {f"epoch_val/{key}": value.detach() for key, value in loss_dict.items() if torch.is_tensor(value)},
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
            batch_size=batchsize,
            sync_dist=True,
        )
        return loss_dict["loss"]

    def configure_optimizers(self):
        if self.lr_scheduler_config:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler,
                    "interval": self.lr_scheduler_config["interval"],
                    "frequency": self.lr_scheduler_config["frequency"],
                },
            }
        return self.optimizer
