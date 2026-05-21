from __future__ import annotations

from typing import Dict, Optional

import lightning.pytorch as pl
import torch

from src.training.lightningmodule.online_teacher_tse import (
    OnlineTeacherTSELightning,
    _freeze_eval,
    _load_model_checkpoint,
)
from src.utils import initialize_config


def _set_requires_grad(model, requires_grad: bool):
    for param in model.parameters():
        param.requires_grad = bool(requires_grad)


class USSScTSEPipelineFinetuneLightning(OnlineTeacherTSELightning):
    """General online USS -> SC -> TSE fine-tuning wrapper.

    ``target_stage="tse"`` trains TSE with frozen USS and SC teachers. This is
    the same data flow as ``OnlineTeacherTSELightning``.

    ``target_stage="sc_after_tse"`` freezes USS and TSE, uses SC once as the
    live label teacher for TSE, then trains SC on the final TSE outputs. This
    matches the deployment order: USS estimates, SC labels, TSE refinement, and
    final SC re-classification.
    """

    def __init__(
        self,
        model: Dict,
        loss: Dict,
        optimizer: Dict,
        uss_model: Dict,
        sc_model: Dict,
        sc_loss: Optional[Dict] = None,
        lr_scheduler: Optional[Dict] = None,
        metric: Optional[Dict] = None,
        target_stage: str = "tse",
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
        sc_active_sample_weight: float = 1.0,
        sc_silence_sample_weight: float = 0.2,
        is_validation: bool = False,
    ):
        pl.LightningModule.__init__(self)
        if target_stage not in {"tse", "sc_after_tse", "sc"}:
            raise ValueError("target_stage must be 'tse' or 'sc_after_tse'")
        self.target_stage = "sc_after_tse" if target_stage == "sc" else target_stage

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

        self.loss_func = initialize_config(loss)
        self.sc_loss_func = initialize_config(sc_loss) if sc_loss is not None else None
        if self.target_stage == "sc_after_tse" and self.sc_loss_func is None:
            raise ValueError("sc_loss is required when target_stage='sc_after_tse'")
        self.metric_func = initialize_config(metric) if metric else None

        self.optimizer_config = optimizer
        trainable_model = self.model if self.target_stage == "tse" else self.sc_model
        self.optimizer_config["args"]["params"] = trainable_model.parameters()
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
        self.sc_active_sample_weight = max(0.0, float(sc_active_sample_weight))
        self.sc_silence_sample_weight = max(0.0, float(sc_silence_sample_weight))
        self.is_validation = bool(is_validation)

        _freeze_eval(self.uss_model)
        if self.target_stage == "tse":
            _freeze_eval(self.sc_model)
            _set_requires_grad(self.model, True)
        else:
            _freeze_eval(self.model)
            _set_requires_grad(self.sc_model, True)

    def train(self, mode=True):
        pl.LightningModule.train(self, mode)
        _freeze_eval(self.uss_model)
        if self.target_stage == "tse":
            _freeze_eval(self.sc_model)
            self.model.train(mode)
        else:
            _freeze_eval(self.model)
            self.sc_model.train(mode)
        return self

    def _current_epoch_value(self):
        try:
            return self.current_epoch
        except (AttributeError, RuntimeError):
            return 0

    def _step_tse(self, batch, stage):
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
            target_dict = second_target_dict
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

    def _run_frozen_tse_pipeline(self, batch):
        _freeze_eval(self.uss_model)
        _freeze_eval(self.model)
        input_dict, target_dict, diagnostics = self._build_teacher_batch(batch)
        first_output = self.model(input_dict)
        final_output = first_output
        final_target = target_dict
        refinement_passes = self._validate_tse_refinement_passes(getattr(self, "tse_refinement_passes", 1))
        if refinement_passes == 2:
            second_input, second_target, second_diagnostics = self._build_second_pass_input(
                input_dict,
                first_output,
                target_dict,
            )
            final_output = self.model(second_input)
            final_target = second_target
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
        return final_output, final_target, diagnostics

    def _sc_target_from_tse_target(self, target_dict, device, dtype, is_training):
        label_vector = target_dict["label_vector"].to(device=device)
        active_mask = target_dict["active_mask"].to(device=device, dtype=torch.bool)
        label_active = label_vector.abs().sum(dim=-1) > 0
        active_mask = active_mask & label_active

        class_index = torch.argmax(label_vector, dim=-1).long()
        class_index = torch.where(active_mask, class_index, torch.zeros_like(class_index))
        is_silence = ~active_mask
        sample_weight = torch.where(
            active_mask,
            torch.full(active_mask.shape, self.sc_active_sample_weight, device=device, dtype=dtype),
            torch.full(active_mask.shape, self.sc_silence_sample_weight, device=device, dtype=dtype),
        )

        out = {
            "class_index": class_index.reshape(-1),
            "is_silence": is_silence.reshape(-1),
            "sample_weight": sample_weight.reshape(-1),
            "current_epoch": self._current_epoch_value(),
            "is_training": is_training,
        }
        if "span_sec" in target_dict:
            out["span_sec"] = target_dict["span_sec"].to(device=device, dtype=dtype).reshape(-1, 2)
        return out

    def _step_sc_after_tse(self, batch, stage):
        teacher_sc_was_training = self.sc_model.training
        self.sc_model.eval()
        with torch.no_grad():
            tse_out, target_dict, diagnostics = self._run_frozen_tse_pipeline(batch)
            if "waveform" not in tse_out:
                raise KeyError("TSE output does not contain 'waveform'")
            waveform = tse_out["waveform"].detach()
        self.sc_model.train(stage == "train" and teacher_sc_was_training)

        batch_size, n_sources, channels, samples = waveform.shape
        waveform = waveform.reshape(batch_size * n_sources, channels, samples)
        class_target = self._sc_target_from_tse_target(
            target_dict,
            device=waveform.device,
            dtype=waveform.dtype,
            is_training=stage == "train",
        )
        sc_input = {
            "waveform": waveform,
            "class_index": class_target["class_index"],
        }
        if "span_sec" in class_target:
            sc_input["span_sec"] = class_target["span_sec"]

        sc_out = self.sc_model(sc_input)
        loss_dict = self.sc_loss_func(sc_out, class_target)
        logits = sc_out.get("plain_logits", sc_out.get("logits"))
        active = ~class_target["is_silence"]
        active_weight = class_target["sample_weight"] * active.to(dtype=class_target["sample_weight"].dtype)
        top1 = logits.new_zeros(())
        if active_weight.sum() > 0:
            pred = logits.argmax(dim=-1)
            top1 = (
                ((pred == class_target["class_index"]).to(dtype=logits.dtype) * active_weight).sum()
                / active_weight.sum().clamp_min(1.0)
                * 100.0
            )

        return {
            **loss_dict,
            **diagnostics,
            "pipeline_sc_top1": top1,
            "pipeline_sc_active_slots": active.to(dtype=logits.dtype).sum(),
            "pipeline_sc_silence_slots": class_target["is_silence"].to(dtype=logits.dtype).sum(),
            "pipeline_sc_sample_weight_mean": class_target["sample_weight"].mean(),
        }

    def _step(self, batch, stage):
        if self.target_stage == "tse":
            return self._step_tse(batch, stage)
        return self._step_sc_after_tse(batch, stage)

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
