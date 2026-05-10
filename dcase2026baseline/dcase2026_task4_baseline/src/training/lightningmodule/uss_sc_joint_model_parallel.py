from __future__ import annotations

import itertools
from typing import Dict, Optional

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from src.temporal import SILENCE_SPAN_SEC
from src.tools.estimated_source_matching import (
    pairwise_match_score,
    quality_and_weight,
    second_best_and_margin,
    source_energy_db,
)
from src.utils import initialize_config


def _strip_lightning_prefix(state_dict):
    out = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and key.startswith("model."):
            key = key[len("model.") :]
        out[key] = value
    return out


def _load_model_checkpoint(model, checkpoint_path, strict=True):
    if not checkpoint_path:
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state = _strip_lightning_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=bool(strict))
    if missing:
        print(f"[USS-SC joint] missing keys from {checkpoint_path}: {len(missing)}")
    if unexpected:
        print(f"[USS-SC joint] unexpected keys from {checkpoint_path}: {len(unexpected)}")


def _active_indices(is_silence_row):
    return [idx for idx, is_sil in enumerate(is_silence_row.tolist()) if not bool(is_sil)]


def _best_assignment(scores, active_refs, n_est):
    if not active_refs:
        return {}
    n_match = min(len(active_refs), n_est)
    refs = active_refs[:n_match]
    best_perm = None
    best_score = None
    for perm in itertools.permutations(range(n_est), n_match):
        vals = torch.stack([scores[ref_idx, est_idx] for ref_idx, est_idx in zip(refs, perm)])
        score = vals.mean()
        if best_score is None or score > best_score:
            best_score = score
            best_perm = perm
    return {int(est_idx): int(ref_idx) for ref_idx, est_idx in zip(refs, best_perm)}


class USSCSJointModelParallelLightning(pl.LightningModule):
    """Opt-in model-parallel joint fine-tuning for USS + SC.

    USS is placed on ``uss_device`` and SC is placed on ``sc_device``. The SC
    loss is computed on USS separated foreground waveforms. If SC is frozen,
    gradients still flow through the SC network input back to USS; only SC
    parameters are not updated.

    This is model parallelism, not DDP. Use a single Lightning process and set
    trainer ``devices: 1`` / ``strategy: auto`` in the config.
    """

    def __init__(
        self,
        uss_model: Dict,
        sc_model: Dict,
        uss_loss: Dict,
        sc_loss: Dict,
        optimizer_uss: Dict,
        optimizer_sc: Optional[Dict] = None,
        uss_lr_scheduler: Optional[Dict] = None,
        sc_lr_scheduler: Optional[Dict] = None,
        uss_pretrained_ckpt: Optional[str] = None,
        sc_pretrained_ckpt: Optional[str] = None,
        uss_pretrained_strict: bool = True,
        sc_pretrained_strict: bool = True,
        uss_device: str = "cuda:0",
        sc_device: str = "cuda:1",
        freeze_sc: bool = True,
        sc_eval_mode_when_frozen: bool = True,
        lambda_sc: float = 0.05,
        lambda_consistency: float = 0.0,
        consistency_temperature: float = 1.0,
        match_metric: str = "sa_sdr",
        min_match_score: float = -1.0e9,
        min_match_margin: float = -1.0e9,
        min_energy_db: float = -80.0,
        clean_match_score: float = -1.0e9,
        clean_match_margin: float = -1.0e9,
        uncertain_weight: float = 0.35,
        use_uncertain_matches: bool = False,
        sc_update_every: int = 1,
        detach_waveform_for_sc: bool = False,
        is_validation: bool = True,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.uss_model = initialize_config(uss_model)
        self.sc_model = initialize_config(sc_model)
        _load_model_checkpoint(self.uss_model, uss_pretrained_ckpt, strict=uss_pretrained_strict)
        _load_model_checkpoint(self.sc_model, sc_pretrained_ckpt, strict=sc_pretrained_strict)

        self.uss_loss_func = initialize_config(uss_loss)
        self.sc_loss_func = initialize_config(sc_loss)
        self.optimizer_uss_config = optimizer_uss
        self.optimizer_sc_config = optimizer_sc
        self.uss_lr_scheduler_config = uss_lr_scheduler
        self.sc_lr_scheduler_config = sc_lr_scheduler

        self.uss_device_name = uss_device
        self.sc_device_name = sc_device
        self.freeze_sc = bool(freeze_sc)
        self.sc_eval_mode_when_frozen = bool(sc_eval_mode_when_frozen)
        self.lambda_sc = float(lambda_sc)
        self.lambda_consistency = float(lambda_consistency)
        self.consistency_temperature = float(consistency_temperature)
        self.match_metric = match_metric
        self.min_match_score = float(min_match_score)
        self.min_match_margin = float(min_match_margin)
        self.min_energy_db = float(min_energy_db)
        self.clean_match_score = float(clean_match_score)
        self.clean_match_margin = float(clean_match_margin)
        self.uncertain_weight = float(uncertain_weight)
        self.use_uncertain_matches = bool(use_uncertain_matches)
        self.sc_update_every = max(1, int(sc_update_every))
        self.detach_waveform_for_sc = bool(detach_waveform_for_sc)
        self.is_validation = bool(is_validation)

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        # Device placement is manual because USS and SC live on different GPUs.
        return batch

    def setup(self, stage=None):
        self.uss_device = torch.device(self.uss_device_name)
        self.sc_device = torch.device(self.sc_device_name)
        if self.uss_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for uss_device")
        if self.sc_device.type == "cuda" and torch.cuda.device_count() <= self.sc_device.index:
            raise RuntimeError(f"Requested sc_device={self.sc_device}, but only {torch.cuda.device_count()} CUDA devices are visible")
        self.uss_model.to(self.uss_device)
        self.sc_model.to(self.sc_device)
        if self.freeze_sc:
            for param in self.sc_model.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "uss_model"):
            self.uss_model.train(mode)
        if hasattr(self, "sc_model"):
            if self.freeze_sc and self.sc_eval_mode_when_frozen:
                self.sc_model.eval()
            else:
                self.sc_model.train(mode)
        return self

    def _to_uss(self, value):
        return value.to(self.uss_device) if torch.is_tensor(value) else value

    def _to_sc(self, value):
        return value.to(self.sc_device) if torch.is_tensor(value) else value

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

    def _build_slot_targets(self, sep, batch):
        """Assign oracle class/span labels to predicted USS foreground slots."""
        ref = self._to_uss(batch["foreground_waveform"])
        class_index_ref = self._to_uss(batch["class_index"])
        is_silence_ref = self._to_uss(batch["is_silence"]).bool()
        span_ref = self._to_uss(batch["foreground_span_sec"]) if "foreground_span_sec" in batch else None

        bsz, n_est = sep.shape[:2]
        class_idx = torch.zeros(bsz, n_est, dtype=torch.long, device=self.uss_device)
        is_silence = torch.ones(bsz, n_est, dtype=torch.bool, device=self.uss_device)
        sample_weight = torch.zeros(bsz, n_est, dtype=sep.dtype, device=self.uss_device)
        quality_code = torch.zeros(bsz, n_est, dtype=torch.long, device=self.uss_device)
        span_sec = None
        if span_ref is not None:
            span_sec = torch.tensor(SILENCE_SPAN_SEC, dtype=sep.dtype, device=self.uss_device).view(1, 1, 2).expand(bsz, n_est, 2).clone()

        with torch.no_grad():
            scores = pairwise_match_score(sep.detach(), ref.detach(), metric=self.match_metric).to(self.uss_device)
            for b in range(bsz):
                active_refs = _active_indices(is_silence_ref[b])
                est_to_ref = _best_assignment(scores[b], active_refs, n_est)
                for est_idx, ref_idx in est_to_ref.items():
                    score = float(scores[b, ref_idx, est_idx].item())
                    _, margin = second_best_and_margin(scores[b], ref_idx, est_idx)
                    energy = source_energy_db(sep[b, est_idx].detach().cpu())
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
                    if not valid or (quality == "uncertain" and not self.use_uncertain_matches):
                        continue
                    class_idx[b, est_idx] = class_index_ref[b, ref_idx]
                    is_silence[b, est_idx] = False
                    sample_weight[b, est_idx] = float(weight)
                    if span_sec is not None:
                        span_sec[b, est_idx] = span_ref[b, ref_idx]
        return class_idx, is_silence, sample_weight, span_sec, quality_code

    def _sc_forward_and_loss(self, uss_out, batch, is_training: bool):
        sep = uss_out["foreground_waveform"]
        if self.detach_waveform_for_sc:
            sep = sep.detach()
        class_idx, is_silence, sample_weight, span_sec, quality_code = self._build_slot_targets(sep, batch)

        bsz, n_slots, channels, samples = sep.shape
        waveform = sep.reshape(bsz * n_slots, channels, samples).to(self.sc_device)
        class_flat = class_idx.reshape(bsz * n_slots).to(self.sc_device)
        silence_flat = is_silence.reshape(bsz * n_slots).to(self.sc_device)
        weight_flat = sample_weight.reshape(bsz * n_slots).to(self.sc_device)

        sc_input = {"waveform": waveform, "class_index": class_flat}
        sc_target = {
            "class_index": class_flat,
            "is_silence": silence_flat,
            "current_epoch": self.current_epoch,
            "is_training": is_training,
        }
        if span_sec is not None:
            span_flat = span_sec.reshape(bsz * n_slots, 2).to(self.sc_device)
            sc_input["span_sec"] = span_flat
            sc_target["span_sec"] = span_flat
        sc_out = self.sc_model(sc_input)
        sc_loss_dict = self.sc_loss_func(sc_out, sc_target)

        # The configured SC loss is still computed for diagnostics. For the joint
        # USS gradient, use an explicit weighted CE on active matched slots so
        # low-quality/unmatched estimates do not dominate.
        logits = sc_out.get("plain_logits", sc_out.get("logits"))
        ce_all = F.cross_entropy(logits.float(), class_flat, reduction="none")
        active_weight = (~silence_flat).to(dtype=ce_all.dtype) * weight_flat.to(dtype=ce_all.dtype)
        loss_sc_weighted = (ce_all * active_weight).sum() / active_weight.sum().clamp_min(1.0)
        top1 = torch.zeros((), device=self.sc_device, dtype=ce_all.dtype)
        if active_weight.sum() > 0:
            pred = logits.argmax(dim=-1)
            top1 = ((pred == class_flat).float() * active_weight).sum() / active_weight.sum().clamp_min(1.0) * 100.0

        out = {f"sc_{k}": v for k, v in sc_loss_dict.items() if torch.is_tensor(v)}
        out["loss_sc_weighted"] = loss_sc_weighted
        out["sc_joint_top1"] = top1
        out["sc_active_weight_mean"] = active_weight.mean()
        quality_flat = quality_code.reshape(bsz * n_slots).to(self.sc_device)
        out["sc_clean_match_count"] = (quality_flat == 1).to(dtype=ce_all.dtype).sum()
        out["sc_uncertain_match_count"] = (quality_flat == 2).to(dtype=ce_all.dtype).sum()
        out["sc_bad_match_count"] = (quality_flat == 3).to(dtype=ce_all.dtype).sum()
        out["sc_used_match_count"] = active_weight.gt(0).to(dtype=ce_all.dtype).sum()

        if self.lambda_consistency > 0.0 and "class_logits" in uss_out:
            uss_logits = uss_out["class_logits"].reshape(bsz * n_slots, -1).to(self.sc_device)
            t = self.consistency_temperature
            teacher = F.softmax(logits.detach().float() / t, dim=-1)
            student = F.log_softmax(uss_logits.float() / t, dim=-1)
            if active_weight.sum() > 0:
                kl_each = F.kl_div(student, teacher, reduction="none").sum(dim=-1) * (t * t)
                loss_consistency = (kl_each * active_weight).sum() / active_weight.sum().clamp_min(1.0)
            else:
                loss_consistency = logits.new_zeros(())
        else:
            loss_consistency = logits.new_zeros(())
        out["loss_consistency"] = loss_consistency
        return out

    def training_step(self, batch, batch_idx):
        self.uss_model.train()
        if self.freeze_sc and self.sc_eval_mode_when_frozen:
            self.sc_model.eval()
        else:
            self.sc_model.train()

        opts = self.optimizers()
        if isinstance(opts, (list, tuple)):
            opt_uss = opts[0]
            opt_sc = opts[1] if len(opts) > 1 else None
        else:
            opt_uss = opts
            opt_sc = None

        opt_uss.zero_grad(set_to_none=True)
        if opt_sc is not None:
            opt_sc.zero_grad(set_to_none=True)

        uss_out = self.uss_model(self._uss_input(batch))
        uss_loss_dict = self.uss_loss_func(uss_out, self._uss_target(batch))
        sc_loss_dict = self._sc_forward_and_loss(uss_out, batch, is_training=True)

        loss = (
            uss_loss_dict["loss"]
            + self.lambda_sc * sc_loss_dict["loss_sc_weighted"].to(self.uss_device)
            + self.lambda_consistency * sc_loss_dict["loss_consistency"].to(self.uss_device)
        )
        self.manual_backward(loss)
        opt_uss.step()
        if opt_sc is not None and (self.global_step % self.sc_update_every == 0):
            opt_sc.step()

        batchsize = batch["mixture"].shape[0]
        log_dict = {"step_train/loss": loss.detach().to(self.uss_device)}
        log_dict.update({f"step_train/uss_{k}": v.detach().to(self.uss_device) for k, v in uss_loss_dict.items() if torch.is_tensor(v)})
        log_dict.update({f"step_train/{k}": v.detach().to(self.uss_device) for k, v in sc_loss_dict.items() if torch.is_tensor(v)})
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=batchsize, sync_dist=False)
        self.log("epoch_train/loss", loss.detach(), prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=batchsize, sync_dist=False)
        return loss.detach()

    def validation_step(self, batch, batch_idx):
        self.uss_model.eval()
        self.sc_model.eval()
        with torch.no_grad():
            uss_out = self.uss_model(self._uss_input(batch))
            uss_loss_dict = self.uss_loss_func(uss_out, self._uss_target(batch))
        # Need gradients disabled for validation, but SC forward can still run normally.
        with torch.no_grad():
            sc_loss_dict = self._sc_forward_and_loss(uss_out, batch, is_training=False)
            loss = (
                uss_loss_dict["loss"]
                + self.lambda_sc * sc_loss_dict["loss_sc_weighted"].to(self.uss_device)
                + self.lambda_consistency * sc_loss_dict["loss_consistency"].to(self.uss_device)
            )
        batchsize = batch["mixture"].shape[0]
        log_dict = {"step_val/loss": loss.detach().to(self.uss_device)}
        log_dict.update({f"step_val/uss_{k}": v.detach().to(self.uss_device) for k, v in uss_loss_dict.items() if torch.is_tensor(v)})
        log_dict.update({f"step_val/{k}": v.detach().to(self.uss_device) for k, v in sc_loss_dict.items() if torch.is_tensor(v)})
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False, batch_size=batchsize, sync_dist=False)
        self.log("epoch_val/loss", loss.detach(), prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=batchsize, sync_dist=False)
        return loss.detach()

    def configure_optimizers(self):
        self.optimizer_uss_config["args"]["params"] = self.uss_model.parameters()
        opt_uss = initialize_config(self.optimizer_uss_config)
        optimizers = [opt_uss]
        schedulers = []
        if not self.freeze_sc and self.optimizer_sc_config is not None:
            self.optimizer_sc_config["args"]["params"] = self.sc_model.parameters()
            opt_sc = initialize_config(self.optimizer_sc_config)
            optimizers.append(opt_sc)
        if self.uss_lr_scheduler_config is not None:
            self.uss_lr_scheduler_config["scheduler"]["args"]["optimizer"] = opt_uss
            schedulers.append(initialize_config(self.uss_lr_scheduler_config["scheduler"]))
        if len(optimizers) == 1:
            return optimizers[0]
        return optimizers
