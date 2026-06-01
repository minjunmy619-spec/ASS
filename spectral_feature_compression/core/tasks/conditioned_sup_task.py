from __future__ import annotations

from typing import Any

import numpy as np

import torch

from spectral_feature_compression.core.tasks.sup_task import SupTask


class EventConditionedSupTask(SupTask):
    """Supervised separation task that forwards event-class query conditions."""

    def __init__(
        self,
        *args,
        event_condition_key: str = "event_condition",
        allow_missing_event_condition: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.event_condition_key = event_condition_key
        self.allow_missing_event_condition = bool(allow_missing_event_condition)

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if isinstance(batch, dict):
            wav = batch.get("wav", batch.get("mixture", batch.get("audio")))
            ref = batch.get("ref", batch.get("target", batch.get("sources")))
            event_condition = batch.get(self.event_condition_key, batch.get("classes", batch.get("class_probs")))
            if wav is None:
                raise KeyError("Conditioned batch dict must contain wav, mixture, or audio.")
            return wav, ref, event_condition

        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                wav, ref, event_condition = batch
                return wav, ref, event_condition
            if len(batch) == 2:
                wav, ref = batch
                return wav, ref, None
            if len(batch) == 1:
                return batch[0], None, None

        return batch, None, None

    @staticmethod
    def _repeat_event_condition_for_segments(event_condition: torch.Tensor, n_segments: int) -> torch.Tensor:
        repeats = (int(n_segments),) + (1,) * (event_condition.ndim - 1)
        return event_condition.repeat(repeats)

    def _css_with_event_condition(
        self,
        model,
        wav: torch.Tensor,
        *,
        ref: torch.Tensor | None,
        event_condition: torch.Tensor,
    ) -> torch.Tensor:
        del ref
        speech_length = wav.shape[-1]
        if speech_length <= model.css_segment_size * model.fs:
            return model(wav, event_condition=event_condition)

        if event_condition.shape[0] == 1 and wav.shape[0] != 1:
            event_condition = event_condition.expand(wav.shape[0], *event_condition.shape[1:])
        if event_condition.shape[0] != wav.shape[0]:
            raise ValueError(
                f"event_condition batch {event_condition.shape[0]} does not match wav batch {wav.shape[0]}."
            )

        overlap_length = int(np.round(model.fs * (model.css_segment_size - model.css_shift_size)))
        num_segments = int(np.ceil((speech_length - overlap_length) / (model.css_shift_size * model.fs)))
        t = t_total = int(model.css_segment_size * model.fs)
        pad_shape = wav[..., :t_total].shape

        segments = []
        is_silent = []
        for idx in range(num_segments):
            start = int(idx * model.css_shift_size * model.fs)
            end = start + t_total
            if end >= speech_length:
                end = speech_length
                segment = wav.new_zeros(pad_shape)
                t = end - start
                segment[..., :t] = wav[..., start:end].clone()
            else:
                segment = wav[..., start:end].clone()
            segments.append(segment)
            is_silent.append(abs(segment).sum().item() == 0.0)

        enh_waves = [None] * num_segments
        valid_indices = [idx for idx, silent in enumerate(is_silent) if not silent]
        if len(valid_indices) > 0:
            css_bs = model.css_batch_size
            for mb_start in range(0, len(valid_indices), css_bs):
                mb_indices = valid_indices[mb_start : mb_start + css_bs]
                seg_batch = torch.cat([segments[idx] for idx in mb_indices], dim=0)
                seg_condition = self._repeat_event_condition_for_segments(event_condition, len(mb_indices))
                processed_batch = model(seg_batch, event_condition=seg_condition)[..., :t_total]
                batch_size = wav.shape[0]
                for local_idx, segment_idx in enumerate(mb_indices):
                    start = local_idx * batch_size
                    end = start + batch_size
                    enh_waves[segment_idx] = processed_batch[start:end]

        for idx in range(num_segments):
            if enh_waves[idx] is None:
                enh_waves[idx] = torch.zeros_like(enh_waves[valid_indices[0]])

        waves = enh_waves[0]
        for idx in range(1, num_segments):
            if idx == num_segments - 1:
                enh_waves[idx][..., t:] = 0
                residual = enh_waves[idx][..., overlap_length:t]
            else:
                residual = enh_waves[idx][..., overlap_length:]

            if overlap_length > 0:
                waves[..., -overlap_length:] = (waves[..., -overlap_length:] + enh_waves[idx][..., :overlap_length]) / 2
            waves = torch.cat([waves, residual], dim=-1)

        if waves.size(-1) != wav.size(-1):
            raise RuntimeError(f"CSS output length mismatch: {waves.shape} vs {wav.shape}")
        return waves

    def _step(
        self,
        wav: torch.Tensor,
        ref: torch.Tensor | None,
        log_prefix: str,
        event_condition: torch.Tensor | None = None,
    ):
        if event_condition is None and not self.allow_missing_event_condition:
            raise ValueError(
                "EventConditionedSupTask requires an event_condition tensor. "
                "Use allow_missing_event_condition=True only for model smoke tests."
            )

        model = self.ema_model.module if self.use_ema_model and log_prefix != "training" else self.model
        model_kwargs = {"event_condition": event_condition} if event_condition is not None else {}
        if log_prefix != "training" and self.css_validation and event_condition is not None:
            est = self._css_with_event_condition(model, wav, ref=ref, event_condition=event_condition)
        elif log_prefix != "training" and self.css_validation:
            est = model.css(wav, ref=ref, **model_kwargs)
        else:
            est = model(wav, **model_kwargs)

        loss = self.loss(est.transpose(1, 2), ref.transpose(1, 2)).mean()

        log_dict = {"step": float(self.trainer.current_epoch), f"{log_prefix}/loss": loss}
        if log_prefix == "validation":
            snr_score = self.snr(est.transpose(1, 2), ref.transpose(1, 2)).mean()
            log_dict[f"{log_prefix}/snr"] = snr_score

        self.log_dict(log_dict, prog_bar=False, on_epoch=True, on_step=False, batch_size=wav.shape[0], sync_dist=True)
        return loss

    @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def training_step(self, batch: torch.Tensor | list | tuple | dict, batch_idx: int):
        wav, ref, event_condition = self._unpack_batch(batch)
        return self._step(wav, ref=ref, event_condition=event_condition, log_prefix="training")

    @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def validation_step(self, batch: torch.Tensor | list | tuple | dict, batch_idx: int):
        wav, ref, event_condition = self._unpack_batch(batch)
        return self._step(wav, ref=ref, event_condition=event_condition, log_prefix="validation")
