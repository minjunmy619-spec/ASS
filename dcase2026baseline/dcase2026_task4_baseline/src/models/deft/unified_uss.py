"""Unified opt-in Modified DeFT USS model.

This class consolidates the additive USS variants under one configurable
memory-efficient model.  Existing variant classes remain importable for old
configs, but new recipes can enable the count, temporal, spatial, residual, and
semantic bridge features from this single class.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.deft.foa_spatial_features import FOASpatialFeatureEncoder
from src.models.deft.modified_deft_bf16_safe import ModifiedDeFTUSSMemoryEfficient
from src.models.deft.modified_deft_semantic_bridge import SemanticAcousticBridgeMixin
from src.models.deft.spatial_heads import ForegroundSpatialHead
from src.models.deft.uss_count_head import ForegroundCountHead


def _mean_optional(outputs: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor | None:
    vals = [out[key] for out in outputs if key in out]
    if not vals:
        return None
    return torch.stack(vals, dim=0).mean(dim=0)


def _normalize_optional(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return F.normalize(value, dim=-1, eps=1e-8)


class UnifiedModifiedDeFTUSS(SemanticAcousticBridgeMixin, ModifiedDeFTUSSMemoryEfficient):
    """Memory-efficient USS with all variant features behind opt-in flags.

    Baseline-compatible output keys are always emitted:
        ``foreground_waveform``, ``interference_waveform``, ``noise_waveform``,
        ``class_logits``, and ``silence_logits``.

    Optional feature flags add only extra keys:
        ``enable_temporal_activity`` adds per-slot activity logits,
        ``enable_count_head`` adds ``count_logits``,
        ``enable_spatial_head`` adds ``spatial_embedding`` and ``doa_vector``,
        ``enable_residual_slots`` adds ``residual_waveform``,
        ``enable_semantic_bridge`` adds bridge/proposal keys for USS -> TSE.
    """

    def __init__(
        self,
        input_channels: int = 4,
        output_channels: int = 1,
        hidden_channels: int = 96,
        n_deft_blocks: int = 6,
        n_heads: int = 4,
        n_foreground: int = 3,
        n_interference: int = 2,
        n_classes: int = 18,
        window_size: int = 1024,
        hop_size: int = 320,
        time_window_size: int = 128,
        freq_group_size: int = 64,
        shift_windows: bool = True,
        inference_chunk_seconds: float | None = 10.0,
        inference_chunk_hop_seconds: float = 8.0,
        sample_rate: int = 32000,
        enable_foa_spatial_features: bool = False,
        include_logmag: bool = True,
        include_aiv: bool = True,
        include_ipd: bool = True,
        spatial_feature_eps: float = 1e-8,
        enable_temporal_activity: bool = False,
        enable_count_head: bool = False,
        count_hidden_dim: int = 64,
        max_count: int = 3,
        enable_spatial_head: bool = False,
        spatial_embedding_dim: int = 16,
        enable_residual_slots: bool = False,
        n_residual: int = 1,
        enable_semantic_bridge: bool = False,
        embedding_dim: int = 256,
        prototype_scale: float = 10.0,
        use_audio_embedding: bool = True,
        use_doa_head: bool = True,
        use_spatial_conditioning: bool = True,
        spatial_dim: int = 3,
        spatial_conditioning_scale: float = 1.0,
        predicted_spatial_prob: float = 0.0,
        spatial_mix_fallback_prob: float = 0.0,
        detach_predicted_spatial_for_condition: bool = False,
        tse_condition_dim: int = 256,
    ):
        self.enable_foa_spatial_features = bool(enable_foa_spatial_features)
        self.enable_temporal_activity = bool(enable_temporal_activity)
        self.enable_count_head = bool(enable_count_head)
        self.enable_spatial_head = bool(enable_spatial_head)
        self.enable_residual_slots = bool(enable_residual_slots)
        self.enable_semantic_bridge = bool(enable_semantic_bridge)
        task_n_interference = int(n_interference)
        residual_slots = int(n_residual) if self.enable_residual_slots else 0
        if task_n_interference < 0 or residual_slots < 0:
            raise ValueError("n_interference and n_residual must be non-negative")

        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=hidden_channels,
            n_deft_blocks=n_deft_blocks,
            n_heads=n_heads,
            n_foreground=n_foreground,
            n_interference=task_n_interference + residual_slots,
            n_classes=n_classes,
            window_size=window_size,
            hop_size=hop_size,
            time_window_size=time_window_size,
            freq_group_size=freq_group_size,
            shift_windows=shift_windows,
            inference_chunk_seconds=inference_chunk_seconds,
            inference_chunk_hop_seconds=inference_chunk_hop_seconds,
            sample_rate=sample_rate,
        )
        self.window_size = int(window_size)
        self.hop_size = int(hop_size)
        self.task_n_interference = task_n_interference
        self.n_residual = residual_slots

        if self.enable_foa_spatial_features:
            self.encoder = FOASpatialFeatureEncoder(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                include_logmag=include_logmag,
                include_aiv=include_aiv,
                include_ipd=include_ipd,
                eps=spatial_feature_eps,
            )
        if self.enable_temporal_activity:
            self.activity_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        if self.enable_count_head:
            self.count_head = ForegroundCountHead(
                n_foreground=n_foreground,
                n_classes=n_classes,
                hidden_dim=count_hidden_dim,
                max_count=max_count,
            )
        if self.enable_spatial_head:
            self.spatial_head = ForegroundSpatialHead(
                hidden_channels=hidden_channels,
                embedding_dim=spatial_embedding_dim,
            )
        if self.enable_semantic_bridge:
            self._init_bridge(
                object_feature_channels=hidden_channels,
                n_classes=n_classes,
                embedding_dim=embedding_dim,
                prototype_scale=prototype_scale,
                use_audio_embedding=use_audio_embedding,
                use_doa_head=use_doa_head,
                use_spatial_conditioning=use_spatial_conditioning,
                spatial_dim=spatial_dim,
                spatial_conditioning_scale=spatial_conditioning_scale,
                predicted_spatial_prob=predicted_spatial_prob,
                spatial_mix_fallback_prob=spatial_mix_fallback_prob,
                detach_predicted_spatial_for_condition=detach_predicted_spatial_for_condition,
                tse_condition_dim=tse_condition_dim,
            )

    def _activity_logits(self, object_features: torch.Tensor) -> torch.Tensor:
        batch_size, n_objects, channels, time_steps, freq_bins = object_features.shape
        logits = self.activity_head(
            object_features.reshape(batch_size * n_objects, channels, time_steps, freq_bins)
        )
        logits = logits.mean(dim=-1).squeeze(1)
        return logits.view(batch_size, n_objects, time_steps)

    def _build_object_features(
        self,
        input_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
        mixture = input_dict["mixture"]
        batch_size, _, samples = mixture.shape
        real, imag = self.waveform_to_complex(mixture.reshape(-1, samples))
        _, _, time_steps, freq_bins = real.shape
        real = real.view(batch_size, self.input_channels, time_steps, freq_bins)
        imag = imag.view(batch_size, self.input_channels, time_steps, freq_bins)

        x = self.encoder(torch.cat([real, imag], dim=1))
        for block in self.blocks:
            x = block(x)
        x = self.object_conv(x)
        x = x.view(batch_size, self.n_objects, -1, time_steps, freq_bins)
        return mixture, real, imag, x, samples, time_steps, freq_bins

    def _add_residual_slots(self, output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.enable_residual_slots:
            return output
        output = dict(output)
        interference = output["interference_waveform"]
        residual_start = self.task_n_interference
        residual_end = residual_start + self.n_residual
        output["interference_waveform"] = interference[:, : self.task_n_interference]
        output["residual_waveform"] = interference[:, residual_start:residual_end]

        if "interference_activity_logits" in output and "residual_activity_logits" not in output:
            activity = output["interference_activity_logits"]
            output["interference_activity_logits"] = activity[:, : self.task_n_interference]
            output["residual_activity_logits"] = activity[:, residual_start:residual_end]
        return output

    def _add_optional_heads(
        self,
        output: dict[str, torch.Tensor],
        object_features: torch.Tensor,
        class_logits: torch.Tensor,
        pred_doa_vector: torch.Tensor | None,
        used_spatial_vector: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        output = dict(output)
        if self.enable_count_head:
            output["count_logits"] = self.count_head(output)
        if self.enable_spatial_head:
            foreground_features = object_features[:, : self.n_foreground]
            spatial_embedding, doa_vector = self.spatial_head(foreground_features)
            output["spatial_embedding"] = spatial_embedding
            output["doa_vector"] = doa_vector
        if self.enable_semantic_bridge:
            if pred_doa_vector is None or used_spatial_vector is None:
                raise RuntimeError("Semantic bridge outputs require DoA/spatial vectors")
            output["pred_doa_vector"] = pred_doa_vector[:, : self.n_foreground]
            output["used_spatial_vector"] = used_spatial_vector[:, : self.n_foreground]
            output.update(self._bridge_outputs(object_features, class_logits, used_spatial_vector))
        return self._add_residual_slots(output)

    def _forward_full(self, input_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        mixture, real, imag, object_features, samples, time_steps, freq_bins = self._build_object_features(input_dict)

        pred_doa_vector = None
        used_spatial_vector = None
        if self.enable_semantic_bridge:
            pred_doa_vector = self.doa_head(object_features)
            used_spatial_vector = self._choose_spatial_condition(input_dict, mixture, pred_doa_vector)
            object_features = self._apply_spatial_conditioning(object_features, used_spatial_vector)

        waveform = self._spatial_mask_to_waveform(object_features, real, imag, samples)
        fg_features = object_features[:, : self.n_foreground]
        class_logits = self.class_head(
            fg_features.reshape(mixture.shape[0] * self.n_foreground, -1, time_steps, freq_bins)
        )
        class_logits = class_logits.view(mixture.shape[0], self.n_foreground, self.n_classes)
        silence_logits = self.silence_head(
            fg_features.reshape(mixture.shape[0] * self.n_foreground, -1, time_steps, freq_bins)
        )
        silence_logits = silence_logits.view(mixture.shape[0], self.n_foreground)

        output = {
            "waveform": waveform,
            "foreground_waveform": waveform[:, : self.n_foreground],
            "interference_waveform": waveform[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_waveform": waveform[:, -1:],
            "class_logits": class_logits,
            "silence_logits": silence_logits,
        }
        if self.enable_temporal_activity:
            activity_logits = self._activity_logits(object_features)
            output.update(
                {
                    "foreground_activity_logits": activity_logits[:, : self.n_foreground],
                    "interference_activity_logits": activity_logits[
                        :, self.n_foreground : self.n_foreground + self.n_interference
                    ],
                    "noise_activity_logits": activity_logits[:, -1:],
                    "duration_sec": mixture.new_full((mixture.shape[0],), float(samples) / float(self.sample_rate)),
                }
            )
        return self._add_optional_heads(output, object_features, class_logits, pred_doa_vector, used_spatial_vector)

    def _stitch_activity(self, chunks: list[dict[str, torch.Tensor]], key: str, samples: int, starts: list[int]) -> torch.Tensor | None:
        first = chunks[0].get(key)
        if first is None:
            return None
        batch_size, n_slots, chunk_frames = first.shape
        total_frames = max(1, int(round(float(samples) / max(float(self.hop_size), 1.0))) + 1)
        stitched = first.new_zeros(batch_size, n_slots, total_frames)
        weight = first.new_zeros(1, 1, total_frames)
        for out, start in zip(chunks, starts):
            activity = out[key]
            frame_start = int(round(float(start) / max(float(self.hop_size), 1.0)))
            frame_end = min(total_frames, frame_start + activity.shape[-1])
            if frame_end <= frame_start:
                continue
            valid = frame_end - frame_start
            stitched[..., frame_start:frame_end] += activity[..., :valid]
            weight[..., frame_start:frame_end] += 1.0
        return stitched / weight.clamp_min(1.0)

    def _chunked_forward(self, input_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        mixture = input_dict["mixture"]
        batch_size, _, samples = mixture.shape
        chunk_samples = int(round(float(self.inference_chunk_seconds) * self.sample_rate))
        hop_samples = int(round(float(self.inference_chunk_hop_seconds) * self.sample_rate))
        if chunk_samples <= 0 or hop_samples <= 0:
            raise ValueError("inference_chunk_seconds and inference_chunk_hop_seconds must be positive")
        if samples <= chunk_samples:
            return self._forward_full(input_dict)

        starts = self._iter_chunk_starts(samples, chunk_samples, hop_samples)
        weight = self._chunk_weight(chunk_samples, mixture.device, mixture.dtype).view(1, 1, 1, chunk_samples)
        waveform_sum = mixture.new_zeros(batch_size, self.n_objects, self.output_channels, samples)
        weight_sum = mixture.new_zeros(1, 1, 1, samples)
        chunks = []

        for start in starts:
            end = start + chunk_samples
            chunk = mixture[..., start:end]
            if chunk.shape[-1] < chunk_samples:
                chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))
            chunk_input = dict(input_dict)
            chunk_input["mixture"] = chunk
            out = self._forward_full(chunk_input)
            chunks.append(out)
            valid = min(chunk_samples, samples - start)
            waveform_sum[..., start : start + valid] += out["waveform"][..., :valid] * weight[..., :valid]
            weight_sum[..., start : start + valid] += weight[..., :valid]

        waveform = waveform_sum / weight_sum.clamp_min(1e-6)
        class_logits = torch.stack([out["class_logits"] for out in chunks], dim=0).mean(dim=0)
        silence_logits = torch.stack([out["silence_logits"] for out in chunks], dim=0).mean(dim=0)
        output = {
            "waveform": waveform,
            "foreground_waveform": waveform[:, : self.n_foreground],
            "interference_waveform": waveform[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_waveform": waveform[:, -1:],
            "class_logits": class_logits,
            "silence_logits": silence_logits,
        }
        if self.enable_temporal_activity:
            output["foreground_activity_logits"] = self._stitch_activity(
                chunks, "foreground_activity_logits", samples, starts
            )
            output["interference_activity_logits"] = self._stitch_activity(
                chunks, "interference_activity_logits", samples, starts
            )
            if "residual_activity_logits" in chunks[0]:
                output["residual_activity_logits"] = self._stitch_activity(
                    chunks, "residual_activity_logits", samples, starts
                )
            output["noise_activity_logits"] = self._stitch_activity(chunks, "noise_activity_logits", samples, starts)
            output["duration_sec"] = mixture.new_full((batch_size,), float(samples) / float(self.sample_rate))

        for key in ("object_embedding", "object_audio_embedding", "prototype_logits"):
            value = _mean_optional(chunks, key)
            if value is not None:
                output[key] = value
        for key in (
            "spatial_embedding",
            "doa_vector",
            "pred_doa_vector",
            "used_spatial_vector",
            "foreground_embedding",
            "foreground_audio_embedding",
            "tse_condition",
        ):
            value = _normalize_optional(_mean_optional(chunks, key))
            if value is not None:
                output[key] = value
        if self.enable_count_head:
            output["count_logits"] = self.count_head(output)
        return self._add_residual_slots(output)

    def forward(self, input_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.training or self.inference_chunk_seconds is None:
            return self._forward_full(input_dict)
        return self._chunked_forward(input_dict)
