from __future__ import annotations

from typing import Any

import torch as _torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from src.models.deft.foa_spatial_features import FOASpatialFeatureEncoder
from src.models.deft.modified_deft import (
    ClassConditioner,
    MemoryEfficientDeFTBlock,
    _BaseSpectralModel,
    _match_query_condition_shape,
    _stack_complex,
    _temporal_film_from_conditioning,
)
from torchlibrosa.stft import magphase

torch: Any = _torch

_CONDITION_KEYS = ("query_condition", "tse_condition", "bridge_condition", "proposal_condition")
_SPATIAL_CONDITION_KEYS = (
    "spatial_condition",
    "pred_doa_vector",
    "doa_vector",
    "used_spatial_vector",
    "spatial_embedding",
)


def _zero_init_last_linear(module: nn.Module) -> None:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            nn.init.zeros_(child.weight)
            nn.init.zeros_(child.bias)
            return


class _PlainComplexEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

    def forward(self, raw_complex: Tensor) -> Tensor:
        return self.net(raw_complex)


class TwoStageRobustSpatialBridgeTSE(_BaseSpectralModel):
    """Two-stage robust spatial USS-conditioned TSE.

    The model is designed for final evaluation where the only external input is
    the mixture waveform.  It requires only mixture, USS enrollment estimates,
    and a label vector.  USS auxiliary outputs such as ``tse_condition``,
    temporal activity, and DoA/spatial vectors are consumed only as optional
    gated residual hints.

    Stage 1 builds a shared mixture scene representation from multichannel FOA
    spectra.  Stage 2 fuses that scene representation with per-query enrollment
    features, applies label/query/temporal/spatial FiLM conditioning, and
    predicts multichannel spatial masks.
    """

    def __init__(
        self,
        mixture_channels: int = 4,
        enrollment_channels: int = 1,
        output_channels: int = 1,
        hidden_channels: int = 96,
        scene_blocks: int = 3,
        query_blocks: int = 3,
        n_heads: int = 4,
        label_dim: int = 18,
        window_size: int = 1024,
        hop_size: int = 320,
        time_window_size: int = 128,
        freq_group_size: int = 64,
        shift_windows: bool = True,
        inference_chunk_seconds: float | None = 10.0,
        inference_chunk_hop_seconds: float = 8.0,
        sample_rate: int = 32000,
        enable_foa_spatial_features: bool = True,
        include_logmag: bool = True,
        include_aiv: bool = True,
        include_ipd: bool = True,
        spatial_feature_eps: float = 1e-8,
        query_condition_dim: int = 256,
        query_condition_hidden_dim: int = 256,
        spatial_condition_dim: int = 3,
        spatial_condition_hidden_dim: int = 64,
        temporal_conditioning_enabled: bool = True,
        auxiliary_gate_init: float = -2.0,
        use_confidence_gates: bool = True,
        confidence_gate_hidden_dim: int = 32,
        condition_dropout: float = 0.0,
        temporal_condition_dropout: float = 0.0,
        spatial_condition_dropout: float = 0.0,
        condition_noise_std: float = 0.0,
        spatial_condition_noise_std: float = 0.0,
        enable_reference_fallback: bool = True,
        spatial_output_gate_init: float = -4.0,
    ):
        super().__init__(window_size=window_size, hop_size=hop_size)
        if output_channels != 1:
            raise ValueError("TwoStageRobustSpatialBridgeTSE currently expects output_channels=1")
        self.mixture_channels = int(mixture_channels)
        self.enrollment_channels = int(enrollment_channels)
        self.output_channels = int(output_channels)
        self.hidden_channels = int(hidden_channels)
        self.label_dim = int(label_dim)
        self.window_size = int(window_size)
        self.hop_size = int(hop_size)
        self.time_window_size = int(time_window_size)
        self.freq_group_size = int(freq_group_size)
        self.shift_windows = bool(shift_windows)
        self.inference_chunk_seconds = inference_chunk_seconds
        self.inference_chunk_hop_seconds = float(inference_chunk_hop_seconds)
        self.sample_rate = int(sample_rate)
        self.query_condition_dim = int(query_condition_dim or 0)
        self.spatial_condition_dim = int(spatial_condition_dim or 0)
        self.temporal_conditioning_enabled = bool(temporal_conditioning_enabled)
        self.condition_dropout = float(condition_dropout)
        self.temporal_condition_dropout = float(temporal_condition_dropout)
        self.spatial_condition_dropout = float(spatial_condition_dropout)
        self.condition_noise_std = float(condition_noise_std)
        self.spatial_condition_noise_std = float(spatial_condition_noise_std)
        self.enable_reference_fallback = bool(enable_reference_fallback)
        self.use_confidence_gates = bool(use_confidence_gates)

        if enable_foa_spatial_features:
            self.scene_encoder = FOASpatialFeatureEncoder(
                input_channels=mixture_channels,
                hidden_channels=hidden_channels,
                include_logmag=include_logmag,
                include_aiv=include_aiv,
                include_ipd=include_ipd,
                eps=spatial_feature_eps,
            )
        else:
            self.scene_encoder = _PlainComplexEncoder(mixture_channels, hidden_channels)

        self.scene_blocks = nn.ModuleList(
            [self._make_block(hidden_channels, n_heads, block_idx) for block_idx in range(int(scene_blocks))]
        )
        self.enrollment_encoder = nn.Sequential(
            nn.Conv2d(enrollment_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.query_blocks = nn.ModuleList(
            [self._make_block(hidden_channels, n_heads, block_idx) for block_idx in range(int(query_blocks))]
        )

        self.class_conditioner = ClassConditioner(label_dim, hidden_channels)
        if self.query_condition_dim > 0:
            self.query_conditioner = nn.Sequential(
                nn.LayerNorm(self.query_condition_dim),
                nn.Linear(self.query_condition_dim, int(query_condition_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(query_condition_hidden_dim), hidden_channels * 2),
            )
        else:
            self.query_conditioner = None

        if self.spatial_condition_dim > 0:
            self.spatial_conditioner = nn.Sequential(
                nn.LayerNorm(self.spatial_condition_dim),
                nn.Linear(self.spatial_condition_dim, int(spatial_condition_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(spatial_condition_hidden_dim), hidden_channels * 2),
            )
        else:
            self.spatial_conditioner = None

        if self.temporal_conditioning_enabled:
            self.temporal_conditioner = nn.Conv1d(1, hidden_channels * 2, kernel_size=3, padding=1)
        else:
            self.temporal_conditioner = None

        # Gates are ordered as query, temporal, spatial.  They are initialized
        # small so noisy USS auxiliary heads cannot dominate early training.
        self.auxiliary_gate_logits = nn.Parameter(torch.full((3,), float(auxiliary_gate_init)))
        if self.use_confidence_gates:
            self.confidence_gate = nn.Sequential(
                nn.LayerNorm(6),
                nn.Linear(6, int(confidence_gate_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(confidence_gate_hidden_dim), 3),
            )
            _zero_init_last_linear(self.confidence_gate)
        else:
            self.confidence_gate = None

        self.spatial_mask_components = 3
        self.spatial_mask_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, mixture_channels * self.spatial_mask_components, kernel_size=1),
        )
        self.out_conv = nn.Conv2d(mixture_channels, output_channels, kernel_size=1, bias=False)

        if self.enable_reference_fallback:
            self.reference_mask_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)
            self.spatial_output_gate_logit = nn.Parameter(torch.tensor(float(spatial_output_gate_init)))
        else:
            self.reference_mask_head = None
            self.spatial_output_gate_logit = None

        self.activity_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def _make_block(self, hidden_channels: int, n_heads: int, block_idx: int) -> MemoryEfficientDeFTBlock:
        use_shift = self.shift_windows and block_idx % 2 == 1
        return MemoryEfficientDeFTBlock(
            hidden_channels,
            n_heads=n_heads,
            time_window_size=self.time_window_size,
            freq_group_size=self.freq_group_size,
            time_shift=self.time_window_size // 2 if use_shift else 0,
            freq_shift=self.freq_group_size // 2 if use_shift else 0,
        )

    def _iter_chunk_starts(self, samples: int, chunk_samples: int, hop_samples: int) -> list[int]:
        if samples <= chunk_samples:
            return [0]
        starts = list(range(0, max(samples - chunk_samples, 0) + 1, hop_samples))
        last_start = samples - chunk_samples
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _chunk_weight(self, chunk_samples: int, device: Any, dtype: Any) -> Tensor:
        if chunk_samples <= 1:
            return torch.ones(1, device=device, dtype=dtype)
        weight = torch.hann_window(chunk_samples, periodic=False, device=device, dtype=dtype)
        return torch.clamp(weight, min=1e-3)

    def _reshape_label_vector(self, label_vector: Tensor, n_queries: int) -> Tensor:
        if label_vector.dim() == 3:
            out = label_vector
        elif label_vector.dim() == 2:
            if label_vector.shape[-1] == self.label_dim:
                out = label_vector.unsqueeze(1).expand(-1, n_queries, -1)
            elif label_vector.shape[-1] % n_queries == 0:
                out = label_vector.view(label_vector.shape[0], n_queries, -1)
            else:
                raise ValueError("label_vector must have shape [B,D], [B,Q*D], or [B,Q,D]")
        else:
            raise ValueError("label_vector must have shape [B,D], [B,Q*D], or [B,Q,D]")

        if out.shape[1] < n_queries:
            pad = out.new_zeros(out.shape[0], n_queries - out.shape[1], out.shape[-1])
            out = torch.cat([out, pad], dim=1)
        elif out.shape[1] > n_queries:
            out = out[:, :n_queries]

        if out.shape[-1] < self.label_dim:
            pad = out.new_zeros(*out.shape[:-1], self.label_dim - out.shape[-1])
            out = torch.cat([out, pad], dim=-1)
        elif out.shape[-1] > self.label_dim:
            out = out[..., : self.label_dim]
        return out

    def _get_first_key(self, input_dict: dict, keys: tuple[str, ...]) -> Tensor | None:
        for key in keys:
            if key in input_dict:
                return input_dict[key]
        return None

    def _maybe_drop_or_noise(self, value: Tensor, dropout: float, noise_std: float) -> Tensor:
        if not self.training:
            return value
        out = value
        if dropout > 0.0:
            keep = torch.rand(*out.shape[:2], 1, device=out.device, dtype=out.dtype) >= float(dropout)
            while keep.dim() < out.dim():
                keep = keep.unsqueeze(-1)
            out = out * keep
        if noise_std > 0.0:
            out = out + torch.randn_like(out) * float(noise_std)
        return out

    def _match_condition(self, condition: Tensor, label_vector: Tensor, condition_dim: int) -> Tensor:
        return _match_query_condition_shape(condition, label_vector, int(condition_dim))

    def _match_slot_tensor(self, value: Tensor, batch_size: int, n_queries: int, device, dtype) -> Tensor:
        value = value.to(device=device, dtype=dtype)
        if value.dim() == 1:
            value = value.view(batch_size, 1).expand(-1, n_queries)
        if value.dim() == 2:
            if value.shape[0] == batch_size and value.shape[1] == n_queries:
                return value
            if value.shape[0] == batch_size:
                return value.max(dim=-1).values.unsqueeze(1).expand(-1, n_queries)
        if value.dim() == 3:
            if value.shape[1] < n_queries:
                pad = value.new_zeros(value.shape[0], n_queries - value.shape[1], value.shape[-1])
                value = torch.cat([value, pad], dim=1)
            elif value.shape[1] > n_queries:
                value = value[:, :n_queries]
            return value.max(dim=-1).values
        raise ValueError(f"Cannot match slot tensor with shape {tuple(value.shape)}")

    def _confidence_features(
        self,
        input_dict: dict,
        enrollment: Tensor,
        label_vector: Tensor,
    ) -> Tensor:
        batch_size, n_queries = enrollment.shape[:2]
        device = enrollment.device
        dtype = enrollment.dtype
        rms = enrollment[:, :, 0].float().pow(2).mean(dim=-1).sqrt().to(device=device, dtype=dtype)
        log_rms = torch.log1p(rms)
        label_strength = label_vector.abs().sum(dim=-1).clamp(max=1.0)

        silence_logits = input_dict.get("silence_logits")
        if silence_logits is not None and torch.is_tensor(silence_logits):
            # Historical key name is ``silence_logits`` in USS, but the loss
            # trains it as an active-slot logit: high value means active.
            active_logit = self._match_slot_tensor(silence_logits, batch_size, n_queries, device, dtype)
            active_prob = torch.sigmoid(active_logit)
            has_active_logit = torch.ones_like(active_prob)
        else:
            active_prob = torch.zeros(batch_size, n_queries, device=device, dtype=dtype)
            has_active_logit = torch.zeros_like(active_prob)

        class_logits = input_dict.get("class_logits")
        if class_logits is not None and torch.is_tensor(class_logits):
            cls = class_logits.to(device=device, dtype=dtype)
            if cls.dim() == 2:
                cls = cls.unsqueeze(1).expand(-1, n_queries, -1)
            if cls.dim() == 3:
                if cls.shape[1] < n_queries:
                    pad = cls.new_zeros(cls.shape[0], n_queries - cls.shape[1], cls.shape[-1])
                    cls = torch.cat([cls, pad], dim=1)
                elif cls.shape[1] > n_queries:
                    cls = cls[:, :n_queries]
                class_conf = cls.softmax(dim=-1).max(dim=-1).values
            else:
                class_conf = label_vector.max(dim=-1).values
            has_class = torch.ones_like(class_conf)
        else:
            class_conf = label_vector.max(dim=-1).values
            has_class = torch.zeros_like(class_conf)

        return torch.stack(
            [log_rms, active_prob, class_conf, label_strength, has_active_logit, has_class],
            dim=-1,
        )

    def _auxiliary_gates(self, input_dict: dict, enrollment: Tensor, label_vector: Tensor) -> Tensor:
        base = self.auxiliary_gate_logits.view(1, 1, 3).to(device=enrollment.device, dtype=enrollment.dtype)
        if self.confidence_gate is None:
            return torch.sigmoid(base).expand(enrollment.shape[0], enrollment.shape[1], -1)
        features = self._confidence_features(input_dict, enrollment, label_vector)
        return torch.sigmoid(base + self.confidence_gate(features))

    def _query_film(self, input_dict: dict, label_vector: Tensor, gates: Tensor):
        if self.query_conditioner is None:
            return None, None
        condition = self._get_first_key(input_dict, _CONDITION_KEYS)
        if condition is None:
            return None, None
        condition = self._match_condition(condition, label_vector, self.query_condition_dim)
        condition = self._maybe_drop_or_noise(condition, self.condition_dropout, self.condition_noise_std)
        batch_size, n_queries = condition.shape[:2]
        beta_gamma = self.query_conditioner(condition.reshape(batch_size * n_queries, -1))
        beta, gamma = beta_gamma.chunk(2, dim=-1)
        gate = gates[..., 0].reshape(batch_size * n_queries, 1).to(dtype=beta.dtype)
        return beta[:, :, None, None] * gate[:, :, None, None], gamma[:, :, None, None] * gate[:, :, None, None]

    def _spatial_film(self, input_dict: dict, label_vector: Tensor, gates: Tensor):
        if self.spatial_conditioner is None:
            return None, None
        condition = self._get_first_key(input_dict, _SPATIAL_CONDITION_KEYS)
        if condition is None:
            return None, None
        condition = self._match_condition(condition, label_vector, self.spatial_condition_dim)
        condition = self._maybe_drop_or_noise(
            condition,
            self.spatial_condition_dropout,
            self.spatial_condition_noise_std,
        )
        if self.spatial_condition_dim == 3:
            condition = F.normalize(condition, dim=-1, eps=1e-6)
        batch_size, n_queries = condition.shape[:2]
        beta_gamma = self.spatial_conditioner(condition.reshape(batch_size * n_queries, -1))
        beta, gamma = beta_gamma.chunk(2, dim=-1)
        gate = gates[..., 2].reshape(batch_size * n_queries, 1).to(dtype=beta.dtype)
        return beta[:, :, None, None] * gate[:, :, None, None], gamma[:, :, None, None] * gate[:, :, None, None]

    def _temporal_film(self, temporal_conditioning, batch_size, n_queries, time_steps, device, dtype, gates):
        if self.temporal_conditioner is None or temporal_conditioning is None:
            return None, None
        temporal_conditioning = temporal_conditioning.to(device=device, dtype=dtype)
        if temporal_conditioning.dim() == 2:
            temporal_conditioning = temporal_conditioning.unsqueeze(1).expand(-1, n_queries, -1)
        if temporal_conditioning.dim() == 3:
            temporal_conditioning = self._maybe_drop_or_noise(
                temporal_conditioning,
                self.temporal_condition_dropout,
                0.0,
            )
        beta, gamma = _temporal_film_from_conditioning(
            self.temporal_conditioner,
            temporal_conditioning,
            batch_size,
            n_queries,
            time_steps,
            device,
            dtype,
        )
        if beta is None:
            return None, None
        gate = gates[..., 1].reshape(batch_size * n_queries, 1, 1, 1).to(dtype=beta.dtype)
        return beta * gate, gamma * gate

    def _encode_scene(self, mixture: Tensor):
        batch_size, _, samples = mixture.shape
        real, imag = self.waveform_to_complex(mixture.reshape(-1, samples))
        _, _, time_steps, freq_bins = real.shape
        real = real.view(batch_size, self.mixture_channels, time_steps, freq_bins)
        imag = imag.view(batch_size, self.mixture_channels, time_steps, freq_bins)
        scene = self.scene_encoder(_stack_complex(real, imag))
        for block in self.scene_blocks:
            scene = block(scene)
        return scene, real, imag, time_steps, freq_bins

    def _encode_enrollment(self, enrollment: Tensor, time_steps: int, freq_bins: int):
        batch_size, n_queries, _, samples = enrollment.shape
        real, imag = self.waveform_to_complex(enrollment.reshape(-1, samples))
        real = real.view(batch_size * n_queries, self.enrollment_channels, time_steps, freq_bins)
        imag = imag.view(batch_size * n_queries, self.enrollment_channels, time_steps, freq_bins)
        return self.enrollment_encoder(_stack_complex(real, imag))

    def _spatial_mask_to_waveform(
        self,
        x: Tensor,
        mixture_real: Tensor,
        mixture_imag: Tensor,
        batch_size: int,
        n_queries: int,
        samples: int,
    ) -> Tensor:
        _, _, time_steps, freq_bins = x.shape
        mask = self.spatial_mask_head(x).view(
            batch_size,
            n_queries,
            self.mixture_channels,
            self.spatial_mask_components,
            time_steps,
            freq_bins,
        )
        mask_mag = torch.sigmoid(mask[:, :, :, 0])
        mask_real = torch.tanh(mask[:, :, :, 1])
        mask_imag = torch.tanh(mask[:, :, :, 2])
        _, mask_cos, mask_sin = magphase(mask_real, mask_imag)

        mixture_mag, mixture_cos, mixture_sin = magphase(mixture_real, mixture_imag)
        out_mag = F.relu(mixture_mag[:, None] * mask_mag)
        out_cos = mixture_cos[:, None] * mask_cos - mixture_sin[:, None] * mask_sin
        out_sin = mixture_sin[:, None] * mask_cos + mixture_cos[:, None] * mask_sin
        est_real = (out_mag * out_cos).reshape(batch_size * n_queries, self.mixture_channels, time_steps, freq_bins)
        est_imag = (out_mag * out_sin).reshape(batch_size * n_queries, self.mixture_channels, time_steps, freq_bins)
        est_real = self.out_conv(est_real)
        est_imag = self.out_conv(est_imag)
        waveform = self.complex_to_waveform(
            est_real.reshape(batch_size * n_queries * self.output_channels, 1, time_steps, freq_bins),
            est_imag.reshape(batch_size * n_queries * self.output_channels, 1, time_steps, freq_bins),
            samples,
        )
        return waveform.view(batch_size, n_queries, self.output_channels, samples)

    def _reference_waveform(
        self,
        x: Tensor,
        mixture_real: Tensor,
        mixture_imag: Tensor,
        batch_size: int,
        n_queries: int,
        samples: int,
    ) -> Tensor:
        if self.reference_mask_head is None:
            raise RuntimeError("reference fallback is disabled")
        _, _, time_steps, freq_bins = x.shape
        mask = torch.tanh(self.reference_mask_head(x))
        mix_ref_real = (
            mixture_real[:, None, 0]
            .expand(-1, n_queries, -1, -1)
            .reshape(batch_size * n_queries, 1, time_steps, freq_bins)
        )
        mix_ref_imag = (
            mixture_imag[:, None, 0]
            .expand(-1, n_queries, -1, -1)
            .reshape(batch_size * n_queries, 1, time_steps, freq_bins)
        )
        est_real = mask[:, 0:1] * mix_ref_real - mask[:, 1:2] * mix_ref_imag
        est_imag = mask[:, 0:1] * mix_ref_imag + mask[:, 1:2] * mix_ref_real
        waveform = self.complex_to_waveform(est_real, est_imag, samples)
        return waveform.view(batch_size, n_queries, 1, samples)

    def _forward_full(self, input_dict: dict) -> dict[str, Tensor]:
        mixture = input_dict["mixture"]
        enrollment = input_dict["enrollment"]
        if enrollment.dim() != 4:
            raise ValueError(f"enrollment must have shape [B,Q,C,T], got {tuple(enrollment.shape)}")
        batch_size, n_queries, _, samples = enrollment.shape
        label_vector = self._reshape_label_vector(input_dict["label_vector"], n_queries)
        label_vector = label_vector.to(device=mixture.device, dtype=mixture.dtype)

        scene, mixture_real, mixture_imag, time_steps, freq_bins = self._encode_scene(mixture)
        enrollment_features = self._encode_enrollment(enrollment, time_steps, freq_bins)
        scene = (
            scene[:, None]
            .expand(-1, n_queries, -1, -1, -1)
            .reshape(
                batch_size * n_queries,
                self.hidden_channels,
                time_steps,
                freq_bins,
            )
        )
        x = self.fusion(torch.cat([scene, enrollment_features], dim=1))

        gates = self._auxiliary_gates(input_dict, enrollment, label_vector)
        beta, gamma = self.class_conditioner(label_vector.reshape(batch_size * n_queries, -1))

        query_beta, query_gamma = self._query_film(input_dict, label_vector, gates)
        if query_beta is not None:
            beta = beta + query_beta
            gamma = gamma + query_gamma

        spatial_beta, spatial_gamma = self._spatial_film(input_dict, label_vector, gates)
        if spatial_beta is not None:
            beta = beta + spatial_beta
            gamma = gamma + spatial_gamma

        temporal_conditioning = input_dict.get("temporal_conditioning")
        if temporal_conditioning is None:
            temporal_conditioning = input_dict.get("foreground_activity_logits")
            if temporal_conditioning is not None:
                temporal_conditioning = temporal_conditioning.sigmoid()
        time_beta, time_gamma = self._temporal_film(
            temporal_conditioning,
            batch_size,
            n_queries,
            time_steps,
            x.device,
            x.dtype,
            gates,
        )
        if time_beta is not None:
            beta = beta + time_beta
            gamma = gamma + time_gamma

        for block in self.query_blocks:
            x = block(x, beta=beta, gamma=gamma)

        spatial_waveform = self._spatial_mask_to_waveform(
            x,
            mixture_real,
            mixture_imag,
            batch_size,
            n_queries,
            samples,
        )
        if self.enable_reference_fallback:
            reference_waveform = self._reference_waveform(
                x,
                mixture_real,
                mixture_imag,
                batch_size,
                n_queries,
                samples,
            )
            assert self.spatial_output_gate_logit is not None
            spatial_gate = torch.sigmoid(self.spatial_output_gate_logit).to(dtype=spatial_waveform.dtype)
            waveform = (1.0 - spatial_gate) * reference_waveform + spatial_gate * spatial_waveform
        else:
            waveform = spatial_waveform

        activity_logits = self.activity_head(x).mean(dim=-1).squeeze(1)
        activity_logits = activity_logits.view(batch_size, n_queries, time_steps)
        duration_sec = mixture.new_full((batch_size,), float(samples) / float(self.sample_rate))
        return {
            "waveform": waveform,
            "activity_logits": activity_logits,
            "duration_sec": duration_sec,
            "auxiliary_gates": gates,
        }

    def _activity_to_samples_for_chunking(self, activity, batch_size, n_queries, samples, device, dtype):
        if activity is None:
            return None
        activity = activity.to(device=device, dtype=dtype)
        if activity.dim() == 2:
            activity = activity.unsqueeze(1).expand(-1, n_queries, -1)
        if activity.dim() != 3:
            raise ValueError("temporal_conditioning must have shape [B,T] or [B,Q,T]")
        if activity.shape[0] != batch_size or activity.shape[1] != n_queries:
            raise ValueError("temporal_conditioning batch/query dimensions do not match TSE input")
        return F.interpolate(
            activity.reshape(batch_size * n_queries, 1, activity.shape[-1]),
            size=samples,
            mode="linear",
            align_corners=False,
        ).view(batch_size, n_queries, samples)

    def _stitch_chunk_activity(
        self,
        chunks: list[tuple[int, int, Tensor]],
        samples: int,
        batch_size: int,
        n_queries: int,
        device,
        dtype,
    ) -> Tensor:
        total_frames = max(1, int(round(float(samples) / max(float(self.hop_size), 1.0))) + 1)
        stitched = torch.zeros(batch_size, n_queries, total_frames, device=device, dtype=dtype)
        weights = torch.zeros(1, 1, total_frames, device=device, dtype=dtype)
        for sample_start, valid_samples, activity in chunks:
            frame_start = int(round(float(sample_start) / max(float(self.hop_size), 1.0)))
            valid_frames = max(1, int(round(float(valid_samples) / max(float(self.hop_size), 1.0))) + 1)
            frame_end = min(total_frames, frame_start + valid_frames, frame_start + activity.shape[-1])
            if frame_end <= frame_start:
                continue
            n_valid = frame_end - frame_start
            stitched[..., frame_start:frame_end] += activity[..., :n_valid]
            weights[..., frame_start:frame_end] += 1.0
        return stitched / weights.clamp_min(1.0)

    def _chunked_forward(self, input_dict: dict) -> dict[str, Tensor]:
        mixture = input_dict["mixture"]
        enrollment = input_dict["enrollment"]
        batch_size, n_queries, _, samples = enrollment.shape
        assert self.inference_chunk_seconds is not None
        chunk_samples = int(round(float(self.inference_chunk_seconds) * self.sample_rate))
        hop_samples = int(round(float(self.inference_chunk_hop_seconds) * self.sample_rate))
        if chunk_samples <= 0 or hop_samples <= 0:
            raise ValueError("inference_chunk_seconds and inference_chunk_hop_seconds must be positive")
        if samples <= chunk_samples:
            return self._forward_full(input_dict)

        starts = self._iter_chunk_starts(samples, chunk_samples, hop_samples)
        weight = self._chunk_weight(chunk_samples, mixture.device, mixture.dtype).view(1, 1, 1, chunk_samples)
        waveform_sum = mixture.new_zeros(batch_size, n_queries, self.output_channels, samples)
        weight_sum = mixture.new_zeros(1, 1, 1, samples)
        activity_chunks = []

        temporal_conditioning = input_dict.get("temporal_conditioning")
        if temporal_conditioning is None and "foreground_activity_logits" in input_dict:
            temporal_conditioning = input_dict["foreground_activity_logits"].sigmoid()
        temporal_samples = self._activity_to_samples_for_chunking(
            temporal_conditioning,
            batch_size,
            n_queries,
            samples,
            mixture.device,
            mixture.dtype,
        )

        for start in starts:
            valid = min(chunk_samples, samples - start)
            mixture_chunk = mixture[..., start : start + valid]
            enrollment_chunk = enrollment[..., start : start + valid]
            temporal_chunk = None
            if temporal_samples is not None:
                temporal_chunk = temporal_samples[..., start : start + valid]
            if valid < chunk_samples:
                mixture_chunk = F.pad(mixture_chunk, (0, chunk_samples - valid))
                enrollment_chunk = F.pad(enrollment_chunk, (0, chunk_samples - valid))
                if temporal_chunk is not None:
                    temporal_chunk = F.pad(temporal_chunk, (0, chunk_samples - valid))

            chunk_input = dict(input_dict)
            chunk_input["mixture"] = mixture_chunk
            chunk_input["enrollment"] = enrollment_chunk
            if temporal_chunk is not None:
                chunk_input["temporal_conditioning"] = temporal_chunk
            out = self._forward_full(chunk_input)
            waveform_sum[..., start : start + valid] += out["waveform"][..., :valid] * weight[..., :valid]
            weight_sum[..., start : start + valid] += weight[..., :valid]
            activity_chunks.append((start, valid, out["activity_logits"]))

        duration_sec = mixture.new_full((batch_size,), float(samples) / float(self.sample_rate))
        activity_logits = self._stitch_chunk_activity(
            activity_chunks,
            samples,
            batch_size,
            n_queries,
            mixture.device,
            mixture.dtype,
        )
        return {
            "waveform": waveform_sum / torch.clamp(weight_sum, min=1e-6),
            "activity_logits": activity_logits,
            "duration_sec": duration_sec,
        }

    def forward(self, input_dict: dict) -> dict[str, Tensor]:
        if self.training or self.inference_chunk_seconds is None:
            return self._forward_full(input_dict)
        return self._chunked_forward(input_dict)
