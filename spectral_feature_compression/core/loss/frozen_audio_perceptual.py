"""Frozen CLAP and Whisper losses for semantic source-separation supervision."""

from __future__ import annotations

from typing import Any

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

import torchaudio.functional as AF

import yaml


def _coverage_window_starts(n_samples: int, window_samples: int) -> tuple[int, ...]:
    if n_samples <= 0:
        raise ValueError(f"Audio must contain at least one sample, got {n_samples}")
    if window_samples <= 0:
        raise ValueError(f"window_samples must be positive, got {window_samples}")
    return tuple(range(0, n_samples, window_samples))


def _as_mono_batch(audio: torch.Tensor, *, name: str) -> torch.Tensor:
    if audio.ndim == 2:
        return audio
    if audio.ndim == 3:
        return audio.mean(dim=1)
    raise ValueError(f"{name} must have shape [batch, samples] or [batch, channels, samples], got {audio.shape}")


def _as_stem_batch(audio: torch.Tensor, *, n_sources: int, name: str) -> torch.Tensor:
    if audio.ndim == 3:
        audio = audio[:, :, None, :]
    if audio.ndim != 4 or audio.shape[1] != n_sources:
        raise ValueError(
            f"{name} must have shape [batch, {n_sources}, channels, samples], got {tuple(audio.shape)}"
        )
    return audio.mean(dim=2)


class _FrozenExternalModelLoss(nn.Module):
    """Keep a frozen loss network off the parent state dict while moving devices."""

    def _set_external_model(self, model: nn.Module) -> None:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        object.__setattr__(self, "_external_model", model)

    @property
    def external_model(self) -> nn.Module:
        model = getattr(self, "_external_model", None)
        if not isinstance(model, nn.Module):
            raise RuntimeError("Frozen external model has not been initialized")
        return model

    def _apply(self, fn):
        super()._apply(fn)
        self.external_model._apply(fn)
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.external_model.eval()
        return self


class WhisperFeatureMatchingLoss(_FrozenExternalModelLoss):
    """Match intermediate frozen Whisper encoder features for the speech stem.

    The official Whisper encoder asserts a fixed 30-second feature shape. This
    implementation runs the same convolution, positional encoding, and encoder
    blocks explicitly so shorter separation crops use only the required prefix
    of the official positional embedding.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        model_name: str = "base",
        selected_layers: Sequence[int] | None = None,
        reference_activity_db: float = -60.0,
        download_root: str | Path | None = None,
        whisper_model: nn.Module | None = None,
        mel_filters: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if not math.isfinite(reference_activity_db):
            raise ValueError(f"reference_activity_db must be finite, got {reference_activity_db}")

        if whisper_model is None:
            try:
                import whisper
            except ModuleNotFoundError as exc:
                raise ImportError(
                    "WhisperFeatureMatchingLoss requires the optional 'openai-whisper' package. "
                    "Install requirements-perceptual.txt before using the CLAP/Whisper recipe."
                ) from exc
            whisper_model = whisper.load_model(
                model_name,
                device="cpu",
                download_root=None if download_root is None else str(Path(download_root).expanduser()),
            )
            if mel_filters is None:
                mel_filters = whisper.audio.mel_filters("cpu", int(whisper_model.dims.n_mels))

        if not hasattr(whisper_model, "encoder") or not hasattr(whisper_model, "dims"):
            raise ValueError("whisper_model must expose official-style 'encoder' and 'dims' attributes")
        n_mels = int(whisper_model.dims.n_mels)
        if mel_filters is None:
            raise ValueError("mel_filters must be provided when injecting whisper_model")
        mel_filters = torch.as_tensor(mel_filters, dtype=torch.float32)
        if tuple(mel_filters.shape) != (n_mels, 201):
            raise ValueError(f"Expected Whisper mel filters [{n_mels}, 201], got {tuple(mel_filters.shape)}")

        n_blocks = len(whisper_model.encoder.blocks)
        if n_blocks <= 0:
            raise ValueError("Whisper encoder must contain at least one block")
        if selected_layers is None:
            selected_layers = (n_blocks // 4, n_blocks // 2, (3 * n_blocks) // 4, n_blocks - 1)
        parsed_layers = tuple(sorted(set(int(layer) for layer in selected_layers)))
        if not parsed_layers or parsed_layers[0] < 0 or parsed_layers[-1] >= n_blocks:
            raise ValueError(f"selected_layers must be within [0, {n_blocks - 1}], got {parsed_layers}")

        self.sample_rate = int(sample_rate)
        self.reference_activity_db = float(reference_activity_db)
        self.selected_layers = parsed_layers
        self.register_buffer("mel_filters", mel_filters, persistent=True)
        self.register_buffer("stft_window", torch.hann_window(400), persistent=False)
        self._set_external_model(whisper_model)

    @property
    def whisper_model(self) -> nn.Module:
        return self.external_model

    def _autocast_disabled(self, audio: torch.Tensor):
        if audio.device.type in {"cpu", "cuda"}:
            return torch.autocast(device_type=audio.device.type, enabled=False)
        return nullcontext()

    def _log_mel(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.float()
        if self.sample_rate != 16000:
            audio = AF.resample(audio, orig_freq=self.sample_rate, new_freq=16000)
        if audio.shape[-1] < 400:
            audio = F.pad(audio, (0, 400 - audio.shape[-1]))
        stft = torch.stft(
            audio,
            n_fft=400,
            hop_length=160,
            window=self.stft_window.float(),
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs().square()
        mel = torch.matmul(self.mel_filters.float(), magnitudes)
        log_spec = mel.clamp_min(1.0e-10).log10()
        maximum = log_spec.flatten(1).amax(dim=-1).view(-1, 1, 1)
        log_spec = torch.maximum(log_spec, maximum - 8.0)
        return (log_spec + 4.0) / 4.0

    def _encoder_features(self, mel: torch.Tensor) -> tuple[torch.Tensor, ...]:
        encoder = self.whisper_model.encoder
        value = F.gelu(encoder.conv1(mel))
        value = F.gelu(encoder.conv2(value)).permute(0, 2, 1)
        positional = encoder.positional_embedding
        if value.shape[1] > positional.shape[0]:
            raise ValueError(
                f"Whisper input produces {value.shape[1]} encoder frames, exceeding its "
                f"{positional.shape[0]}-frame context"
            )
        value = (value + positional[: value.shape[1]]).to(value.dtype)
        selected = []
        selected_set = set(self.selected_layers)
        for layer_idx, block in enumerate(encoder.blocks):
            value = block(value)
            if layer_idx in selected_set:
                selected.append(value)
        return tuple(selected)

    def forward(self, estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        estimate = _as_mono_batch(estimate, name="estimate")
        reference = _as_mono_batch(reference, name="reference")
        if estimate.shape != reference.shape:
            raise ValueError(f"Whisper estimate/reference shapes differ: {estimate.shape} != {reference.shape}")
        reference_power = reference.float().square().mean(dim=-1)
        active = reference_power > 10.0 ** (self.reference_activity_db / 10.0)
        if not active.any():
            return estimate.float().sum() * 0.0

        with self._autocast_disabled(estimate):
            active_estimate = estimate[active].float()
            active_reference = reference[active].float()
            with torch.no_grad():
                target_features = self._encoder_features(self._log_mel(active_reference))
            estimate_features = self._encoder_features(self._log_mel(active_estimate))
            losses = [
                F.l1_loss(estimate_feature.float(), target_feature.float())
                for estimate_feature, target_feature in zip(estimate_features, target_features, strict=True)
            ]
        return torch.stack(losses).mean()


class ClapSemanticLoss(_FrozenExternalModelLoss):
    """CLAP text-audio semantic anchoring and anti-bleed loss."""

    max_audio_seconds = 10.0

    _DEFAULT_POSITIVE_PROMPTS = {
        "speech": "clear human speech and dialogue",
        "music": "background music and instrumental music",
        "effects": "sound effects, foley, and environmental ambience",
    }
    _DEFAULT_NEGATIVE_PROMPTS = {
        "music": ("human speaking, dialogue, or singing voice",),
        "effects": ("human speaking or dialogue", "music or singing"),
    }

    def __init__(
        self,
        *,
        sample_rate: int,
        source_order: Sequence[str] = ("speech", "music", "effects"),
        prompt_config_path: str | Path | None = None,
        positive_prompts: Mapping[str, str] | None = None,
        negative_prompts: Mapping[str, Sequence[str] | str] | None = None,
        positive_weight: float = 0.5,
        negative_weight: float = 1.0,
        negative_margin: float = 0.0,
        reference_activity_db: float = -60.0,
        checkpoint_path: str | Path | None = None,
        model_id: int = 1,
        allow_download: bool = False,
        clap_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        self.source_order = tuple(str(source) for source in source_order)
        if not self.source_order or len(set(self.source_order)) != len(self.source_order):
            raise ValueError(f"source_order must contain unique source names, got {self.source_order}")
        for name, value in {
            "positive_weight": positive_weight,
            "negative_weight": negative_weight,
        }.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if not math.isfinite(negative_margin):
            raise ValueError(f"negative_margin must be finite, got {negative_margin}")
        if not math.isfinite(reference_activity_db):
            raise ValueError(f"reference_activity_db must be finite, got {reference_activity_db}")

        if clap_model is None:
            try:
                import laion_clap
            except ModuleNotFoundError as exc:
                raise ImportError(
                    "ClapSemanticLoss requires the optional 'laion-clap' package. "
                    "Install requirements-perceptual.txt before using the CLAP/Whisper recipe."
                ) from exc
            if checkpoint_path is None and not allow_download:
                raise ValueError("Set checkpoint_path or allow_download=True before constructing ClapSemanticLoss")
            clap_model = laion_clap.CLAP_Module(enable_fusion=False, device="cpu")
            clap_model.load_ckpt(
                ckpt=None if checkpoint_path is None else str(Path(checkpoint_path).expanduser()),
                model_id=int(model_id),
            )
        for method_name in ("get_text_embedding", "get_audio_embedding_from_data"):
            if not hasattr(clap_model, method_name):
                raise ValueError(f"clap_model must provide {method_name}()")

        configured_positive: Mapping[str, Any] = {}
        configured_negative: Mapping[str, Any] = {}
        if prompt_config_path is not None:
            prompt_path = Path(prompt_config_path).expanduser()
            if not prompt_path.is_file():
                raise FileNotFoundError(f"CLAP prompt config does not exist: {prompt_path}")
            loaded_prompts = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_prompts, Mapping):
                raise ValueError(f"CLAP prompt config must contain a mapping: {prompt_path}")
            configured_positive = loaded_prompts.get("positive_prompts", {})
            configured_negative = loaded_prompts.get("negative_prompts", {})
            if not isinstance(configured_positive, Mapping) or not isinstance(configured_negative, Mapping):
                raise ValueError("positive_prompts and negative_prompts in the CLAP prompt config must be mappings")

        positive = {
            source: self._DEFAULT_POSITIVE_PROMPTS[source]
            for source in self.source_order
            if source in self._DEFAULT_POSITIVE_PROMPTS
        }
        positive.update({str(key): str(value) for key, value in configured_positive.items()})
        if positive_prompts is not None:
            positive.update({str(key): str(value) for key, value in positive_prompts.items()})
        missing_positive = sorted(set(self.source_order) - set(positive))
        if missing_positive:
            raise ValueError(f"positive_prompts is missing sources: {missing_positive}")

        negative: dict[str, tuple[str, ...]] = {
            source: self._DEFAULT_NEGATIVE_PROMPTS[source]
            for source in self.source_order
            if source in self._DEFAULT_NEGATIVE_PROMPTS
        }
        for source, prompts in configured_negative.items():
            negative[str(source)] = (str(prompts),) if isinstance(prompts, str) else tuple(map(str, prompts))
        if negative_prompts is not None:
            for source, prompts in negative_prompts.items():
                negative[str(source)] = (str(prompts),) if isinstance(prompts, str) else tuple(map(str, prompts))
        unknown_prompt_sources = sorted((set(positive) | set(negative)) - set(self.source_order))
        if unknown_prompt_sources:
            raise ValueError(f"Prompt mappings contain sources outside source_order: {unknown_prompt_sources}")

        self.sample_rate = int(sample_rate)
        self.positive_weight = float(positive_weight)
        self.negative_weight = float(negative_weight)
        self.negative_margin = float(negative_margin)
        self.reference_activity_db = float(reference_activity_db)
        self._set_external_model(clap_model)

        with torch.no_grad():
            positive_embeddings = self.clap_model.get_text_embedding(
                [positive[source] for source in self.source_order],
                use_tensor=True,
            ).float()
            positive_embeddings = F.normalize(positive_embeddings, dim=-1)
            negative_embeddings = []
            has_negative = []
            for source_idx, source in enumerate(self.source_order):
                prompts = negative.get(source, ())
                has_negative.append(bool(prompts))
                if prompts:
                    embeddings = self.clap_model.get_text_embedding(list(prompts), use_tensor=True).float()
                    negative_embeddings.append(F.normalize(F.normalize(embeddings, dim=-1).mean(dim=0), dim=0))
                else:
                    negative_embeddings.append(positive_embeddings[source_idx])
        self.register_buffer("positive_text_embeddings", positive_embeddings, persistent=True)
        self.register_buffer("negative_text_embeddings", torch.stack(negative_embeddings), persistent=True)
        self.register_buffer("has_negative_prompt", torch.tensor(has_negative, dtype=torch.bool), persistent=True)

    @property
    def clap_model(self) -> Any:
        return self.external_model

    def window_bounds(self, n_samples: int) -> tuple[tuple[int, int], ...]:
        window_samples = int(round(self.max_audio_seconds * self.sample_rate))
        starts = _coverage_window_starts(n_samples, window_samples)
        return tuple((start, min(n_samples, start + window_samples)) for start in starts)

    def _audio_embeddings_by_window(self, stems: torch.Tensor) -> torch.Tensor:
        batch, n_sources, n_samples = stems.shape
        flattened = stems.reshape(batch * n_sources, n_samples).float()
        if self.sample_rate != 48000:
            flattened = AF.resample(flattened, orig_freq=self.sample_rate, new_freq=48000)
        max_samples = 480000
        starts = _coverage_window_starts(flattened.shape[-1], max_samples)
        window_embeddings = []
        for start in starts:
            window = flattened[..., start : start + max_samples]
            embeddings = self.clap_model.get_audio_embedding_from_data(window, use_tensor=True).float()
            window_embeddings.append(F.normalize(embeddings, dim=-1).reshape(batch, n_sources, -1))
        return torch.stack(window_embeddings, dim=2)

    def semantic_window_scores(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stems = _as_stem_batch(audio, n_sources=len(self.source_order), name="audio")
        context = (
            torch.autocast(device_type=stems.device.type, enabled=False)
            if stems.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with context:
            embeddings = self._audio_embeddings_by_window(stems.float())
            positive = (embeddings * self.positive_text_embeddings[None, :, None].float()).sum(dim=-1)
            negative = (embeddings * self.negative_text_embeddings[None, :, None].float()).sum(dim=-1)
        return positive, negative

    def semantic_scores(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positive, negative = self.semantic_window_scores(audio)
        bounds = self.window_bounds(audio.shape[-1])
        durations = positive.new_tensor([end - start for start, end in bounds])
        weights = durations / durations.sum()
        return (positive * weights).sum(dim=-1), (negative * weights).sum(dim=-1)

    def source_has_negative_prompt(self, source: str) -> bool:
        try:
            source_index = self.source_order.index(str(source))
        except ValueError as exc:
            raise ValueError(f"Unknown CLAP source: {source!r}") from exc
        return bool(self.has_negative_prompt[source_index])

    def forward(self, estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        estimate_stems = _as_stem_batch(estimate, n_sources=len(self.source_order), name="estimate")
        reference_stems = _as_stem_batch(reference, n_sources=len(self.source_order), name="reference")
        if estimate_stems.shape != reference_stems.shape:
            raise ValueError(
                f"CLAP estimate/reference shapes differ: {estimate_stems.shape} != {reference_stems.shape}"
            )
        active = reference_stems.float().square().mean(dim=-1) > 10.0 ** (self.reference_activity_db / 10.0)
        if not active.any():
            return estimate_stems.float().sum() * 0.0

        positive_similarity, negative_similarity = self.semantic_scores(estimate_stems)
        positive_loss = ((1.0 - positive_similarity) * active).sum() / active.sum().clamp_min(1)
        negative_mask = active & self.has_negative_prompt[None]
        if negative_mask.any():
            negative_penalty = F.relu(negative_similarity - self.negative_margin)
            negative_loss = (negative_penalty * negative_mask).sum() / negative_mask.sum().clamp_min(1)
        else:
            negative_loss = estimate_stems.float().sum() * 0.0
        return self.positive_weight * positive_loss + self.negative_weight * negative_loss
