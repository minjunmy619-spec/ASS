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
        selected_layer_policy: str = "quartiles",
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
        selected_layer_policy = str(selected_layer_policy).lower()
        if selected_layer_policy not in {"quartiles", "middle"}:
            raise ValueError("selected_layer_policy must be 'quartiles' or 'middle'")
        if selected_layers is None:
            if selected_layer_policy == "quartiles":
                selected_layers = (n_blocks // 4, n_blocks // 2, (3 * n_blocks) // 4, n_blocks - 1)
            else:
                count = min(3, n_blocks)
                start = (n_blocks - count) // 2
                selected_layers = tuple(range(start, start + count))
        parsed_layers = tuple(sorted(set(int(layer) for layer in selected_layers)))
        if not parsed_layers or parsed_layers[0] < 0 or parsed_layers[-1] >= n_blocks:
            raise ValueError(f"selected_layers must be within [0, {n_blocks - 1}], got {parsed_layers}")

        self.sample_rate = int(sample_rate)
        self.reference_activity_db = float(reference_activity_db)
        self.selected_layers = parsed_layers
        self.selected_layer_policy = selected_layer_policy
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


class WhisperStemPerceptualLoss(WhisperFeatureMatchingLoss):
    """Frozen Whisper supervision for Speech fidelity and relative cross-stem bleed.

    Unlike a zero-target penalty, cross-stem similarity is compared with the
    clean target/reference similarity. This preserves legitimate singing in
    Music and voice-like Effects that occur in the labelled target stem.
    """

    def __init__(
        self,
        *,
        source_order: Sequence[str] = ("speech", "music", "effects"),
        speech_source: str = "speech",
        bleed_target_sources: Sequence[str] | None = None,
        speech_feature_match_weight: float = 1.0,
        cross_stem_bleed_weight: float = 0.0,
        speech_active_db: float = -45.0,
        target_quiet_relative_db: float = 12.0,
        mask_softness_db: float = 3.0,
        relative_bleed_margin: float = 0.0,
        **kwargs: Any,
    ) -> None:
        source_order = tuple(str(source) for source in source_order)
        if not source_order or len(set(source_order)) != len(source_order):
            raise ValueError(f"source_order must contain unique source names, got {source_order}")
        speech_source = str(speech_source)
        if speech_source not in source_order:
            raise ValueError(f"speech_source={speech_source!r} is not in source_order={source_order}")
        targets = (
            tuple(source for source in source_order if source != speech_source)
            if bleed_target_sources is None
            else tuple(str(source) for source in bleed_target_sources)
        )
        if not targets:
            raise ValueError("bleed_target_sources must contain at least one non-speech source")
        unknown = sorted(set(targets) - set(source_order))
        if unknown:
            raise ValueError(f"bleed_target_sources contains unknown sources: {unknown}")
        if speech_source in targets:
            raise ValueError("bleed_target_sources must not contain speech_source")
        for name, value in {
            "speech_feature_match_weight": speech_feature_match_weight,
            "cross_stem_bleed_weight": cross_stem_bleed_weight,
            "speech_active_db": speech_active_db,
            "target_quiet_relative_db": target_quiet_relative_db,
            "mask_softness_db": mask_softness_db,
            "relative_bleed_margin": relative_bleed_margin,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if speech_feature_match_weight < 0.0 or cross_stem_bleed_weight < 0.0:
            raise ValueError("Whisper stem perceptual weights must be non-negative")
        if mask_softness_db <= 0.0:
            raise ValueError("mask_softness_db must be positive")

        super().__init__(**kwargs)
        self.source_order = source_order
        self.speech_source = speech_source
        self.speech_source_index = source_order.index(speech_source)
        self.bleed_target_sources = targets
        self.bleed_target_indices = tuple(source_order.index(source) for source in targets)
        self.speech_feature_match_weight = float(speech_feature_match_weight)
        self.cross_stem_bleed_weight = float(cross_stem_bleed_weight)
        self.speech_active_db = float(speech_active_db)
        self.target_quiet_relative_db = float(target_quiet_relative_db)
        self.mask_softness_db = float(mask_softness_db)
        self.relative_bleed_margin = float(relative_bleed_margin)

    def _stem_features(self, stems: torch.Tensor, *, require_grad: bool) -> tuple[torch.Tensor, ...]:
        batch, source_count, samples = stems.shape
        flattened = stems.reshape(batch * source_count, samples)
        context = nullcontext() if require_grad else torch.no_grad()
        with context:
            features = self._encoder_features(self._log_mel(flattened))
        return tuple(
            feature.reshape(batch, source_count, feature.shape[1], feature.shape[2]) for feature in features
        )

    @staticmethod
    def _frame_power(stems: torch.Tensor, n_frames: int) -> torch.Tensor:
        return F.adaptive_avg_pool1d(stems.float().square(), n_frames)

    def loss_components(self, estimate: torch.Tensor, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        estimate_stems = _as_stem_batch(estimate, n_sources=len(self.source_order), name="estimate")
        reference_stems = _as_stem_batch(reference, n_sources=len(self.source_order), name="reference")
        if estimate_stems.shape != reference_stems.shape:
            raise ValueError(
                f"Whisper estimate/reference shapes differ: {estimate_stems.shape} != {reference_stems.shape}"
            )
        zero = estimate_stems.float().sum() * 0.0
        components = {
            "speech_feature_match": zero,
            "cross_stem_bleed": zero,
            **{f"cross_stem_bleed_{source}": zero for source in self.bleed_target_sources},
        }
        if self.speech_feature_match_weight == 0.0 and self.cross_stem_bleed_weight == 0.0:
            return components

        with self._autocast_disabled(estimate_stems):
            estimate_features = self._stem_features(estimate_stems.float(), require_grad=True)
            reference_features = self._stem_features(reference_stems.float(), require_grad=False)
            speech_reference = reference_stems[:, self.speech_source_index]
            speech_active_samples = speech_reference.float().square().mean(dim=-1)
            batch_active = speech_active_samples > 10.0 ** (self.reference_activity_db / 10.0)
            if self.speech_feature_match_weight > 0.0 and batch_active.any():
                feature_losses = [
                    F.l1_loss(
                        estimate_feature[batch_active, self.speech_source_index].float(),
                        reference_feature[batch_active, self.speech_source_index].float(),
                    )
                    for estimate_feature, reference_feature in zip(
                        estimate_features,
                        reference_features,
                        strict=True,
                    )
                ]
                components["speech_feature_match"] = torch.stack(feature_losses).mean()

            if self.cross_stem_bleed_weight > 0.0:
                target_losses: dict[str, list[torch.Tensor]] = {
                    source: [] for source in self.bleed_target_sources
                }
                for estimate_feature, reference_feature in zip(estimate_features, reference_features, strict=True):
                    n_frames = estimate_feature.shape[2]
                    frame_power = self._frame_power(reference_stems, n_frames)
                    speech_power = frame_power[:, self.speech_source_index]
                    speech_db = 10.0 * torch.log10(speech_power.clamp_min(1.0e-10))
                    speech_active = torch.sigmoid((speech_db - self.speech_active_db) / self.mask_softness_db)
                    speech_reference_feature = F.normalize(
                        reference_feature[:, self.speech_source_index].float(), dim=-1
                    )
                    for source, target_index in zip(
                        self.bleed_target_sources,
                        self.bleed_target_indices,
                        strict=True,
                    ):
                        target_power = frame_power[:, target_index]
                        target_db = 10.0 * torch.log10(target_power.clamp_min(1.0e-10))
                        target_quiet = torch.sigmoid(
                            (speech_db - target_db - self.target_quiet_relative_db) / self.mask_softness_db
                        )
                        mask = speech_active * target_quiet
                        estimate_target = F.normalize(estimate_feature[:, target_index].float(), dim=-1)
                        reference_target = F.normalize(reference_feature[:, target_index].float(), dim=-1)
                        estimate_similarity = (estimate_target * speech_reference_feature).sum(dim=-1)
                        reference_similarity = (reference_target * speech_reference_feature).sum(dim=-1)
                        excess = F.relu(estimate_similarity - reference_similarity - self.relative_bleed_margin)
                        denominator = mask.sum()
                        value = zero if not bool(denominator > 0) else (mask * excess).sum() / denominator
                        target_losses[source].append(value)
                for source, losses in target_losses.items():
                    components[f"cross_stem_bleed_{source}"] = torch.stack(losses).mean()
                components["cross_stem_bleed"] = torch.stack(
                    [components[f"cross_stem_bleed_{source}"] for source in self.bleed_target_sources]
                ).mean()
        return components

    def combine_loss_components(self, components: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.speech_feature_match_weight * components["speech_feature_match"]
            + self.cross_stem_bleed_weight * components["cross_stem_bleed"]
        )

    def forward_with_components(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = self.loss_components(estimate, reference)
        return self.combine_loss_components(components), components

    def forward(self, estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        total, _ = self.forward_with_components(estimate, reference)
        return total


class ClapSemanticLoss(_FrozenExternalModelLoss):
    """CLAP text and reference-audio supervision for separated stems.

    The legacy positive/negative text losses remain available. Optional
    reference matching, relative cross-stem anti-bleed, and prompt-bank
    classification share the same frozen CLAP model and audio embeddings.
    """

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
        audio_match_weight: float = 0.0,
        audio_antibleed_weight: float = 0.0,
        audio_antibleed_margin: float = 0.0,
        prompt_banks: Mapping[str, Sequence[str] | str] | None = None,
        prompt_bank_weight: float = 0.0,
        prompt_bank_temperature: float = 0.07,
        reference_activity_db: float = -60.0,
        checkpoint_path: str | Path | None = None,
        model_id: int = 1,
        amodel: str = "HTSAT-tiny",
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
            "audio_match_weight": audio_match_weight,
            "audio_antibleed_weight": audio_antibleed_weight,
            "prompt_bank_weight": prompt_bank_weight,
        }.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        for name, value in {
            "negative_margin": negative_margin,
            "audio_antibleed_margin": audio_antibleed_margin,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if not math.isfinite(prompt_bank_temperature) or prompt_bank_temperature <= 0.0:
            raise ValueError(
                f"prompt_bank_temperature must be finite and positive, got {prompt_bank_temperature}"
            )
        if not math.isfinite(reference_activity_db):
            raise ValueError(f"reference_activity_db must be finite, got {reference_activity_db}")
        amodel = str(amodel).strip()
        if not amodel:
            raise ValueError("amodel must not be empty")

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
            clap_model = laion_clap.CLAP_Module(enable_fusion=False, device="cpu", amodel=amodel)
            clap_model.load_ckpt(
                ckpt=None if checkpoint_path is None else str(Path(checkpoint_path).expanduser()),
                model_id=int(model_id),
            )
        for method_name in ("get_text_embedding", "get_audio_embedding_from_data"):
            if not hasattr(clap_model, method_name):
                raise ValueError(f"clap_model must provide {method_name}()")

        configured_positive: Mapping[str, Any] = {}
        configured_negative: Mapping[str, Any] = {}
        configured_banks: Mapping[str, Any] = {}
        if prompt_config_path is not None:
            prompt_path = Path(prompt_config_path).expanduser()
            if not prompt_path.is_file():
                raise FileNotFoundError(f"CLAP prompt config does not exist: {prompt_path}")
            loaded_prompts = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_prompts, Mapping):
                raise ValueError(f"CLAP prompt config must contain a mapping: {prompt_path}")
            configured_positive = loaded_prompts.get("positive_prompts", {})
            configured_negative = loaded_prompts.get("negative_prompts", {})
            configured_banks = loaded_prompts.get("prompt_banks", {})
            if not all(
                isinstance(value, Mapping)
                for value in (configured_positive, configured_negative, configured_banks)
            ):
                raise ValueError(
                    "positive_prompts, negative_prompts, and prompt_banks in the CLAP prompt config "
                    "must be mappings"
                )

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

        banks: dict[str, tuple[str, ...]] = {}
        for source, prompts in configured_banks.items():
            banks[str(source)] = (str(prompts),) if isinstance(prompts, str) else tuple(map(str, prompts))
        if prompt_banks is not None:
            for source, prompts in prompt_banks.items():
                banks[str(source)] = (str(prompts),) if isinstance(prompts, str) else tuple(map(str, prompts))
        for source, prompts in banks.items():
            if not prompts or any(not prompt.strip() for prompt in prompts):
                raise ValueError(f"prompt_banks[{source!r}] must contain non-empty prompts")

        unknown_prompt_sources = sorted((set(positive) | set(negative) | set(banks)) - set(self.source_order))
        if unknown_prompt_sources:
            raise ValueError(f"Prompt mappings contain sources outside source_order: {unknown_prompt_sources}")
        if banks:
            missing_banks = sorted(set(self.source_order) - set(banks))
            if missing_banks:
                raise ValueError(f"prompt_banks is missing sources: {missing_banks}")
        elif prompt_bank_weight > 0.0:
            raise ValueError("prompt_bank_weight > 0 requires prompt_banks for every source")

        self.sample_rate = int(sample_rate)
        self.positive_weight = float(positive_weight)
        self.negative_weight = float(negative_weight)
        self.negative_margin = float(negative_margin)
        self.audio_match_weight = float(audio_match_weight)
        self.audio_antibleed_weight = float(audio_antibleed_weight)
        self.audio_antibleed_margin = float(audio_antibleed_margin)
        self.prompt_bank_weight = float(prompt_bank_weight)
        self.prompt_bank_temperature = float(prompt_bank_temperature)
        self.reference_activity_db = float(reference_activity_db)
        self.amodel = amodel
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
            bank_prompts = [prompt for source in self.source_order for prompt in banks.get(source, ())]
            bank_source_indices = [
                source_idx
                for source_idx, source in enumerate(self.source_order)
                for _ in banks.get(source, ())
            ]
            if bank_prompts:
                prompt_bank_embeddings = self.clap_model.get_text_embedding(
                    bank_prompts,
                    use_tensor=True,
                ).float()
                prompt_bank_embeddings = F.normalize(prompt_bank_embeddings, dim=-1)
            else:
                prompt_bank_embeddings = positive_embeddings.new_empty((0, positive_embeddings.shape[-1]))
        self.register_buffer("positive_text_embeddings", positive_embeddings, persistent=False)
        self.register_buffer("negative_text_embeddings", torch.stack(negative_embeddings), persistent=False)
        self.register_buffer("has_negative_prompt", torch.tensor(has_negative, dtype=torch.bool), persistent=False)
        self.register_buffer(
            "prompt_bank_text_embeddings",
            prompt_bank_embeddings,
            persistent=False,
        )
        self.register_buffer(
            "prompt_bank_source_indices",
            torch.tensor(bank_source_indices, dtype=torch.long),
            persistent=False,
        )

    @property
    def clap_model(self) -> Any:
        return self.external_model

    @property
    def has_prompt_banks(self) -> bool:
        return self.prompt_bank_text_embeddings.shape[0] > 0

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

    def _reference_activity_by_window(self, reference_stems: torch.Tensor) -> torch.Tensor:
        threshold = 10.0 ** (self.reference_activity_db / 10.0)
        return torch.stack(
            [
                reference_stems[..., start:end].float().square().mean(dim=-1) > threshold
                for start, end in self.window_bounds(reference_stems.shape[-1])
            ],
            dim=-1,
        )

    def _window_durations(self, n_samples: int, *, like: torch.Tensor) -> torch.Tensor:
        return like.new_tensor([end - start for start, end in self.window_bounds(n_samples)])

    def _duration_masked_mean(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        *,
        n_samples: int,
    ) -> torch.Tensor:
        if values.shape != mask.shape:
            raise ValueError(f"values and mask shapes differ: {values.shape} != {mask.shape}")
        durations = self._window_durations(n_samples, like=values)
        duration_shape = (1,) * (values.ndim - 1) + (durations.numel(),)
        weights = mask.to(values.dtype) * durations.reshape(duration_shape)
        denominator = weights.sum()
        if not bool(denominator > 0):
            return values.sum() * 0.0
        return (values * weights).sum() / denominator

    def _semantic_window_scores_from_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positive = (embeddings * self.positive_text_embeddings[None, :, None].float()).sum(dim=-1)
        negative = (embeddings * self.negative_text_embeddings[None, :, None].float()).sum(dim=-1)
        return positive, negative

    def _prompt_bank_window_scores_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.prompt_bank_text_embeddings.shape[0] == 0:
            raise RuntimeError("CLAP prompt-bank scores require configured prompt_banks")
        prompt_logits = torch.einsum(
            "bswe,pe->bswp",
            embeddings,
            self.prompt_bank_text_embeddings.float(),
        ) / self.prompt_bank_temperature
        bank_scores = []
        for source_idx in range(len(self.source_order)):
            source_logits = prompt_logits[..., self.prompt_bank_source_indices == source_idx]
            bank_scores.append(torch.logsumexp(source_logits, dim=-1) - math.log(source_logits.shape[-1]))
        return torch.stack(bank_scores, dim=-1)

    def _reference_scores_from_embeddings(
        self,
        estimate_embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        same_stem_similarity = (estimate_embeddings * reference_embeddings).sum(dim=-1).clamp(-1.0, 1.0)
        estimate_to_reference = torch.einsum(
            "biwe,bjwe->bijw",
            estimate_embeddings,
            reference_embeddings,
        ).clamp(-1.0, 1.0)
        reference_to_reference = torch.einsum(
            "biwe,bjwe->bijw",
            reference_embeddings,
            reference_embeddings,
        ).clamp(-1.0, 1.0)
        relative_cross_stem_excess = F.relu(
            estimate_to_reference - reference_to_reference - self.audio_antibleed_margin
        )
        return same_stem_similarity, relative_cross_stem_excess

    def _reference_window_metrics_from_embeddings(
        self,
        estimate_embeddings: torch.Tensor,
        reference_embeddings: torch.Tensor,
        reference_stems: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        same_stem_similarity, pair_excess = self._reference_scores_from_embeddings(
            estimate_embeddings,
            reference_embeddings,
        )
        active = self._reference_activity_by_window(reference_stems)
        source_count = len(self.source_order)
        off_diagonal = ~torch.eye(source_count, dtype=torch.bool, device=active.device)[None, :, :, None]
        pair_active = active[:, :, None, :] & active[:, None, :, :] & off_diagonal
        pair_count = pair_active.sum(dim=2)
        cross_stem_excess = (pair_excess * pair_active).sum(dim=2) / pair_count.clamp_min(1)
        return {
            "same_stem_similarity": same_stem_similarity,
            "relative_cross_stem_excess": cross_stem_excess,
            "reference_active": active,
            "cross_stem_valid": pair_count > 0,
        }

    def window_metrics(
        self,
        audio: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode estimates once and return all configured CLAP window metrics."""

        stems = _as_stem_batch(audio, n_sources=len(self.source_order), name="audio")
        reference_stems = None
        if reference is not None:
            reference_stems = _as_stem_batch(
                reference,
                n_sources=len(self.source_order),
                name="reference",
            )
            if stems.shape != reference_stems.shape:
                raise ValueError(f"CLAP estimate/reference shapes differ: {stems.shape} != {reference_stems.shape}")
        context = (
            torch.autocast(device_type=stems.device.type, enabled=False)
            if stems.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with context:
            embeddings = self._audio_embeddings_by_window(stems.float())
            positive, negative = self._semantic_window_scores_from_embeddings(embeddings)
            metrics = {
                "positive_similarity": positive,
                "negative_similarity": negative,
            }
            if self.has_prompt_banks:
                metrics["prompt_bank_scores"] = self._prompt_bank_window_scores_from_embeddings(embeddings)
            if reference_stems is not None:
                with torch.no_grad():
                    reference_embeddings = self._audio_embeddings_by_window(reference_stems.float())
                metrics.update(
                    self._reference_window_metrics_from_embeddings(
                        embeddings,
                        reference_embeddings,
                        reference_stems,
                    )
                )
        return metrics

    def semantic_window_scores(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        metrics = self.window_metrics(audio)
        return metrics["positive_similarity"], metrics["negative_similarity"]

    def semantic_scores(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positive, negative = self.semantic_window_scores(audio)
        bounds = self.window_bounds(audio.shape[-1])
        durations = positive.new_tensor([end - start for start, end in bounds])
        weights = durations / durations.sum()
        return (positive * weights).sum(dim=-1), (negative * weights).sum(dim=-1)

    def prompt_bank_window_scores(self, audio: torch.Tensor) -> torch.Tensor:
        """Return normalized multi-prompt class logits as ``[B, stem, window, class]``."""

        if not self.has_prompt_banks:
            raise RuntimeError("CLAP prompt-bank scores require configured prompt_banks")
        metrics = self.window_metrics(audio)
        return metrics["prompt_bank_scores"]

    def reference_window_metrics(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return reference-aware CLAP diagnostics without reducing windows."""

        metrics = self.window_metrics(estimate, reference)
        return {
            name: metrics[name]
            for name in (
                "same_stem_similarity",
                "relative_cross_stem_excess",
                "reference_active",
                "cross_stem_valid",
            )
        }

    def source_has_negative_prompt(self, source: str) -> bool:
        try:
            source_index = self.source_order.index(str(source))
        except ValueError as exc:
            raise ValueError(f"Unknown CLAP source: {source!r}") from exc
        return bool(self.has_negative_prompt[source_index])

    def loss_components(self, estimate: torch.Tensor, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute unweighted CLAP objective components using one estimate encoding pass."""

        estimate_stems = _as_stem_batch(estimate, n_sources=len(self.source_order), name="estimate")
        reference_stems = _as_stem_batch(reference, n_sources=len(self.source_order), name="reference")
        if estimate_stems.shape != reference_stems.shape:
            raise ValueError(
                f"CLAP estimate/reference shapes differ: {estimate_stems.shape} != {reference_stems.shape}"
            )
        zero = estimate_stems.float().sum() * 0.0
        components = {
            "text_positive": zero,
            "text_negative": zero,
            "audio_match": zero,
            "audio_antibleed": zero,
            "prompt_bank": zero,
        }
        active = reference_stems.float().square().mean(dim=-1) > 10.0 ** (
            self.reference_activity_db / 10.0
        )
        if not active.any() or not any(
            weight > 0.0
            for weight in (
                self.positive_weight,
                self.negative_weight,
                self.audio_match_weight,
                self.audio_antibleed_weight,
                self.prompt_bank_weight,
            )
        ):
            return components

        context = (
            torch.autocast(device_type=estimate_stems.device.type, enabled=False)
            if estimate_stems.device.type in {"cpu", "cuda"}
            else nullcontext()
        )
        with context:
            estimate_embeddings = self._audio_embeddings_by_window(estimate_stems.float())
            window_activity = self._reference_activity_by_window(reference_stems)
            if self.positive_weight > 0.0 or self.negative_weight > 0.0:
                positive, negative = self._semantic_window_scores_from_embeddings(estimate_embeddings)
                durations = self._window_durations(estimate_stems.shape[-1], like=positive)
                duration_weights = durations / durations.sum()
                positive = (positive * duration_weights).sum(dim=-1)
                negative = (negative * duration_weights).sum(dim=-1)
                if self.positive_weight > 0.0:
                    components["text_positive"] = ((1.0 - positive) * active).sum() / active.sum()
                negative_mask = active & self.has_negative_prompt[None]
                if self.negative_weight > 0.0 and negative_mask.any():
                    penalty = F.relu(negative - self.negative_margin)
                    components["text_negative"] = (penalty * negative_mask).sum() / negative_mask.sum()

            if self.audio_match_weight > 0.0 or self.audio_antibleed_weight > 0.0:
                with torch.no_grad():
                    reference_embeddings = self._audio_embeddings_by_window(reference_stems.float())
                same_stem_similarity, pair_excess = self._reference_scores_from_embeddings(
                    estimate_embeddings,
                    reference_embeddings,
                )
                if self.audio_match_weight > 0.0:
                    components["audio_match"] = self._duration_masked_mean(
                        1.0 - same_stem_similarity,
                        window_activity,
                        n_samples=estimate_stems.shape[-1],
                    )
                if self.audio_antibleed_weight > 0.0:
                    source_count = len(self.source_order)
                    off_diagonal = ~torch.eye(
                        source_count,
                        dtype=torch.bool,
                        device=window_activity.device,
                    )[None, :, :, None]
                    pair_activity = (
                        window_activity[:, :, None, :]
                        & window_activity[:, None, :, :]
                        & off_diagonal
                    )
                    components["audio_antibleed"] = self._duration_masked_mean(
                        pair_excess,
                        pair_activity,
                        n_samples=estimate_stems.shape[-1],
                    )

            if self.prompt_bank_weight > 0.0:
                bank_scores = self._prompt_bank_window_scores_from_embeddings(estimate_embeddings)
                targets = torch.arange(
                    len(self.source_order),
                    device=bank_scores.device,
                )[None, :, None].expand(bank_scores.shape[:-1])
                bank_loss = F.cross_entropy(
                    bank_scores.reshape(-1, len(self.source_order)),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape(targets.shape)
                components["prompt_bank"] = self._duration_masked_mean(
                    bank_loss,
                    window_activity,
                    n_samples=estimate_stems.shape[-1],
                )
        return components

    def combine_loss_components(self, components: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.positive_weight * components["text_positive"]
            + self.negative_weight * components["text_negative"]
            + self.audio_match_weight * components["audio_match"]
            + self.audio_antibleed_weight * components["audio_antibleed"]
            + self.prompt_bank_weight * components["prompt_bank"]
        )

    def forward_with_components(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        components = self.loss_components(estimate, reference)
        return self.combine_loss_components(components), components

    def forward(self, estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        total, _ = self.forward_with_components(estimate, reference)
        return total
