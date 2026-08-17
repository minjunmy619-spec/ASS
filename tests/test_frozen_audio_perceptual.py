from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from spectral_feature_compression.core.loss.frozen_audio_perceptual import (
    ClapSemanticLoss,
    WhisperFeatureMatchingLoss,
)


class _DummyWhisperBlock(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(channels, channels, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + torch.tanh(self.projection(value))


class _DummyWhisperEncoder(torch.nn.Module):
    def __init__(self, n_mels: int = 4, channels: int = 8) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv1d(n_mels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", torch.randn(64, channels))
        self.blocks = torch.nn.ModuleList([_DummyWhisperBlock(channels) for _ in range(3)])
        self.ln_post = torch.nn.LayerNorm(channels)


class _DummyWhisper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dims = SimpleNamespace(n_mels=4)
        self.encoder = _DummyWhisperEncoder()


class _DummyClap(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 3, bias=False)

    def get_text_embedding(self, prompts: list[str], *, use_tensor: bool) -> torch.Tensor:
        assert use_tensor is True
        embeddings = []
        for prompt in prompts:
            normalized = prompt.lower()
            if "speech" in normalized or "dialogue" in normalized or "speaking" in normalized:
                embeddings.append([1.0, 0.0, 0.0])
            elif "music" in normalized:
                embeddings.append([0.0, 1.0, 0.0])
            else:
                embeddings.append([0.0, 0.0, 1.0])
        return torch.tensor(embeddings, dtype=torch.float32)

    def get_audio_embedding_from_data(self, value: torch.Tensor, *, use_tensor: bool) -> torch.Tensor:
        assert use_tensor is True
        features = torch.stack(
            [
                value.mean(dim=-1),
                value.abs().mean(dim=-1),
                value.square().mean(dim=-1).sqrt(),
            ],
            dim=-1,
        )
        return self.projection(features)


class _RecordingClap(_DummyClap):
    def __init__(self) -> None:
        super().__init__()
        self.audio_input_shapes: list[tuple[int, ...]] = []

    def get_audio_embedding_from_data(self, value: torch.Tensor, *, use_tensor: bool) -> torch.Tensor:
        self.audio_input_shapes.append(tuple(value.shape))
        return super().get_audio_embedding_from_data(value, use_tensor=use_tensor)


def test_whisper_feature_matching_is_frozen_but_backpropagates_to_estimate() -> None:
    whisper = _DummyWhisper()
    mel_filters = torch.rand(4, 201)
    loss_module = WhisperFeatureMatchingLoss(
        sample_rate=16000,
        selected_layers=(0, 2),
        whisper_model=whisper,
        mel_filters=mel_filters,
    )
    estimate = torch.randn(2, 1, 1600, requires_grad=True)
    reference = torch.randn_like(estimate)

    loss = loss_module(estimate, reference)
    loss.backward()

    assert loss > 0.0
    assert estimate.grad is not None
    assert float(estimate.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in whisper.parameters())
    assert all(not parameter.requires_grad for parameter in whisper.parameters())
    assert tuple(loss_module.state_dict()) == ("mel_filters",)
    assert tuple(loss_module.named_parameters()) == ()


def test_whisper_feature_matching_ignores_reference_silent_examples() -> None:
    loss_module = WhisperFeatureMatchingLoss(
        sample_rate=16000,
        selected_layers=(1,),
        reference_activity_db=-80.0,
        whisper_model=_DummyWhisper(),
        mel_filters=torch.rand(4, 201),
    )
    estimate = torch.randn(2, 1600, requires_grad=True)
    reference = torch.zeros_like(estimate)

    loss = loss_module(estimate, reference)
    loss.backward()

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    torch.testing.assert_close(estimate.grad, torch.zeros_like(estimate))


def test_clap_semantic_loss_is_frozen_and_returns_per_stem_scores() -> None:
    clap = _DummyClap()
    loss_module = ClapSemanticLoss(
        sample_rate=48000,
        source_order=("speech", "music", "effects"),
        positive_prompts={
            "speech": "clear speech dialogue",
            "music": "background music",
            "effects": "sound effects foley",
        },
        negative_prompts={
            "music": ("human speech",),
            "effects": ("human speaking", "music"),
        },
        clap_model=clap,
    )
    estimate = torch.randn(2, 3, 1, 2400, requires_grad=True)
    reference = torch.ones_like(estimate)

    loss = loss_module(estimate, reference)
    positive, negative = loss_module.semantic_scores(estimate)
    loss.backward()

    assert loss.ndim == 0
    assert tuple(positive.shape) == (2, 3)
    assert tuple(negative.shape) == (2, 3)
    assert estimate.grad is not None
    assert float(estimate.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in clap.parameters())
    assert all(not parameter.requires_grad for parameter in clap.parameters())
    assert set(loss_module.state_dict()) == {
        "positive_text_embeddings",
        "negative_text_embeddings",
        "has_negative_prompt",
    }
    assert tuple(loss_module.named_parameters()) == ()


def test_clap_semantic_loss_masks_reference_silent_stems() -> None:
    loss_module = ClapSemanticLoss(
        sample_rate=48000,
        source_order=("speech", "music", "effects"),
        clap_model=_DummyClap(),
    )
    estimate = torch.randn(1, 3, 1, 2400, requires_grad=True)
    reference = torch.zeros_like(estimate)

    loss = loss_module(estimate, reference)
    loss.backward()

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    torch.testing.assert_close(estimate.grad, torch.zeros_like(estimate))


def test_clap_semantic_scores_cover_all_long_audio_windows() -> None:
    clap = _RecordingClap()
    loss_module = ClapSemanticLoss(sample_rate=48000, clap_model=clap)
    audio = torch.ones(1, 3, 1, 15 * 48000)

    positive, negative = loss_module.semantic_window_scores(audio)

    assert tuple(positive.shape) == (1, 3, 2)
    assert tuple(negative.shape) == (1, 3, 2)
    assert clap.audio_input_shapes[-2:] == [(3, 480000), (3, 240000)]
    assert loss_module.window_bounds(audio.shape[-1]) == ((0, 480000), (480000, 720000))
    assert loss_module.source_has_negative_prompt("speech") is False
    assert loss_module.source_has_negative_prompt("music") is True


def test_clap_prompt_config_is_shared_and_can_be_overridden(tmp_path: Path) -> None:
    prompt_config = tmp_path / "clap_prompts.yaml"
    prompt_config.write_text(
        """positive_prompts:
  speech: television dialogue
  music: instrumental soundtrack
  effects: environmental effects
negative_prompts:
  speech: music or sound effects
  music: spoken dialogue
  effects: spoken dialogue
""",
        encoding="utf-8",
    )

    loss_module = ClapSemanticLoss(
        sample_rate=48000,
        prompt_config_path=prompt_config,
        clap_model=_DummyClap(),
    )

    assert loss_module.source_has_negative_prompt("speech") is True
    assert loss_module.source_has_negative_prompt("music") is True
    assert loss_module.source_has_negative_prompt("effects") is True
