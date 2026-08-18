from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import torch

import pytest

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
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(3))

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
    assert tuple(loss_module.state_dict()) == ()
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
prompt_banks:
  speech: [human speech, spoken dialogue]
  music: [background music]
  effects: [foley effects]
""",
        encoding="utf-8",
    )

    loss_module = ClapSemanticLoss(
        sample_rate=48000,
        prompt_config_path=prompt_config,
        prompt_bank_weight=0.1,
        clap_model=_DummyClap(),
    )

    assert loss_module.source_has_negative_prompt("speech") is True
    assert loss_module.source_has_negative_prompt("music") is True
    assert loss_module.source_has_negative_prompt("effects") is True
    assert loss_module.has_prompt_banks is True
    assert tuple(loss_module.prompt_bank_window_scores(_distinct_reference_stems()).shape) == (1, 3, 1, 3)
    assert tuple(loss_module.state_dict()) == ()


def _distinct_reference_stems(samples: int = 2400) -> torch.Tensor:
    alternating = torch.ones(samples)
    alternating[1::2] = -1.0
    return torch.stack(
        (
            torch.ones(samples),
            -torch.ones(samples),
            alternating,
        )
    )[None, :, None, :]


def test_clap_audio_to_audio_loss_matches_references_and_penalizes_swaps() -> None:
    loss_module = ClapSemanticLoss(
        sample_rate=48000,
        positive_weight=0.0,
        negative_weight=0.0,
        audio_match_weight=1.0,
        audio_antibleed_weight=1.0,
        clap_model=_DummyClap(),
    )
    reference = _distinct_reference_stems()
    exact_components = loss_module.loss_components(reference, reference)
    swapped = reference[:, (1, 0, 2)].clone().requires_grad_(True)
    swapped_components = loss_module.loss_components(swapped, reference)

    torch.testing.assert_close(exact_components["audio_match"], torch.tensor(0.0), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(
        exact_components["audio_antibleed"],
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0.0,
    )
    assert swapped_components["audio_match"] > exact_components["audio_match"]
    assert swapped_components["audio_antibleed"] > exact_components["audio_antibleed"]

    perturbed = (swapped.detach() + 0.05 * torch.randn_like(swapped)).requires_grad_(True)
    active_reference = reference.clone().requires_grad_(True)
    total = loss_module(perturbed, active_reference)
    total.backward()
    assert perturbed.grad is not None
    assert float(perturbed.grad.abs().sum()) > 0.0
    assert active_reference.grad is None
    assert all(parameter.grad is None for parameter in loss_module.clap_model.parameters())


def test_clap_audio_to_audio_reference_branch_is_frozen_and_silence_is_masked() -> None:
    loss_module = ClapSemanticLoss(
        sample_rate=48000,
        positive_weight=0.0,
        negative_weight=0.0,
        audio_match_weight=1.0,
        audio_antibleed_weight=1.0,
        clap_model=_DummyClap(),
    )
    estimate = torch.randn(1, 3, 1, 2400, requires_grad=True)
    reference = torch.zeros_like(estimate, requires_grad=True)

    loss = loss_module(estimate, reference)
    loss.backward()

    torch.testing.assert_close(loss, torch.zeros_like(loss))
    torch.testing.assert_close(estimate.grad, torch.zeros_like(estimate))
    assert reference.grad is None


def test_clap_audio_to_audio_masks_inactive_final_window() -> None:
    loss_module = ClapSemanticLoss(
        sample_rate=48000,
        positive_weight=0.0,
        negative_weight=0.0,
        audio_match_weight=1.0,
        clap_model=_DummyClap(),
    )
    reference = torch.zeros(1, 3, 1, 15 * 48000)
    reference[..., : 10 * 48000] = _distinct_reference_stems(10 * 48000)
    estimate = reference.clone()
    estimate[..., 10 * 48000 :] = torch.randn_like(estimate[..., 10 * 48000 :])

    components = loss_module.loss_components(estimate, reference)

    torch.testing.assert_close(components["audio_match"], torch.tensor(0.0), atol=1e-6, rtol=0.0)


def test_clap_prompt_bank_uses_prompt_count_normalized_class_scores() -> None:
    short_banks = {
        "speech": ("human speech",),
        "music": ("background music",),
        "effects": ("foley effects",),
    }
    repeated_banks = {source: prompts * 3 for source, prompts in short_banks.items()}
    short = ClapSemanticLoss(
        sample_rate=48000,
        prompt_banks=short_banks,
        prompt_bank_weight=1.0,
        positive_weight=0.0,
        negative_weight=0.0,
        clap_model=_DummyClap(),
    )
    repeated = ClapSemanticLoss(
        sample_rate=48000,
        prompt_banks=repeated_banks,
        prompt_bank_weight=1.0,
        positive_weight=0.0,
        negative_weight=0.0,
        clap_model=_DummyClap(),
    )
    estimate = _distinct_reference_stems().requires_grad_(True)
    reference = _distinct_reference_stems()

    short_scores = short.prompt_bank_window_scores(estimate)
    repeated_scores = repeated.prompt_bank_window_scores(estimate)
    torch.testing.assert_close(short_scores, repeated_scores)
    torch.testing.assert_close(short(estimate, reference), repeated(estimate, reference))


def test_clap_combined_window_metrics_share_estimate_encoding() -> None:
    clap = _RecordingClap()
    loss_module = ClapSemanticLoss(
        sample_rate=48000,
        prompt_banks={
            "speech": ("human speech",),
            "music": ("background music",),
            "effects": ("foley effects",),
        },
        clap_model=clap,
    )
    reference = _distinct_reference_stems()

    metrics = loss_module.window_metrics(reference, reference)

    assert tuple(metrics["positive_similarity"].shape) == (1, 3, 1)
    assert tuple(metrics["prompt_bank_scores"].shape) == (1, 3, 1, 3)
    assert tuple(metrics["same_stem_similarity"].shape) == (1, 3, 1)
    assert clap.audio_input_shapes == [(3, 2400), (3, 2400)]


def test_clap_prompt_bank_requires_complete_nonempty_source_banks() -> None:
    with pytest.raises(ValueError, match="missing sources"):
        ClapSemanticLoss(
            sample_rate=48000,
            prompt_banks={"speech": ("human speech",)},
            prompt_bank_weight=1.0,
            clap_model=_DummyClap(),
        )

    with pytest.raises(ValueError, match="non-empty prompts"):
        ClapSemanticLoss(
            sample_rate=48000,
            prompt_banks={"speech": (), "music": ("music",), "effects": ("effects",)},
            prompt_bank_weight=1.0,
            clap_model=_DummyClap(),
        )


def test_clap_constructs_the_requested_audio_encoder_and_excludes_prompt_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class _LoadableDummyClap(_DummyClap):
        def load_ckpt(self, *, ckpt: str | None, model_id: int) -> None:
            calls["checkpoint"] = ckpt
            calls["model_id"] = model_id

    def construct_clap(**kwargs: object) -> _LoadableDummyClap:
        calls["constructor"] = kwargs
        return _LoadableDummyClap()

    monkeypatch.setitem(sys.modules, "laion_clap", SimpleNamespace(CLAP_Module=construct_clap))
    loss_module = ClapSemanticLoss(
        sample_rate=48000,
        checkpoint_path="/models/music_speech_audioset.pt",
        amodel="HTSAT-base",
    )

    assert calls["constructor"] == {
        "enable_fusion": False,
        "device": "cpu",
        "amodel": "HTSAT-base",
    }
    assert calls["checkpoint"] == "/models/music_speech_audioset.pt"
    assert calls["model_id"] == 1
    assert loss_module.amodel == "HTSAT-base"
    assert tuple(loss_module.state_dict()) == ()
