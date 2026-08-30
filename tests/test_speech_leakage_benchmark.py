from __future__ import annotations

import torch

from spectral_feature_compression.core.loss.composite_separation import CompositeSeparationSpectralLoss
from tools.benchmark_speech_leakage_losses import inject_speech_leakage, probe_speech_leakage_levels


def test_controlled_speech_leakage_injection_preserves_references_and_is_monotonic() -> None:
    samples = 1024
    timeline = torch.arange(samples, dtype=torch.float32) / samples
    speech = 0.5 * torch.sin(2.0 * torch.pi * 16.0 * timeline)
    reference = torch.zeros(1, 3, 1, samples)
    reference[:, 0, 0] = speech
    scorer = CompositeSeparationSpectralLoss(
        n_fft=128,
        hop_length=32,
        source_order=("speech", "music", "effects"),
        speech_leakage_weight=1.0,
        speech_leakage_n_fft=128,
        speech_leakage_hop_length=32,
        speech_leakage_speech_active_db=-40.0,
        speech_leakage_target_relative_db=12.0,
    )

    injected = inject_speech_leakage(
        reference,
        source_order=("speech", "music", "effects"),
        target_source="music",
        leakage_db=-20.0,
        start_sample=256,
        duration_samples=256,
    )
    results = probe_speech_leakage_levels(
        reference,
        scorer=scorer,
        source_order=("speech", "music", "effects"),
        target_source="music",
        leakage_db_values=(-40.0, -20.0, -5.0),
        start_sample=256,
        duration_samples=256,
    )

    torch.testing.assert_close(reference[:, 1], torch.zeros_like(reference[:, 1]))
    assert float(injected[:, 1].abs().sum()) > 0.0
    values = [record["speech_leakage_tf_music"] for record in results]
    assert values[0] <= values[1] <= values[2]
