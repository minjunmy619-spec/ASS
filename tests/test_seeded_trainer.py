from __future__ import annotations

from spectral_feature_compression.common import seeded_trainer


def test_seeded_trainer_seeds_before_trainer_construction(monkeypatch) -> None:
    events: list[tuple[str, object]] = []
    sentinel = object()

    def fake_seed_everything(seed: int, *, workers: bool) -> None:
        events.append(("seed", (seed, workers)))

    def fake_trainer(**kwargs):
        events.append(("trainer", kwargs))
        return sentinel

    monkeypatch.setattr(seeded_trainer.lt, "seed_everything", fake_seed_everything)
    monkeypatch.setattr(seeded_trainer.lt, "Trainer", fake_trainer)

    trainer = seeded_trainer.build_seeded_trainer(seed=2026, max_steps=20_000)

    assert trainer is sentinel
    assert events == [
        ("seed", (2026, True)),
        ("trainer", {"max_steps": 20_000}),
    ]
