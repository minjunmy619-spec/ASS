from __future__ import annotations

from typing import Any

import lightning as lt


def build_seeded_trainer(*, seed: int, **trainer_kwargs: Any) -> lt.Trainer:
    """Seed all ranks before task/model construction, then build the trainer."""

    lt.seed_everything(int(seed), workers=True)
    return lt.Trainer(**trainer_kwargs)
