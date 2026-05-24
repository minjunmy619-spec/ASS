from __future__ import annotations


def _resolve_attr(obj, dotted_path: str):
    cur = obj
    for part in dotted_path.split("."):
        cur = getattr(cur, part)
    return cur


def _linear(epoch: int, warmup_epochs: int, ramp_epochs: int, start: float, end: float) -> float:
    if epoch < warmup_epochs:
        return float(start)
    if ramp_epochs <= 0:
        return float(end)
    progress = min(1.0, max(0.0, float(epoch - warmup_epochs) / float(ramp_epochs)))
    return float(start + progress * (end - start))


try:
    import lightning.pytorch as pl

    _CallbackBase = pl.Callback
except ModuleNotFoundError:  # pragma: no cover
    pl = None

    class _CallbackBase:
        pass


class RobustTSEConditionCurriculum(_CallbackBase):
    """Schedule robust TSE condition dropout/noise attributes.

    This callback is intentionally small and model-specific.  It progressively
    exposes a robust TSE model to noisy USS auxiliary hints by starting with
    heavy condition dropout and relaxing it over training.  Learned gate
    parameters are not overwritten; only plain scalar attributes such as
    ``condition_dropout`` and ``spatial_condition_noise_std`` are updated.
    """

    def __init__(
        self,
        model_attr: str = "model",
        warmup_epochs: int = 5,
        ramp_epochs: int = 25,
        start_condition_dropout: float = 0.8,
        end_condition_dropout: float = 0.3,
        start_temporal_condition_dropout: float = 0.8,
        end_temporal_condition_dropout: float = 0.4,
        start_spatial_condition_dropout: float = 1.0,
        end_spatial_condition_dropout: float = 0.5,
        start_condition_noise_std: float = 0.05,
        end_condition_noise_std: float = 0.02,
        start_spatial_condition_noise_std: float = 0.10,
        end_spatial_condition_noise_std: float = 0.05,
        strict: bool = True,
    ):
        super().__init__()
        self.model_attr = model_attr
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.start_condition_dropout = float(start_condition_dropout)
        self.end_condition_dropout = float(end_condition_dropout)
        self.start_temporal_condition_dropout = float(start_temporal_condition_dropout)
        self.end_temporal_condition_dropout = float(end_temporal_condition_dropout)
        self.start_spatial_condition_dropout = float(start_spatial_condition_dropout)
        self.end_spatial_condition_dropout = float(end_spatial_condition_dropout)
        self.start_condition_noise_std = float(start_condition_noise_std)
        self.end_condition_noise_std = float(end_condition_noise_std)
        self.start_spatial_condition_noise_std = float(start_spatial_condition_noise_std)
        self.end_spatial_condition_noise_std = float(end_spatial_condition_noise_std)
        self.strict = bool(strict)

    def _set_if_present(self, model, name: str, value: float) -> None:
        if hasattr(model, name):
            setattr(model, name, float(value))
            return
        if self.strict:
            raise AttributeError(f"{type(model).__name__} does not expose attribute '{name}'")

    def _apply(self, trainer, pl_module) -> None:
        model = _resolve_attr(pl_module, self.model_attr)
        epoch = int(getattr(trainer, "current_epoch", 0))
        values = {
            "condition_dropout": _linear(
                epoch,
                self.warmup_epochs,
                self.ramp_epochs,
                self.start_condition_dropout,
                self.end_condition_dropout,
            ),
            "temporal_condition_dropout": _linear(
                epoch,
                self.warmup_epochs,
                self.ramp_epochs,
                self.start_temporal_condition_dropout,
                self.end_temporal_condition_dropout,
            ),
            "spatial_condition_dropout": _linear(
                epoch,
                self.warmup_epochs,
                self.ramp_epochs,
                self.start_spatial_condition_dropout,
                self.end_spatial_condition_dropout,
            ),
            "condition_noise_std": _linear(
                epoch,
                self.warmup_epochs,
                self.ramp_epochs,
                self.start_condition_noise_std,
                self.end_condition_noise_std,
            ),
            "spatial_condition_noise_std": _linear(
                epoch,
                self.warmup_epochs,
                self.ramp_epochs,
                self.start_spatial_condition_noise_std,
                self.end_spatial_condition_noise_std,
            ),
        }
        for name, value in values.items():
            self._set_if_present(model, name, value)
            if hasattr(pl_module, "log"):
                pl_module.log(f"condition_curriculum/{name}", value, prog_bar=False, logger=True, sync_dist=True)

    def on_train_epoch_start(self, trainer, pl_module):
        self._apply(trainer, pl_module)
