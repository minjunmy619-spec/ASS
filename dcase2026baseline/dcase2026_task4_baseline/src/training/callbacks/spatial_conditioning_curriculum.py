import lightning.pytorch as pl


class SpatialConditioningCurriculum(pl.Callback):
    """Anneal USS bridge spatial conditioning from oracle-mixed to predicted-only.

    The callback only changes the conditioning probabilities used inside the
    semantic bridge. DoA metadata can still flow through the target/loss path;
    the final phase makes the model input distribution match evaluation by using
    predicted spatial conditioning only.
    """

    def __init__(
        self,
        model_attr="model",
        warmup_epochs=25,
        anneal_epochs=150,
        start_predicted_spatial_prob=0.25,
        end_predicted_spatial_prob=1.0,
        start_spatial_mix_fallback_prob=0.05,
        end_spatial_mix_fallback_prob=0.0,
        strict=True,
        log_prefix="spatial_conditioning",
    ):
        super().__init__()
        self.model_attr = str(model_attr)
        self.warmup_epochs = int(warmup_epochs)
        self.anneal_epochs = int(anneal_epochs)
        self.start_predicted_spatial_prob = float(start_predicted_spatial_prob)
        self.end_predicted_spatial_prob = float(end_predicted_spatial_prob)
        self.start_spatial_mix_fallback_prob = float(start_spatial_mix_fallback_prob)
        self.end_spatial_mix_fallback_prob = float(end_spatial_mix_fallback_prob)
        self.strict = bool(strict)
        self.log_prefix = str(log_prefix)
        if self.warmup_epochs < 0 or self.anneal_epochs < 0:
            raise ValueError("warmup_epochs and anneal_epochs must be non-negative")
        for name, value in (
            ("start_predicted_spatial_prob", self.start_predicted_spatial_prob),
            ("end_predicted_spatial_prob", self.end_predicted_spatial_prob),
            ("start_spatial_mix_fallback_prob", self.start_spatial_mix_fallback_prob),
            ("end_spatial_mix_fallback_prob", self.end_spatial_mix_fallback_prob),
        ):
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    def _resolve_model(self, pl_module):
        target = pl_module
        if self.model_attr:
            for attr in self.model_attr.split("."):
                target = getattr(target, attr)
        return target

    def _phase_progress(self, epoch):
        if epoch < self.warmup_epochs:
            return 0.0
        if self.anneal_epochs == 0:
            return 1.0
        return min(1.0, float(epoch - self.warmup_epochs + 1) / float(self.anneal_epochs))

    def values_for_epoch(self, epoch):
        progress = self._phase_progress(int(epoch))
        pred = self.start_predicted_spatial_prob + progress * (
            self.end_predicted_spatial_prob - self.start_predicted_spatial_prob
        )
        fallback = self.start_spatial_mix_fallback_prob + progress * (
            self.end_spatial_mix_fallback_prob - self.start_spatial_mix_fallback_prob
        )
        return float(pred), float(fallback)

    def _apply(self, pl_module, epoch):
        model = self._resolve_model(pl_module)
        if not hasattr(model, "set_predicted_spatial_prob"):
            if self.strict:
                raise AttributeError(
                    f"{model.__class__.__name__} does not expose set_predicted_spatial_prob"
                )
            return
        if not hasattr(model, "spatial_mix_fallback_prob") and self.strict:
            raise AttributeError(
                f"{model.__class__.__name__} does not expose spatial_mix_fallback_prob"
            )

        pred, fallback = self.values_for_epoch(epoch)
        model.set_predicted_spatial_prob(pred)
        if hasattr(model, "spatial_mix_fallback_prob"):
            model.spatial_mix_fallback_prob = fallback

        pl_module.log(
            f"{self.log_prefix}/predicted_spatial_prob",
            pred,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        pl_module.log(
            f"{self.log_prefix}/spatial_mix_fallback_prob",
            fallback,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    def on_fit_start(self, trainer, pl_module):
        self._apply(pl_module, int(getattr(trainer, "current_epoch", 0)))

    def on_train_epoch_start(self, trainer, pl_module):
        self._apply(pl_module, int(trainer.current_epoch))

