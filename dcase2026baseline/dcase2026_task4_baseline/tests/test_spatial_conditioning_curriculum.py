import pytest

from src.training.callbacks.spatial_conditioning_curriculum import SpatialConditioningCurriculum


class DummyBridgeModel:
    def __init__(self):
        self.predicted_spatial_prob = 0.0
        self.spatial_mix_fallback_prob = 0.0

    def set_predicted_spatial_prob(self, value):
        self.predicted_spatial_prob = float(value)


class DummyLightningModule:
    def __init__(self):
        self.model = DummyBridgeModel()
        self.logged = {}

    def log(self, key, value, **kwargs):
        self.logged[key] = value


class DummyTrainer:
    current_epoch = 0


def test_spatial_conditioning_curriculum_reaches_predicted_only_phase():
    callback = SpatialConditioningCurriculum(
        warmup_epochs=2,
        anneal_epochs=3,
        start_predicted_spatial_prob=0.25,
        end_predicted_spatial_prob=1.0,
        start_spatial_mix_fallback_prob=0.05,
        end_spatial_mix_fallback_prob=0.0,
    )

    assert callback.values_for_epoch(0) == pytest.approx((0.25, 0.05))
    assert callback.values_for_epoch(1) == pytest.approx((0.25, 0.05))
    assert callback.values_for_epoch(2) == pytest.approx((0.5, 0.0333333333))
    assert callback.values_for_epoch(4) == pytest.approx((1.0, 0.0))
    assert callback.values_for_epoch(10) == pytest.approx((1.0, 0.0))


def test_spatial_conditioning_curriculum_updates_lightning_module_model():
    callback = SpatialConditioningCurriculum(
        warmup_epochs=0,
        anneal_epochs=1,
        start_predicted_spatial_prob=0.25,
        end_predicted_spatial_prob=1.0,
        start_spatial_mix_fallback_prob=0.05,
        end_spatial_mix_fallback_prob=0.0,
    )
    module = DummyLightningModule()
    trainer = DummyTrainer()

    callback.on_train_epoch_start(trainer, module)

    assert module.model.predicted_spatial_prob == pytest.approx(1.0)
    assert module.model.spatial_mix_fallback_prob == pytest.approx(0.0)
    assert module.logged["spatial_conditioning/predicted_spatial_prob"] == pytest.approx(1.0)
    assert module.logged["spatial_conditioning/spatial_mix_fallback_prob"] == pytest.approx(0.0)

