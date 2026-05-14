import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.callbacks.lambda_scheduler import LambdaScheduler, _Schedule


def _aclose(a, b, eps=1e-6):
    assert abs(float(a) - float(b)) < eps, f"{a!r} != {b!r}"


class _FakeLossFunc:
    def __init__(self, lambdas):
        self.lambdas = lambdas

    def __call__(self, *args, **kwargs):  # pragma: no cover - not used in tests
        return {}


class _FakePLModule:
    def __init__(self, lambdas):
        self.loss_func = _FakeLossFunc(lambdas)
        self.logged = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = float(value)


class _FakeTrainer:
    def __init__(self, epoch=0, step=0):
        self.current_epoch = int(epoch)
        self.global_step = int(step)


def test_constant_schedule():
    s = _Schedule("c", {"kind": "constant", "value": 0.4})
    _aclose(s(0, 0), 0.4)
    _aclose(s(100, 999), 0.4)


def test_linear_schedule_with_warmup():
    s = _Schedule(
        "l",
        {"kind": "linear", "warmup": 10, "duration": 20, "start": 0.0, "end": 1.0},
    )
    _aclose(s(0, 0), 0.0)
    _aclose(s(10, 0), 0.0)
    _aclose(s(20, 0), 0.5)
    _aclose(s(30, 0), 1.0)
    _aclose(s(50, 0), 1.0)


def test_cosine_schedule_endpoints_and_midpoint():
    s = _Schedule("c", {"kind": "cosine", "warmup": 0, "duration": 10, "start": 1.0, "end": 0.0})
    _aclose(s(0, 0), 1.0)
    _aclose(s(10, 0), 0.0)
    _aclose(s(5, 0), 0.5)


def test_exponential_with_clamp():
    s = _Schedule(
        "e",
        {"kind": "exponential", "warmup": 0, "start": 1.0, "gamma": 0.9, "min": 0.5},
    )
    _aclose(s(0, 0), 1.0)
    _aclose(s(1, 0), 0.9)
    _aclose(s(100, 0), 0.5)


def test_piecewise_step_jumps_at_boundaries():
    s = _Schedule(
        "p",
        {"kind": "piecewise", "points": [[0, 0.1], [10, 0.5], [20, 1.0]]},
    )
    _aclose(s(0, 0), 0.1)
    _aclose(s(5, 0), 0.1)
    _aclose(s(10, 0), 0.5)
    _aclose(s(15, 0), 0.5)
    _aclose(s(20, 0), 1.0)
    _aclose(s(100, 0), 1.0)


def test_piecewise_linear_interpolates():
    s = _Schedule(
        "pl",
        {"kind": "piecewise_linear", "points": [[0, 0.0], [10, 1.0], [20, 0.0]]},
    )
    _aclose(s(0, 0), 0.0)
    _aclose(s(5, 0), 0.5)
    _aclose(s(10, 0), 1.0)
    _aclose(s(15, 0), 0.5)
    _aclose(s(20, 0), 0.0)


def test_step_unit_selection():
    s = _Schedule(
        "step",
        {
            "kind": "linear",
            "unit": "step",
            "warmup": 0,
            "duration": 100,
            "start": 0.0,
            "end": 1.0,
        },
    )
    assert s.is_step_unit()
    _aclose(s(0, 50), 0.5)
    _aclose(s(999, 100), 1.0)


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        _Schedule("x", {"kind": "totally-bogus"})


def test_lambda_scheduler_updates_lambdas_each_epoch():
    lambdas = {
        "lambda_class_ce": 0.8,
        "lambda_doa": 0.0,
        "lambda_extra_unknown": 0.5,
    }
    sched = LambdaScheduler(
        schedules={
            "lambda_class_ce": {
                "kind": "linear",
                "warmup": 10,
                "duration": 10,
                "start": 0.2,
                "end": 0.8,
            },
            "lambda_doa": {"kind": "cosine", "warmup": 0, "duration": 10, "start": 0.0, "end": 0.05},
        },
        defaults={"kind": "constant", "value": 0.123},
        global_scale={"kind": "linear", "warmup": 0, "duration": 20, "start": 0.0, "end": 1.0},
        strict=True,
    )
    pl_mod = _FakePLModule(lambdas)
    sched.on_fit_start(_FakeTrainer(epoch=0, step=0), pl_mod)
    _aclose(lambdas["lambda_class_ce"], 0.2 * 0.0)
    _aclose(lambdas["lambda_doa"], 0.0)
    _aclose(lambdas["lambda_extra_unknown"], 0.123 * 0.0)

    sched.on_train_epoch_start(_FakeTrainer(epoch=20, step=0), pl_mod)
    _aclose(lambdas["lambda_class_ce"], 0.8)
    _aclose(lambdas["lambda_doa"], 0.05)
    _aclose(lambdas["lambda_extra_unknown"], 0.123)


def test_lambda_scheduler_step_unit_runs_on_batch_end():
    lambdas = {"lambda_class_ce": 0.0}
    sched = LambdaScheduler(
        schedules={
            "lambda_class_ce": {
                "kind": "linear",
                "unit": "step",
                "warmup": 0,
                "duration": 100,
                "start": 0.0,
                "end": 1.0,
            }
        },
        strict=True,
    )
    pl_mod = _FakePLModule(lambdas)
    sched.on_fit_start(_FakeTrainer(epoch=0, step=0), pl_mod)
    _aclose(lambdas["lambda_class_ce"], 0.0)

    # epoch_start should not advance for step-unit schedules but it does run
    # the schedule with the current global_step which is still 0.
    sched.on_train_batch_end(_FakeTrainer(epoch=0, step=50), pl_mod, outputs=None, batch=None, batch_idx=0)
    _aclose(lambdas["lambda_class_ce"], 0.5)

    sched.on_train_batch_end(_FakeTrainer(epoch=0, step=100), pl_mod, outputs=None, batch=None, batch_idx=0)
    _aclose(lambdas["lambda_class_ce"], 1.0)


def test_strict_mode_raises_on_unknown_lambda():
    sched = LambdaScheduler(
        schedules={"lambda_does_not_exist": {"kind": "constant", "value": 0.0}},
        strict=True,
    )
    with pytest.raises(KeyError):
        sched.on_fit_start(_FakeTrainer(), _FakePLModule({"lambda_real": 0.1}))


def test_non_strict_mode_skips_unknown_lambda():
    lambdas = {"lambda_real": 0.1}
    sched = LambdaScheduler(
        schedules={"lambda_does_not_exist": {"kind": "constant", "value": 0.0}},
        strict=False,
    )
    sched.on_fit_start(_FakeTrainer(), _FakePLModule(lambdas))
    # Untouched
    _aclose(lambdas["lambda_real"], 0.1)


def test_strict_mode_requires_lambdas_attribute():
    class _NoLambdasLoss:
        def __call__(self, *a, **k):  # pragma: no cover
            return {}

    class _BadModule:
        def __init__(self):
            self.loss_func = _NoLambdasLoss()

        def log(self, *a, **k):
            pass

    sched = LambdaScheduler(
        schedules={"x": {"kind": "constant", "value": 0.0}},
        strict=True,
    )
    with pytest.raises(AttributeError):
        sched.on_fit_start(_FakeTrainer(), _BadModule())


def _maybe_skip_torch():
    try:
        import torch  # noqa: F401
    except (ImportError, OSError) as exc:  # pragma: no cover - env specific
        pytest.skip(f"torch unavailable in this env: {exc}")


def test_uss_loss_factory_exposes_mutable_lambdas():
    _maybe_skip_torch()
    from src.training.loss.uss_loss import get_loss_func as get_uss_loss

    loss = get_uss_loss(lambda_class_ce=0.123)
    assert hasattr(loss, "lambdas")
    assert loss.lambdas["lambda_class_ce"] == pytest.approx(0.123)
    loss.lambdas["lambda_class_ce"] = 0.999
    assert loss.lambdas["lambda_class_ce"] == pytest.approx(0.999)


def test_uss_bridge_loss_shares_lambdas_with_base():
    _maybe_skip_torch()
    from src.training.loss.uss_bridge_loss import get_loss_func as get_bridge_loss

    loss = get_bridge_loss(
        lambda_bridge_proto=0.07,
        lambda_class_ce=0.4,
    )
    assert hasattr(loss, "lambdas")
    # Bridge-specific
    assert loss.lambdas["lambda_bridge_proto"] == pytest.approx(0.07)
    # Base USS lambdas live in the same dict
    assert loss.lambdas["lambda_class_ce"] == pytest.approx(0.4)
    # Mutating one updates both views (they are the same dict).
    loss.lambdas["lambda_bridge_proto"] = 0.5
    loss.lambdas["lambda_class_ce"] = 0.1
    assert loss.lambdas["lambda_bridge_proto"] == 0.5
    assert loss.lambdas["lambda_class_ce"] == 0.1



# ---------------------------------------------------------------------------
# Checkpoint / resume round-trip
# ---------------------------------------------------------------------------


def _make_scheduler(**overrides):
    kwargs = dict(
        schedules={
            "lambda_class_ce": {
                "kind": "linear",
                "warmup": 10,
                "duration": 10,
                "start": 0.2,
                "end": 0.8,
            },
            "lambda_doa": {
                "kind": "cosine",
                "warmup": 0,
                "duration": 10,
                "start": 0.0,
                "end": 0.05,
            },
        },
        defaults={"kind": "constant", "value": 0.123},
        global_scale={
            "kind": "linear",
            "warmup": 0,
            "duration": 20,
            "start": 0.0,
            "end": 1.0,
        },
        strict=True,
    )
    kwargs.update(overrides)
    return LambdaScheduler(**kwargs)


def _initial_lambdas():
    return {
        "lambda_class_ce": 0.8,
        "lambda_doa": 0.0,
        "lambda_extra_unknown": 0.5,
    }


def test_state_dict_includes_originals_last_applied_and_fingerprint():
    sched = _make_scheduler()
    pl_mod = _FakePLModule(_initial_lambdas())

    sched.on_fit_start(_FakeTrainer(epoch=0, step=0), pl_mod)
    sched.on_train_epoch_start(_FakeTrainer(epoch=15, step=0), pl_mod)

    state = sched.state_dict()
    assert set(state.keys()) == {"original_values", "last_applied", "config_fingerprint"}
    # Originals are the YAML defaults, NOT the schedule-modified values.
    assert state["original_values"] == {
        "lambda_class_ce": 0.8,
        "lambda_doa": 0.0,
        "lambda_extra_unknown": 0.5,
    }
    # Last applied reflects what the scheduler actually wrote at epoch 15.
    assert "lambda_class_ce" in state["last_applied"]
    assert "lambda_doa" in state["last_applied"]
    assert isinstance(state["config_fingerprint"], str)
    assert state["config_fingerprint"]  # non-empty


def test_resume_uses_loaded_originals_and_recomputes_at_resumed_epoch():
    # First run: train to epoch 15, capture state.
    fresh_lambdas = _initial_lambdas()
    sched1 = _make_scheduler()
    pl1 = _FakePLModule(fresh_lambdas)
    sched1.on_fit_start(_FakeTrainer(epoch=0, step=0), pl1)
    sched1.on_train_epoch_start(_FakeTrainer(epoch=15, step=0), pl1)
    saved = sched1.state_dict()

    # Second run: same config, but the loss factory was rebuilt and
    # ``loss_func.lambdas`` is back at YAML defaults. Lightning loads the
    # callback state BEFORE on_fit_start runs.
    rebuilt_lambdas = _initial_lambdas()
    sched2 = _make_scheduler()
    pl2 = _FakePLModule(rebuilt_lambdas)
    sched2.load_state_dict(saved)
    sched2.on_fit_start(_FakeTrainer(epoch=15, step=0), pl2)

    # Originals from the checkpoint are preserved (not overwritten by the
    # YAML-default snapshot taken on resume).
    assert sched2._original_values == saved["original_values"]
    # And the schedule has been re-evaluated for the resumed epoch, so the
    # values match what the original run would have applied at epoch 15.
    _aclose(rebuilt_lambdas["lambda_class_ce"], pl1.loss_func.lambdas["lambda_class_ce"])
    _aclose(rebuilt_lambdas["lambda_doa"], pl1.loss_func.lambdas["lambda_doa"])
    _aclose(rebuilt_lambdas["lambda_extra_unknown"], pl1.loss_func.lambdas["lambda_extra_unknown"])


def test_on_fit_end_restores_originals_only_when_enabled():
    lambdas = _initial_lambdas()
    sched = _make_scheduler()
    pl_mod = _FakePLModule(lambdas)
    sched.on_fit_start(_FakeTrainer(epoch=0, step=0), pl_mod)
    # Use a mid-ramp epoch so the schedule definitely produces values
    # different from the YAML defaults.
    sched.on_train_epoch_start(_FakeTrainer(epoch=14, step=0), pl_mod)
    # Mid-training: scheduler has mutated values.
    assert lambdas["lambda_class_ce"] != pytest.approx(0.8)
    assert lambdas["lambda_doa"] != pytest.approx(0.0)

    sched.on_fit_end(_FakeTrainer(epoch=14, step=0), pl_mod)
    # All YAML defaults restored.
    _aclose(lambdas["lambda_class_ce"], 0.8)
    _aclose(lambdas["lambda_doa"], 0.0)
    _aclose(lambdas["lambda_extra_unknown"], 0.5)

    # Now opt out of restoration.
    lambdas2 = _initial_lambdas()
    sched_no_restore = _make_scheduler(restore_on_fit_end=False)
    pl2 = _FakePLModule(lambdas2)
    sched_no_restore.on_fit_start(_FakeTrainer(epoch=0, step=0), pl2)
    sched_no_restore.on_train_epoch_start(_FakeTrainer(epoch=14, step=0), pl2)
    mid = dict(lambdas2)
    sched_no_restore.on_fit_end(_FakeTrainer(epoch=14, step=0), pl2)
    assert lambdas2 == mid  # untouched


def test_config_fingerprint_changes_with_schedule_changes():
    sched_a = _make_scheduler()
    sched_b = _make_scheduler(
        schedules={
            "lambda_class_ce": {
                "kind": "linear",
                "warmup": 10,
                "duration": 10,
                "start": 0.2,
                "end": 0.9,  # changed end value
            },
            "lambda_doa": {
                "kind": "cosine",
                "warmup": 0,
                "duration": 10,
                "start": 0.0,
                "end": 0.05,
            },
        }
    )
    assert sched_a._config_fingerprint() != sched_b._config_fingerprint()

    # Whereas changing only book-keeping flags must NOT affect fingerprint.
    sched_c = _make_scheduler(strict=False, log_prefix="other_prefix", restore_on_fit_end=False)
    assert sched_a._config_fingerprint() == sched_c._config_fingerprint()


def test_load_state_dict_warns_on_fingerprint_mismatch_but_keeps_originals(caplog):
    sched1 = _make_scheduler()
    pl1 = _FakePLModule(_initial_lambdas())
    sched1.on_fit_start(_FakeTrainer(epoch=0, step=0), pl1)
    sched1.on_train_epoch_start(_FakeTrainer(epoch=15, step=0), pl1)
    saved = sched1.state_dict()

    sched2 = _make_scheduler(
        schedules={
            "lambda_class_ce": {
                "kind": "linear",
                "warmup": 10,
                "duration": 10,
                "start": 0.2,
                "end": 0.9,  # different config
            },
        }
    )
    with caplog.at_level("INFO"):
        sched2.load_state_dict(saved)
    # Originals from the checkpoint are still preserved on mismatch.
    assert sched2._original_values == saved["original_values"]
    # And we logged something about the mismatch.
    assert any("fingerprint" in rec.getMessage().lower() for rec in caplog.records)


def test_load_state_dict_handles_empty_or_missing_keys():
    sched = _make_scheduler()
    sched.load_state_dict({})  # must not raise
    assert sched._original_values == {}
    assert sched._last_applied == {}
    assert sched._loaded_from_checkpoint is True
