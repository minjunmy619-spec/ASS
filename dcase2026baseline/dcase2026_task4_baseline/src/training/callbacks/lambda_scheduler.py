"""Opt-in scheduler that updates ``loss_func.lambdas`` during training.

The companion loss factories (``src.training.loss.uss_loss.get_loss_func`` and
``src.training.loss.uss_bridge_loss.get_loss_func``) expose a single mutable
dict ``loss_func.lambdas`` that holds every scalar ``lambda_*`` weight used to
combine the constituent losses. Each forward call reads the current value, so
mutating that dict here takes effect on the very next training step without
needing to rebuild the loss.

The schedule is fully described in YAML and is opt-in: if the callback is not
listed in ``train.callbacks`` the loss behaves exactly as before. Any new
``lambda_*`` added to the loss is automatically schedulable through the same
callback as long as the loss factory exposes it via ``loss_func.lambdas``.

Schedule kinds
--------------
Each named lambda gets a per-key schedule with the following supported kinds:

* ``constant`` -- ``value`` (or fall back to ``start``).
* ``linear`` -- linear interpolation from ``start`` to ``end`` between
  ``warmup`` and ``warmup + duration``.
* ``cosine`` -- half-cosine ramp from ``start`` to ``end`` between
  ``warmup`` and ``warmup + duration``.
* ``exponential`` -- multiply by ``gamma`` every full unit (epoch or step) past
  ``warmup``; clamped to ``min``/``max`` if provided.
* ``piecewise`` (alias ``step``) -- list of ``[boundary, value]`` pairs; the
  schedule jumps to ``value`` once the unit counter crosses ``boundary``.
* ``piecewise_linear`` -- list of ``[boundary, value]`` pairs interpolated
  linearly between consecutive points.

Each schedule defaults to ``unit: epoch`` but can opt into ``unit: step`` to
update on every batch. ``warmup`` and ``duration`` are expressed in the chosen
unit. Boundaries in ``piecewise`` and ``piecewise_linear`` are also expressed
in the chosen unit.

Two top-level conveniences are also supported:

* ``defaults``: a schedule applied to any lambda not explicitly listed.
* ``global_scale``: a schedule whose value is multiplied into every scheduled
  lambda after its own schedule is evaluated. Useful for "ramp the whole loss
  weight up" experiments while still tweaking individual ratios.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

_log = logging.getLogger(__name__)

try:
    import lightning.pytorch as pl

    _CallbackBase = pl.Callback
except ModuleNotFoundError:  # pragma: no cover - lightning is the runtime dep
    pl = None  # type: ignore[assignment]

    class _CallbackBase:  # minimal shim so the schedule logic stays importable
        pass


_PIECEWISE_KINDS = ("piecewise", "step", "piecewise_linear")
_RAMP_KINDS = ("linear", "cosine", "exponential")
_VALID_KINDS = ("constant",) + _RAMP_KINDS + _PIECEWISE_KINDS


def _coerce_points(points: Sequence[Any]) -> list:
    coerced = []
    for entry in points:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"piecewise points must be [boundary, value] pairs, got {entry!r}"
            )
        coerced.append((float(entry[0]), float(entry[1])))
    coerced.sort(key=lambda x: x[0])
    if not coerced:
        raise ValueError("piecewise schedule requires at least one point")
    return coerced


def _normalize_for_fingerprint(value: Any) -> Any:
    """Convert a schedule spec into JSON-stable scalars for hashing.

    Mappings are dict-ified with sorted keys, sequences become lists, and
    every scalar is coerced to a JSON-safe primitive. Floats are rounded
    to a tight tolerance so that load-from-yaml roundtrips compare equal.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): _normalize_for_fingerprint(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_fingerprint(v) for v in value]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return round(float(value), 12)
    return str(value)


class _Schedule:
    """Compiled per-lambda schedule."""

    def __init__(self, name: str, spec: Mapping[str, Any]):
        if not isinstance(spec, Mapping):
            raise TypeError(f"schedule for {name!r} must be a mapping, got {type(spec)!r}")

        kind = str(spec.get("kind", "constant")).lower()
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"unknown schedule kind {kind!r} for {name!r}; "
                f"expected one of {_VALID_KINDS}"
            )

        self.name = str(name)
        self.kind = kind
        self.unit = str(spec.get("unit", "epoch")).lower()
        if self.unit not in ("epoch", "step"):
            raise ValueError(f"schedule unit must be 'epoch' or 'step', got {self.unit!r}")

        self.warmup = float(spec.get("warmup", 0.0))
        self.duration = float(spec.get("duration", 0.0))
        self.start = float(spec.get("start", spec.get("value", 0.0)))
        self.end = float(spec.get("end", self.start))
        self.gamma = float(spec.get("gamma", 1.0))
        self.min_value = spec.get("min", None)
        self.max_value = spec.get("max", None)
        self.min_value = None if self.min_value is None else float(self.min_value)
        self.max_value = None if self.max_value is None else float(self.max_value)

        if kind in _PIECEWISE_KINDS:
            self.points = _coerce_points(spec.get("points", []))
        else:
            self.points = None

        if kind == "constant" and "value" in spec:
            self.start = float(spec["value"])
            self.end = self.start

    # ------------------------------------------------------------------
    def is_step_unit(self) -> bool:
        return self.unit == "step"

    # ------------------------------------------------------------------
    def __call__(self, epoch: int, step: int) -> float:
        t = float(step if self.unit == "step" else epoch)
        if self.kind == "constant":
            return self._clamp(self.start)

        if self.kind in _RAMP_KINDS:
            if t < self.warmup:
                return self._clamp(self.start)
            if self.kind == "exponential":
                value = self.start * (self.gamma ** max(0.0, t - self.warmup))
                return self._clamp(value)
            if self.duration <= 0.0:
                return self._clamp(self.end)
            progress = (t - self.warmup) / self.duration
            progress = max(0.0, min(1.0, progress))
            if self.kind == "linear":
                value = self.start + progress * (self.end - self.start)
            else:  # cosine
                value = self.end + 0.5 * (self.start - self.end) * (1.0 + math.cos(math.pi * progress))
            return self._clamp(value)

        # piecewise variants
        assert self.points is not None
        if self.kind in ("piecewise", "step"):
            value = self.points[0][1]
            for boundary, point_value in self.points:
                if t >= boundary:
                    value = point_value
                else:
                    break
            return self._clamp(value)

        # piecewise_linear
        if t <= self.points[0][0]:
            return self._clamp(self.points[0][1])
        if t >= self.points[-1][0]:
            return self._clamp(self.points[-1][1])
        for (b0, v0), (b1, v1) in zip(self.points[:-1], self.points[1:]):
            if b0 <= t <= b1:
                if b1 == b0:
                    return self._clamp(v1)
                fraction = (t - b0) / (b1 - b0)
                return self._clamp(v0 + fraction * (v1 - v0))
        return self._clamp(self.points[-1][1])  # pragma: no cover

    # ------------------------------------------------------------------
    def _clamp(self, value: float) -> float:
        if self.min_value is not None:
            value = max(self.min_value, value)
        if self.max_value is not None:
            value = min(self.max_value, value)
        return float(value)


class LambdaScheduler(_CallbackBase):
    """Adjust ``loss_func.lambdas`` over the course of training.

    Parameters
    ----------
    schedules : Mapping[str, Mapping]
        Per-lambda schedule specs. Keys must match entries in
        ``loss_func.lambdas`` (e.g. ``lambda_bridge_proto``,
        ``lambda_class_ce``). See module docstring for supported kinds.
    defaults : Optional[Mapping]
        Fallback schedule applied to any lambda found in ``loss_func.lambdas``
        that is not explicitly listed in ``schedules``. Useful for sweeping
        all lambdas with one common ramp.
    global_scale : Optional[Mapping]
        Schedule whose value is multiplied into every scheduled lambda after
        its own schedule is evaluated.
    log_prefix : str
        Prefix for the per-epoch logged scalar values.
    strict : bool
        If True, raise when ``loss_func`` does not expose ``.lambdas`` or when
        a scheduled key is unknown. If False, fall back silently.
    """

    def __init__(
        self,
        schedules: Optional[Mapping[str, Mapping[str, Any]]] = None,
        defaults: Optional[Mapping[str, Any]] = None,
        global_scale: Optional[Mapping[str, Any]] = None,
        log_prefix: str = "lambda_schedule",
        strict: bool = True,
        restore_on_fit_end: bool = True,
    ):
        super().__init__()
        self.schedules_spec: Dict[str, Mapping[str, Any]] = dict(schedules or {})
        self.defaults_spec: Optional[Mapping[str, Any]] = defaults
        self.global_scale_spec: Optional[Mapping[str, Any]] = global_scale
        self.log_prefix = str(log_prefix)
        self.strict = bool(strict)
        self.restore_on_fit_end = bool(restore_on_fit_end)

        # Compiled at on_fit_start so every entry validates eagerly.
        self._compiled: Dict[str, _Schedule] = {}
        self._defaults: Optional[_Schedule] = None
        self._global_scale: Optional[_Schedule] = None
        self._has_step_unit: bool = False

        # Persisted state (covered by state_dict / load_state_dict):
        # - ``_original_values`` snapshots the YAML-default lambdas captured the
        #   first time fit started, so we can restore them at fit end.
        # - ``_last_applied`` records the most recent value the scheduler wrote
        #   into ``loss_func.lambdas`` for diagnostics / forensics.
        # - ``_loaded_from_checkpoint`` distinguishes a fresh fit from a resumed
        #   one so ``on_fit_start`` knows whether to seed ``_original_values``.
        self._original_values: Dict[str, float] = {}
        self._last_applied: Dict[str, float] = {}
        self._loaded_from_checkpoint: bool = False

    # ------------------------------------------------------------------
    def _resolve_lambdas(self, pl_module) -> Optional[Dict[str, float]]:
        loss_func = getattr(pl_module, "loss_func", None)
        if loss_func is None:
            return None
        lambdas = getattr(loss_func, "lambdas", None)
        if lambdas is None or not isinstance(lambdas, dict):
            return None
        return lambdas

    # ------------------------------------------------------------------
    def _compile(self, lambdas: Mapping[str, float]) -> None:
        compiled: Dict[str, _Schedule] = {}
        unknown = []
        for name, spec in self.schedules_spec.items():
            if name not in lambdas:
                unknown.append(name)
                if self.strict:
                    continue
            compiled[name] = _Schedule(name, spec)

        if unknown and self.strict:
            raise KeyError(
                "LambdaScheduler received schedules for lambdas not exposed by "
                f"loss_func.lambdas: {unknown}. Available keys: {sorted(lambdas)}"
            )

        defaults = None
        if self.defaults_spec is not None:
            defaults = _Schedule("__defaults__", self.defaults_spec)

        global_scale = None
        if self.global_scale_spec is not None:
            global_scale = _Schedule("__global_scale__", self.global_scale_spec)

        self._compiled = compiled
        self._defaults = defaults
        self._global_scale = global_scale
        self._has_step_unit = any(s.is_step_unit() for s in compiled.values())
        if defaults is not None and defaults.is_step_unit():
            self._has_step_unit = True
        if global_scale is not None and global_scale.is_step_unit():
            self._has_step_unit = True

    # ------------------------------------------------------------------
    def _names_to_update(self, lambdas: Mapping[str, float]) -> Iterable[str]:
        emitted = set()
        for name in self._compiled.keys():
            if name in lambdas:
                emitted.add(name)
                yield name
        if self._defaults is not None:
            for name in lambdas.keys():
                if name in emitted:
                    continue
                yield name

    # ------------------------------------------------------------------
    def _value_for(self, name: str, epoch: int, step: int) -> float:
        schedule = self._compiled.get(name, self._defaults)
        if schedule is None:
            raise KeyError(f"no schedule registered for {name!r}")
        value = schedule(epoch, step)
        if self._global_scale is not None:
            value = value * self._global_scale(epoch, step)
        return float(value)

    # ------------------------------------------------------------------
    def _apply(self, trainer, pl_module, on_step: bool) -> None:
        lambdas = self._resolve_lambdas(pl_module)
        if lambdas is None:
            if self.strict:
                raise AttributeError(
                    "LambdaScheduler requires pl_module.loss_func.lambdas to be a dict; "
                    "ensure the loss factory exposes its lambda state."
                )
            return

        epoch = int(getattr(trainer, "current_epoch", 0))
        step = int(getattr(trainer, "global_step", 0))
        any_logged = False

        for name in list(self._names_to_update(lambdas)):
            schedule = self._compiled.get(name, self._defaults)
            if schedule is None:
                continue
            if on_step and not schedule.is_step_unit():
                continue
            new_value = self._value_for(name, epoch, step)
            lambdas[name] = new_value
            self._last_applied[name] = new_value
            try:
                pl_module.log(
                    f"{self.log_prefix}/{name}",
                    new_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                any_logged = True
            except Exception:
                # Logging may fail outside an active trainer context (e.g. in
                # ``on_fit_start`` before any batch). Mutating the dict is the
                # important side effect; logging is best-effort.
                pass

        if (
            self._global_scale is not None
            and (not on_step or self._global_scale.is_step_unit())
        ):
            try:
                pl_module.log(
                    f"{self.log_prefix}/__global_scale__",
                    self._global_scale(epoch, step),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )
                any_logged = True
            except Exception:
                pass

        # Suppress unused-variable lint without changing behaviour.
        del any_logged

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def on_fit_start(self, trainer, pl_module):
        lambdas = self._resolve_lambdas(pl_module)
        if lambdas is None:
            if self.strict:
                raise AttributeError(
                    "LambdaScheduler requires pl_module.loss_func.lambdas to be a dict; "
                    "ensure the loss factory exposes its lambda state."
                )
            return

        # Snapshot YAML defaults only the first time this scheduler runs
        # (a fresh fit). On a resumed fit, ``load_state_dict`` already
        # populated ``_original_values`` from the checkpoint, so the fresh
        # ``loss_func.lambdas`` we'd see here reflects YAML defaults but
        # we want the originals from before training started.
        if not self._loaded_from_checkpoint and not self._original_values:
            self._original_values = dict(lambdas)

        self._compile(lambdas)
        # Sanity check: scheduled keys that exist now are validated by
        # ``_compile``. The originals snapshot may include extra keys that
        # are still valid; nothing to do.
        self._apply(trainer, pl_module, on_step=False)

    def on_train_epoch_start(self, trainer, pl_module):
        self._apply(trainer, pl_module, on_step=False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._has_step_unit:
            self._apply(trainer, pl_module, on_step=True)

    def on_fit_end(self, trainer, pl_module):
        if not self.restore_on_fit_end:
            return
        # Restore original values so a subsequent fit (e.g. with a different
        # schedule) starts from a clean slate.
        lambdas = self._resolve_lambdas(pl_module)
        if lambdas is None or not self._original_values:
            return
        for name, value in self._original_values.items():
            lambdas[name] = value

    # ------------------------------------------------------------------
    # Checkpoint integration
    # ------------------------------------------------------------------
    def _config_fingerprint(self) -> str:
        """Stable hash of the schedule config to detect mismatches on resume.

        The fingerprint covers all knobs that change the trajectory of the
        lambdas over time. It deliberately ignores ``log_prefix``,
        ``strict``, and ``restore_on_fit_end``, which are book-keeping flags
        that don't affect numerical values.
        """
        payload = {
            "schedules": _normalize_for_fingerprint(self.schedules_spec),
            "defaults": _normalize_for_fingerprint(self.defaults_spec),
            "global_scale": _normalize_for_fingerprint(self.global_scale_spec),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def state_dict(self) -> Dict[str, Any]:
        """Persist enough state to faithfully resume from a checkpoint.

        Lightning calls this for every callback whose class name is unique
        and saves the result inside the checkpoint under
        ``callbacks[<callback_state_key>]``. ``load_state_dict`` is then
        invoked before ``on_fit_start`` on resume.
        """
        return {
            "original_values": dict(self._original_values),
            "last_applied": dict(self._last_applied),
            "config_fingerprint": self._config_fingerprint(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if not isinstance(state_dict, Mapping):
            return
        self._original_values = {
            str(k): float(v) for k, v in (state_dict.get("original_values") or {}).items()
        }
        self._last_applied = {
            str(k): float(v) for k, v in (state_dict.get("last_applied") or {}).items()
        }
        loaded_fp = state_dict.get("config_fingerprint")
        current_fp = self._config_fingerprint()
        if loaded_fp is not None and loaded_fp != current_fp:
            msg = (
                "LambdaScheduler config fingerprint changed since the "
                "checkpoint was saved. The schedule will follow the *new* "
                "config from this point onward; ``original_values`` from "
                "the checkpoint are preserved so ``on_fit_end`` can still "
                "restore them. Set ``strict=False`` to silence this."
            )
            if self.strict:
                _log.warning(msg)
            else:
                _log.info(msg)
        self._loaded_from_checkpoint = True
