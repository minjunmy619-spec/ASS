from __future__ import annotations

from typing import Any

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F

import torchaudio.functional as AF

import soundfile as sf


def _as_range(value: Any, *, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        result = (float(value), float(value))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        result = (float(value[0]), float(value[1]))
    else:
        raise ValueError(f"Expected a scalar or two-value range, got {value!r}")
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"Range values must be finite: {result}")
    if result[0] > result[1]:
        raise ValueError(f"Range minimum exceeds maximum: {result}")
    return result


def _sample_range(rng: random.Random, value: Any, *, default: tuple[float, float]) -> float:
    low, high = _as_range(value, default=default)
    return rng.uniform(low, high)


def _probability(config: Mapping[str, Any], key: str = "probability") -> float:
    value = float(config.get(key, 0.0))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} must be in [0, 1], got {value}")
    return value


def _frame_power(audio: torch.Tensor, frame_samples: int, hop_samples: int | None = None) -> torch.Tensor:
    if audio.ndim != 2:
        raise ValueError(f"Expected [channels, samples], got {tuple(audio.shape)}")
    frame_samples = max(1, int(frame_samples))
    hop_samples = frame_samples if hop_samples is None else max(1, int(hop_samples))
    n_samples = int(audio.shape[-1])
    n_frames = max(1, math.ceil(max(0, n_samples - frame_samples) / hop_samples) + 1)
    padded_samples = (n_frames - 1) * hop_samples + frame_samples
    pad = max(0, padded_samples - n_samples)
    squared = audio.float().square().mean(dim=0, keepdim=True).unsqueeze(0)
    if pad:
        squared = F.pad(squared, (0, pad))
    return F.avg_pool1d(squared, kernel_size=frame_samples, stride=hop_samples).flatten()


def active_rms(
    audio: torch.Tensor,
    *,
    sr: int,
    frame_ms: float = 40.0,
    hop_ms: float = 20.0,
    activity_threshold_db: float = -48.0,
) -> torch.Tensor:
    """Measure RMS over active non-overlapping frames of ``[channels, samples]`` audio."""

    if int(sr) <= 0:
        raise ValueError(f"sr must be positive, got {sr}")
    if float(frame_ms) <= 0.0 or not math.isfinite(float(frame_ms)):
        raise ValueError(f"frame_ms must be positive and finite, got {frame_ms}")
    if float(hop_ms) <= 0.0 or not math.isfinite(float(hop_ms)):
        raise ValueError(f"hop_ms must be positive and finite, got {hop_ms}")
    frame_samples = max(1, int(round(float(frame_ms) * int(sr) / 1000.0)))
    hop_samples = max(1, int(round(float(hop_ms) * int(sr) / 1000.0)))
    power = _frame_power(audio, frame_samples, hop_samples)
    active = power > float(10.0 ** (float(activity_threshold_db) / 10.0))
    if not active.any():
        return power.new_zeros(())
    return power[active].mean().sqrt()


def _expand_frame_gain(gain: torch.Tensor, *, frame_samples: int, n_samples: int) -> torch.Tensor:
    expanded = gain.repeat_interleave(frame_samples)
    if expanded.numel() < n_samples:
        expanded = F.pad(expanded, (0, n_samples - expanded.numel()), value=float(gain[-1]))
    return expanded[:n_samples]


def _smooth_gain(
    target: torch.Tensor,
    *,
    frame_ms: float,
    attack_ms: float,
    release_ms: float,
) -> torch.Tensor:
    if target.numel() == 0:
        return target
    attack_coeff = math.exp(-float(frame_ms) / max(float(attack_ms), 1.0e-3))
    release_coeff = math.exp(-float(frame_ms) / max(float(release_ms), 1.0e-3))
    result = torch.empty_like(target)
    state = target.new_tensor(1.0)
    for frame_idx in range(target.numel()):
        current = target[frame_idx]
        coeff = attack_coeff if float(current) < float(state) else release_coeff
        state = state * coeff + current * (1.0 - coeff)
        result[frame_idx] = state
    return result


def _compressor_gain(audio: torch.Tensor, *, sr: int, config: Mapping[str, Any], rng: random.Random) -> torch.Tensor:
    frame_ms = float(config.get("frame_ms", 20.0))
    frame_samples = max(1, int(round(frame_ms * sr / 1000.0)))
    power = _frame_power(audio, frame_samples)
    level_db = 10.0 * torch.log10(power.clamp_min(1.0e-12))
    threshold_db = _sample_range(rng, config.get("threshold_db"), default=(-24.0, -12.0))
    ratio = _sample_range(rng, config.get("ratio"), default=(2.0, 4.0))
    if ratio < 1.0:
        raise ValueError(f"Compressor ratio must be at least 1, got {ratio}")
    over_db = (level_db - threshold_db).clamp_min(0.0)
    target_gain = torch.pow(10.0, -(over_db * (1.0 - 1.0 / ratio)) / 20.0)
    smoothed = _smooth_gain(
        target_gain,
        frame_ms=frame_ms,
        attack_ms=float(config.get("attack_ms", 10.0)),
        release_ms=float(config.get("release_ms", 100.0)),
    )
    return _expand_frame_gain(smoothed, frame_samples=frame_samples, n_samples=audio.shape[-1])


def _ducking_gain(speech: torch.Tensor, *, sr: int, config: Mapping[str, Any], rng: random.Random) -> torch.Tensor:
    frame_ms = float(config.get("frame_ms", 20.0))
    frame_samples = max(1, int(round(frame_ms * sr / 1000.0)))
    power = _frame_power(speech, frame_samples)
    threshold = float(10.0 ** (float(config.get("activity_threshold_db", -48.0)) / 10.0))
    attenuation_db = _sample_range(rng, config.get("attenuation_db"), default=(-10.0, -3.0))
    if attenuation_db > 0.0:
        raise ValueError(f"Ducking attenuation must be non-positive dB, got {attenuation_db}")
    duck_gain = float(10.0 ** (attenuation_db / 20.0))
    target_gain = torch.where(power > threshold, power.new_tensor(duck_gain), power.new_tensor(1.0))
    smoothed = _smooth_gain(
        target_gain,
        frame_ms=frame_ms,
        attack_ms=float(config.get("attack_ms", 20.0)),
        release_ms=float(config.get("release_ms", 250.0)),
    )
    return _expand_frame_gain(smoothed, frame_samples=frame_samples, n_samples=speech.shape[-1])


def _shared_channel_eq(audio: torch.Tensor, *, sr: int, config: Mapping[str, Any], rng: random.Random) -> torch.Tensor:
    n_samples = int(audio.shape[-1])
    frequencies = torch.linspace(0.0, sr / 2.0, n_samples // 2 + 1, device=audio.device)
    low_hz = _sample_range(rng, config.get("low_cut_hz"), default=(20.0, 180.0))
    high_hz = _sample_range(rng, config.get("high_cut_hz"), default=(8000.0, min(20000.0, sr / 2.0)))
    high_hz = min(high_hz, sr / 2.0)
    if low_hz >= high_hz:
        raise ValueError(f"channel_eq requires low_cut_hz < high_cut_hz, got {low_hz} >= {high_hz}")
    transition_hz = max(1.0, float(config.get("transition_hz", 200.0)))
    high_pass = ((frequencies - (low_hz - transition_hz)) / transition_hz).clamp(0.0, 1.0)
    low_pass = (((high_hz + transition_hz) - frequencies) / transition_hz).clamp(0.0, 1.0)
    response = high_pass * low_pass
    spectrum = torch.fft.rfft(audio.float(), dim=-1)
    return torch.fft.irfft(spectrum * response, n=n_samples, dim=-1).to(audio.dtype)


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "wet"}:
        return True
    if normalized in {"0", "false", "no", "n", "dry"}:
        return False
    return None


class BroadcastStemRenderer:
    """Apply label-consistent acoustic and broadcast processing to stem targets.

    Every operation transforms the returned stems themselves. The caller must
    construct the mixture after rendering, which preserves exact additivity.
    """

    _VALID_KEYS = {
        "room",
        "channel_eq",
        "source_compression",
        "ducking",
        "bus_compression",
        "bus_peak_limit_db",
    }

    def __init__(self, *, sr: int, source_order: Sequence[str], config: Mapping[str, Any] | None = None) -> None:
        self.sr = int(sr)
        self.source_order = tuple(str(stem) for stem in source_order)
        if self.sr <= 0:
            raise ValueError(f"sr must be positive, got {sr}")
        if not self.source_order or len(set(self.source_order)) != len(self.source_order):
            raise ValueError(f"source_order must contain unique stem names, got {self.source_order}")
        self.config = dict(config or {})
        unknown = sorted(set(self.config) - self._VALID_KEYS)
        if unknown:
            raise ValueError(f"broadcast config contains unknown fields: {unknown}")
        for key in ("room", "channel_eq", "source_compression", "ducking", "bus_compression"):
            value = self.config.get(key, {})
            if not isinstance(value, Mapping):
                raise ValueError(f"broadcast.{key} must be a mapping")
            _probability(value)

        for section_name in ("source_compression", "bus_compression"):
            section = self.config.get(section_name, {})
            self._validate_dynamics(section, section_name=section_name)
            ratio = _as_range(section.get("ratio"), default=(2.0, 4.0))
            if ratio[0] < 1.0:
                raise ValueError(f"broadcast.{section_name}.ratio must be at least 1, got {ratio}")
        source_stems = tuple(self.config.get("source_compression", {}).get("stems", self.source_order))
        self._validate_stems(source_stems, field="source_compression.stems")

        ducking = self.config.get("ducking", {})
        self._validate_dynamics(ducking, section_name="ducking")
        attenuation = _as_range(ducking.get("attenuation_db"), default=(-10.0, -3.0))
        if attenuation[1] > 0.0:
            raise ValueError(f"broadcast.ducking.attenuation_db must be non-positive, got {attenuation}")
        if ducking:
            speech_stem = str(ducking.get("speech_stem", "speech"))
            self._validate_stems((speech_stem,), field="ducking.speech_stem")
            self._validate_stems(
                tuple(ducking.get("target_stems", ("music", "effects"))),
                field="ducking.target_stems",
            )

        channel_eq = self.config.get("channel_eq", {})
        if channel_eq:
            transition_hz = float(channel_eq.get("transition_hz", 200.0))
            if not math.isfinite(transition_hz) or transition_hz <= 0.0:
                raise ValueError(f"broadcast.channel_eq.transition_hz must be positive, got {transition_hz}")
            low_cut = _as_range(channel_eq.get("low_cut_hz"), default=(20.0, 180.0))
            high_cut = _as_range(
                channel_eq.get("high_cut_hz"),
                default=(min(8000.0, self.sr / 2.0), min(20000.0, self.sr / 2.0)),
            )
            if low_cut[0] < 0.0 or high_cut[0] <= 0.0 or low_cut[1] >= min(high_cut[0], self.sr / 2.0):
                raise ValueError(
                    "broadcast.channel_eq cutoffs must satisfy 0 <= low_cut_hz < high_cut_hz <= Nyquist"
                )

        room = self.config.get("room", {})
        if room:
            wet_mix = _as_range(room.get("wet_mix"), default=(0.15, 0.45))
            if wet_mix[0] < 0.0 or wet_mix[1] > 1.0:
                raise ValueError(f"broadcast.room.wet_mix must be in [0, 1], got {wet_mix}")
            max_rir_seconds = float(room.get("max_rir_seconds", 1.0))
            if not math.isfinite(max_rir_seconds) or max_rir_seconds <= 0.0:
                raise ValueError(
                    f"broadcast.room.max_rir_seconds must be positive, got {max_rir_seconds}"
                )
            unknown_policy = str(room.get("unknown_wet_policy", "assume_wet"))
            if unknown_policy not in {"assume_wet", "assume_dry"}:
                raise ValueError("broadcast.room.unknown_wet_policy must be 'assume_wet' or 'assume_dry'")
            self._validate_stems(
                tuple(room.get("shared_stems", ("speech", "effects"))),
                field="room.shared_stems",
            )
        self.rir_paths = tuple(Path(path).expanduser() for path in room.get("rir_paths", ()))
        if _probability(room) > 0.0:
            if not self.rir_paths:
                raise ValueError("broadcast.room.rir_paths must not be empty when room probability is positive")
            missing = [path for path in self.rir_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Room impulse responses do not exist: {missing}")

        peak_limit_db = self.config.get("bus_peak_limit_db")
        if peak_limit_db is not None:
            peak_limit_db = float(peak_limit_db)
            if not math.isfinite(peak_limit_db) or peak_limit_db > 0.0:
                raise ValueError(f"broadcast.bus_peak_limit_db must be finite and non-positive, got {peak_limit_db}")

    def _validate_stems(self, stems: Sequence[str], *, field: str) -> None:
        unknown = sorted(set(stems) - set(self.source_order))
        if unknown:
            raise ValueError(f"broadcast.{field} contains unknown stems: {unknown}")

    @staticmethod
    def _validate_dynamics(config: Mapping[str, Any], *, section_name: str) -> None:
        if not config:
            return
        for field, default in (("frame_ms", 20.0), ("attack_ms", 10.0), ("release_ms", 100.0)):
            value = float(config.get(field, default))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"broadcast.{section_name}.{field} must be positive, got {value}")
        _as_range(config.get("threshold_db"), default=(-24.0, -12.0))

    def _load_rir(self, path: Path, *, max_seconds: float) -> torch.Tensor:
        info = sf.info(path)
        max_frames = max(1, int(round(max_seconds * info.samplerate)))
        audio_np, source_sr = sf.read(path, frames=max_frames, always_2d=True, dtype="float32")
        rir = torch.from_numpy(audio_np.T.copy()).float().mean(dim=0, keepdim=True)
        if int(source_sr) != self.sr:
            rir = AF.resample(rir, orig_freq=int(source_sr), new_freq=self.sr)
        rir = rir.flatten()
        energy = rir.square().sum().sqrt()
        if not torch.isfinite(energy) or float(energy) <= 1.0e-8:
            raise ValueError(f"RIR has no energy: {path}")
        return rir / energy

    def _convolve_rir(self, audio: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
        n_samples = int(audio.shape[-1])
        fft_size = n_samples + int(rir.numel()) - 1
        spectrum = torch.fft.rfft(audio.float(), n=fft_size, dim=-1)
        rir_spectrum = torch.fft.rfft(rir.to(audio.device), n=fft_size)
        return torch.fft.irfft(spectrum * rir_spectrum, n=fft_size, dim=-1)[..., :n_samples].to(audio.dtype)

    def _path_is_dry(
        self,
        path: str,
        *,
        source_metadata: Mapping[str, Mapping[str, Any]],
        unknown_wet_policy: str,
    ) -> bool:
        row = source_metadata.get(str(Path(path)), source_metadata.get(path, {}))
        is_wet = _parse_bool(row.get("is_wet")) if row else None
        if is_wet is None:
            return unknown_wet_policy == "assume_dry"
        return not is_wet

    def _apply_room(
        self,
        stems: torch.Tensor,
        *,
        rng: random.Random,
        source_paths: Mapping[str, Sequence[str]],
        source_metadata: Mapping[str, Mapping[str, Any]],
    ) -> list[str]:
        config = self.config.get("room", {})
        if rng.random() >= _probability(config):
            return []
        shared_stems = tuple(config.get("shared_stems", ("speech", "effects")))
        unknown_policy = str(config.get("unknown_wet_policy", "assume_wet"))
        if unknown_policy not in {"assume_wet", "assume_dry"}:
            raise ValueError("broadcast.room.unknown_wet_policy must be 'assume_wet' or 'assume_dry'")
        candidates = [
            stem
            for stem in shared_stems
            if stem in self.source_order and bool(torch.count_nonzero(stems[self.source_order.index(stem)]))
        ]
        all_dry = all(
            source_paths.get(stem)
            and all(
                self._path_is_dry(path, source_metadata=source_metadata, unknown_wet_policy=unknown_policy)
                for path in source_paths[stem]
            )
            for stem in candidates
        )
        if not candidates or not all_dry:
            return []

        rir = self._load_rir(
            rng.choice(self.rir_paths),
            max_seconds=float(config.get("max_rir_seconds", 1.0)),
        )
        wet_mix = _sample_range(rng, config.get("wet_mix"), default=(0.15, 0.45))
        if not 0.0 <= wet_mix <= 1.0:
            raise ValueError(f"broadcast.room.wet_mix must be in [0, 1], got {wet_mix}")
        preserve_rms = bool(config.get("preserve_rms", True))
        for stem in candidates:
            stem_idx = self.source_order.index(stem)
            dry = stems[stem_idx]
            wet = self._convolve_rir(dry, rir)
            rendered = dry * (1.0 - wet_mix) + wet * wet_mix
            if preserve_rms:
                dry_rms = dry.float().square().mean().sqrt()
                rendered_rms = rendered.float().square().mean().sqrt().clamp_min(1.0e-8)
                rendered = rendered * (dry_rms / rendered_rms)
            stems[stem_idx] = rendered
        return candidates

    def render(
        self,
        stems: torch.Tensor,
        *,
        rng: random.Random,
        source_paths: Mapping[str, Sequence[str]] | None = None,
        source_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if stems.ndim != 3 or stems.shape[0] != len(self.source_order):
            raise ValueError(
                f"Expected [n_src, channels, samples] with n_src={len(self.source_order)}, got {tuple(stems.shape)}"
            )
        out = stems.clone()
        paths = source_paths or {}
        metadata_rows = source_metadata or {}
        render_metadata: dict[str, Any] = {
            "room_applied_stems": self._apply_room(
                out,
                rng=rng,
                source_paths=paths,
                source_metadata=metadata_rows,
            ),
            "channel_eq_applied": False,
            "source_compression_applied_stems": [],
            "ducking_applied": False,
            "bus_compression_applied": False,
        }

        channel_eq = self.config.get("channel_eq", {})
        if rng.random() < _probability(channel_eq):
            out = _shared_channel_eq(out, sr=self.sr, config=channel_eq, rng=rng)
            render_metadata["channel_eq_applied"] = True

        source_compression = self.config.get("source_compression", {})
        if rng.random() < _probability(source_compression):
            configured_stems = tuple(source_compression.get("stems", self.source_order))
            applied = []
            for stem in configured_stems:
                if stem not in self.source_order:
                    raise ValueError(f"Unknown source_compression stem: {stem!r}")
                stem_idx = self.source_order.index(stem)
                if not torch.count_nonzero(out[stem_idx]):
                    continue
                gain = _compressor_gain(out[stem_idx], sr=self.sr, config=source_compression, rng=rng)
                out[stem_idx] *= gain
                applied.append(stem)
            render_metadata["source_compression_applied_stems"] = applied

        ducking = self.config.get("ducking", {})
        if rng.random() < _probability(ducking):
            speech_stem = str(ducking.get("speech_stem", "speech"))
            if speech_stem not in self.source_order:
                raise ValueError(f"Unknown ducking speech_stem: {speech_stem!r}")
            speech = out[self.source_order.index(speech_stem)]
            if torch.count_nonzero(speech):
                gain = _ducking_gain(speech, sr=self.sr, config=ducking, rng=rng)
                for stem in ducking.get("target_stems", ("music", "effects")):
                    if stem not in self.source_order:
                        raise ValueError(f"Unknown ducking target stem: {stem!r}")
                    out[self.source_order.index(stem)] *= gain
                render_metadata["ducking_applied"] = True

        bus_compression = self.config.get("bus_compression", {})
        if rng.random() < _probability(bus_compression):
            mixture = out.sum(dim=0)
            gain = _compressor_gain(mixture, sr=self.sr, config=bus_compression, rng=rng)
            out *= gain
            render_metadata["bus_compression_applied"] = True

        peak_limit_db = self.config.get("bus_peak_limit_db")
        if peak_limit_db is not None:
            mixture_peak = out.sum(dim=0).abs().max()
            limit = float(10.0 ** (float(peak_limit_db) / 20.0))
            if float(mixture_peak) > limit:
                out *= limit / mixture_peak
                render_metadata["bus_peak_limited"] = True
            else:
                render_metadata["bus_peak_limited"] = False
        return out, render_metadata
