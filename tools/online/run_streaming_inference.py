#!/usr/bin/env python3

from __future__ import annotations

import importlib
import json
import re
from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path
import sys
from typing import Iterable

from einops import rearrange
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_feature_compression.core.model.online_sfc_2d import pack_complex_stft_as_2d, unpack_2d_to_complex_stft


INTERPOLATION_RE = re.compile(r"^\$\{([^}]+)\}$")


def import_object(target: str):
    module_name, _, attr_name = target.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid import target: {target}")
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def parse_scalar(value: str):
    if not value:
        return ""
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def get_concrete_base_path(path: Path) -> Path | None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line.startswith(" ") or raw_line.startswith("\t"):
            continue
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        if key.strip() != "_base_":
            continue
        base_value = parse_scalar(value.strip())
        if isinstance(base_value, str) and base_value and "${" not in base_value:
            base_path = (path.parent / base_value).resolve()
            if base_path.exists():
                return base_path
        return None
    return None


def merge_top_level_scalars(path: Path) -> dict[str, object]:
    merged: dict[str, object] = {}
    parsed: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line.startswith(" ") or raw_line.startswith("\t"):
            continue
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        parsed[key.strip()] = parse_scalar(value.strip())

    base_path = get_concrete_base_path(path)
    if base_path is not None:
        merged.update(merge_top_level_scalars(base_path))

    for key, value in parsed.items():
        if key == "_base_":
            continue
        merged[key] = value
    return merged


def merge_task_model_mapping(path: Path) -> dict[str, object]:
    merged: dict[str, object] = {}
    base_path = get_concrete_base_path(path)
    if base_path is not None:
        merged.update(merge_task_model_mapping(base_path))

    in_task = False
    in_model = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if indent == 0:
            in_task = stripped.startswith("task:")
            in_model = False
            continue
        if in_task and indent == 2 and stripped.startswith("model:"):
            in_model = True
            continue
        if in_model:
            if indent < 4:
                in_model = False
                continue
            if indent == 4 and ":" in stripped:
                key, value = stripped.split(":", 1)
                merged[key.strip()] = parse_scalar(value.strip())
    return merged


def resolve_value(value, context: dict[str, object], *, stack: tuple[str, ...] = ()):
    if isinstance(value, list):
        return [resolve_value(item, context, stack=stack) for item in value]
    if isinstance(value, str):
        match = INTERPOLATION_RE.fullmatch(value)
        if match is None:
            return value
        ref = match.group(1)
        if ref in stack:
            raise ValueError(f"Cyclic interpolation detected: {' -> '.join((*stack, ref))}")
        if ref not in context:
            raise KeyError(f"Unknown interpolation key: {ref}")
        return resolve_value(context[ref], context, stack=(*stack, ref))
    return value


def build_model_system_from_recipe_config(config_path: Path):
    top_level = merge_top_level_scalars(config_path)
    model_cfg = merge_task_model_mapping(config_path)
    if "_target_" not in model_cfg:
        raise ValueError(f"Could not find task.model._target_ in {config_path}")

    resolved_model_cfg: dict[str, object] = {}
    for key, value in model_cfg.items():
        resolution_context = {**top_level, **model_cfg, **resolved_model_cfg}
        if isinstance(value, str):
            match = INTERPOLATION_RE.fullmatch(value)
            if match is not None and match.group(1) == key and key in top_level:
                resolved_model_cfg[key] = resolve_value(top_level[key], resolution_context, stack=(key,))
                continue
        resolved_model_cfg[key] = resolve_value(value, resolution_context, stack=(key,))
    target = resolved_model_cfg.pop("_target_")
    builder = import_object(target)
    return builder(**resolved_model_cfg)


def load_trained_task(model_path: Path, device: str):
    from aiaccel.config import load_config
    from hydra.utils import instantiate

    if model_path.is_dir():
        checkpoint_path = model_path / "checkpoints"
        config_path = model_path
    elif model_path.suffix in [".ckpt", ".pth", ".pt"]:
        checkpoint_path = model_path
        config_path = checkpoint_path.parent.parent
    else:
        raise ValueError("model_path must be either a model directory or a checkpoint file")

    config = load_config(config_path / "merged_config.yaml")
    task = instantiate(config.task).to(device)
    task.load_average_checkpoint(checkpoint_path)
    task.eval()
    return task, config


def infer_recipe_config_path(model_path: Path) -> Path | None:
    if model_path.is_file() and model_path.name in {"config.yaml", "merged_config.yaml"}:
        return model_path
    if model_path.is_dir():
        merged_config = model_path / "merged_config.yaml"
        config_yaml = model_path / "config.yaml"
        if merged_config.exists():
            return merged_config
        if config_yaml.exists():
            return config_yaml
        return None
    if model_path.is_file() and model_path.suffix in {".ckpt", ".pth", ".pt"}:
        candidate = model_path.parent.parent / "merged_config.yaml"
        return candidate if candidate.exists() else None
    return None


def load_streaming_model(model_path: Path, device: str):
    source_mode: str
    config_path = infer_recipe_config_path(model_path)
    top_level = merge_top_level_scalars(config_path) if config_path is not None else {}

    if model_path.is_file() and model_path.suffix in {".ckpt", ".pth", ".pt"}:
        task, _ = load_trained_task(model_path, device)
        wrapper = task.ema_model.module if getattr(task, "use_ema_model", False) else task.model
        source_mode = "trained_checkpoint"
    elif model_path.is_dir():
        merged_config = model_path / "merged_config.yaml"
        config_yaml = model_path / "config.yaml"
        checkpoint_dir = model_path / "checkpoints"
        if merged_config.exists() and checkpoint_dir.exists():
            task, _ = load_trained_task(model_path, device)
            wrapper = task.ema_model.module if getattr(task, "use_ema_model", False) else task.model
            source_mode = "trained_directory"
        elif merged_config.exists():
            wrapper = build_model_system_from_recipe_config(merged_config).to(device)
            source_mode = "config_only_merged"
        elif config_yaml.exists():
            wrapper = build_model_system_from_recipe_config(config_yaml).to(device)
            source_mode = "config_only_recipe"
        else:
            raise ValueError(
                f"{model_path} must contain merged_config.yaml or config.yaml, and checkpoints/ for trained inference."
            )
    elif model_path.is_file() and model_path.name in {"config.yaml", "merged_config.yaml"}:
        wrapper = build_model_system_from_recipe_config(model_path).to(device)
        source_mode = "config_only_merged" if model_path.name == "merged_config.yaml" else "config_only_recipe"
    else:
        raise ValueError(
            "model_path must be a checkpoint, a model directory, or a config.yaml / merged_config.yaml file."
        )

    wrapper = wrapper.to(device)
    wrapper.eval()

    online_model = getattr(wrapper, "model", None)
    if online_model is None or not hasattr(online_model, "forward_stream"):
        raise ValueError(f"Expected an OnlineModelWrapper, got {type(wrapper).__name__}")

    return wrapper, top_level, source_mode


def default_stem_names(n_src: int) -> list[str]:
    if n_src == 4:
        return ["bass", "drums", "vocals", "other"]
    if n_src == 3:
        return ["speech", "music", "sfx"]
    return [f"est{i}" for i in range(n_src)]


class StreamingISTFTWriter:
    def __init__(
        self,
        *,
        out_dir: Path,
        stem_names: list[str],
        sample_rate: int,
        n_chan: int,
        subtype: str = "PCM_24",
        n_fft: int,
        hop_length: int,
        window: torch.Tensor,
        total_length: int,
        left_trim: int,
        eps: float = 1e-8,
    ):
        self.out_dir = out_dir
        self.stem_names = stem_names
        self.sample_rate = sample_rate
        self.n_chan = n_chan
        self.subtype = subtype
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.tail = n_fft - hop_length
        self.eps = eps
        self.total_length = total_length
        self.remaining_drop = left_trim
        self.remaining_keep = total_length

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.files = [
            sf.SoundFile(
                self.out_dir / f"{idx:02d}_{stem}.wav",
                mode="w",
                samplerate=sample_rate,
                channels=n_chan,
                subtype=subtype,
            )
            for idx, stem in enumerate(stem_names)
        ]

        window = window.to(torch.float32)
        self.window = window
        self.window_sq = window.square()
        self.signal_tail: torch.Tensor | None = None
        self.env_tail: torch.Tensor | None = None

    def _fold_chunk(self, frames: torch.Tensor, window_sq_cols: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_length = self.n_fft + self.hop_length * max(frames.shape[-1] - 1, 0)
        signal = F.fold(
            frames,
            output_size=(1, local_length),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).squeeze(-2).squeeze(-2)
        envelope = F.fold(
            window_sq_cols,
            output_size=(1, local_length),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).squeeze(-2).squeeze(-2)
        return signal, envelope

    def _write_ready_audio(self, signal: torch.Tensor, envelope: torch.Tensor) -> None:
        if signal.numel() == 0 or self.remaining_keep <= 0:
            return

        audio = torch.where(envelope > self.eps, signal / envelope.clamp_min(self.eps), torch.zeros_like(signal))
        audio = audio.reshape(1, len(self.stem_names), self.n_chan, -1)[0]

        if self.remaining_drop > 0:
            drop = min(self.remaining_drop, audio.shape[-1])
            audio = audio[..., drop:]
            self.remaining_drop -= drop
            if audio.shape[-1] == 0:
                return

        take = min(self.remaining_keep, audio.shape[-1])
        if take <= 0:
            return
        audio = audio[..., :take]
        self.remaining_keep -= take

        audio_np = audio.transpose(1, 2).detach().cpu().to(torch.float32).numpy()
        for idx, handle in enumerate(self.files):
            handle.write(audio_np[idx])

    def push(self, stft_chunk: torch.Tensor, *, final: bool) -> None:
        if not torch.is_complex(stft_chunk):
            raise TypeError(f"Expected complex STFT chunk, got {stft_chunk.dtype}")
        if stft_chunk.ndim != 5 or stft_chunk.shape[0] != 1:
            raise ValueError(f"Expected STFT chunk shape (1, N, M, F, T), got {tuple(stft_chunk.shape)}")

        n_frames = stft_chunk.shape[-1]
        flat = stft_chunk.reshape(-1, stft_chunk.shape[-2], n_frames)

        if self.signal_tail is None:
            device = flat.device
            self.signal_tail = torch.zeros(flat.shape[0], self.tail, device=device, dtype=torch.float32)
            self.env_tail = torch.zeros_like(self.signal_tail)

        if n_frames == 0:
            if final and self.signal_tail is not None:
                self._write_ready_audio(self.signal_tail, self.env_tail)
                self.signal_tail.zero_()
                self.env_tail.zero_()
            return

        frames = torch.fft.irfft(flat.to(torch.complex64), n=self.n_fft, dim=1)
        frames = frames.to(torch.float32) * self.window.view(1, self.n_fft, 1).to(flat.device)
        window_sq_cols = self.window_sq.view(1, self.n_fft, 1).expand(flat.shape[0], self.n_fft, n_frames).to(flat.device)
        signal, envelope = self._fold_chunk(frames, window_sq_cols)

        if self.tail > 0:
            signal[:, : self.tail] += self.signal_tail
            envelope[:, : self.tail] += self.env_tail

        if final:
            ready_signal = signal
            ready_envelope = envelope
            self.signal_tail = torch.zeros_like(self.signal_tail)
            self.env_tail = torch.zeros_like(self.env_tail)
        else:
            ready_length = signal.shape[-1] - self.tail
            ready_signal = signal[:, :ready_length]
            ready_envelope = envelope[:, :ready_length]
            self.signal_tail = signal[:, ready_length:]
            self.env_tail = envelope[:, ready_length:]

        self._write_ready_audio(ready_signal, ready_envelope)

    def close(self) -> None:
        for handle in self.files:
            handle.close()


def iter_input_files(input_path: Path, *, patterns: list[str], recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = input_path.rglob(pattern) if recursive else input_path.glob(pattern)
        for match in sorted(matches):
            if match.is_file() and match not in seen:
                seen.add(match)
                files.append(match)
    return files


def infer_sample_out_dir(
    src_file: Path,
    *,
    input_root: Path,
    output_root: Path,
    output_group: str | None,
) -> Path:
    relative = src_file.relative_to(input_root) if input_root.is_dir() else Path(src_file.name)
    if src_file.stem.lower() in {"mix", "mixture"}:
        sample_rel = relative.parent if str(relative.parent) not in {"", "."} else Path(src_file.stem)
    else:
        sample_rel = relative.with_suffix("")
    if output_group:
        sample_rel = Path(output_group) / sample_rel
    return output_root / sample_rel


def write_run_manifest(
    *,
    path: Path,
    model_path: Path,
    source_mode: str,
    input_path: Path,
    output_root: Path,
    chunk_frames: int,
    stem_names: list[str],
    wrapper,
) -> None:
    online_model = wrapper.model
    freq_preprocess = None
    if hasattr(online_model, "frequency_preprocess_manifest"):
        freq_preprocess = online_model.frequency_preprocess_manifest()

    manifest = {
        "model_path": str(model_path),
        "source_mode": source_mode,
        "input_path": str(input_path),
        "output_root": str(output_root),
        "chunk_frames": chunk_frames,
        "chunk_samples": int(chunk_frames * wrapper.hop_length),
        "sample_rate": int(wrapper.fs),
        "n_fft": int(wrapper.istft.n_fft),
        "hop_length": int(wrapper.hop_length),
        "left_pad_samples": int(wrapper.wave_context_samples),
        "right_pad_samples": int(wrapper.wave_tail_pad_samples),
        "n_src": len(stem_names),
        "n_chan": int(getattr(online_model, "n_chan")),
        "stem_names": stem_names,
        "stream_context_frames": int(online_model.stream_context_frames()),
        "frequency_preprocessing": freq_preprocess,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def autocast_context(device: str, enabled: bool):
    if not enabled or not device.startswith("cuda"):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


@torch.inference_mode()
def run_streaming_file(
    *,
    src_file: Path,
    out_dir: Path,
    wrapper,
    stem_names: list[str],
    device: str,
    chunk_frames: int,
    subtype: str,
    use_autocast: bool,
) -> None:
    wav_np, sample_rate = sf.read(src_file, dtype="float32", always_2d=True)
    expected_sr = int(wrapper.fs)
    if int(sample_rate) != expected_sr:
        raise ValueError(f"{src_file} sample rate {sample_rate} != model fs {expected_sr}")

    wav = rearrange(torch.from_numpy(wav_np), "t m -> 1 m t").to(device=device, dtype=torch.float32)
    online_model = wrapper.model
    n_src = len(stem_names)
    n_chan = int(getattr(online_model, "n_chan"))
    if wav.shape[1] != n_chan:
        raise ValueError(f"{src_file} has {wav.shape[1]} channels, but model expects {n_chan}")

    state = online_model.init_stream_state(batch_size=1, device=device, dtype=torch.float32)
    chunk_samples = int(chunk_frames * wrapper.hop_length)
    left_pad = int(wrapper.wave_context_samples)
    right_pad = int(wrapper.wave_tail_pad_samples)

    analysis_buffer = torch.zeros(1, n_chan, left_pad, device=device, dtype=torch.float32)

    writer = StreamingISTFTWriter(
        out_dir=out_dir,
        stem_names=stem_names,
        sample_rate=sample_rate,
        n_chan=n_chan,
        subtype=subtype,
        n_fft=int(wrapper.istft.n_fft),
        hop_length=int(wrapper.hop_length),
        window=wrapper.istft.window.detach(),
        total_length=wav.shape[-1],
        left_trim=left_pad,
        eps=float(wrapper.istft.eps),
    )

    try:
        with autocast_context(device, enabled=use_autocast):
            for start in range(0, wav.shape[-1], chunk_samples):
                chunk = wav[..., start : start + chunk_samples]
                buf = torch.cat([analysis_buffer, chunk], dim=-1)
                x = wrapper.stft(buf)
                produced_samples = int(x.shape[-1] * wrapper.hop_length)
                analysis_buffer = buf[..., produced_samples:]
                if x.shape[-1] == 0:
                    continue
                x2d = pack_complex_stft_as_2d(x)
                y2d, state = online_model.forward_stream(x2d, state)
                y = unpack_2d_to_complex_stft(y2d, n_src=n_src, n_chan=n_chan)
                writer.push(y, final=False)

            flush = torch.zeros(1, n_chan, right_pad, device=device, dtype=torch.float32)
            buf = torch.cat([analysis_buffer, flush], dim=-1)
            x = wrapper.stft(buf)
            if x.shape[-1] > 0:
                x2d = pack_complex_stft_as_2d(x)
                y2d, state = online_model.forward_stream(x2d, state)
                y = unpack_2d_to_complex_stft(y2d, n_src=n_src, n_chan=n_chan)
                writer.push(y, final=True)
            else:
                writer.push(
                    torch.zeros(1, n_src, n_chan, wrapper.istft.n_fft // 2 + 1, 0, device=device, dtype=torch.complex64),
                    final=True,
                )
    finally:
        writer.close()


def main() -> None:
    parser = ArgumentParser(description="Run true stateful streaming inference for online models on wav files.")
    parser.add_argument(
        "model_path",
        type=Path,
        help="Checkpoint, trained model directory, or config.yaml / merged_config.yaml.",
    )
    parser.add_argument("input_path", type=Path, help="Input wav file or directory.")
    parser.add_argument("output_root", type=Path, help="Output root directory.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Glob pattern for wav discovery when input_path is a directory. Can be passed multiple times.",
    )
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive directory search.")
    parser.add_argument("--chunk-frames", type=int, default=8, help="Streaming STFT frames per model call.")
    parser.add_argument(
        "--stem-name",
        action="append",
        default=[],
        help="Override output stem names. Repeat once per source in model order.",
    )
    parser.add_argument(
        "--output-group",
        type=str,
        default=None,
        help="Optional extra directory inserted before each sample directory, useful for metrics layout.",
    )
    parser.add_argument("--subtype", type=str, default="PCM_24", help="soundfile subtype for written wavs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-sample output directories.")
    parser.add_argument("--manifest-out", type=Path, help="Optional JSON manifest for this inference run.")
    parser.add_argument(
        "--cuda-autocast",
        action="store_true",
        help="Enable bfloat16 autocast on CUDA for faster inference.",
    )

    args = parser.parse_args()

    wrapper, top_level, source_mode = load_streaming_model(args.model_path, args.device)
    online_model = wrapper.model
    n_src = int(getattr(online_model, "n_src"))
    stem_names = args.stem_name or default_stem_names(n_src)
    if len(stem_names) != n_src:
        raise ValueError(f"Expected {n_src} stem names, got {len(stem_names)}")
    if not getattr(getattr(online_model, "core", online_model), "causal", True):
        raise ValueError("This script only supports causal online models.")
    if args.chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive.")

    patterns = args.pattern or ["mix*.wav", "mixture.wav", "Mix*.wav"]
    src_files = iter_input_files(args.input_path, patterns=patterns, recursive=not args.no_recursive)
    if not src_files:
        raise ValueError(f"No input wav files found under {args.input_path} with patterns {patterns}")

    if args.manifest_out is not None:
        write_run_manifest(
            path=args.manifest_out,
            model_path=args.model_path,
            source_mode=source_mode,
            input_path=args.input_path,
            output_root=args.output_root,
            chunk_frames=args.chunk_frames,
            stem_names=stem_names,
            wrapper=wrapper,
        )

    print("=" * 48)
    print(f"model_path: {args.model_path}")
    print(f"source_mode: {source_mode}")
    print(f"input_path: {args.input_path}")
    print(f"num_files: {len(src_files)}")
    print(f"device: {args.device}")
    print(f"chunk_frames: {args.chunk_frames}")
    print(f"chunk_samples: {args.chunk_frames * wrapper.hop_length}")
    print(f"sample_rate: {wrapper.fs}")
    print(f"n_fft/hop: {wrapper.istft.n_fft}/{wrapper.hop_length}")
    print(f"stems: {stem_names}")
    if top_level.get("online_freq_preprocess_enabled", False):
        print(
            "frequency_preprocess: "
            f"keep={top_level.get('online_freq_preprocess_keep_bins')} "
            f"target={top_level.get('online_freq_preprocess_target_bins')} "
            f"mode={top_level.get('online_freq_preprocess_mode', 'triangular')}"
        )
    print("=" * 48)

    for idx, src_file in enumerate(src_files, start=1):
        out_dir = infer_sample_out_dir(
            src_file,
            input_root=args.input_path if args.input_path.is_dir() else src_file.parent,
            output_root=args.output_root,
            output_group=args.output_group,
        )
        if out_dir.exists():
            existing = sorted(out_dir.glob("*.wav"))
            if existing and not args.overwrite:
                print(f"[{idx}/{len(src_files)}] skip existing: {src_file} -> {out_dir}")
                continue
            if existing and args.overwrite:
                for wav_file in existing:
                    wav_file.unlink()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{len(src_files)}] streaming: {src_file} -> {out_dir}")
        run_streaming_file(
            src_file=src_file,
            out_dir=out_dir,
            wrapper=wrapper,
            stem_names=stem_names,
            device=args.device,
            chunk_frames=args.chunk_frames,
            subtype=args.subtype,
            use_autocast=args.cuda_autocast,
        )


if __name__ == "__main__":
    main()
