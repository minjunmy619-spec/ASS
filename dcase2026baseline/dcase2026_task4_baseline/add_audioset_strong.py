#!/usr/bin/env python3
"""
Extract isolated AudioSet-Strong events into a DCASE2026 Task4-compatible
folder layout.

The script intentionally has a two-stage workflow:
  1. Parse annotations and print/write candidate statistics.
  2. Extract audio only when --execute is provided.

Output layout, where --output_dir is a DCASE dev_set-compatible root:
  <output_dir>/sound_event/{train,valid}/<DCASE2026_LABEL>/*.wav
  <output_dir>/interference/{train,valid}/<AudioSet_LABEL>/*.wav
  <output_dir>/noise/{train,valid}/<background_label>/*.wav
  <output_dir>/metadata/audioset_strong/*.jsonl|*.json

Example dry run:
  python add_audioset_strong.py \
      --input_dir /data/audioset_waves \
      --annotations_dir audioset_strong_annotations \
      --output_dir data/dev_set_audioset

Example extraction:
  python add_audioset_strong.py \
      --input_dir /data/audioset_waves \
      --annotations_dir audioset_strong_annotations \
      --output_dir data/dev_set_audioset \
      --execute
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import soundfile as sf
except Exception as exc:  # pragma: no cover - dependency failure is reported at runtime
    sf = None
    _SOUNDFILE_IMPORT_ERROR = exc
else:
    _SOUNDFILE_IMPORT_ERROR = None

try:
    import librosa
except Exception as exc:  # pragma: no cover
    librosa = None
    _LIBROSA_IMPORT_ERROR = exc
else:
    _LIBROSA_IMPORT_ERROR = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **_: object):
        return x


DCASE2026_LABELS = [
    "AlarmClock",
    "BicycleBell",
    "Blender",
    "Buzzer",
    "Clapping",
    "Cough",
    "CupboardOpenClose",
    "Dishes",
    "Doorbell",
    "FootSteps",
    "HairDryer",
    "MechanicalFans",
    "MusicalKeyboard",
    "Percussion",
    "Pour",
    "Speech",
    "Typing",
    "VacuumCleaner",
]

# Display-name aliases. MIDs are resolved through class_labels_indices.csv when available.
# The mapping is deliberately conservative. Broad AudioSet classes such as "Music" are not
# mapped to a DCASE target class.
AUDIOSET_DISPLAY_TO_DCASE: Dict[str, str] = {
    "alarm clock": "AlarmClock",
    "alarm": "AlarmClock",
    "bicycle bell": "BicycleBell",
    "blender": "Blender",
    "buzzer": "Buzzer",
    "buzz": "Buzzer",
    "beep, bleep": "Buzzer",
    "clapping": "Clapping",
    "applause": "Clapping",
    "cough": "Cough",
    "cupboard open or close": "CupboardOpenClose",
    "drawer open or close": "CupboardOpenClose",
    "doorbell": "Doorbell",
    "ding-dong": "Doorbell",
    "walk, footsteps": "FootSteps",
    "footsteps": "FootSteps",
    "walking": "FootSteps",
    "hair dryer": "HairDryer",
    "fan": "MechanicalFans",
    "mechanical fan": "MechanicalFans",
    "air conditioning": "MechanicalFans",
    "keyboard (musical)": "MusicalKeyboard",
    "electric piano": "MusicalKeyboard",
    "piano": "MusicalKeyboard",
    "percussion": "Percussion",
    "drum": "Percussion",
    "drum kit": "Percussion",
    "drum roll": "Percussion",
    "snare drum": "Percussion",
    "bass drum": "Percussion",
    "tabla": "Percussion",
    "cymbal": "Percussion",
    "hi-hat": "Percussion",
    "pour": "Pour",
    "speech": "Speech",
    "male speech, man speaking": "Speech",
    "female speech, woman speaking": "Speech",
    "child speech, kid speaking": "Speech",
    "conversation": "Speech",
    "narration, monologue": "Speech",
    "typing": "Typing",
    "computer keyboard": "Typing",
    "typewriter": "Typing",
    "dishes, pots, and pans": "Dishes",
    "cutlery, silverware": "Dishes",
    "clink": "Dishes",
    "chink, clink": "Dishes",
    "vacuum cleaner": "VacuumCleaner",
}

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac")


@dataclass(frozen=True)
class StrongEvent:
    uid: str
    segment_id: str
    ytid: str
    start: float
    end: float
    raw_label: str
    display_label: str
    dcase_label: Optional[str]
    split: str
    annotation_file: str
    line_no: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class Candidate:
    uid: str
    kind: str  # sound_event, interference, noise
    split: str
    label: str
    segment_id: str
    start: float
    end: float
    duration: float
    annotation_file: str
    raw_label: str = ""
    display_label: str = ""
    reject_reason: str = ""


@dataclass
class ExtractResult:
    uid: str
    kind: str
    split: str
    label: str
    source_path: str
    output_path: str
    start: float
    end: float
    duration: float
    status: str
    reason: str = ""
    source_sr: Optional[int] = None
    target_sr: Optional[int] = None
    peak: Optional[float] = None
    rms: Optional[float] = None


def norm_key(text: object) -> str:
    s = str(text or "").strip().lower()
    s = s.replace("#", "")
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s.strip("_")


def norm_label(text: object) -> str:
    s = str(text or "").strip()
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def display_label_key(text: object) -> str:
    return norm_label(text).lower()


def safe_name(text: object, max_len: int = 96) -> str:
    s = str(text or "unknown").strip()
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._+\-=]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    if not s:
        s = "unknown"
    return s[:max_len]


def parse_float(value: object) -> Optional[float]:
    try:
        v = float(str(value).strip())
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def choose_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        sample = "".join([f.readline() for _ in range(20)])
    if sample.count("\t") >= sample.count(","):
        return "\t"
    return ","


def read_table(path: Path) -> Iterator[Tuple[int, Dict[str, str]]]:
    """Yield (line_number, row_dict) from a CSV/TSV file.

    Handles the common AudioSet formats:
      - class_labels_indices.csv: index,mid,display_name
      - strong TSV: segment_id,start_time_seconds,end_time_seconds,label
      - headerless 4-col rows: segment_id,start,end,label
    """
    delimiter = choose_delimiter(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        first_row: Optional[List[str]] = None
        first_line_no = 0
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            first_row = row
            first_line_no = line_no
            break
        if first_row is None:
            return

        normalized_first = [norm_key(c) for c in first_row]
        header_tokens = {
            "segment_id",
            "ytid",
            "youtube_id",
            "video_id",
            "start_time_seconds",
            "end_time_seconds",
            "start_seconds",
            "end_seconds",
            "onset",
            "offset",
            "label",
            "labels",
            "mid",
            "display_name",
            "positive_labels",
        }
        has_header = bool(set(normalized_first) & header_tokens)

        if has_header:
            header = normalized_first
        else:
            # Common strong-label no-header fallback.
            if len(first_row) >= 4:
                header = ["segment_id", "start_time_seconds", "end_time_seconds", "label"]
                if len(first_row) >= 5:
                    header.append("display_name")
            else:
                return
            row_dict = {header[i]: first_row[i].strip() for i in range(min(len(header), len(first_row)))}
            yield first_line_no, row_dict

        for line_no, row in enumerate(reader, start=first_line_no + 1):
            if not row or all(not cell.strip() for cell in row):
                continue
            # Skip full-line comments but keep '# YTID' headers handled above.
            if row[0].lstrip().startswith("#") and not any(cell.strip() for cell in row[1:]):
                continue
            row_dict = {header[i]: row[i].strip() for i in range(min(len(header), len(row)))}
            yield line_no, row_dict


def discover_annotation_files(annotations_dir: Path) -> List[Path]:
    return sorted(
        p for p in annotations_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt"}
    )


def collect_mid_to_display_name(files: Sequence[Path]) -> Dict[str, str]:
    mid_to_name: Dict[str, str] = {}
    for path in files:
        try:
            rows = list(read_table(path))
        except Exception:
            continue
        for _, row in rows[:5]:
            if "mid" in row and "display_name" in row:
                break
        else:
            continue
        for _, row in rows:
            mid = row.get("mid", "").strip()
            display = row.get("display_name", "").strip()
            if mid and display:
                mid_to_name[mid] = display
    return mid_to_name


def is_mapping_file(path: Path, rows_preview: Sequence[Tuple[int, Dict[str, str]]]) -> bool:
    if "class_labels_indices" in path.name.lower():
        return True
    return any("mid" in row and "display_name" in row and "start_time_seconds" not in row for _, row in rows_preview)


def find_first(row: Dict[str, str], keys: Sequence[str]) -> str:
    for key in keys:
        if key in row and str(row[key]).strip() != "":
            return row[key]
    return ""


def infer_split(path: Path, train_keywords: Sequence[str], valid_keywords: Sequence[str]) -> str:
    low = "/".join(part.lower() for part in path.parts)
    if any(k and k.lower() in low for k in valid_keywords):
        return "valid"
    if any(k and k.lower() in low for k in train_keywords):
        return "train"
    return "train"


def split_label_values(raw: str, column_name: str) -> List[str]:
    raw = str(raw or "").strip().strip('"')
    if not raw:
        return []
    col = norm_key(column_name)
    if col in {"positive_labels", "labels", "mids"}:
        return [x.strip().strip('"') for x in raw.split(",") if x.strip()]
    # MIDs can be comma-separated in a generic label column.
    if raw.startswith("/m/") and "," in raw:
        return [x.strip().strip('"') for x in raw.split(",") if x.strip()]
    return [raw]


def parse_segment_offsets(segment_id: str) -> Tuple[str, Optional[float], Optional[float]]:
    """Return (ytid_like, clip_start, clip_end) from common AudioSet segment IDs.

    Examples:
      abcdefghijk_30.000_40.000 -> (abcdefghijk, 30.0, 40.0)
      abcdefghijk-000030-000040 -> (abcdefghijk, 30.0, 40.0)
    """
    stem = Path(str(segment_id)).stem.strip()
    m = re.match(r"^(.+?)[_\-]([0-9]+(?:\.[0-9]+)?)[_\-]([0-9]+(?:\.[0-9]+)?)$", stem)
    if m:
        return m.group(1), float(m.group(2)), float(m.group(3))
    if len(stem) >= 11:
        return stem[:11], None, None
    return stem, None, None


def audio_key_candidates(identifier: str) -> List[str]:
    stem = Path(str(identifier)).stem.strip()
    candidates: List[str] = []

    def add(x: str) -> None:
        x = x.strip()
        if x and x not in candidates:
            candidates.append(x)

    add(stem)
    ytid, clip_start, clip_end = parse_segment_offsets(stem)
    add(ytid)
    if clip_start is not None and clip_end is not None:
        add(f"{ytid}_{clip_start:g}_{clip_end:g}")
        add(f"{ytid}_{int(round(clip_start)):06d}_{int(round(clip_end)):06d}")
        add(f"{ytid}-{int(round(clip_start)):06d}-{int(round(clip_end)):06d}")
    if len(stem) >= 11:
        add(stem[:11])
    return candidates


def resolve_dcase_label(display_or_raw: str) -> Optional[str]:
    label = norm_label(display_or_raw)
    if label in DCASE2026_LABELS:
        return label
    compact = re.sub(r"[^a-z0-9]", "", label.lower())
    for dcase in DCASE2026_LABELS:
        if re.sub(r"[^a-z0-9]", "", dcase.lower()) == compact:
            return dcase
    return AUDIOSET_DISPLAY_TO_DCASE.get(display_label_key(label))


def parse_strong_annotations(
    annotations_dir: Path,
    train_keywords: Sequence[str],
    valid_keywords: Sequence[str],
) -> Tuple[List[StrongEvent], Dict[str, str], Counter]:
    files = discover_annotation_files(annotations_dir)
    mid_to_name = collect_mid_to_display_name(files)
    parse_status = Counter()
    events: List[StrongEvent] = []

    for path in files:
        try:
            rows_all = list(read_table(path))
        except Exception as exc:
            parse_status[f"skip_read_error:{path.name}:{type(exc).__name__}"] += 1
            continue
        if not rows_all:
            parse_status[f"skip_empty:{path.name}"] += 1
            continue
        if is_mapping_file(path, rows_all[:5]):
            parse_status[f"skip_mapping:{path.name}"] += 1
            continue

        split = infer_split(path, train_keywords=train_keywords, valid_keywords=valid_keywords)
        file_events = 0
        for line_no, row in rows_all:
            segment_id = find_first(
                row,
                [
                    "segment_id",
                    "ytid",
                    "youtube_id",
                    "video_id",
                    "file_name",
                    "filename",
                    "audio_id",
                    "wav",
                ],
            )
            start = parse_float(find_first(row, ["start_time_seconds", "start_seconds", "onset_seconds", "onset", "start"]))
            end = parse_float(find_first(row, ["end_time_seconds", "end_seconds", "offset_seconds", "offset", "end"]))

            label_col = ""
            raw_label = ""
            for candidate_col in ["label", "event_label", "class_label", "mid", "positive_labels", "labels", "display_name"]:
                if candidate_col in row and row[candidate_col].strip():
                    label_col = candidate_col
                    raw_label = row[candidate_col].strip()
                    break

            if not segment_id or start is None or end is None or not raw_label:
                parse_status[f"skip_bad_row:{path.name}"] += 1
                continue

            if end <= start:
                parse_status[f"skip_nonpositive_duration:{path.name}"] += 1
                continue

            for label_value in split_label_values(raw_label, label_col):
                display = mid_to_name.get(label_value, label_value)
                dcase_label = resolve_dcase_label(display)
                ytid, _, _ = parse_segment_offsets(segment_id)
                uid_base = f"{path.relative_to(annotations_dir)}:{line_no}:{segment_id}:{start:.6f}:{end:.6f}:{label_value}"
                uid = hashlib.sha1(uid_base.encode("utf-8")).hexdigest()[:16]
                events.append(
                    StrongEvent(
                        uid=uid,
                        segment_id=Path(str(segment_id)).stem.strip(),
                        ytid=ytid,
                        start=float(start),
                        end=float(end),
                        raw_label=label_value,
                        display_label=display,
                        dcase_label=dcase_label,
                        split=split,
                        annotation_file=str(path.relative_to(annotations_dir)),
                        line_no=line_no,
                    )
                )
                file_events += 1
        parse_status[f"parsed:{path.name}"] += file_events
    return events, mid_to_name, parse_status


def intervals_overlap(a_start: float, a_end: float, b_start: float, b_end: float, margin: float) -> bool:
    return max(a_start - margin, b_start) < min(a_end + margin, b_end)


def build_candidates(
    events: Sequence[StrongEvent],
    min_event_duration: float,
    max_event_duration: float,
    min_noise_duration: float,
    default_clip_duration: float,
    overlap_margin: float,
    keep_unmapped_interference: bool,
    background_label: str,
    max_background_per_clip: int,
) -> Tuple[List[Candidate], List[Candidate]]:
    by_segment: Dict[Tuple[str, str], List[StrongEvent]] = defaultdict(list)
    for event in events:
        by_segment[(event.split, event.segment_id)].append(event)

    accepted: List[Candidate] = []
    rejected: List[Candidate] = []

    for (split, segment_id), seg_events in by_segment.items():
        seg_events = sorted(seg_events, key=lambda e: (e.start, e.end, e.display_label))
        for idx, event in enumerate(seg_events):
            reason = ""
            if event.duration < min_event_duration:
                reason = "too_short"
            elif event.duration > max_event_duration:
                reason = "too_long"
            else:
                for j, other in enumerate(seg_events):
                    if j == idx:
                        continue
                    if intervals_overlap(event.start, event.end, other.start, other.end, overlap_margin):
                        reason = "overlap_with_other_annotated_event"
                        break

            if reason:
                rejected.append(
                    Candidate(
                        uid=event.uid,
                        kind="reject",
                        split=split,
                        label=event.dcase_label or event.display_label,
                        segment_id=segment_id,
                        start=event.start,
                        end=event.end,
                        duration=event.duration,
                        annotation_file=event.annotation_file,
                        raw_label=event.raw_label,
                        display_label=event.display_label,
                        reject_reason=reason,
                    )
                )
                continue

            if event.dcase_label is not None:
                kind = "sound_event"
                label = event.dcase_label
            elif keep_unmapped_interference:
                kind = "interference"
                label = safe_name(event.display_label)
            else:
                rejected.append(
                    Candidate(
                        uid=event.uid,
                        kind="reject",
                        split=split,
                        label=event.display_label,
                        segment_id=segment_id,
                        start=event.start,
                        end=event.end,
                        duration=event.duration,
                        annotation_file=event.annotation_file,
                        raw_label=event.raw_label,
                        display_label=event.display_label,
                        reject_reason="unmapped_label",
                    )
                )
                continue

            accepted.append(
                Candidate(
                    uid=event.uid,
                    kind=kind,
                    split=split,
                    label=label,
                    segment_id=segment_id,
                    start=event.start,
                    end=event.end,
                    duration=event.duration,
                    annotation_file=event.annotation_file,
                    raw_label=event.raw_label,
                    display_label=event.display_label,
                )
            )

        # Background/noise candidates from annotation-free gaps.
        _, clip_start, clip_end = parse_segment_offsets(segment_id)
        clip_duration = default_clip_duration
        if clip_start is not None and clip_end is not None and clip_end > clip_start:
            clip_duration = clip_end - clip_start
        if seg_events:
            clip_duration = max(clip_duration, max(e.end for e in seg_events))

        merged: List[Tuple[float, float]] = []
        for e in seg_events:
            s = max(0.0, e.start - overlap_margin)
            t = min(clip_duration, e.end + overlap_margin)
            if not merged or s > merged[-1][1]:
                merged.append((s, t))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], t))

        gaps: List[Tuple[float, float]] = []
        cursor = 0.0
        for s, t in merged:
            if s - cursor >= min_noise_duration:
                gaps.append((cursor, s))
            cursor = max(cursor, t)
        if clip_duration - cursor >= min_noise_duration:
            gaps.append((cursor, clip_duration))

        # Use longest gaps first to avoid flooding noise folders.
        gaps = sorted(gaps, key=lambda x: x[1] - x[0], reverse=True)[:max_background_per_clip]
        for gap_idx, (s, t) in enumerate(gaps):
            uid_base = f"noise:{split}:{segment_id}:{s:.6f}:{t:.6f}:{gap_idx}"
            uid = hashlib.sha1(uid_base.encode("utf-8")).hexdigest()[:16]
            accepted.append(
                Candidate(
                    uid=uid,
                    kind="noise",
                    split=split,
                    label=background_label,
                    segment_id=segment_id,
                    start=s,
                    end=t,
                    duration=t - s,
                    annotation_file="<derived_from_strong_gaps>",
                    raw_label="",
                    display_label=background_label,
                )
            )

    return accepted, rejected


def summarize_candidates(
    events: Sequence[StrongEvent],
    accepted: Sequence[Candidate],
    rejected: Sequence[Candidate],
    parse_status: Counter,
    top_k: int,
) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    summary["parse_status"] = dict(parse_status)
    summary["n_annotation_events"] = len(events)
    summary["n_candidates"] = len(accepted)
    summary["n_rejected"] = len(rejected)
    summary["accepted_by_kind"] = dict(Counter(c.kind for c in accepted))
    summary["accepted_by_split"] = dict(Counter(c.split for c in accepted))
    summary["rejected_by_reason"] = dict(Counter(c.reject_reason for c in rejected))

    by_kind_label: Dict[str, Counter] = defaultdict(Counter)
    duration_by_kind = defaultdict(float)
    duration_by_kind_label = defaultdict(float)
    for c in accepted:
        by_kind_label[c.kind][c.label] += 1
        duration_by_kind[c.kind] += c.duration
        duration_by_kind_label[(c.kind, c.label)] += c.duration

    summary["duration_hours_by_kind"] = {k: round(v / 3600.0, 4) for k, v in sorted(duration_by_kind.items())}
    summary["labels"] = {}
    for kind, counter in sorted(by_kind_label.items()):
        items = []
        limit = top_k if kind == "interference" else None
        for label, count in counter.most_common(limit):
            items.append(
                {
                    "label": label,
                    "count": count,
                    "duration_hours": round(duration_by_kind_label[(kind, label)] / 3600.0, 4),
                }
            )
        summary["labels"][kind] = items
    return summary


def print_summary(summary: Dict[str, object]) -> None:
    print("\n================ AudioSet-Strong annotation statistics ================")
    print(f"Annotation events parsed : {summary['n_annotation_events']}")
    print(f"Accepted candidates      : {summary['n_candidates']}")
    print(f"Rejected annotations     : {summary['n_rejected']}")
    print(f"Accepted by kind         : {summary['accepted_by_kind']}")
    print(f"Accepted by split        : {summary['accepted_by_split']}")
    print(f"Duration hours by kind   : {summary['duration_hours_by_kind']}")
    print(f"Rejected by reason       : {summary['rejected_by_reason']}")
    print("\nTarget/interference/noise label summary:")
    labels_summary = summary.get("labels", {})
    if isinstance(labels_summary, dict):
        for kind, rows in labels_summary.items():
            print(f"  [{kind}]")
            if isinstance(rows, list):
                for row in rows:
                    print(f"    {row['label']}: count={row['count']}, hours={row['duration_hours']}")
    print("======================================================================\n")


def write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                obj = asdict(row)
            else:
                obj = row
            f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary_files(output_dir: Path, summary: Dict[str, object], accepted: Sequence[Candidate], rejected: Sequence[Candidate]) -> None:
    meta_dir = output_dir / "metadata" / "audioset_strong"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    write_jsonl(meta_dir / "accepted_candidates.jsonl", accepted)
    write_jsonl(meta_dir / "rejected_annotations.jsonl", rejected)


def build_audio_index(input_dir: Path, extensions: Sequence[str]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    duplicates: Counter = Counter()
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    for path in tqdm(files, desc="Index raw audio", unit="file"):
        for key in audio_key_candidates(path.stem):
            if key in index and index[key] != path:
                duplicates[key] += 1
                # Keep the first path deterministically.
                continue
            index[key] = path
    if duplicates:
        print(f"Warning: {len(duplicates)} duplicate audio keys found; keeping first path per key.")
    print(f"Indexed {len(files)} raw audio files with {len(index)} lookup keys.")
    return index


def resolve_audio(candidate: Candidate, audio_index: Dict[str, Path]) -> Tuple[Optional[Path], float]:
    # offset is normally zero because AudioSet-Strong timestamps are relative to each
    # 10 s AudioSet segment. Keep the return value explicit for future extension.
    for key in audio_key_candidates(candidate.segment_id):
        if key in audio_index:
            return audio_index[key], 0.0
    return None, 0.0


def load_audio_segment(path: Path, start: float, end: float, mono: bool) -> Tuple[np.ndarray, int]:
    if sf is None:
        raise RuntimeError(f"soundfile import failed: {_SOUNDFILE_IMPORT_ERROR}")
    if end <= start:
        raise ValueError(f"invalid segment: start={start}, end={end}")

    try:
        with sf.SoundFile(str(path), "r") as f:
            sr = int(f.samplerate)
            start_frame = max(0, int(round(start * sr)))
            n_frames = max(1, int(round((end - start) * sr)))
            f.seek(min(start_frame, len(f)))
            audio = f.read(frames=n_frames, dtype="float32", always_2d=True)
        # [frames, channels] -> [channels, frames]
        audio = audio.T
    except Exception:
        if librosa is None:
            raise RuntimeError(f"librosa import failed and soundfile could not read {path}: {_LIBROSA_IMPORT_ERROR}")
        audio, sr = librosa.load(str(path), sr=None, mono=False, offset=float(start), duration=float(end - start))
        if audio.ndim == 1:
            audio = audio[None, :]
        audio = audio.astype(np.float32, copy=False)

    if mono and audio.ndim == 2 and audio.shape[0] > 1:
        audio = np.mean(audio, axis=0, keepdims=True).astype(np.float32)
    elif audio.ndim == 1:
        audio = audio[None, :].astype(np.float32)
    return audio, int(sr)


def trim_edge_low_amplitude(audio: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0 or audio.size == 0:
        return audio
    mono_abs = np.max(np.abs(audio), axis=0)
    idx = np.flatnonzero(mono_abs >= threshold)
    if idx.size == 0:
        return audio[:, :0]
    return audio[:, idx[0]: idx[-1] + 1]


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    if librosa is None:
        raise RuntimeError(f"librosa is required for resampling: {_LIBROSA_IMPORT_ERROR}")
    channels = []
    for ch in audio:
        channels.append(librosa.resample(ch.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr))
    return np.stack(channels, axis=0).astype(np.float32)


def save_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    if sf is None:
        raise RuntimeError(f"soundfile import failed: {_SOUNDFILE_IMPORT_ERROR}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # DCASE additional dry source scripts expect mono files; still support multi-channel if requested.
    if audio.ndim == 2 and audio.shape[0] == 1:
        out = audio[0]
    else:
        out = audio.T
    sf.write(str(path), out, sr)


def candidate_output_path(output_dir: Path, candidate: Candidate, suffix: str = ".wav") -> Path:
    label_dir = safe_name(candidate.label)
    filename = (
        f"{safe_name(candidate.segment_id, max_len=80)}__"
        f"{int(round(candidate.start * 1000)):08d}_{int(round(candidate.end * 1000)):08d}__"
        f"{safe_name(candidate.display_label or candidate.label, max_len=48)}__{candidate.uid}{suffix}"
    )
    return output_dir / candidate.kind / candidate.split / label_dir / filename


def extract_candidates(
    candidates: Sequence[Candidate],
    input_dir: Path,
    output_dir: Path,
    target_sr: int,
    amp_threshold: float,
    min_rms: float,
    trim_edges: bool,
    mono: bool,
    overwrite: bool,
    extensions: Sequence[str],
) -> List[ExtractResult]:
    audio_index = build_audio_index(input_dir, extensions=extensions)
    results: List[ExtractResult] = []

    for c in tqdm(candidates, desc="Extract candidates", unit="seg"):
        src_path, offset = resolve_audio(c, audio_index)
        out_path = candidate_output_path(output_dir, c)
        if src_path is None:
            results.append(
                ExtractResult(
                    uid=c.uid,
                    kind=c.kind,
                    split=c.split,
                    label=c.label,
                    source_path="",
                    output_path=str(out_path),
                    start=c.start,
                    end=c.end,
                    duration=c.duration,
                    status="skip",
                    reason="missing_raw_audio",
                    target_sr=target_sr,
                )
            )
            continue
        if out_path.exists() and not overwrite:
            results.append(
                ExtractResult(
                    uid=c.uid,
                    kind=c.kind,
                    split=c.split,
                    label=c.label,
                    source_path=str(src_path),
                    output_path=str(out_path),
                    start=c.start,
                    end=c.end,
                    duration=c.duration,
                    status="skip",
                    reason="output_exists",
                    target_sr=target_sr,
                )
            )
            continue

        try:
            audio, sr = load_audio_segment(src_path, c.start + offset, c.end + offset, mono=mono)
            if trim_edges:
                audio = trim_edge_low_amplitude(audio, amp_threshold)
            if audio.size == 0:
                raise ValueError("empty_after_trim")
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(np.square(audio))))
            if peak < amp_threshold:
                raise ValueError(f"low_peak:{peak:.6g}")
            if rms < min_rms:
                raise ValueError(f"low_rms:{rms:.6g}")
            audio = resample_audio(audio, orig_sr=sr, target_sr=target_sr)
            save_audio(out_path, audio, target_sr)
            status = "saved"
            reason = ""
        except Exception as exc:
            peak = None
            rms = None
            sr = None  # type: ignore[assignment]
            status = "skip"
            reason = f"{type(exc).__name__}:{exc}"

        results.append(
            ExtractResult(
                uid=c.uid,
                kind=c.kind,
                split=c.split,
                label=c.label,
                source_path=str(src_path),
                output_path=str(out_path),
                start=c.start,
                end=c.end,
                duration=c.duration,
                status=status,
                reason=reason,
                source_sr=sr if isinstance(sr, int) else None,
                target_sr=target_sr,
                peak=peak,
                rms=rms,
            )
        )
    return results


def print_extraction_summary(results: Sequence[ExtractResult]) -> None:
    print("\n================ Audio extraction summary ================")
    print(f"Segments processed : {len(results)}")
    print(f"By status          : {dict(Counter(r.status for r in results))}")
    print(f"Skip reasons       : {dict(Counter(r.reason for r in results if r.status != 'saved'))}")
    saved_by_kind = Counter(r.kind for r in results if r.status == "saved")
    print(f"Saved by kind      : {dict(saved_by_kind)}")
    print("==========================================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract non-overlapping AudioSet-Strong clips into DCASE2026 Task4-compatible folders."
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Root folder containing raw AudioSet audio. Searched recursively.")
    parser.add_argument("--annotations_dir", type=Path, required=True, help="Folder containing AudioSet strong CSV/TSV annotations.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output dev_set-compatible root folder.")
    parser.add_argument("--execute", action="store_true", help="Actually extract audio. Without this flag, only statistics/manifests are produced.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already extracted WAV files.")

    parser.add_argument("--target_sr", type=int, default=32000, help="Output sample rate.")
    parser.add_argument("--min_event_duration", type=float, default=0.15, help="Minimum target/interference event duration in seconds.")
    parser.add_argument("--max_event_duration", type=float, default=10.0, help="Maximum target/interference event duration in seconds.")
    parser.add_argument("--min_noise_duration", type=float, default=1.0, help="Minimum derived background/noise gap duration in seconds.")
    parser.add_argument("--default_clip_duration", type=float, default=10.0, help="AudioSet clip duration used when segment IDs do not encode duration.")
    parser.add_argument("--overlap_margin", type=float, default=0.03, help="Safety margin around annotations when checking overlap and background gaps.")
    parser.add_argument("--max_background_per_clip", type=int, default=2, help="Maximum annotation-free gaps exported per AudioSet segment.")
    parser.add_argument("--background_label", type=str, default="AudioSetBackground", help="Folder label for annotation-free background/noise gaps.")

    parser.add_argument("--keep_unmapped_interference", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep isolated non-DCASE AudioSet labels as interference examples.")
    parser.add_argument("--mono", action=argparse.BooleanOptionalAction, default=True, help="Convert extracted audio to mono.")
    parser.add_argument("--trim_edges", action=argparse.BooleanOptionalAction, default=True, help="Trim low-amplitude leading/trailing samples after crop.")
    parser.add_argument("--amp_threshold", type=float, default=0.005, help="Peak threshold for extraction-time silence rejection/trimming.")
    parser.add_argument("--min_rms", type=float, default=1e-4, help="RMS threshold for extraction-time silence rejection.")
    parser.add_argument("--audio_extensions", nargs="+", default=list(AUDIO_EXTENSIONS), help="Audio file extensions to index.")

    parser.add_argument("--train_keywords", nargs="+", default=["train", "balanced", "unbalanced"],
                        help="Path/name keywords used to infer train split from annotation files.")
    parser.add_argument("--valid_keywords", nargs="+", default=["valid", "validation", "eval", "evaluate", "test"],
                        help="Path/name keywords used to infer valid split from annotation files.")
    parser.add_argument("--top_k_interference_stats", type=int, default=40, help="Number of interference labels to print/write in summary.")
    parser.add_argument("--no_write_manifests", action="store_true", help="Do not write summary/candidate JSONL files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.annotations_dir.is_dir():
        raise FileNotFoundError(f"annotations_dir does not exist or is not a folder: {args.annotations_dir}")
    if args.execute and not args.input_dir.is_dir():
        raise FileNotFoundError(f"input_dir does not exist or is not a folder: {args.input_dir}")

    print("====================================================")
    print("AudioSet-Strong -> DCASE2026 Task4 extraction")
    print(f"input_dir       : {args.input_dir}")
    print(f"annotations_dir : {args.annotations_dir}")
    print(f"output_dir      : {args.output_dir}")
    print(f"execute         : {args.execute}")
    print(f"target_sr       : {args.target_sr}")
    print(f"overlap_margin  : {args.overlap_margin}")
    print("====================================================")

    events, mid_to_display, parse_status = parse_strong_annotations(
        args.annotations_dir,
        train_keywords=args.train_keywords,
        valid_keywords=args.valid_keywords,
    )
    print(f"Resolved {len(mid_to_display)} AudioSet MID -> display-name mappings.")

    accepted, rejected = build_candidates(
        events,
        min_event_duration=args.min_event_duration,
        max_event_duration=args.max_event_duration,
        min_noise_duration=args.min_noise_duration,
        default_clip_duration=args.default_clip_duration,
        overlap_margin=args.overlap_margin,
        keep_unmapped_interference=args.keep_unmapped_interference,
        background_label=args.background_label,
        max_background_per_clip=args.max_background_per_clip,
    )

    summary = summarize_candidates(
        events,
        accepted,
        rejected,
        parse_status=parse_status,
        top_k=args.top_k_interference_stats,
    )
    print_summary(summary)

    if not args.no_write_manifests:
        write_summary_files(args.output_dir, summary, accepted, rejected)
        print(f"Wrote annotation statistics/manifests to: {args.output_dir / 'metadata' / 'audioset_strong'}")

    if not args.execute:
        print("Dry run only. Re-run with --execute to extract audio after reviewing the statistics.")
        return 0

    results = extract_candidates(
        accepted,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_sr=args.target_sr,
        amp_threshold=args.amp_threshold,
        min_rms=args.min_rms,
        trim_edges=args.trim_edges,
        mono=args.mono,
        overwrite=args.overwrite,
        extensions=args.audio_extensions,
    )
    print_extraction_summary(results)

    if not args.no_write_manifests:
        meta_dir = args.output_dir / "metadata" / "audioset_strong"
        write_jsonl(meta_dir / "extraction_results.jsonl", results)
        extraction_summary = {
            "processed": len(results),
            "by_status": dict(Counter(r.status for r in results)),
            "skip_reasons": dict(Counter(r.reason for r in results if r.status != "saved")),
            "saved_by_kind": dict(Counter(r.kind for r in results if r.status == "saved")),
        }
        (meta_dir / "extraction_summary.json").write_text(
            json.dumps(extraction_summary, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Wrote extraction manifest to: {meta_dir / 'extraction_results.jsonl'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
