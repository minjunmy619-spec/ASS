#!/usr/bin/env python3
"""
Extract isolated AudioSet-Strong events into a DCASE2026 Task4-compatible
folder layout.

The script is intentionally conservative:
  1. It first parses annotations and writes/prints candidate statistics.
  2. It extracts WAV files only when --execute is provided.
  3. Target/interference folders are restricted to the exact DCASE2026 Task4
     folder names used by the baseline.
  4. Output WAVs are always written at 32 kHz.

Output layout, where --output_dir is the DCASE dev_set root:
  <output_dir>/sound_event/{train,valid}/<DCASE2026_LABEL>/*.wav
  <output_dir>/interference/{train,valid}/<DCASE_INTERFERENCE_LABEL>/*.wav
  <output_dir>/noise/{train,valid}/*.wav
  <output_dir>/metadata/audioset_strong/*.jsonl|*.json

Example dry run:
  python add_audioset_strong.py \
      --input_dir /data/audioset_waves \
      --annotations_dir audioset_strong_annotations \
      --output_dir data/dev_set

Example extraction:
  python add_audioset_strong.py \
      --input_dir /data/audioset_waves \
      --annotations_dir audioset_strong_annotations \
      --output_dir data/dev_set \
      --execute
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import soundfile as sf
except Exception as exc:  # pragma: no cover
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


TARGET_SR = 32000

DCASE2026_LABELS = [
    "AlarmClock", "BicycleBell", "Blender", "Buzzer",
    "Clapping", "Cough", "CupboardOpenClose", "Dishes",
    "Doorbell", "FootSteps", "HairDryer", "MechanicalFans",
    "MusicalKeyboard", "Percussion", "Pour", "Speech",
    "Typing", "VacuumCleaner",
]

# Must match the baseline interference folders exactly. Do not sanitize these
# names when creating directories.
DCASE_INTERFERENCE_LABELS = [
    "Air conditioning", "Aircraft", "Bird flight, flapping wings", "Bleat", "Boiling", "Boom",
    "Burping, eructation", "Burst, pop", "Bus", "Camera", "Car passing by",
    "Cattle, bovinae", "Chainsaw", "Chewing, mastication", "Chink, clink", "Clip-clop",
    "Cluck", "Clunk", "Coin (dropping)", "Crack", "Crackle", "Creak", "Croak", "Crow",
    "Crumpling, crinkling", "Crushing", "Drill", "Drip", "Electric toothbrush", "Engine",
    "Fart", "Finger snapping", "Fire", "Fire alarm", "Firecracker", "Fireworks",
    "Fixed-wing aircraft, airplane", "Frog", "Gears", "Growling", "Gurgling", "Helicopter",
    "Hiss", "Hoot", "Howl", "Howl (wind)", "Jackhammer", "Keys jangling", "Lawn mower",
    "Light engine (high frequency)", "Microwave oven", "Moo", "Oink",
    "Packing tape, duct tape", "Pig", "Printer", "Purr", "Rain", "Rain on surface",
    "Raindrop", "Ratchet, pawl", "Rattle", "Sanding", "Sawing", "Scissors", "Screech",
    "Sheep", "Ship", "Shuffling cards", "Skateboard", "Slam", "Sliding door", "Sneeze",
    "Sniff", "Snoring", "Splinter", "Squeak", "Stream", "Subway, metro, underground",
    "Tap", "Tearing", "Thump, thud", "Tick", "Tick-tock", "Toothbrush",
    "Traffic noise, roadway noise", "Train", "Train horn", "Velcro, hook and loop fastener",
    "Waterfall", "Whoosh, swoosh, swish", "Wind", "Writing", "Zipper (clothing)",
]

# Conservative target aliases. Exact DCASE target labels are also accepted.
# Labels that are exact baseline interference labels are routed to interference
# before this table is consulted.
AUDIOSET_DISPLAY_TO_DCASE: Dict[str, str] = {
    "alarm clock": "AlarmClock",
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
    interference_label: Optional[str]
    split: str
    annotation_file: str
    line_no: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class Candidate:
    uid: str
    kind: str  # sound_event, interference, noise, reject
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
    target_sr: int = TARGET_SR
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


def safe_filename_part(text: object, max_len: int = 96) -> str:
    s = str(text or "unknown").strip()
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^A-Za-z0-9._+\-=]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    return (s or "unknown")[:max_len]


def parse_float(value: object) -> Optional[float]:
    try:
        v = float(str(value).strip())
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


INTERFERENCE_BY_KEY = {display_label_key(x): x for x in DCASE_INTERFERENCE_LABELS}


def resolve_interference_label(display_or_raw: str) -> Optional[str]:
    return INTERFERENCE_BY_KEY.get(display_label_key(display_or_raw))


def resolve_dcase_label(display_or_raw: str) -> Optional[str]:
    label = norm_label(display_or_raw)
    if label in DCASE2026_LABELS:
        return label
    compact = re.sub(r"[^a-z0-9]", "", label.lower())
    for dcase in DCASE2026_LABELS:
        if re.sub(r"[^a-z0-9]", "", dcase.lower()) == compact:
            return dcase
    return AUDIOSET_DISPLAY_TO_DCASE.get(display_label_key(label))


def choose_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        sample = "".join([f.readline() for _ in range(20)])
    return "\t" if sample.count("\t") >= sample.count(",") else ","


def read_table(path: Path) -> Iterator[Tuple[int, Dict[str, str]]]:
    """Yield (line_number, row_dict) from a CSV/TSV file.

    Handles common AudioSet formats:
      - class_labels_indices.csv: index,mid,display_name
      - strong TSV/CSV: segment_id,start_time_seconds,end_time_seconds,label
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
            "segment_id", "ytid", "youtube_id", "video_id",
            "start_time_seconds", "end_time_seconds", "start_seconds", "end_seconds",
            "onset", "offset", "label", "labels", "mid", "display_name", "positive_labels",
        }
        has_header = bool(set(normalized_first) & header_tokens)

        if has_header:
            header = normalized_first
        elif len(first_row) >= 4:
            header = ["segment_id", "start_time_seconds", "end_time_seconds", "label"]
            if len(first_row) >= 5:
                header.append("display_name")
            row_dict = {header[i]: first_row[i].strip() for i in range(min(len(header), len(first_row)))}
            yield first_line_no, row_dict
        else:
            return

        for line_no, row in enumerate(reader, start=first_line_no + 1):
            if not row or all(not cell.strip() for cell in row):
                continue
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
        if not any("mid" in row and "display_name" in row for _, row in rows[:5]):
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
        if key in row and str(row[key]).strip():
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
    if raw.startswith("/m/") and "," in raw:
        return [x.strip().strip('"') for x in raw.split(",") if x.strip()]
    return [raw]


def parse_segment_offsets(segment_id: str) -> Tuple[str, Optional[float], Optional[float]]:
    """Return (ytid_like, clip_start, clip_end) from common AudioSet segment IDs."""
    stem = Path(str(segment_id)).stem.strip()
    m = re.match(r"^(.+?)[_\-]([0-9]+(?:\.[0-9]+)?)[_\-]([0-9]+(?:\.[0-9]+)?)$", stem)
    if m:
        return m.group(1), float(m.group(2)), float(m.group(3))
    if len(stem) >= 11:
        return stem[:11], None, None
    return stem, None, None


def audio_key_candidates(identifier: str) -> List[Tuple[str, str]]:
    """Return candidate lookup keys with mode.

    mode == "segment" means the raw file is probably the 10 s AudioSet segment.
    mode == "ytid" means the raw file may be a full YouTube audio; use clip offset if available.
    """
    stem = Path(str(identifier)).stem.strip()
    candidates: List[Tuple[str, str]] = []

    def add(key: str, mode: str) -> None:
        key = key.strip()
        pair = (key, mode)
        if key and pair not in candidates:
            candidates.append(pair)

    add(stem, "segment")
    ytid, clip_start, clip_end = parse_segment_offsets(stem)
    if clip_start is not None and clip_end is not None:
        add(f"{ytid}_{clip_start:g}_{clip_end:g}", "segment")
        add(f"{ytid}_{int(round(clip_start)):06d}_{int(round(clip_end)):06d}", "segment")
        add(f"{ytid}-{int(round(clip_start)):06d}-{int(round(clip_end)):06d}", "segment")
    add(ytid, "ytid")
    if len(stem) >= 11:
        add(stem[:11], "ytid")
    return candidates


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
            segment_id = find_first(row, [
                "segment_id", "ytid", "youtube_id", "video_id", "file_name", "filename", "audio_id", "wav",
            ])
            start = parse_float(find_first(row, [
                "start_time_seconds", "start_seconds", "onset_seconds", "onset", "start",
            ]))
            end = parse_float(find_first(row, [
                "end_time_seconds", "end_seconds", "offset_seconds", "offset", "end",
            ]))

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
                interference_label = resolve_interference_label(display)
                dcase_label = None if interference_label else resolve_dcase_label(display)
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
                        interference_label=interference_label,
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
    keep_noise_gaps: bool,
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
                    if j != idx and intervals_overlap(event.start, event.end, other.start, other.end, overlap_margin):
                        reason = "overlap_with_other_annotated_event"
                        break

            label = event.dcase_label or event.interference_label or event.display_label
            if reason:
                rejected.append(Candidate(event.uid, "reject", split, label, segment_id, event.start, event.end,
                                          event.duration, event.annotation_file, event.raw_label, event.display_label, reason))
                continue

            if event.dcase_label is not None:
                accepted.append(Candidate(event.uid, "sound_event", split, event.dcase_label, segment_id, event.start,
                                          event.end, event.duration, event.annotation_file, event.raw_label, event.display_label))
            elif event.interference_label is not None:
                accepted.append(Candidate(event.uid, "interference", split, event.interference_label, segment_id,
                                          event.start, event.end, event.duration, event.annotation_file,
                                          event.raw_label, event.display_label))
            else:
                rejected.append(Candidate(event.uid, "reject", split, event.display_label, segment_id, event.start,
                                          event.end, event.duration, event.annotation_file, event.raw_label,
                                          event.display_label, "unmapped_not_dcase_target_or_interference"))

        if not keep_noise_gaps:
            continue

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

        gaps = sorted(gaps, key=lambda x: x[1] - x[0], reverse=True)[:max_background_per_clip]
        for gap_idx, (s, t) in enumerate(gaps):
            uid_base = f"noise:{split}:{segment_id}:{s:.6f}:{t:.6f}:{gap_idx}"
            uid = hashlib.sha1(uid_base.encode("utf-8")).hexdigest()[:16]
            accepted.append(Candidate(uid, "noise", split, "noise", segment_id, s, t, t - s,
                                      "<derived_from_strong_gaps>", "", "noise"))

    return accepted, rejected


def summarize_candidates(
    events: Sequence[StrongEvent],
    accepted: Sequence[Candidate],
    rejected: Sequence[Candidate],
    parse_status: Counter,
    top_k: int,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "target_sr": TARGET_SR,
        "parse_status": dict(parse_status),
        "n_annotation_events": len(events),
        "n_candidates": len(accepted),
        "n_rejected": len(rejected),
        "accepted_by_kind": dict(Counter(c.kind for c in accepted)),
        "accepted_by_split": dict(Counter(c.split for c in accepted)),
        "rejected_by_reason": dict(Counter(c.reject_reason for c in rejected)),
    }

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
            items.append({
                "label": label,
                "count": count,
                "duration_hours": round(duration_by_kind_label[(kind, label)] / 3600.0, 4),
            })
        summary["labels"][kind] = items
    return summary


def print_summary(summary: Dict[str, object]) -> None:
    print("\n================ AudioSet-Strong annotation statistics ================")
    print(f"Output sample rate       : {summary['target_sr']} Hz")
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
            obj = asdict(row) if hasattr(row, "__dataclass_fields__") else row
            f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def create_compatible_dirs(output_dir: Path) -> None:
    for split in ("train", "valid"):
        for label in DCASE2026_LABELS:
            (output_dir / "sound_event" / split / label).mkdir(parents=True, exist_ok=True)
        for label in DCASE_INTERFERENCE_LABELS:
            (output_dir / "interference" / split / label).mkdir(parents=True, exist_ok=True)
        (output_dir / "noise" / split).mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata" / "valid").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata" / "audioset_strong").mkdir(parents=True, exist_ok=True)


def write_summary_files(
    output_dir: Path,
    summary: Dict[str, object],
    accepted: Sequence[Candidate],
    rejected: Sequence[Candidate],
) -> None:
    meta_dir = output_dir / "metadata" / "audioset_strong"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
                                           encoding="utf-8")
    write_jsonl(meta_dir / "accepted_candidates.jsonl", accepted)
    write_jsonl(meta_dir / "rejected_annotations.jsonl", rejected)


def build_audio_index(input_dir: Path, extensions: Sequence[str]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    duplicates: Counter = Counter()
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    for path in tqdm(files, desc="Index raw audio", unit="file"):
        for key, _mode in audio_key_candidates(path.stem):
            if key in index and index[key] != path:
                duplicates[key] += 1
                continue
            index[key] = path
    if duplicates:
        print(f"Warning: {len(duplicates)} duplicate audio keys found; keeping first path per key.")
    print(f"Indexed {len(files)} raw audio files with {len(index)} lookup keys.")
    return index


def resolve_audio(candidate: Candidate, audio_index: Dict[str, Path]) -> Tuple[Optional[Path], float]:
    ytid, clip_start, _clip_end = parse_segment_offsets(candidate.segment_id)
    for key, mode in audio_key_candidates(candidate.segment_id):
        if key in audio_index:
            if mode == "ytid" and key == ytid and clip_start is not None:
                return audio_index[key], float(clip_start)
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
            audio = f.read(frames=n_frames, dtype="float32", always_2d=True).T
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


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    if librosa is None:
        raise RuntimeError(f"librosa is required for resampling: {_LIBROSA_IMPORT_ERROR}")
    channels = [librosa.resample(ch.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr) for ch in audio]
    return np.stack(channels, axis=0).astype(np.float32)


def save_audio_32k(path: Path, audio: np.ndarray) -> None:
    if sf is None:
        raise RuntimeError(f"soundfile import failed: {_SOUNDFILE_IMPORT_ERROR}")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = audio[0] if audio.ndim == 2 and audio.shape[0] == 1 else audio.T
    sf.write(str(path), out, TARGET_SR)


def candidate_output_path(output_dir: Path, candidate: Candidate, suffix: str = ".wav") -> Path:
    filename = (
        f"{safe_filename_part(candidate.segment_id, max_len=80)}__"
        f"{int(round(candidate.start * 1000)):08d}_{int(round(candidate.end * 1000)):08d}__"
        f"{safe_filename_part(candidate.display_label or candidate.label, max_len=48)}__"
        f"{candidate.uid}{suffix}"
    )
    if candidate.kind == "noise":
        return output_dir / "noise" / candidate.split / filename
    return output_dir / candidate.kind / candidate.split / candidate.label / filename


def extract_candidates(
    candidates: Sequence[Candidate],
    input_dir: Path,
    output_dir: Path,
    amp_threshold: float,
    min_rms: float,
    trim_edges: bool,
    mono: bool,
    overwrite: bool,
    extensions: Sequence[str],
) -> List[ExtractResult]:
    create_compatible_dirs(output_dir)
    audio_index = build_audio_index(input_dir, extensions=extensions)
    results: List[ExtractResult] = []

    for c in tqdm(candidates, desc="Extract candidates", unit="seg"):
        src_path, offset = resolve_audio(c, audio_index)
        out_path = candidate_output_path(output_dir, c)
        if src_path is None:
            results.append(ExtractResult(c.uid, c.kind, c.split, c.label, "", str(out_path), c.start, c.end,
                                         c.duration, "skip", "missing_raw_audio"))
            continue
        if out_path.exists() and not overwrite:
            results.append(ExtractResult(c.uid, c.kind, c.split, c.label, str(src_path), str(out_path),
                                         c.start, c.end, c.duration, "skip", "output_exists"))
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
            audio = resample_audio(audio, orig_sr=sr, target_sr=TARGET_SR)
            save_audio_32k(out_path, audio)
            status, reason = "saved", ""
        except Exception as exc:
            sr = None  # type: ignore[assignment]
            peak = None
            rms = None
            status, reason = "skip", f"{type(exc).__name__}:{exc}"

        results.append(ExtractResult(c.uid, c.kind, c.split, c.label, str(src_path), str(out_path),
                                     c.start, c.end, c.duration, status, reason,
                                     source_sr=sr if isinstance(sr, int) else None,
                                     target_sr=TARGET_SR, peak=peak, rms=rms))
    return results


def print_extraction_summary(results: Sequence[ExtractResult]) -> None:
    print("\n================ Audio extraction summary ================")
    print(f"Output sample rate : {TARGET_SR} Hz")
    print(f"Segments processed : {len(results)}")
    print(f"By status          : {dict(Counter(r.status for r in results))}")
    print(f"Skip reasons       : {dict(Counter(r.reason for r in results if r.status != 'saved'))}")
    print(f"Saved by kind      : {dict(Counter(r.kind for r in results if r.status == 'saved'))}")
    print("==========================================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract non-overlapping AudioSet-Strong clips into DCASE2026 Task4-compatible folders."
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Root folder containing raw AudioSet audio; searched recursively.")
    parser.add_argument("--annotations_dir", type=Path, required=True, help="Folder containing AudioSet Strong CSV/TSV annotations.")
    parser.add_argument("--output_dir", type=Path, required=True, help="DCASE dev_set root or compatible output root.")
    parser.add_argument("--execute", action="store_true", help="Actually extract audio. Without this flag, only statistics/manifests are produced.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted WAV files.")
    parser.add_argument("--target_sr", type=int, default=TARGET_SR, choices=[TARGET_SR],
                        help="Output sample rate. Fixed to 32000 for DCASE2026 Task4 compatibility.")

    parser.add_argument("--min_event_duration", type=float, default=0.15)
    parser.add_argument("--max_event_duration", type=float, default=10.0)
    parser.add_argument("--min_noise_duration", type=float, default=1.0)
    parser.add_argument("--default_clip_duration", type=float, default=10.0)
    parser.add_argument("--overlap_margin", type=float, default=0.03)
    parser.add_argument("--max_background_per_clip", type=int, default=2)
    parser.add_argument("--keep_noise_gaps", action=argparse.BooleanOptionalAction, default=True,
                        help="Export annotation-free gaps as background/noise WAVs under noise/{train,valid}.")

    parser.add_argument("--mono", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trim_edges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp_threshold", type=float, default=0.005)
    parser.add_argument("--min_rms", type=float, default=1e-4)
    parser.add_argument("--audio_extensions", nargs="+", default=list(AUDIO_EXTENSIONS))

    parser.add_argument("--train_keywords", nargs="+", default=["train", "balanced", "unbalanced"])
    parser.add_argument("--valid_keywords", nargs="+", default=["valid", "validation", "eval", "evaluate", "test"])
    parser.add_argument("--top_k_interference_stats", type=int, default=40)
    parser.add_argument("--no_write_manifests", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.target_sr != TARGET_SR:
        raise ValueError("DCASE2026 Task4 augmentation WAVs must be 32000 Hz.")
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
    print(f"target_sr       : {TARGET_SR}")
    print(f"overlap_margin  : {args.overlap_margin}")
    print("====================================================")

    create_compatible_dirs(args.output_dir)

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
        keep_noise_gaps=args.keep_noise_gaps,
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
        print("Dry run only. Re-run with --execute to extract 32 kHz WAVs after reviewing the statistics.")
        return 0

    results = extract_candidates(
        accepted,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
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
            "target_sr": TARGET_SR,
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
