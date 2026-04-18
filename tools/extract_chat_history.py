#!/usr/bin/env python3

from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TEXT_ITEM_TYPES = {
    "input_text",
    "output_text",
    "text",
}

ROLE_TITLES = {
    "user": "User",
    "assistant": "Assistant",
}

ROLE_BADGES = {
    "user": "USER",
    "assistant": "ASSISTANT",
}

NOISE_TAGS = {
    "environment_context",
    "permissions instructions",
    "collaboration_mode",
    "skills_instructions",
}

NOISE_PATTERNS = [
    re.compile(r"<environment_context>.*?</environment_context>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<permissions instructions>.*?</permissions instructions>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<collaboration_mode>.*?</collaboration_mode>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<skills_instructions>.*?</skills_instructions>", re.DOTALL | re.IGNORECASE),
]

NOISE_SUBSTRINGS = (
    "Filesystem sandboxing defines which files can be read or written.",
    "The writable roots are",
    "Your active mode changes only when new developer instructions",
    "The request_user_input tool is unavailable in Default mode.",
    "Below is the list of skills that can be used in this session.",
)

NOISE_LINE_PREFIXES = (
    "Context from my IDE setup:",
    "Open tabs:",
)

DROP_LINE_PREFIXES = (
    "Context from my IDE setup:",
    "Open tabs:",
    "Active file:",
)

REQUEST_LINE_PREFIXES = (
    "My request for Codex:",
    "My request:",
)

NOISE_LINE_PATTERNS = [
    re.compile(r"^[A-Za-z0-9_.\-/ ]+\s+\([^)]+\)$"),
]

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "summary_border": "#d7deea",
        "summary_accent": "#5b8def",
        "summary_bg": "#f8fbff",
        "summary_title": "#172033",
        "summary_label": "#5a6b85",
        "summary_value": "#172033",
        "meta_tile_bg": "rgba(255,255,255,0.72)",
        "meta_value": "#223049",
        "user_border": "#dce5f3",
        "user_accent": "#4f7cff",
        "user_bg": "#f7faff",
        "user_shadow": "rgba(79,124,255,0.08)",
        "user_badge_bg": "#e2ebff",
        "user_badge_fg": "#234ca8",
        "assistant_border": "#d7eadf",
        "assistant_accent": "#3fa56b",
        "assistant_bg": "#f5fcf7",
        "assistant_shadow": "rgba(63,165,107,0.08)",
        "assistant_badge_bg": "#dff5e7",
        "assistant_badge_fg": "#1c6b41",
        "body_text": "#1d2738",
        "title_text": "#172033",
        "inline_code_bg": "#eef3ff",
        "inline_code_fg": "#213b86",
        "inline_code_border": "#cfdbff",
        "code_header_bg": "#1f2937",
        "code_header_fg": "#e5edf7",
        "code_bg": "#0f1720",
        "code_fg": "#e8eef6",
        "code_border": "#d6dce8",
        "table_border": "#d6dce8",
        "table_header_bg": "#edf4ff",
        "table_header_fg": "#1d355f",
        "table_row_alt": "#fafcff",
    },
    "soft": {
        "summary_border": "#e7dcc8",
        "summary_accent": "#d88b4a",
        "summary_bg": "#fffaf3",
        "summary_title": "#2f2418",
        "summary_label": "#7a634f",
        "summary_value": "#2f2418",
        "meta_tile_bg": "rgba(255,248,239,0.86)",
        "meta_value": "#3e3124",
        "user_border": "#eadfcf",
        "user_accent": "#c9784d",
        "user_bg": "#fff8f3",
        "user_shadow": "rgba(201,120,77,0.10)",
        "user_badge_bg": "#ffe7d9",
        "user_badge_fg": "#9b4e27",
        "assistant_border": "#d9eadf",
        "assistant_accent": "#5f9d76",
        "assistant_bg": "#f6fcf8",
        "assistant_shadow": "rgba(95,157,118,0.10)",
        "assistant_badge_bg": "#dff3e6",
        "assistant_badge_fg": "#2f6a45",
        "body_text": "#2e2a25",
        "title_text": "#2f2418",
        "inline_code_bg": "#fff1e6",
        "inline_code_fg": "#92451c",
        "inline_code_border": "#f1d0ba",
        "code_header_bg": "#3c2f2a",
        "code_header_fg": "#f7eadf",
        "code_bg": "#201917",
        "code_fg": "#f5eee7",
        "code_border": "#e7d8cb",
        "table_border": "#e3d7cb",
        "table_header_bg": "#fff0e2",
        "table_header_fg": "#6f4528",
        "table_row_alt": "#fffaf6",
    },
    "high-contrast": {
        "summary_border": "#a9b3c7",
        "summary_accent": "#2457ff",
        "summary_bg": "#f4f7ff",
        "summary_title": "#0e1320",
        "summary_label": "#33415c",
        "summary_value": "#0e1320",
        "meta_tile_bg": "#ffffff",
        "meta_value": "#0e1320",
        "user_border": "#b7c7ff",
        "user_accent": "#2457ff",
        "user_bg": "#f5f8ff",
        "user_shadow": "rgba(36,87,255,0.12)",
        "user_badge_bg": "#2457ff",
        "user_badge_fg": "#ffffff",
        "assistant_border": "#b7e4c9",
        "assistant_accent": "#127a43",
        "assistant_bg": "#f3fff7",
        "assistant_shadow": "rgba(18,122,67,0.12)",
        "assistant_badge_bg": "#127a43",
        "assistant_badge_fg": "#ffffff",
        "body_text": "#111827",
        "title_text": "#0e1320",
        "inline_code_bg": "#e8eeff",
        "inline_code_fg": "#14358f",
        "inline_code_border": "#9bb2ff",
        "code_header_bg": "#111827",
        "code_header_fg": "#f9fafb",
        "code_bg": "#020617",
        "code_fg": "#f8fafc",
        "code_border": "#94a3b8",
        "table_border": "#94a3b8",
        "table_header_bg": "#dbe7ff",
        "table_header_fg": "#102a75",
        "table_row_alt": "#f8fbff",
    },
}


@dataclass(order=True)
class MessageEvent:
    timestamp: datetime
    path: Path
    line_number: int
    session_id: str
    role: str
    text: str


@dataclass
class SessionInfo:
    session_id: str
    path: Path
    start_time: datetime | None = None


@dataclass
class MergedMessage:
    role: str
    start_time: datetime
    end_time: datetime
    sources: list[Path]
    session_ids: list[str]
    text: str


MIN_FORK_OVERLAP_MESSAGES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract meaningful chat history from JSONL logs into a Markdown file."
    )
    parser.add_argument("input_dir", type=Path, help="Folder to scan recursively for .jsonl files.")
    parser.add_argument("output_md", type=Path, help="Markdown file to write.")
    parser.add_argument(
        "--group-by",
        choices=("timeline", "session"),
        default="timeline",
        help="How to organize the exported Markdown.",
    )
    parser.add_argument(
        "--theme",
        choices=tuple(THEMES),
        default="light",
        help="Visual theme for the generated Markdown + HTML.",
    )
    parser.add_argument(
        "--output-html",
        action="store_true",
        help="Write a full standalone HTML document instead of Markdown.",
    )
    return parser.parse_args()


def iter_jsonl_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.jsonl") if path.is_file())


def extract_file_data(path: Path) -> tuple[SessionInfo, list[MessageEvent]]:
    session_id = path.stem
    session_start: datetime | None = None
    events: list[MessageEvent] = []
    seen: set[tuple[datetime, str, str]] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            if record.get("type") == "session_meta":
                payload = record.get("payload")
                if isinstance(payload, dict):
                    payload_id = payload.get("id")
                    if isinstance(payload_id, str) and payload_id.strip():
                        session_id = payload_id.strip()
                    session_start = parse_timestamp(payload.get("timestamp")) or session_start

            event = extract_event_from_record(record, path, line_number, session_id)
            if event is None:
                continue

            key = (event.timestamp, event.role, event.text)
            if key in seen:
                continue

            seen.add(key)
            events.append(event)

            if session_start is None or event.timestamp < session_start:
                session_start = event.timestamp

    return SessionInfo(session_id=session_id, path=path, start_time=session_start), events


def extract_event_from_record(
    record: dict[str, Any],
    path: Path,
    line_number: int,
    session_id: str,
) -> MessageEvent | None:
    if record.get("type") != "response_item":
        return None

    payload = record.get("payload")
    if not isinstance(payload, dict) or payload.get("type") != "message":
        return None

    role = payload.get("role")
    if role not in ROLE_TITLES:
        return None

    content = payload.get("content")
    if not isinstance(content, list):
        return None

    text = collect_message_text(content)
    if not text:
        return None

    timestamp = parse_timestamp(record.get("timestamp"))
    if timestamp is None:
        timestamp = parse_timestamp(payload.get("timestamp"))
    if timestamp is None:
        return None

    return MessageEvent(
        timestamp=timestamp,
        path=path,
        line_number=line_number,
        session_id=session_id,
        role=role,
        text=text,
    )


def collect_message_text(content: list[dict[str, Any]]) -> str:
    parts: list[str] = []

    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in TEXT_ITEM_TYPES:
            continue

        text = item.get("text")
        if not isinstance(text, str):
            continue

        cleaned = clean_text(text)
        if cleaned:
            parts.append(cleaned)

    return normalize_spacing("\n\n".join(parts))


def clean_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""

    for pattern in NOISE_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    cleaned = strip_noise_paragraphs(cleaned)
    cleaned = normalize_spacing(cleaned)
    if not cleaned or is_noise_text(cleaned):
        return ""

    return cleaned


def strip_noise_paragraphs(text: str) -> str:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    kept: list[str] = []

    for paragraph in paragraphs:
        cleaned_paragraph = strip_noise_lines(paragraph)
        if cleaned_paragraph and not is_noise_text(cleaned_paragraph):
            kept.append(cleaned_paragraph)

    return "\n\n".join(kept)


def strip_noise_lines(text: str) -> str:
    lines = text.splitlines()
    kept: list[str] = []
    skip_block = False

    for line in lines:
        stripped = line.strip()
        normalized = normalize_noise_line(stripped)
        if not stripped:
            skip_block = False
            kept.append("")
            continue

        request_prefix = next((prefix for prefix in REQUEST_LINE_PREFIXES if normalized.startswith(prefix)), None)
        if request_prefix is not None:
            remainder = normalized[len(request_prefix):].strip()
            if remainder:
                kept.append(remainder)
            skip_block = False
            continue

        if any(normalized.startswith(prefix) for prefix in DROP_LINE_PREFIXES):
            skip_block = normalized.startswith("Open tabs:")
            continue

        if skip_block and looks_like_context_line(stripped):
            continue

        if skip_block:
            skip_block = False

        if looks_like_context_line(stripped):
            continue

        kept.append(line.rstrip())

    return normalize_spacing("\n".join(kept))


def looks_like_context_line(line: str) -> bool:
    normalized = normalize_noise_line(line)
    if normalized.startswith("- ") and "/" in normalized:
        return True
    if normalized.startswith("- ") and "." in normalized and "/" not in normalized:
        return True
    if normalized.startswith("* ") and "/" in normalized:
        return True
    return any(pattern.fullmatch(normalized) for pattern in NOISE_LINE_PATTERNS)


def is_noise_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True

    lower = stripped.lower()
    for tag in NOISE_TAGS:
        if lower.startswith(f"<{tag}>") and lower.endswith(f"</{tag}>"):
            return True

    if stripped.startswith("<") and stripped.endswith(">") and "\n" not in stripped:
        return True

    if any(marker in stripped for marker in NOISE_SUBSTRINGS):
        return True

    normalized = normalize_noise_line(stripped)
    return any(normalized.startswith(prefix) for prefix in DROP_LINE_PREFIXES)


def normalize_noise_line(line: str) -> str:
    normalized = re.sub(r"^#+\s*", "", line.strip())
    return normalized.strip()


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value or "item"


def normalize_spacing(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None

    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def merge_consecutive_messages(events: list[MessageEvent]) -> list[MergedMessage]:
    merged: list[MergedMessage] = []

    for event in events:
        if merged and merged[-1].role == event.role:
            merged[-1].end_time = event.timestamp
            if event.path not in merged[-1].sources:
                merged[-1].sources.append(event.path)
            if event.session_id not in merged[-1].session_ids:
                merged[-1].session_ids.append(event.session_id)
            merged[-1].text = f"{merged[-1].text}\n\n{event.text}".strip()
            continue

        merged.append(
            MergedMessage(
                role=event.role,
                start_time=event.timestamp,
                end_time=event.timestamp,
                sources=[event.path],
                session_ids=[event.session_id],
                text=event.text,
            )
        )

    return merged


def remove_forked_history(
    sessions: list[SessionInfo],
    events: list[MessageEvent],
) -> tuple[list[MessageEvent], dict[str, int]]:
    by_file: dict[Path, list[MessageEvent]] = defaultdict(list)
    for event in events:
        by_file[event.path].append(event)

    ordered_sessions = sorted(
        sessions,
        key=lambda session: (
            session.start_time or datetime.max.replace(tzinfo=timezone.utc),
            str(session.path),
        ),
    )

    accepted_sequences_by_session: dict[str, list[list[tuple[str, str]]]] = defaultdict(list)
    accepted_sequences_global: list[list[tuple[str, str]]] = []
    kept_events: list[MessageEvent] = []
    removed_counts: dict[str, int] = {}

    for session in ordered_sessions:
        session_events = sorted(
            by_file.get(session.path, []),
            key=lambda event: (event.timestamp, str(event.path), event.line_number),
        )
        if not session_events:
            continue

        session_sequence = [(event.role, event.text) for event in session_events]
        overlap_same_session = find_longest_prefix_overlap(
            session_sequence,
            accepted_sequences_by_session[session.session_id],
        )
        overlap_global = find_longest_prefix_overlap(session_sequence, accepted_sequences_global)
        overlap = max(overlap_same_session, overlap_global)
        removed_counts[str(session.path)] = overlap

        trimmed_events = session_events[overlap:]
        kept_events.extend(trimmed_events)
        accepted_sequences_by_session[session.session_id].append(session_sequence)
        accepted_sequences_global.append(session_sequence)

    kept_events.sort(key=lambda event: (event.timestamp, str(event.path), event.line_number))
    return kept_events, removed_counts


def find_longest_prefix_overlap(
    session_sequence: list[tuple[str, str]],
    accepted_sequences: list[list[tuple[str, str]]],
) -> int:
    best = 0
    for previous in accepted_sequences:
        max_overlap = min(len(session_sequence), len(previous))
        overlap = 0
        while overlap < max_overlap and session_sequence[overlap] == previous[overlap]:
            overlap += 1
        if overlap > best:
            best = overlap

    if best < MIN_FORK_OVERLAP_MESSAGES:
        return 0
    return best


def render_markdown(
    root: Path,
    jsonl_files: list[Path],
    merged_messages: list[MergedMessage],
    grouped_sessions: list[tuple[SessionInfo, list[MergedMessage]]],
    group_by: str,
    theme: dict[str, str],
    removed_fork_messages: int,
) -> str:
    if group_by == "session":
        return render_session_markdown(root, jsonl_files, grouped_sessions, theme, removed_fork_messages)
    return render_timeline_markdown(root, jsonl_files, merged_messages, theme, removed_fork_messages)


def render_html_document(markdown_body: str, theme_name: str, theme: dict[str, str]) -> str:
    title = "Extracted Chat History"
    body_html = convert_markdown_shell_to_html(markdown_body)
    navigation_html = build_navigation_html(body_html)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Noto Sans SC", "Helvetica Neue", Arial, sans-serif;
      background:
        radial-gradient(circle at top left, {theme['summary_bg']} 0%, transparent 35%),
        linear-gradient(180deg, #ffffff 0%, #f6f8fc 100%);
      color: {theme['body_text']};
    }}
    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px 20px 48px;
      display: grid;
      grid-template-columns: 290px minmax(0, 1fr);
      gap: 24px;
    }}
    .page-header {{
      margin-bottom: 24px;
      grid-column: 1 / -1;
    }}
    .eyebrow {{
      display: inline-block;
      padding: 6px 12px;
      border-radius: 999px;
      background: {theme['summary_bg']};
      border: 1px solid {theme['summary_border']};
      color: {theme['summary_label']};
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 14px 0 8px;
      color: {theme['title_text']};
      font-size: 40px;
      line-height: 1.1;
    }}
    .subtitle {{
      margin: 0;
      color: {theme['summary_label']};
      font-size: 16px;
    }}
    .content > *:first-child {{
      margin-top: 0;
    }}
    .content {{
      font-size: 16px;
      min-width: 0;
    }}
    .sidebar {{
      position: sticky;
      top: 20px;
      align-self: start;
      background: rgba(255,255,255,0.84);
      backdrop-filter: blur(8px);
      border: 1px solid {theme['summary_border']};
      border-radius: 20px;
      padding: 18px 16px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
      max-height: calc(100vh - 40px);
      overflow: auto;
    }}
    .sidebar h2 {{
      margin: 0 0 12px;
      font-size: 16px;
      color: {theme['title_text']};
    }}
    .nav-group {{
      margin-bottom: 18px;
    }}
    .nav-group-title {{
      margin: 0 0 8px;
      color: {theme['summary_label']};
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .nav-list {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: grid;
      gap: 6px;
    }}
    .nav-link {{
      display: block;
      padding: 8px 10px;
      border-radius: 12px;
      color: {theme['body_text']};
      text-decoration: none;
      font-size: 14px;
      background: transparent;
    }}
    .nav-link .role-chip {{
      display: inline-block;
      min-width: 72px;
      margin-right: 8px;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }}
    .nav-link .role-chip.user {{
      background: {theme['user_badge_bg']};
      color: {theme['user_badge_fg']};
    }}
    .nav-link .role-chip.assistant {{
      background: {theme['assistant_badge_bg']};
      color: {theme['assistant_badge_fg']};
    }}
    .nav-link:hover {{
      background: {theme['summary_bg']};
    }}
    .nav-link.active {{
      background: {theme['summary_bg']};
      box-shadow: inset 0 0 0 1px {theme['summary_accent']};
      color: {theme['title_text']};
      font-weight: 700;
    }}
    details.nav-session-group {{
      margin: 0 0 10px 0;
      border-radius: 14px;
      background: rgba(255,255,255,0.5);
      border: 1px solid transparent;
    }}
    details.nav-session-group[open] {{
      border-color: {theme['summary_border']};
      background: rgba(255,255,255,0.72);
    }}
    details.nav-session-group > summary {{
      list-style: none;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border-radius: 12px;
    }}
    details.nav-session-group > summary::-webkit-details-marker {{
      display: none;
    }}
    .nav-session-label {{
      color: {theme['title_text']};
      font-size: 13px;
      font-weight: 800;
    }}
    .nav-session-count {{
      margin-left: auto;
      color: {theme['summary_label']};
      font-size: 11px;
      font-weight: 800;
    }}
    .nav-session-chevron {{
      color: {theme['summary_label']};
      font-size: 15px;
      transition: transform 0.18s ease;
    }}
    details.nav-session-group[open] > summary .nav-session-chevron {{
      transform: rotate(90deg);
    }}
    .nav-session-body {{
      padding: 0 6px 8px 6px;
    }}
    .search-input {{
      width: 100%;
      border: 1px solid {theme['summary_border']};
      background: white;
      color: {theme['body_text']};
      border-radius: 12px;
      padding: 12px 14px;
      font-size: 14px;
      outline: none;
      margin-bottom: 10px;
    }}
    .filter-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 14px;
    }}
    .filter-chip {{
      border: 1px solid {theme['summary_border']};
      background: white;
      color: {theme['title_text']};
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 12px;
      font-weight: 800;
      cursor: pointer;
    }}
    .filter-chip.active {{
      background: {theme['summary_bg']};
      border-color: {theme['summary_accent']};
    }}
    .control-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .control-btn {{
      border: 1px solid {theme['summary_border']};
      background: white;
      color: {theme['title_text']};
      border-radius: 10px;
      padding: 9px 10px;
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
    }}
    .control-btn:hover {{
      background: {theme['summary_bg']};
    }}
    details.message-card, details.session-card {{
      margin: 0 0 20px 0;
    }}
    details.message-card > summary,
    details.session-card > summary {{
      list-style: none;
      cursor: pointer;
    }}
    details.message-card > summary::-webkit-details-marker,
    details.session-card > summary::-webkit-details-marker {{
      display: none;
    }}
    .summary-chevron {{
      margin-left: auto;
      font-size: 18px;
      color: {theme['summary_label']};
      transition: transform 0.18s ease;
    }}
    details[open] > summary .summary-chevron {{
      transform: rotate(90deg);
    }}
    .copy-btn {{
      margin-left: auto;
      border: 1px solid rgba(255,255,255,0.18);
      background: rgba(255,255,255,0.08);
      color: {theme['code_header_fg']};
      border-radius: 9px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 800;
      cursor: pointer;
    }}
    .copy-btn:hover {{
      background: rgba(255,255,255,0.16);
    }}
    mark.search-hit {{
      background: #ffe58f;
      color: #1f2937;
      padding: 0 2px;
      border-radius: 4px;
    }}
    @media (max-width: 720px) {{
      .page {{
        grid-template-columns: 1fr;
        padding: 20px 14px 40px;
      }}
      h1 {{
        font-size: 30px;
      }}
      .sidebar {{
        position: static;
        max-height: none;
      }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="page-header">
      <div class="eyebrow">Codex Session Export</div>
      <h1>{html.escape(title)}</h1>
      <p class="subtitle">Standalone HTML export. Theme: {html.escape(theme_name)}.</p>
    </header>
    <aside class="sidebar">
{navigation_html}
    </aside>
    <section class="content">
{body_html}
    </section>
  </main>
  <script>
    function setOpenBySelector(selector, openValue) {{
      document.querySelectorAll(selector).forEach((node) => {{
        node.open = openValue;
      }});
    }}
    function setOpenByRole(role, openValue) {{
      document.querySelectorAll(`details.message-card[data-role="${{role}}"]`).forEach((node) => {{
        node.open = openValue;
      }});
    }}
    async function copyCode(button) {{
      const code = button.closest('.code-block').querySelector('code');
      if (!code) return;
      try {{
        await navigator.clipboard.writeText(code.innerText);
        const prev = button.innerText;
        button.innerText = 'Copied';
        setTimeout(() => {{
          button.innerText = prev;
        }}, 1200);
      }} catch (_err) {{
        button.innerText = 'Copy failed';
        setTimeout(() => {{
          button.innerText = 'Copy';
        }}, 1200);
      }}
    }}
    let activeRoleFilter = 'all';
    function filterByRole(role) {{
      activeRoleFilter = role;
      document.querySelectorAll('.filter-chip[data-role]').forEach((chip) => {{
        chip.classList.toggle('active', chip.dataset.role === role);
      }});
      applyFilters();
    }}
    function applyFilters() {{
      const query = (document.getElementById('searchBox')?.value || '').toLowerCase();
      const messages = document.querySelectorAll('details.message-card');
      messages.forEach((node) => {{
        clearHighlights(node);
        const text = (node.dataset.search || '').toLowerCase();
        const roleOk = activeRoleFilter === 'all' || node.dataset.role === activeRoleFilter;
        const textOk = !query || text.includes(query);
        node.style.display = roleOk && textOk ? '' : 'none';
        if (roleOk && textOk && query) {{
          highlightInNode(node.querySelector('.message-body'), query);
        }}
      }});
      document.querySelectorAll('details.session-card').forEach((session) => {{
        const visibleMessages = session.querySelectorAll('details.message-card:not([style*="display: none"])');
        session.style.display = visibleMessages.length ? '' : 'none';
      }});
    }}
    function clearHighlights(root) {{
      if (!root) return;
      root.querySelectorAll('mark.search-hit').forEach((mark) => {{
        const parent = mark.parentNode;
        if (!parent) return;
        parent.replaceChild(document.createTextNode(mark.textContent), mark);
        parent.normalize();
      }});
    }}
    function highlightInNode(root, query) {{
      if (!root || !query) return;
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {{
        acceptNode(node) {{
          if (!node.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
          const parent = node.parentElement;
          if (!parent) return NodeFilter.FILTER_REJECT;
          if (parent.closest('code, pre, button, .copy-btn')) return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        }}
      }});
      const nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);
      nodes.forEach((textNode) => {{
        const value = textNode.nodeValue;
        const lower = value.toLowerCase();
        const idx = lower.indexOf(query);
        if (idx === -1) return;
        const frag = document.createDocumentFragment();
        let start = 0;
        let pos = lower.indexOf(query, start);
        while (pos !== -1) {{
          frag.appendChild(document.createTextNode(value.slice(start, pos)));
          const mark = document.createElement('mark');
          mark.className = 'search-hit';
          mark.textContent = value.slice(pos, pos + query.length);
          frag.appendChild(mark);
          start = pos + query.length;
          pos = lower.indexOf(query, start);
        }}
        frag.appendChild(document.createTextNode(value.slice(start)));
        textNode.parentNode.replaceChild(frag, textNode);
      }});
    }}
    function setupActiveNav() {{
      const links = Array.from(document.querySelectorAll('.nav-link[href^="#"]'));
      const map = new Map();
      links.forEach((link) => {{
        const id = link.getAttribute('href').slice(1);
        const target = document.getElementById(id);
        if (target) map.set(target, link);
      }});
      const observer = new IntersectionObserver((entries) => {{
        let best = null;
        for (const entry of entries) {{
          if (entry.isIntersecting) {{
            if (!best || entry.intersectionRatio > best.intersectionRatio) best = entry;
          }}
        }}
        if (!best) return;
      links.forEach((link) => link.classList.remove('active'));
      const activeLink = map.get(best.target);
      if (activeLink) {{
        activeLink.classList.add('active');
        const sessionGroup = activeLink.closest('details.nav-session-group');
        if (sessionGroup) sessionGroup.open = true;
      }}
      }}, {{ rootMargin: '-20% 0px -65% 0px', threshold: [0.1, 0.25, 0.5, 0.75] }});
      map.forEach((_link, target) => observer.observe(target));
    }}
    window.addEventListener('DOMContentLoaded', () => {{
      setupActiveNav();
    }});
  </script>
</body>
</html>
"""


def convert_markdown_shell_to_html(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    body = "\n".join(lines).strip()
    return "\n".join(f"      {line}" if line else "" for line in body.splitlines())


def build_navigation_html(body_html: str) -> str:
    sessions = re.findall(
        r'<meta class="nav-session" data-anchor="([^"]+)" data-label="([^"]+)"',
        body_html,
    )
    messages = re.findall(
        r'<meta class="nav-message" data-anchor="([^"]+)" data-role="([^"]+)" data-label="([^"]+)" data-session="([^"]+)"',
        body_html,
    )

    session_items = "".join(
        f'<li><a class="nav-link" href="#{html.escape(anchor)}">{html.escape(label)}</a></li>'
        for anchor, label in sessions
    )
    messages_by_session: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for anchor, role, label, session_anchor in messages:
        messages_by_session[session_anchor].append((anchor, role, label))

    grouped_message_sections: list[str] = []
    for index, (session_anchor, session_label) in enumerate(sessions, start=1):
        session_messages = messages_by_session.get(session_anchor, [])
        items = "".join(
            f'<li><a class="nav-link" href="#{html.escape(anchor)}"><span class="role-chip {html.escape(role)}">{html.escape(label)}</span>{html.escape(label)}</a></li>'
            for anchor, role, label in session_messages
        )
        grouped_message_sections.append(
            f'<details class="nav-session-group" {"open" if index == 1 else ""}>'
            f'<summary>'
            f'<span class="nav-session-chevron">&#8250;</span>'
            f'<span class="nav-session-label">{html.escape(session_label)}</span>'
            f'<span class="nav-session-count">{len(session_messages)} messages</span>'
            f'</summary>'
            f'<div class="nav-session-body"><ul class="nav-list">{items}</ul></div>'
            f'</details>'
        )
    grouped_message_html = "".join(grouped_message_sections)

    return f"""
      <div class="nav-group">
        <h2>Explore</h2>
        <input id="searchBox" class="search-input" type="search" placeholder="Search messages..." oninput="applyFilters()">
        <div class="filter-row">
          <button class="filter-chip active" data-role="all" onclick="filterByRole('all')">All</button>
          <button class="filter-chip" data-role="user" onclick="filterByRole('user')">User</button>
          <button class="filter-chip" data-role="assistant" onclick="filterByRole('assistant')">Assistant</button>
        </div>
      </div>
      <div class="nav-group">
        <h2>Controls</h2>
        <div class="control-grid">
          <button class="control-btn" onclick="setOpenBySelector('details.session-card', true)">Open Sessions</button>
          <button class="control-btn" onclick="setOpenBySelector('details.session-card', false)">Close Sessions</button>
          <button class="control-btn" onclick="setOpenByRole('user', true)">Open User</button>
          <button class="control-btn" onclick="setOpenByRole('user', false)">Close User</button>
          <button class="control-btn" onclick="setOpenByRole('assistant', true)">Open Assistant</button>
          <button class="control-btn" onclick="setOpenByRole('assistant', false)">Close Assistant</button>
        </div>
      </div>
      <div class="nav-group">
        <div class="nav-group-title">Sessions</div>
        <ul class="nav-list">{session_items}</ul>
      </div>
      <div class="nav-group">
        <div class="nav-group-title">Messages</div>
        {grouped_message_html}
      </div>
    """


def render_timeline_markdown(
    root: Path,
    jsonl_files: list[Path],
    merged_messages: list[MergedMessage],
    theme: dict[str, str],
    removed_fork_messages: int,
) -> str:
    time_range = format_timerange(merged_messages[0].start_time, merged_messages[-1].end_time)
    lines = [
        "# Extracted Chat History",
        "",
        render_summary_panel(
            title="Timeline View",
            rows=[
                ("Source folder", str(root)),
                ("JSONL files scanned", str(len(jsonl_files))),
                ("Time range", time_range),
                ("Fork-history duplicates removed", str(removed_fork_messages)),
                ("Messages written after merge", str(len(merged_messages))),
            ],
            theme=theme,
        ),
        "",
    ]

    for index, message in enumerate(merged_messages, start=1):
        lines.extend(render_message_block(root, message, index=index, theme=theme))

    return "\n".join(lines).rstrip() + "\n"


def render_session_markdown(
    root: Path,
    jsonl_files: list[Path],
    grouped_sessions: list[tuple[SessionInfo, list[MergedMessage]]],
    theme: dict[str, str],
    removed_fork_messages: int,
) -> str:
    lines = [
        "# Extracted Chat History",
        "",
        render_summary_panel(
            title="Session View",
            rows=[
                ("Source folder", str(root)),
                ("JSONL files scanned", str(len(jsonl_files))),
                ("Fork-history duplicates removed", str(removed_fork_messages)),
                ("Sessions written", str(len(grouped_sessions))),
            ],
            theme=theme,
        ),
        "",
    ]

    for session_index, (session, messages) in enumerate(grouped_sessions, start=1):
        if session.start_time and messages:
            timerange = format_timerange(messages[0].start_time, messages[-1].end_time)
        else:
            timerange = "unknown"
        session_sources = collect_session_sources(root, messages)
        session_label = f"Session {session_index}"
        session_anchor = f"session-{session_index}"
        lines.append(f"<meta class=\"nav-session\" data-anchor=\"{session_anchor}\" data-label=\"{html.escape(session_label, quote=True)}\">")
        lines.append(f"<details id=\"{session_anchor}\" class=\"session-card\" open>")
        lines.append("<summary>")
        lines.append(
            render_summary_panel(
                title=session_label,
                rows=[
                    ("Session ID", session.session_id),
                    ("Source files", session_sources),
                    ("Time range", timerange),
                    ("Messages written after merge", str(len(messages))),
                ],
                theme=theme,
                collapsible=True,
            )
        )
        lines.append("</summary>")
        lines.append("")

        for message_index, message in enumerate(messages, start=1):
            lines.extend(
                render_message_block(
                    root,
                    message,
                    index=message_index,
                    theme=theme,
                    session_anchor=session_anchor,
                )
            )

        lines.append("</details>")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_message_block(
    root: Path,
    message: MergedMessage,
    index: int,
    theme: dict[str, str],
    session_anchor: str | None = None,
) -> list[str]:
    title = ROLE_TITLES.get(message.role, message.role.title())
    badge = ROLE_BADGES.get(message.role, f"[{title.upper()}]")
    body = format_message_body_html(message.text, theme)
    card_style = get_card_style(message.role, theme)
    meta_row = [
        ("Time", format_timerange(message.start_time, message.end_time)),
        ("Source", format_sources(root, message.sources)),
    ]

    if len(message.session_ids) == 1:
        meta_row.append(("Session", message.session_ids[0]))
    else:
        meta_row.append(("Sessions", ", ".join(message.session_ids)))

    anchor_id = f"msg-{message.role}-{index}-{slugify(format_timerange(message.start_time, message.end_time))[:24]}"
    message_label = f"{index}. {title}"
    search_text = html.escape(" ".join([title, message.text, " ".join(message.session_ids)]), quote=True)
    session_anchor = session_anchor or "timeline"
    return [
        "",
        f"<meta class=\"nav-message\" data-anchor=\"{anchor_id}\" data-role=\"{message.role}\" data-label=\"{html.escape(message_label, quote=True)}\" data-session=\"{session_anchor}\">",
        (
            f"<details id=\"{anchor_id}\" class=\"message-card\" data-role=\"{message.role}\" data-search=\"{search_text}\" open>"
            f"<summary style=\"{card_style}padding-bottom:14px;\">"
            f"<div style=\"display:flex;align-items:center;gap:10px;\">"
            f"<span style=\"font-size:12px;font-weight:800;letter-spacing:0.08em;padding:4px 10px;border-radius:999px;{get_badge_style(message.role, theme)}\">"
            f"{html.escape(badge)}</span>"
            f"<span style=\"font-size:26px;font-weight:800;color:{theme['title_text']};\">{index}. {html.escape(title)}</span>"
            f"<span class=\"summary-chevron\">&#8250;</span>"
            f"</div></summary>"
            f"<div style=\"{card_style}margin-top:-8px;\">"
            f"{render_meta_grid(meta_row, theme)}"
            f"<div class=\"message-body\" style=\"margin-top:16px;font-size:{get_body_font_size(message.role)};line-height:1.75;"
            f"color:{theme['body_text']};white-space:pre-wrap;\">{body}</div>"
            f"</div>"
            f"</details>"
        ),
        "",
    ]


def render_summary_panel(
    title: str,
    rows: list[tuple[str, str]],
    theme: dict[str, str],
    collapsible: bool = False,
) -> str:
    items = "".join(
        (
            "<div style=\"margin:8px 0;\">"
            f"<div style=\"font-size:12px;font-weight:700;letter-spacing:0.08em;color:{theme['summary_label']};text-transform:uppercase;\">"
            f"{html.escape(label)}</div>"
            f"<div style=\"font-size:16px;font-weight:600;color:{theme['summary_value']};margin-top:2px;\">{html.escape(value)}</div>"
            "</div>"
        )
        for label, value in rows
    )
    panel = (
        f"<div style=\"border:1px solid {theme['summary_border']};border-left:6px solid {theme['summary_accent']};background:{theme['summary_bg']};"
        "border-radius:16px;padding:20px 22px;margin:8px 0 24px 0;\">"
        f"<div style=\"display:flex;align-items:center;gap:10px;\">"
        f"<span style=\"font-size:28px;font-weight:800;color:{theme['summary_title']};margin-bottom:10px;\">{html.escape(title)}</span>"
        f"{'<span class=\"summary-chevron\">&#8250;</span>' if collapsible else ''}"
        f"</div>"
        f"{items}"
        "</div>"
    )
    return panel


def render_meta_grid(rows: list[tuple[str, str]], theme: dict[str, str]) -> str:
    items = "".join(
        (
            f"<div style=\"min-width:180px;flex:1 1 220px;background:{theme['meta_tile_bg']};"
            "border-radius:12px;padding:10px 12px;\">"
            f"<div style=\"font-size:11px;font-weight:800;letter-spacing:0.08em;color:{theme['summary_label']};text-transform:uppercase;\">"
            f"{html.escape(label)}</div>"
            f"<div style=\"font-size:15px;font-weight:600;color:{theme['meta_value']};margin-top:4px;\">{html.escape(value)}</div>"
            "</div>"
        )
        for label, value in rows
    )
    return f"<div style=\"display:flex;flex-wrap:wrap;gap:10px;\">{items}</div>"


def format_message_body_html(text: str, theme: dict[str, str]) -> str:
    rendered: list[str] = []
    for block_type, payload in split_rich_text(text):
        if block_type == "code":
            language, code = payload
            rendered.append(render_code_block_html(language, code, theme))
            continue
        if block_type == "table":
            rendered.append(render_markdown_table_html(payload, theme))
            continue

        paragraphs = payload.split("\n\n")
        for paragraph in paragraphs:
            rendered.append(
                f"<p style=\"margin:0 0 14px 0;\">{render_inline_html(paragraph, theme).replace(chr(10), '<br>')}</p>"
            )

    return "".join(rendered)


def split_rich_text(text: str) -> list[tuple[str, Any]]:
    parts: list[tuple[str, Any]] = []
    pattern = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
    last_end = 0

    for match in pattern.finditer(text):
        prose = text[last_end:match.start()].strip()
        if prose:
            parts.extend(split_text_and_tables(prose))

        language = match.group(1).strip() or "text"
        code = match.group(2).rstrip()
        parts.append(("code", (language, code)))
        last_end = match.end()

    tail = text[last_end:].strip()
    if tail:
        parts.extend(split_text_and_tables(tail))

    if not parts:
        parts.extend(split_text_and_tables(text))

    return parts


def render_code_block_html(language: str, code: str, theme: dict[str, str]) -> str:
    escaped_language = html.escape(language)
    escaped_code = html.escape(code)
    return (
        f"<div class=\"code-block\" style=\"margin:10px 0 18px 0;border:1px solid {theme['code_border']};border-radius:14px;overflow:hidden;"
        "box-shadow:inset 0 1px 0 rgba(255,255,255,0.5);\">"
        f"<div style=\"display:flex;align-items:center;gap:10px;background:{theme['code_header_bg']};color:{theme['code_header_fg']};font-size:12px;font-weight:800;letter-spacing:0.08em;"
        "text-transform:uppercase;padding:9px 12px;\">"
        f"<span>{escaped_language}</span>"
        f"<button class=\"copy-btn\" type=\"button\" onclick=\"copyCode(this)\">Copy</button></div>"
        f"<pre style=\"margin:0;background:{theme['code_bg']};color:{theme['code_fg']};padding:16px 18px;overflow:auto;"
        "font-size:14px;line-height:1.6;\"><code>"
        f"{escaped_code}</code></pre></div>"
    )


def get_card_style(role: str, theme: dict[str, str]) -> str:
    if role == "assistant":
        return (
            f"border:1px solid {theme['assistant_border']};border-left:8px solid {theme['assistant_accent']};background:{theme['assistant_bg']};"
            "border-radius:18px;padding:22px 24px;margin:14px 0 22px 0;"
            f"box-shadow:0 6px 18px {theme['assistant_shadow']};"
        )
    return (
        f"border:1px solid {theme['user_border']};border-left:8px solid {theme['user_accent']};background:{theme['user_bg']};"
        "border-radius:18px;padding:22px 24px;margin:14px 0 22px 0;"
        f"box-shadow:0 6px 18px {theme['user_shadow']};"
    )


def get_badge_style(role: str, theme: dict[str, str]) -> str:
    if role == "assistant":
        return f"background:{theme['assistant_badge_bg']};color:{theme['assistant_badge_fg']};"
    return f"background:{theme['user_badge_bg']};color:{theme['user_badge_fg']};"


def split_text_and_tables(text: str) -> list[tuple[str, Any]]:
    blocks: list[tuple[str, Any]] = []
    current: list[str] = []
    lines = text.splitlines()
    index = 0

    while index < len(lines):
        if is_markdown_table_start(lines, index):
            if current:
                blocks.append(("text", "\n".join(current).strip()))
                current = []
            table_lines = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index]:
                table_lines.append(lines[index])
                index += 1
            blocks.append(("table", table_lines))
            continue

        current.append(lines[index])
        index += 1

    if current:
        blocks.append(("text", "\n".join(current).strip()))

    return [block for block in blocks if block[1]]


def is_markdown_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    header = lines[index].strip()
    separator = lines[index + 1].strip()
    if "|" not in header or "|" not in separator:
        return False
    return bool(re.fullmatch(r"\|?[\s:\-|\t]+\|?", separator))


def render_markdown_table_html(lines: list[str], theme: dict[str, str]) -> str:
    rows = [parse_markdown_table_row(line) for line in lines]
    if len(rows) < 2:
        return ""
    headers = rows[0]
    body_rows = rows[2:]
    header_html = "".join(
        f"<th style=\"border:1px solid {theme['table_border']};padding:10px 12px;background:{theme['table_header_bg']};"
        f"color:{theme['table_header_fg']};font-size:14px;text-align:left;\">{render_inline_html(cell, theme)}</th>"
        for cell in headers
    )
    body_html = []
    for idx, row in enumerate(body_rows):
        bg = theme["table_row_alt"] if idx % 2 == 0 else "#ffffff"
        cells = "".join(
            f"<td style=\"border:1px solid {theme['table_border']};padding:10px 12px;background:{bg};font-size:14px;vertical-align:top;\">"
            f"{render_inline_html(cell, theme)}</td>"
            for cell in row
        )
        body_html.append(f"<tr>{cells}</tr>")
    return (
        "<div style=\"margin:10px 0 18px 0;overflow:auto;\">"
        f"<table style=\"border-collapse:collapse;min-width:420px;width:100%;border:1px solid {theme['table_border']};\">"
        f"<thead><tr>{header_html}</tr></thead><tbody>{''.join(body_html)}</tbody></table></div>"
    )


def parse_markdown_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def render_inline_html(text: str, theme: dict[str, str]) -> str:
    chunks: list[str] = []
    last_end = 0
    for match in re.finditer(r"`([^`]+)`", text):
        chunks.append(html.escape(text[last_end:match.start()]))
        chunks.append(
            f"<code style=\"background:{theme['inline_code_bg']};color:{theme['inline_code_fg']};"
            f"border:1px solid {theme['inline_code_border']};border-radius:6px;padding:1px 6px;"
            f"font-size:0.92em;font-family:SFMono-Regular,Consolas,'Liberation Mono',Menlo,monospace;\">"
            f"{html.escape(match.group(1))}</code>"
        )
        last_end = match.end()
    chunks.append(html.escape(text[last_end:]))
    return "".join(chunks)


def get_body_font_size(role: str) -> str:
    if role == "assistant":
        return "17px"
    return "18px"


def group_messages_by_session(
    sessions: list[SessionInfo],
    events: list[MessageEvent],
) -> list[tuple[SessionInfo, list[MergedMessage]]]:
    by_session: dict[str, list[MessageEvent]] = defaultdict(list)
    for event in events:
        by_session[event.session_id].append(event)

    session_map = {session.session_id: session for session in sessions}
    grouped: list[tuple[SessionInfo, list[MergedMessage]]] = []

    for session_id, session_events in by_session.items():
        session_events.sort(key=lambda event: (event.timestamp, str(event.path), event.line_number))
        session = session_map[session_id]
        grouped.append((session, merge_consecutive_messages(session_events)))

    grouped.sort(
        key=lambda item: (
            item[0].start_time or item[1][0].start_time,
            str(item[0].path),
        )
    )
    return grouped


def collect_session_sources(root: Path, messages: list[MergedMessage]) -> str:
    seen: list[Path] = []
    for message in messages:
        for path in message.sources:
            if path not in seen:
                seen.append(path)
    return format_sources(root, seen)


def format_sources(root: Path, sources: list[Path]) -> str:
    labels = [str(path.relative_to(root)) for path in sources]
    if len(labels) <= 3:
        return ", ".join(labels)
    head = ", ".join(labels[:3])
    return f"{head}, ... ({len(labels)} files)"


def format_timerange(start: datetime, end: datetime) -> str:
    start_label = format_timestamp(start)
    end_label = format_timestamp(end)
    if start_label == end_label:
        return start_label
    return f"{start_label} -> {end_label}"


def format_timestamp(value: datetime) -> str:
    local_value = value.astimezone()
    return local_value.strftime("%Y-%m-%d %H:%M:%S %Z")


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_md = args.output_md.resolve()
    theme = THEMES[args.theme]

    if not input_dir.exists():
        raise SystemExit(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise SystemExit(f"Input path is not a folder: {input_dir}")

    jsonl_files = iter_jsonl_files(input_dir)
    if not jsonl_files:
        raise SystemExit(f"No .jsonl files found under: {input_dir}")

    sessions: list[SessionInfo] = []
    events: list[MessageEvent] = []
    for path in jsonl_files:
        session, file_events = extract_file_data(path)
        sessions.append(session)
        events.extend(file_events)

    if not events:
        raise SystemExit("No meaningful user/assistant chat messages were found in the JSONL files.")

    events.sort(key=lambda event: (event.timestamp, str(event.path), event.line_number))
    events, removed_by_session = remove_forked_history(sessions, events)
    removed_fork_messages = sum(removed_by_session.values())
    merged_messages = merge_consecutive_messages(events)
    grouped_sessions = group_messages_by_session(sessions, events)

    rendered = render_markdown(
        input_dir,
        jsonl_files,
        merged_messages,
        grouped_sessions,
        args.group_by,
        theme,
        removed_fork_messages,
    )
    if args.output_html:
        rendered = render_html_document(rendered, args.theme, theme)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
