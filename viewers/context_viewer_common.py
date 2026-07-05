"""Shared helpers for clip-level context viewers."""

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq
import streamlit as st

from vidaforge.common import join_data_dir
from vidaforge.index import resolve_parquet_paths

try:
    from .coerce import coerce_float, coerce_int
except ImportError:
    from coerce import coerce_float, coerce_int

LARGE_SUMMARY_SKIP_KEYS = {"task_outputs", "failed_task_outputs", "failed_examples"}
CONTACT_SHEET_THUMB_HEIGHT = 120
CONTACT_SHEET_PADDING = 8
CONTACT_SHEET_LABEL_HEIGHT = 22


def format_seconds(value: object) -> str:
    number = coerce_float(value)
    return "-" if number is None else f"{number:.3f}s"


def resolve_context_shards(
    metadata_path: Path,
    summary: dict[str, object],
) -> tuple[tuple[str, int], ...]:
    metadata_root = metadata_path.expanduser().resolve()
    shards = summary.get("shards")
    resolved: list[tuple[str, int]] = []
    if isinstance(shards, list):
        for item in shards:
            if not isinstance(item, dict):
                continue
            shard_name = str(item.get("path") or "").strip()
            if not shard_name:
                continue
            shard_path = metadata_root / shard_name
            if not shard_path.exists() or not shard_path.is_file():
                continue
            rows = coerce_int(item.get("rows"), default=-1)
            if rows < 0:
                rows = pq.ParquetFile(shard_path).metadata.num_rows
            resolved.append((str(shard_path), rows))
    if resolved:
        return tuple(resolved)

    for path in resolve_parquet_paths(metadata_root, unit="clip"):
        resolved.append((str(path), pq.ParquetFile(path).metadata.num_rows))
    return tuple(resolved)


def total_rows_from_shards(shards: tuple[tuple[str, int], ...]) -> int:
    return sum(rows for _, rows in shards)


def _strip_top_level_json_key(text: str, key: str) -> str:
    marker = json.dumps(key)
    key_index = text.find(marker)
    if key_index < 0:
        return text

    colon_index = text.find(":", key_index + len(marker))
    if colon_index < 0:
        return text

    value_start = colon_index + 1
    while value_start < len(text) and text[value_start].isspace():
        value_start += 1
    if value_start >= len(text):
        return text

    stack: list[str] = []
    in_string = False
    escape = False
    value_end = value_start
    for index in range(value_start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "[{":
            stack.append("]" if char == "[" else "}")
            continue
        if stack:
            if char == stack[-1]:
                stack.pop()
                if not stack:
                    value_end = index + 1
                    break
            continue
        if char in ",\n\r}":
            value_end = index
            break

    remove_start = key_index
    while remove_start > 0 and text[remove_start - 1].isspace():
        remove_start -= 1
    if remove_start > 0 and text[remove_start - 1] == ",":
        remove_start -= 1
    else:
        value_end_with_comma = value_end
        while value_end_with_comma < len(text) and text[value_end_with_comma].isspace():
            value_end_with_comma += 1
        if value_end_with_comma < len(text) and text[value_end_with_comma] == ",":
            value_end = value_end_with_comma + 1

    return text[:remove_start] + text[value_end:]


@st.cache_data(show_spinner=False)
def load_summary_light(summary_path: str) -> dict[str, object]:
    path = Path(summary_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    for key in LARGE_SUMMARY_SKIP_KEYS:
        text = _strip_top_level_json_key(text, key)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


@st.cache_data(show_spinner=False)
def load_rows_by_indices(
    shards: tuple[tuple[str, int], ...],
    *,
    row_indices: tuple[int, ...],
    row_index_key: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not row_indices:
        return rows

    requested = set(index for index in row_indices if index >= 0)
    shard_start = 0
    for shard_path, shard_rows in shards:
        shard_end = shard_start + shard_rows
        global_to_local = {
            index: index - shard_start
            for index in requested
            if shard_start <= index < shard_end
        }
        if global_to_local:
            frame = pq.read_table(shard_path).to_pandas()
            for global_index in sorted(global_to_local):
                local_index = global_to_local[global_index]
                if local_index < len(frame):
                    row = frame.iloc[local_index].to_dict()
                    row[row_index_key] = global_index
                    rows.append(row)
        shard_start = shard_end
    return rows


def page_indices(*, total_rows: int, page_number: int, per_page: int) -> tuple[int, ...]:
    start = max(0, (page_number - 1) * per_page)
    end = min(total_rows, start + per_page)
    return tuple(range(start, end))


def parse_json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def resolve_frame_paths(
    frame_json: dict[str, Any],
) -> list[Path]:
    raw_paths = frame_json.get("frame_paths")
    paths: list[Path] = []
    if isinstance(raw_paths, list):
        for raw_path in raw_paths:
            path_text = str(raw_path or "").strip()
            if not path_text:
                continue
            paths.append(join_data_dir(path_text))
    return paths


def frame_timestamps(frame_json: dict[str, Any]) -> list[float | None]:
    raw_timestamps = frame_json.get("timestamps_sec")
    if not isinstance(raw_timestamps, list):
        return []
    return [coerce_float(value) for value in raw_timestamps]


def resolve_audio_path(row: dict[str, object]) -> Path | None:
    audio_json = parse_json_object(row.get("audio_json"))
    paths = audio_json.get("audio_paths")
    if not isinstance(paths, list) or not paths:
        return None

    path = join_data_dir(str(paths[0]))
    return path if path.exists() and path.is_file() else None


@st.cache_data(show_spinner=False)
def build_contact_sheet_bytes(
    frame_paths: tuple[str, ...],
    timestamps: tuple[float | None, ...],
    *,
    thumb_height: int = CONTACT_SHEET_THUMB_HEIGHT,
) -> tuple[bytes | None, int]:
    images: list[tuple[Image.Image, str]] = []
    missing = 0
    for index, path_text in enumerate(frame_paths):
        path = Path(path_text)
        try:
            image = Image.open(path).convert("RGB")
        except OSError:
            missing += 1
            continue

        scale = float(thumb_height) / float(image.height)
        thumb_width = max(1, int(round(image.width * scale)))
        resized = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        timestamp = timestamps[index] if index < len(timestamps) else None
        label = f"#{index:03d}"
        if timestamp is not None:
            label = f"{label} {timestamp:.3f}s"
        images.append((resized, label))

    if not images:
        return None, missing

    padding = CONTACT_SHEET_PADDING
    label_height = CONTACT_SHEET_LABEL_HEIGHT
    sheet_width = padding + sum(image.width + padding for image, _ in images)
    sheet_height = padding + thumb_height + label_height + padding
    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(245, 245, 242))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    x = padding
    for image, label in images:
        sheet.paste(image, (x, padding))
        draw.text((x, padding + thumb_height + 5), label, fill=(40, 40, 40), font=font)
        x += image.width + padding

    buffer = BytesIO()
    sheet.save(buffer, format="JPEG", quality=88, optimize=True)
    return buffer.getvalue(), missing


def render_frame_strip(
    frame_paths: list[Path],
    *,
    timestamps: list[float | None],
) -> None:
    if not frame_paths:
        st.warning("Current clip has no frames to display.")
        return

    sheet_bytes, missing = build_contact_sheet_bytes(
        tuple(str(path) for path in frame_paths),
        tuple(timestamps),
    )
    if sheet_bytes is None:
        st.warning("Current clip has no readable frame files.")
        return
    encoded = base64.b64encode(sheet_bytes).decode("ascii")
    st.markdown(
        f"""
<div style="overflow-x: auto; overflow-y: hidden; padding: 6px 0 10px 0;">
  <img
    src="data:image/jpeg;base64,{encoded}"
    style="display: block; max-width: none; height: auto; border: 1px solid rgba(49, 51, 63, 0.18); border-radius: 6px;"
  />
</div>
""",
        unsafe_allow_html=True,
    )
    if missing:
        st.caption(f"{missing} frame files are missing or unreadable and were skipped.")


def render_audio_player(row: dict[str, object]) -> None:
    audio_ok = coerce_int(row.get("audio_ok"))
    audio_error = str(row.get("audio_error") or "").strip()
    audio_path = resolve_audio_path(row)

    if audio_path is not None:
        st.audio(str(audio_path))
        st.caption(str(audio_path))
        return

    if audio_ok:
        st.warning("audio_ok=1, but the audio file does not exist or is unreadable.")
    elif audio_error:
        st.caption(f"audio unavailable: {audio_error}")


def render_clip_video(row: dict[str, object], *, missing_label: str = "Current row is missing `clip_path`.") -> None:
    clip_path = str(row.get("clip_path") or "").strip()
    if not clip_path:
        st.warning(missing_label)
        return

    path = join_data_dir(clip_path)
    if not path.exists() or not path.is_file():
        st.error(f"clip file does not exist: {clip_path}")
        return

    try:
        st.video(str(path), autoplay=False)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"clip file cannot be opened and was skipped: {clip_path}")
        st.caption(str(exc))
        return
    st.caption(clip_path)


def context_row_detail(row: dict[str, object], *, frame_json: dict[str, Any]) -> dict[str, object]:
    fields = (
        "clip_id",
        "video_id",
        "clip_path",
        "duration_sec",
        "width",
        "height",
        "fps",
        "audio_ok",
        "audio_error",
        "input_run_id",
        "run_id",
        "context_ok",
        "context_error",
        "frame_ok",
        "frame_error",
    )
    detail = {field: row.get(field) for field in fields if field in row and pd.notna(row.get(field))}
    detail["frame"] = {
        key: frame_json.get(key)
        for key in (
            "sampling_method",
            "sampled_fps",
            "sampled_frame_count",
            "frame_width",
            "frame_height",
        )
    }
    audio_json = parse_json_object(row.get("audio_json"))
    if audio_json:
        detail["audio"] = audio_json
    return detail
