"""Helpers for media asset names and structured asset metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Mapping


_SAFE_FILE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_file_name(value: str, *, default: str) -> str:
    safe = _SAFE_FILE_NAME_RE.sub("_", value.strip()).strip("._-")
    if safe:
        return safe
    safe_default = _SAFE_FILE_NAME_RE.sub("_", default.strip()).strip("._-")
    if not safe_default:
        raise ValueError("default must produce a non-empty safe file name")
    return safe_default


def video_id_from_raw_path(
    *,
    source: str,
    source_batch: str,
    raw_path: str | Path,
    raw_member_path: str | Path = "",
) -> str:
    member_text = str(raw_member_path)
    key_parts = [source, source_batch, str(raw_path)]
    if member_text:
        key_parts.append(member_text)
    key = "\n".join(key_parts)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()
    return f"video-{digest}"


def hash_bucketed_path(
    root: str | Path,
    key: str,
    *,
    depth: int = 2,
    width: int = 2,
) -> Path:
    if depth <= 0:
        raise ValueError("depth must be > 0")
    if width <= 0:
        raise ValueError("width must be > 0")

    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    path = Path(root)
    for index in range(depth):
        path = path / digest[index * width : (index + 1) * width]
    return path / key


def frame_timestamps_from_json(
    frame_json: Mapping[str, object],
    *,
    expected_count: int,
) -> tuple[float, ...]:
    """Read selected frame timestamps and pad missing values with -1.0."""
    raw_timestamps = frame_json["timestamps_sec"]
    if not isinstance(raw_timestamps, list):
        raise TypeError("frame_json.timestamps_sec must be a list")
    timestamps: list[float] = []
    for value in raw_timestamps:
        timestamps.append(round(float(value), 6))
    if len(timestamps) < expected_count:
        timestamps.extend([-1.0] * (expected_count - len(timestamps)))
    return tuple(timestamps[:expected_count])


def frame_timeline_text(frame_count: int, timestamps_sec: tuple[float, ...]) -> str:
    lines: list[str] = []
    for index in range(frame_count):
        timestamp = timestamps_sec[index] if index < len(timestamps_sec) else -1.0
        timestamp_text = "unknown" if timestamp < 0 else f"{timestamp:.3f}s"
        lines.append(f"- frame_{index:04d}: {timestamp_text}")
    return "\n".join(lines)
