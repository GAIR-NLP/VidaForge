from __future__ import annotations

import math


def valid_frame_count_at_or_below(frame_count: int, *, stride: int) -> int:
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    return ((int(frame_count) - 1) // int(stride)) * int(stride) + 1


def duration_bucket_frame_counts(
    durations_sec: list[float],
    *,
    fps: float,
    stride: int,
) -> list[int]:
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError(f"fps must be finite and > 0, got {fps!r}")
    if not durations_sec:
        raise ValueError("bucket.durations_sec must not be empty")

    frame_counts: list[int] = []
    for duration_sec in durations_sec:
        duration = float(duration_sec)
        if duration <= 0 or not math.isfinite(duration):
            raise ValueError(
                f"bucket.durations_sec values must be finite and > 0, got {duration_sec!r}"
            )
        target_frame_count = max(1, int(math.floor(duration * fps + 1e-9)))
        valid_frame_count = valid_frame_count_at_or_below(
            target_frame_count,
            stride=stride,
        )
        if valid_frame_count not in frame_counts:
            frame_counts.append(valid_frame_count)

    frame_counts.sort()
    return frame_counts


def select_bucket_frame_count(
    *,
    source_duration_sec: float,
    fps: float,
    durations_sec: list[float],
    stride: int,
) -> int:
    duration = float(source_duration_sec)
    if duration <= 0 or not math.isfinite(duration):
        raise ValueError(f"duration_sec must be finite and > 0, got {source_duration_sec!r}")

    input_frame_count = int(math.floor(duration * fps + 1e-9))
    if input_frame_count < 1:
        raise ValueError(
            f"duration_sec and fps produce no input frames: "
            f"duration_sec={source_duration_sec!r}, fps={fps!r}"
        )

    candidates = duration_bucket_frame_counts(
        durations_sec,
        fps=fps,
        stride=stride,
    )
    eligible = [frame_count for frame_count in candidates if frame_count <= input_frame_count]
    if not eligible:
        raise ValueError(
            "clip is shorter than the smallest temporal bucket: "
            f"input_frame_count={input_frame_count}, "
            f"min_bucket_frame_count={candidates[0]}, fps={fps}"
        )
    return eligible[-1]


__all__ = [
    "duration_bucket_frame_counts",
    "select_bucket_frame_count",
    "valid_frame_count_at_or_below",
]
