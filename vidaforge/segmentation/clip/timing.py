from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


_TIME_EPSILON_SEC = 1e-6


@dataclass(frozen=True, slots=True)
class ClipTiming:
    """Final clip timing selected by Stage 2 Step 2 policy."""

    start_sec: float
    end_sec: float
    detect_start_sec: float
    detect_end_sec: float
    clip_index: int
    split_index: int

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def detect_duration_sec(self) -> float:
        return max(0.0, self.detect_end_sec - self.detect_start_sec)

    def to_dict(self) -> dict[str, object]:
        return {
            "start_sec": round(float(self.start_sec), 6),
            "end_sec": round(float(self.end_sec), 6),
            "duration_sec": round(float(self.duration_sec), 6),
            "detect_start_sec": round(float(self.detect_start_sec), 6),
            "detect_end_sec": round(float(self.detect_end_sec), 6),
            "detect_duration_sec": round(float(self.detect_duration_sec), 6),
            "clip_index": self.clip_index,
            "split_index": self.split_index,
        }


def _build_clip_boundaries(
    ticks_sec: Iterable[object],
    *,
    video_duration_sec: float | None,
) -> list[float]:
    if video_duration_sec is not None:
        if video_duration_sec <= 0:
            raise ValueError("video_duration_sec must be > 0 when provided.")

    ticks: list[float] = []
    for value in ticks_sec:
        tick = float(value)
        if not math.isfinite(tick):
            raise ValueError(f"ticks_sec contains non-finite value: {value!r}")
        if tick < 0:
            raise ValueError(f"ticks_sec contains negative value: {value!r}")
        if (
            video_duration_sec is not None
            and tick > video_duration_sec + _TIME_EPSILON_SEC
        ):
            raise ValueError(
                f"ticks_sec contains value beyond video duration: {value!r}"
            )
        ticks.append(tick)
    if not ticks:
        return []

    merged: list[float] = []
    for tick in sorted(ticks):
        if merged and abs(tick - merged[-1]) <= _TIME_EPSILON_SEC:
            continue
        merged.append(round(float(tick), 6))
    return merged


def _split_detect_range(
    *,
    start_sec: float,
    end_sec: float,
    overlong_split_len_sec: float,
    first_clip_index: int,
) -> list[ClipTiming]:
    clip_timings: list[ClipTiming] = []
    cursor = start_sec
    split_index = 0

    while cursor < end_sec - _TIME_EPSILON_SEC:
        split_end_sec = min(cursor + overlong_split_len_sec, end_sec)
        clip_timings.append(
            ClipTiming(
                start_sec=round(cursor, 6),
                end_sec=round(split_end_sec, 6),
                detect_start_sec=round(start_sec, 6),
                detect_end_sec=round(end_sec, 6),
                clip_index=first_clip_index + split_index,
                split_index=split_index,
            )
        )
        cursor = split_end_sec
        split_index += 1

    return clip_timings


def _apply_boundary_trim(
    clip_timings: list[ClipTiming],
    *,
    boundary_trim_sec: float,
) -> list[ClipTiming]:
    if boundary_trim_sec <= _TIME_EPSILON_SEC:
        return clip_timings

    min_duration_to_trim_sec = (2 * boundary_trim_sec) + _TIME_EPSILON_SEC

    def can_trim_boundary(left: ClipTiming, right: ClipTiming) -> bool:
        is_detect_boundary = (
            abs(left.detect_start_sec - right.detect_start_sec) > _TIME_EPSILON_SEC
            or abs(left.detect_end_sec - right.detect_end_sec) > _TIME_EPSILON_SEC
        )
        if not is_detect_boundary:
            return False
        return (
            left.duration_sec > min_duration_to_trim_sec
            and right.duration_sec > min_duration_to_trim_sec
        )

    keep_timings: list[ClipTiming] = []
    for index, clip_timing in enumerate(clip_timings):
        start_sec = clip_timing.start_sec
        end_sec = clip_timing.end_sec
        if index > 0 and can_trim_boundary(
            clip_timings[index - 1],
            clip_timing,
        ):
            start_sec += boundary_trim_sec
        if index < len(clip_timings) - 1 and can_trim_boundary(
            clip_timing,
            clip_timings[index + 1],
        ):
            end_sec -= boundary_trim_sec
        if end_sec <= start_sec + _TIME_EPSILON_SEC:
            continue
        keep_timings.append(
            ClipTiming(
                start_sec=round(float(start_sec), 6),
                end_sec=round(float(end_sec), 6),
                detect_start_sec=clip_timing.detect_start_sec,
                detect_end_sec=clip_timing.detect_end_sec,
                clip_index=clip_timing.clip_index,
                split_index=clip_timing.split_index,
            )
        )
    return keep_timings


def build_clip_timings_from_ticks(
    ticks_sec: Iterable[object],
    *,
    video_duration_sec: float | None = None,
    min_len_sec: float = 1.0,
    max_len_sec: float = 10.0,
    overlong_split_len_sec: float = 10.0,
    boundary_trim_sec: float = 0.0,
) -> list[ClipTiming]:
    """Convert detected timeline ticks into final clip timings.

    Stage 2 Step 1 detects candidate boundaries. Stage 2 Step 2 keeps short
    clips as assets and only splits ranges that exceed the maximum duration.
    """
    if min_len_sec <= 0:
        raise ValueError("min_len_sec must be > 0.")
    if max_len_sec <= 0:
        raise ValueError("max_len_sec must be > 0.")
    if min_len_sec > max_len_sec:
        raise ValueError("min_len_sec must be <= max_len_sec.")
    if overlong_split_len_sec <= 0:
        raise ValueError("overlong_split_len_sec must be > 0.")
    if overlong_split_len_sec > max_len_sec:
        raise ValueError("overlong_split_len_sec must be <= max_len_sec.")
    if boundary_trim_sec < 0:
        raise ValueError("boundary_trim_sec must be >= 0.")

    video_duration = float(video_duration_sec) if video_duration_sec is not None else None
    if video_duration is not None and not math.isfinite(video_duration):
        raise ValueError("video_duration_sec must be finite when provided.")
    ticks = _build_clip_boundaries(ticks_sec, video_duration_sec=video_duration)
    if len(ticks) < 2:
        return []

    clip_timings: list[ClipTiming] = []
    for start_sec, end_sec in zip(ticks, ticks[1:]):
        duration_sec = end_sec - start_sec
        if duration_sec <= _TIME_EPSILON_SEC:
            continue
        if duration_sec <= max_len_sec + _TIME_EPSILON_SEC:
            clip_timings.append(
                ClipTiming(
                    start_sec=round(start_sec, 6),
                    end_sec=round(end_sec, 6),
                    detect_start_sec=round(start_sec, 6),
                    detect_end_sec=round(end_sec, 6),
                    clip_index=len(clip_timings),
                    split_index=0,
                )
            )
        else:
            clip_timings.extend(
                _split_detect_range(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    overlong_split_len_sec=overlong_split_len_sec,
                    first_clip_index=len(clip_timings),
                )
            )

    trimmed_timings = _apply_boundary_trim(
        clip_timings,
        boundary_trim_sec=boundary_trim_sec,
    )
    return [
        clip_timing
        for clip_timing in trimmed_timings
        if clip_timing.duration_sec >= min_len_sec - _TIME_EPSILON_SEC
    ]
