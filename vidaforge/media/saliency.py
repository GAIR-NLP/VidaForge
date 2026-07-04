from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import NamedTuple

import cv2
import numpy as np


TIMESTAMP_EPSILON_SEC = 1e-6


class ContentChangeWeights(NamedTuple):
    delta_hue: float = 1.0
    delta_sat: float = 1.0
    delta_lum: float = 1.0


DEFAULT_CONTENT_CHANGE_WEIGHTS = ContentChangeWeights()


@dataclass(slots=True)
class ContentChangeScore:
    frame_index: int
    timestamp_sec: float
    score: float
    delta_hue: float
    delta_sat: float
    delta_lum: float


@dataclass(slots=True)
class ContentChangeSampling:
    timestamps_sec: list[float]
    selected_frame_indices: list[int]
    decoded_frame_count: int
    analysis_width: int
    analysis_height: int
    scores: list[ContentChangeScore]


def _mean_pixel_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return mean absolute 8-bit pixel distance for two single-channel frames."""
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("left and right must be 2D single-channel frames.")
    if left.shape != right.shape:
        raise ValueError("left and right must have the same shape.")
    num_pixels = float(left.shape[0] * left.shape[1])
    return float(np.sum(np.abs(left.astype(np.int32) - right.astype(np.int32))) / num_pixels)


def _scaled_size(*, source_width: int, source_height: int, short_side: int) -> tuple[int, int]:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source_width and source_height must be > 0.")
    if short_side <= 0:
        raise ValueError("short_side must be > 0.")

    if source_width <= source_height:
        width = short_side
        height = int(round(source_height * short_side / source_width))
    else:
        height = short_side
        width = int(round(source_width * short_side / source_height))
    if width % 2:
        width += 1
    if height % 2:
        height += 1
    return width, height


def build_ffmpeg_decode_frames_cmd(
    input_path: str | Path,
    *,
    short_side: int,
    ffmpeg_bin: str = "ffmpeg",
    gpu_device: int | None = None,
) -> list[str]:
    """Build a command that decodes all video frames to resized BGR rawvideo."""
    if short_side <= 0:
        raise ValueError("short_side must be > 0.")

    input_file = Path(input_path).expanduser().resolve()
    scale_filter = (
        f"scale='if(lte(iw,ih),{short_side},-2)':"
        f"'if(lte(iw,ih),-2,{short_side})'"
    )
    cmd = [ffmpeg_bin, "-v", "error", "-nostdin"]
    if gpu_device is not None:
        cmd.extend(["-hwaccel", "cuda", "-hwaccel_device", str(gpu_device)])
    cmd.extend(
        [
            "-i",
            str(input_file),
            "-map",
            "0:v:0",
            "-vf",
            scale_filter,
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
    )
    return cmd


def decode_video_frames_bgr(
    input_path: str | Path,
    *,
    source_width: int,
    source_height: int,
    short_side: int,
    ffmpeg_bin: str = "ffmpeg",
    gpu_device: int | None = None,
    timeout_sec: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """Decode all frames into a low-resolution BGR ndarray of shape N,H,W,3."""
    width, height = _scaled_size(
        source_width=int(source_width),
        source_height=int(source_height),
        short_side=int(short_side),
    )
    cmd = build_ffmpeg_decode_frames_cmd(
        input_path,
        short_side=short_side,
        ffmpeg_bin=ffmpeg_bin,
        gpu_device=gpu_device,
    )
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_sec,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode(errors="replace").strip() or "ffmpeg failed"
        raise RuntimeError(f"ffmpeg frame analysis decode error for {input_path}: {message}")

    frame_bytes = width * height * 3
    if frame_bytes <= 0:
        raise ValueError("decoded frame size must be > 0.")
    if not completed.stdout:
        raise RuntimeError(f"ffmpeg decoded no analysis frames for {input_path}")
    if len(completed.stdout) % frame_bytes != 0:
        raise RuntimeError(
            "ffmpeg rawvideo output size is not divisible by expected frame size "
            f"(bytes={len(completed.stdout)}, frame_bytes={frame_bytes})"
        )

    frame_count = len(completed.stdout) // frame_bytes
    frames = np.frombuffer(completed.stdout, dtype=np.uint8)
    return frames.reshape((frame_count, height, width, 3)), width, height


def compute_content_change_scores(
    frames_bgr: np.ndarray,
    *,
    fps: float,
    weights: ContentChangeWeights = DEFAULT_CONTENT_CHANGE_WEIGHTS,
) -> list[ContentChangeScore]:
    """Compute PySceneDetect-like HSV content-change scores for decoded frames."""
    if fps <= 0:
        raise ValueError("fps must be > 0.")
    if frames_bgr.ndim != 4 or frames_bgr.shape[-1] != 3:
        raise ValueError("frames_bgr must have shape N,H,W,3.")

    weight_sum = sum(abs(weight) for weight in weights)
    if weight_sum <= 0:
        raise ValueError("at least one content-change weight must be non-zero.")

    scores: list[ContentChangeScore] = []
    last_hue: np.ndarray | None = None
    last_sat: np.ndarray | None = None
    last_lum: np.ndarray | None = None

    for frame_index, frame_bgr in enumerate(frames_bgr):
        hue, sat, lum = cv2.split(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV))
        if last_hue is None or last_sat is None or last_lum is None:
            score = ContentChangeScore(
                frame_index=frame_index,
                timestamp_sec=round(frame_index / fps, 6),
                score=0.0,
                delta_hue=0.0,
                delta_sat=0.0,
                delta_lum=0.0,
            )
        else:
            delta_hue = _mean_pixel_distance(hue, last_hue)
            delta_sat = _mean_pixel_distance(sat, last_sat)
            delta_lum = _mean_pixel_distance(lum, last_lum)
            content_score = (
                delta_hue * weights.delta_hue
                + delta_sat * weights.delta_sat
                + delta_lum * weights.delta_lum
            ) / weight_sum
            score = ContentChangeScore(
                frame_index=frame_index,
                timestamp_sec=round(frame_index / fps, 6),
                score=round(content_score, 6),
                delta_hue=round(delta_hue, 6),
                delta_sat=round(delta_sat, 6),
                delta_lum=round(delta_lum, 6),
            )

        scores.append(score)
        last_hue = hue
        last_sat = sat
        last_lum = lum

    return scores


def _target_bin_ranges(*, duration_sec: float, sampled_fps: float) -> list[tuple[float, float]]:
    duration = float(duration_sec)
    fps = float(sampled_fps)
    if duration <= 0:
        raise ValueError("duration_sec must be > 0.")
    if fps <= 0:
        raise ValueError("sampled_fps must be > 0.")

    ranges: list[tuple[float, float]] = []
    step = 1.0 / fps
    start = 0.0
    while start < duration - TIMESTAMP_EPSILON_SEC:
        end = min(start + step, duration)
        ranges.append((start, end))
        start += step
    return ranges or [(0.0, duration)]


def select_content_change_timestamps(
    scores: list[ContentChangeScore],
    *,
    duration_sec: float,
    sampled_fps: float,
) -> tuple[list[float], list[int]]:
    """Select one high-change frame per temporal bin, with unique frame indices."""
    if not scores:
        raise ValueError("scores must not be empty.")

    selected: list[ContentChangeScore] = []
    selected_indices: set[int] = set()
    for start, end in _target_bin_ranges(duration_sec=duration_sec, sampled_fps=sampled_fps):
        candidates = [
            score
            for score in scores
            if score.frame_index not in selected_indices
            and start - TIMESTAMP_EPSILON_SEC <= score.timestamp_sec < end - TIMESTAMP_EPSILON_SEC
        ]
        if not candidates:
            center = (start + end) / 2.0
            candidates = [
                score
                for score in scores
                if score.frame_index not in selected_indices
            ]
            if not candidates:
                break
            best = min(candidates, key=lambda score: abs(score.timestamp_sec - center))
        else:
            best = max(candidates, key=lambda score: (score.score, -score.frame_index))
        selected.append(best)
        selected_indices.add(best.frame_index)

    selected.sort(key=lambda score: score.frame_index)
    return (
        [round(score.timestamp_sec, 6) for score in selected],
        [score.frame_index for score in selected],
    )


def sample_content_change_timestamps(
    input_path: str | Path,
    *,
    duration_sec: float,
    fps: float,
    sampled_fps: float,
    source_width: int,
    source_height: int,
    analysis_short_side: int = 128,
    ffmpeg_bin: str = "ffmpeg",
    gpu_device: int | None = None,
    timeout_sec: int | None = None,
) -> ContentChangeSampling:
    """Decode low-res frames and select high-change timestamps for frame sampling."""
    frames, width, height = decode_video_frames_bgr(
        input_path,
        source_width=source_width,
        source_height=source_height,
        short_side=analysis_short_side,
        ffmpeg_bin=ffmpeg_bin,
        gpu_device=gpu_device,
        timeout_sec=timeout_sec,
    )
    scores = compute_content_change_scores(frames, fps=float(fps))
    timestamps, frame_indices = select_content_change_timestamps(
        scores,
        duration_sec=float(duration_sec),
        sampled_fps=float(sampled_fps),
    )
    return ContentChangeSampling(
        timestamps_sec=timestamps,
        selected_frame_indices=frame_indices,
        decoded_frame_count=int(frames.shape[0]),
        analysis_width=width,
        analysis_height=height,
        scores=scores,
    )
