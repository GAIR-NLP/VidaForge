from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


@dataclass(frozen=True, slots=True)
class FFmpegMotionResult:
    vmafmotion_mean: float
    vmafmotion_frame_count: int
    scdet_score_max: float
    scdet_cut_ratio: float
    scdet_frame_count: int


_VMAFMOTION_PATTERN = re.compile(r"\bmotion:(?P<value>[-+0-9.eE]+)\b")
_SCDET_SCORE_PATTERN = re.compile(r"^lavfi\.scd\.score=(?P<value>[-+0-9.eE]+)$")


def _read_vmafmotion_log(log_path: Path) -> list[float]:
    values: list[float] = []
    for line in log_path.read_text().splitlines():
        match = _VMAFMOTION_PATTERN.search(line)
        if match is not None:
            values.append(float(match.group("value")))
    if not values:
        raise RuntimeError("vmafmotion did not produce frame scores.")
    return values


def _read_scdet_log(log_path: Path) -> list[float]:
    values: list[float] = []
    for line in log_path.read_text().splitlines():
        match = _SCDET_SCORE_PATTERN.match(line.strip())
        if match is not None:
            values.append(float(match.group("value")))
    if not values:
        raise RuntimeError("scdet did not produce frame scores.")
    return values


def run_ffmpeg_motion(
    video_path: str | Path,
    *,
    scdet_threshold: float = 10.0,
    scdet_sample_fps: float = 8.0,
    scdet_sample_width: int = 320,
    ffmpeg_bin: str = "ffmpeg",
) -> FFmpegMotionResult:
    if not 0 <= scdet_threshold <= 100:
        raise ValueError("scdet_threshold must be in [0, 100].")
    if scdet_sample_fps <= 0:
        raise ValueError("scdet_sample_fps must be > 0.")
    if scdet_sample_width <= 0:
        raise ValueError("scdet_sample_width must be > 0.")

    with tempfile.TemporaryDirectory(prefix="motion_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        vmafmotion_log_path = tmp_path / "vmafmotion.log"
        scdet_log_path = tmp_path / "scdet.log"
        filter_graph = (
            "[0:v]split=2[vmaf_in][scd_in];"
            f"[vmaf_in]vmafmotion=stats_file={vmafmotion_log_path}[vmaf_out];"
            f"[scd_in]scale={scdet_sample_width}:-2,"
            f"fps={scdet_sample_fps:g},"
            f"scdet=threshold={scdet_threshold:g},"
            f"metadata=mode=print:file={scdet_log_path}[scd_out]"
        )
        video = Path(video_path).expanduser().resolve()
        completed = subprocess.run(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-v",
                "error",
                "-i",
                str(video),
                "-filter_complex",
                filter_graph,
                "-map",
                "[vmaf_out]",
                "-map",
                "[scd_out]",
                "-an",
                "-f",
                "null",
                "-",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.strip() or "ffmpeg failed"
            raise RuntimeError(f"ffmpeg motion error for {video}: {message}")
        vmafmotion_values = _read_vmafmotion_log(vmafmotion_log_path)
        scdet_scores = _read_scdet_log(scdet_log_path)

    vmafmotion_array = np.asarray(vmafmotion_values, dtype=np.float64)
    scdet_array = np.asarray(scdet_scores, dtype=np.float64)
    return FFmpegMotionResult(
        vmafmotion_mean=round(float(vmafmotion_array.mean()), 6),
        vmafmotion_frame_count=len(vmafmotion_values),
        scdet_score_max=round(float(scdet_array.max()), 6),
        scdet_cut_ratio=round(
            float(np.mean(scdet_array >= float(scdet_threshold))),
            6,
        ),
        scdet_frame_count=len(scdet_scores),
    )
