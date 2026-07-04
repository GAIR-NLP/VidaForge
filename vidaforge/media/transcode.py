from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time

from .frames import build_short_side_scale_filter


TRANSCODE_MODE_CPU = "cpu"


@dataclass(slots=True)
class FFmpegTranscodeResult:
    input_path: Path
    output_path: Path
    transcode_mode: str
    elapsed_sec: float


def build_ffmpeg_transcode_cmd(
    input_path: str | Path,
    output_path: str | Path,
    *,
    min_edge: int,
    fps: int,
    crf: int,
    pix_fmt: str,
    audio_bitrate: str,
    ffmpeg_threads: int | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    input_file = Path(input_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_file),
        "-vf",
        build_short_side_scale_filter(
            min_edge,
            allow_upscale=False,
            use_gpu=False,
        ),
        "-c:v",
        "libx265",
        "-crf",
        str(crf),
        "-preset",
        "fast",
    ]
    if ffmpeg_threads is not None:
        cmd.extend(["-threads", str(ffmpeg_threads)])

    cmd.extend(
        [
            "-pix_fmt",
            pix_fmt,
            "-r",
            str(fps),
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(output_file),
        ]
    )
    return cmd


def run_ffmpeg_transcode(
    input_path: str | Path,
    output_path: str | Path,
    *,
    min_edge: int,
    fps: int,
    crf: int,
    pix_fmt: str,
    audio_bitrate: str,
    ffmpeg_threads: int | None = None,
    ffmpeg_bin: str = "ffmpeg",
    timeout_sec: int | None = None,
) -> FFmpegTranscodeResult:
    input_file = Path(input_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()

    if not input_file.is_file():
        raise FileNotFoundError(f"input video not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_ffmpeg_transcode_cmd(
        input_file,
        output_file,
        min_edge=min_edge,
        fps=fps,
        crf=crf,
        pix_fmt=pix_fmt,
        audio_bitrate=audio_bitrate,
        ffmpeg_threads=ffmpeg_threads,
        ffmpeg_bin=ffmpeg_bin,
    )

    started_at = time.perf_counter()
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    elapsed_sec = time.perf_counter() - started_at

    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "ffmpeg failed"
        raise RuntimeError(f"ffmpeg error for {input_file}: {message}")

    if not output_file.is_file():
        raise RuntimeError(f"ffmpeg finished but output file is missing: {output_file}")

    return FFmpegTranscodeResult(
        input_path=input_file,
        output_path=output_file,
        transcode_mode=TRANSCODE_MODE_CPU,
        elapsed_sec=round(elapsed_sec, 3),
    )
