from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
from typing import Literal, Sequence


CLIP_ENCODE_BACKEND_CPU = "cpu"
CLIP_ENCODE_BACKEND_CONSUMER_GPU = "consumer_gpu"
ClipEncodeBackend = Literal["cpu", "consumer_gpu"]


@dataclass(slots=True)
class FFmpegClipResult:
    elapsed_sec: float
    filesize_bytes: int


@dataclass(slots=True)
class FFmpegSegmentMuxerResult:
    output_paths: list[Path]
    elapsed_sec: float
    filesize_bytes: list[int]


def _ffmpeg_time(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _tail_lines(value: str, *, max_lines: int = 30) -> str:
    lines = value.strip().splitlines()
    return "\n".join(lines[-max_lines:])


def build_ffmpeg_clip_cmd(
    input_path: str | Path,
    output_path: str | Path,
    *,
    start_sec: float,
    duration_sec: float,
    stream_copy: bool = True,
    encode_backend: ClipEncodeBackend = CLIP_ENCODE_BACKEND_CPU,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = True,
    gpu_device: int | None = None,
) -> list[str]:
    """Build a single-clip ffmpeg command.

    Stream copy is only safe as a fast smoke mode. Scene boundaries usually do
    not land on keyframes, so exact, decodable clips require re-encoding.
    """
    if start_sec < 0:
        raise ValueError("start_sec must be >= 0.")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be > 0.")
    if encode_backend not in {CLIP_ENCODE_BACKEND_CPU, CLIP_ENCODE_BACKEND_CONSUMER_GPU}:
        raise ValueError("encode_backend must be one of: cpu, consumer_gpu.")

    input_file = Path(input_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    base_cmd = [
        ffmpeg_bin,
        "-y" if overwrite else "-n",
    ]
    if stream_copy:
        # Keep -ss after -i for stream-copy clipping. Input seeking can jump to
        # an earlier keyframe and produce clips much longer than requested.
        return [
            *base_cmd,
            "-i",
            str(input_file),
            "-ss",
            f"{start_sec:.6f}",
            "-t",
            f"{duration_sec:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            str(output_file),
        ]

    return [
        *base_cmd,
        "-ss",
        f"{start_sec:.6f}",
        *(
            [
                "-hwaccel",
                "cuda",
                *([] if gpu_device is None else ["-hwaccel_device", str(gpu_device)]),
            ]
            if encode_backend == CLIP_ENCODE_BACKEND_CONSUMER_GPU
            else []
        ),
        "-i",
        str(input_file),
        "-t",
        f"{duration_sec:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        *(
            [
                "-c:v",
                "hevc_nvenc",
                *([] if gpu_device is None else ["-gpu", str(gpu_device)]),
                "-preset",
                "p6",
                "-tune",
                "hq",
                "-cq",
                "23",
                "-forced-idr",
                "1",
            ]
            if encode_backend == CLIP_ENCODE_BACKEND_CONSUMER_GPU
            else [
                "-c:v",
                "libx265",
                "-crf",
                "23",
                "-preset",
                "fast",
                "-x265-params",
                "forced-idr=1:open-gop=0:min-keyint=1",
                "-pix_fmt",
                "yuv420p",
            ]
        ),
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(output_file),
    ]


def build_ffmpeg_segment_muxer_cmd(
    input_path: str | Path,
    output_pattern: str | Path,
    *,
    segment_times_sec: Sequence[float],
    end_sec: float,
    stream_copy: bool = False,
    encode_backend: ClipEncodeBackend = CLIP_ENCODE_BACKEND_CPU,
    ffmpeg_bin: str = "ffmpeg",
    overwrite: bool = True,
    gpu_device: int | None = None,
    segment_time_delta_sec: float = 0.04,
) -> list[str]:
    """Build a one-video multi-clip ffmpeg segment muxer command."""
    if stream_copy:
        raise ValueError("segment_muxer requires stream_copy=false.")
    if end_sec <= 0:
        raise ValueError("end_sec must be > 0.")
    if segment_time_delta_sec <= 0:
        raise ValueError("segment_time_delta_sec must be > 0.")
    if encode_backend not in {CLIP_ENCODE_BACKEND_CPU, CLIP_ENCODE_BACKEND_CONSUMER_GPU}:
        raise ValueError("encode_backend must be one of: cpu, consumer_gpu.")

    input_file = Path(input_path).expanduser().resolve()
    output_file_pattern = str(output_pattern)
    segment_times = [
        float(time_sec)
        for time_sec in segment_times_sec
        if 0 < float(time_sec) < end_sec
    ]
    segment_times_arg = ",".join(_ffmpeg_time(time_sec) for time_sec in segment_times)

    base_cmd = [
        ffmpeg_bin,
        "-y" if overwrite else "-n",
        *(
            [
                "-hwaccel",
                "cuda",
                *([] if gpu_device is None else ["-hwaccel_device", str(gpu_device)]),
            ]
            if encode_backend == CLIP_ENCODE_BACKEND_CONSUMER_GPU
            else []
        ),
        "-i",
        str(input_file),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        *(
            [
                "-c:v",
                "hevc_nvenc",
                *([] if gpu_device is None else ["-gpu", str(gpu_device)]),
                "-preset",
                "p6",
                "-tune",
                "hq",
                "-cq",
                "23",
                "-forced-idr",
                "1",
            ]
            if encode_backend == CLIP_ENCODE_BACKEND_CONSUMER_GPU
            else [
                "-c:v",
                "libx265",
                "-crf",
                "23",
                "-preset",
                "fast",
                "-x265-params",
                "forced-idr=1:open-gop=0:min-keyint=1",
                "-pix_fmt",
                "yuv420p",
            ]
        ),
    ]
    force_keyframe_args = (
        ["-force_key_frames", segment_times_arg]
        if segment_times_arg
        else []
    )
    segment_args = (
        ["-segment_times", segment_times_arg]
        if segment_times_arg
        else ["-segment_time", _ffmpeg_time(end_sec + 1.0)]
    )
    return [
        *base_cmd,
        *force_keyframe_args,
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ac",
        "2",
        "-to",
        _ffmpeg_time(end_sec),
        "-f",
        "segment",
        *segment_args,
        "-segment_time_delta",
        _ffmpeg_time(segment_time_delta_sec),
        "-reset_timestamps",
        "1",
        "-segment_format",
        "mp4",
        "-segment_format_options",
        "movflags=+faststart",
        output_file_pattern,
    ]


def run_ffmpeg_clip(
    input_path: str | Path,
    output_path: str | Path,
    *,
    start_sec: float,
    duration_sec: float,
    stream_copy: bool = True,
    encode_backend: ClipEncodeBackend = CLIP_ENCODE_BACKEND_CPU,
    ffmpeg_bin: str = "ffmpeg",
    gpu_device: int | None = None,
    overwrite: bool = True,
    timeout_sec: int | None = None,
) -> FFmpegClipResult:
    """Execute single-clip ffmpeg extraction and return execution details."""
    input_file = Path(input_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()

    if not input_file.is_file():
        raise FileNotFoundError(f"input video not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_ffmpeg_clip_cmd(
        input_file,
        output_file,
        start_sec=start_sec,
        duration_sec=duration_sec,
        stream_copy=stream_copy,
        encode_backend=encode_backend,
        ffmpeg_bin=ffmpeg_bin,
        gpu_device=gpu_device,
        overwrite=overwrite,
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
        raise RuntimeError(f"ffmpeg clip error for {input_file}: {message}")

    if not output_file.is_file():
        raise RuntimeError(f"ffmpeg finished but output clip is missing: {output_file}")

    filesize_bytes = output_file.stat().st_size
    if filesize_bytes <= 0:
        raise RuntimeError(f"ffmpeg finished but output clip is empty: {output_file}")

    return FFmpegClipResult(
        elapsed_sec=round(elapsed_sec, 3),
        filesize_bytes=filesize_bytes,
    )


def run_ffmpeg_segment_muxer(
    input_path: str | Path,
    output_pattern: str | Path,
    *,
    segment_times_sec: Sequence[float],
    end_sec: float,
    expected_output_count: int,
    stream_copy: bool = False,
    encode_backend: ClipEncodeBackend = CLIP_ENCODE_BACKEND_CPU,
    ffmpeg_bin: str = "ffmpeg",
    gpu_device: int | None = None,
    overwrite: bool = True,
    timeout_sec: int | None = None,
    segment_time_delta_sec: float = 0.04,
) -> FFmpegSegmentMuxerResult:
    """Execute ffmpeg segment muxer and verify expected output files."""
    input_file = Path(input_path).expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"input video not found: {input_file}")
    if expected_output_count <= 0:
        raise ValueError("expected_output_count must be > 0.")

    output_pattern_text = str(output_pattern)
    output_paths = [
        Path(output_pattern_text % index).expanduser().resolve()
        for index in range(expected_output_count)
    ]
    output_paths[0].parent.mkdir(parents=True, exist_ok=True)
    cmd = build_ffmpeg_segment_muxer_cmd(
        input_file,
        output_pattern,
        segment_times_sec=segment_times_sec,
        end_sec=end_sec,
        stream_copy=stream_copy,
        encode_backend=encode_backend,
        ffmpeg_bin=ffmpeg_bin,
        gpu_device=gpu_device,
        overwrite=overwrite,
        segment_time_delta_sec=segment_time_delta_sec,
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
        raise RuntimeError(f"ffmpeg segment muxer error for {input_file}: {message}")

    missing_paths = [path for path in output_paths if not path.is_file()]
    if missing_paths:
        generated_paths = [path for path in output_paths if path.is_file()]
        stderr_tail = _tail_lines(completed.stderr)
        stdout_tail = _tail_lines(completed.stdout)
        process_output = stderr_tail or stdout_tail or "ffmpeg produced no output"
        raise RuntimeError(
            "ffmpeg segment muxer finished but expected output is missing "
            f"(expected={expected_output_count}, generated={len(generated_paths)}, "
            f"end_sec={end_sec}, segment_times={len(segment_times_sec)}). "
            f"missing: {', '.join(str(path) for path in missing_paths[:5])}. "
            f"ffmpeg_tail: {process_output}"
        )

    filesize_bytes = [path.stat().st_size for path in output_paths]
    empty_paths = [
        path
        for path, file_size in zip(output_paths, filesize_bytes)
        if file_size <= 0
    ]
    if empty_paths:
        raise RuntimeError(
            "ffmpeg segment muxer finished but expected output is empty: "
            + ", ".join(str(path) for path in empty_paths[:5])
        )

    return FFmpegSegmentMuxerResult(
        output_paths=output_paths,
        elapsed_sec=round(elapsed_sec, 3),
        filesize_bytes=filesize_bytes,
    )
