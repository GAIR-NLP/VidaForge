"""FFmpeg frame sampling and optional audio extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from .frames import build_short_side_scale_filter, scaled_size


FRAME_SAMPLING_METHOD_UNIFORM = "uniform"


@dataclass(slots=True)
class FFmpegFrameAudioExtractResult:
    sampling_method: str
    frame_paths: list[Path]
    timestamps_sec: list[float]
    frame_width: int
    frame_height: int
    audio_paths: list[Path]
    audio_format: str


def _append_audio_output_args(
    cmd: list[str],
    output_path: str | Path,
    *,
    audio_format: str,
    audio_sample_rate: int,
    audio_channels: int,
) -> None:
    output_file = Path(output_path).expanduser().resolve()
    cmd.extend(["-map", "0:a:0?"])
    if audio_format == "m4a":
        cmd.extend(["-c:a", "copy"])
    elif audio_format == "wav":
        cmd.extend(
            [
                "-ac",
                str(audio_channels),
                "-ar",
                str(audio_sample_rate),
                "-c:a",
                "pcm_s16le",
            ]
        )
    else:
        raise ValueError(f"unsupported audio format: {audio_format!r}")
    cmd.append(str(output_file))


def build_ffmpeg_extract_frames_audio_cmd(
    input_path: str | Path,
    frame_output_path: str | Path,
    *,
    frame_sampled_fps: float,
    frame_short_side: int,
    frame_jpeg_qscale: int,
    audio_output_path: str | Path | None,
    audio_format: str,
    audio_sample_rate: int,
    audio_channels: int,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """Build one FFmpeg command that writes sampled frames and optional audio."""
    input_file = Path(input_path).expanduser().resolve()
    frame_output = Path(frame_output_path).expanduser().resolve()
    vf = (
        f"fps={float(frame_sampled_fps):g},"
        f"{build_short_side_scale_filter(frame_short_side)}"
    )
    cmd = [
        ffmpeg_bin,
        "-y",
    ]
    cmd.extend(
        [
            "-i",
            str(input_file),
            "-map",
            "0:v:0",
            "-vf",
            vf,
            "-vsync",
            "0",
            "-q:v",
            str(frame_jpeg_qscale),
            "-start_number",
            "0",
            str(frame_output),
        ]
    )
    if audio_output_path is not None:
        _append_audio_output_args(
            cmd,
            audio_output_path,
            audio_format=audio_format,
            audio_sample_rate=audio_sample_rate,
            audio_channels=audio_channels,
        )
    return cmd


def _uniform_frame_timestamps_sec(
    *,
    frame_count: int,
    frame_sampled_fps: float,
    duration_sec: float,
) -> list[float]:
    if frame_count <= 0:
        return []
    fps = float(frame_sampled_fps)
    return [
        round(min(index / fps, max(float(duration_sec), 0.0)), 6)
        for index in range(frame_count)
    ]


def run_ffmpeg_extract_frames_audio(
    input_path: str | Path,
    output_path: str | Path,
    *,
    frame_sampled_fps: float,
    frame_short_side: int,
    frame_jpeg_qscale: int,
    frame_sampling_method: str,
    has_audio: bool,
    audio_format: str,
    audio_sample_rate: int,
    audio_channels: int,
    duration_sec: float,
    input_width: int,
    input_height: int,
    ffmpeg_bin: str = "ffmpeg",
    timeout_sec: int | None = None,
) -> FFmpegFrameAudioExtractResult:
    """Extract uniformly sampled JPEG frames and optional audio in one FFmpeg process."""
    output_dir = Path(output_path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 2 <= int(frame_jpeg_qscale) <= 31:
        raise ValueError("frame_jpeg_qscale must be in [2, 31].")

    frame_path = output_dir / "frame_%06d.jpg"
    audio_path = output_dir / f"audio.{audio_format}" if has_audio else None

    for pattern in ("frame_*.jpg", "frame_*.jpeg", "frame_*.png"):
        for stale_frame in output_dir.glob(pattern):
            stale_frame.unlink()
    if audio_path is not None and audio_path.exists():
        audio_path.unlink()

    frame_width, frame_height = scaled_size(
        input_width=input_width,
        input_height=input_height,
        short_side=frame_short_side,
    )
    cmd = build_ffmpeg_extract_frames_audio_cmd(
        input_path,
        frame_output_path=frame_path,
        frame_sampled_fps=frame_sampled_fps,
        frame_short_side=frame_short_side,
        frame_jpeg_qscale=frame_jpeg_qscale,
        audio_output_path=audio_path,
        audio_format=audio_format,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
        ffmpeg_bin=ffmpeg_bin,
    )
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "ffmpeg failed"
        raise RuntimeError(f"ffmpeg frame/audio extraction error for {input_path}: {message}")

    frame_paths = sorted(output_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"ffmpeg finished but output frames are missing: {output_dir}")

    audio_paths: list[Path] = []
    if audio_path is not None:
        if not audio_path.is_file() or audio_path.stat().st_size <= 0:
            raise RuntimeError(f"ffmpeg finished but output audio is missing: {audio_path}")
        audio_paths.append(audio_path)

    if frame_sampling_method == FRAME_SAMPLING_METHOD_UNIFORM:
        timestamps_sec = _uniform_frame_timestamps_sec(
            frame_count=len(frame_paths),
            frame_sampled_fps=frame_sampled_fps,
            duration_sec=duration_sec,
        )
    else:
        raise ValueError(
            "run_ffmpeg_extract_frames_audio only supports "
            f"frame_sampling_method={FRAME_SAMPLING_METHOD_UNIFORM!r}."
        )

    return FFmpegFrameAudioExtractResult(
        sampling_method=frame_sampling_method,
        frame_paths=frame_paths,
        timestamps_sec=timestamps_sec,
        frame_width=frame_width,
        frame_height=frame_height,
        audio_paths=audio_paths,
        audio_format=audio_format if audio_path is not None else "",
    )
