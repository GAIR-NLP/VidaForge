from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FFProbeResult:
    duration_sec: float | None = None
    width: int | None = None
    height: int | None = None
    codec: str = ""
    avg_frame_rate: str = ""
    fps: float | None = None
    bit_rate: int | None = None
    has_audio: bool | None = None


def _parse_frame_rate(raw_value: object) -> float | None:
    if raw_value in (None, ""):
        return None

    text = str(raw_value).strip()
    if not text:
        return None

    try:
        value = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return None

    return value if value > 0 else None


def run_ffprobe(
    video_path: str | Path, ffprobe_bin: str = "ffprobe", timeout_sec: int = 30
) -> FFProbeResult:
    """Extract core media properties with ffprobe."""
    video = Path(video_path).expanduser().resolve()
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,avg_frame_rate,bit_rate:format=duration,bit_rate",
        "-of",
        "json",
        str(video),
    ]

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
        check=False,
    )

    if completed.returncode != 0:
        message = completed.stderr.strip() or "ffprobe failed"
        raise RuntimeError(f"ffprobe error for {video}: {message}")

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for {video}") from exc

    streams = payload.get("streams", [])
    video_stream = {}
    has_audio = False
    for stream in streams:
        if stream.get("codec_type") == "video" and not video_stream:
            video_stream = stream
        if stream.get("codec_type") == "audio":
            has_audio = True
    fmt = payload.get("format", {})

    duration = fmt.get("duration")
    width = video_stream.get("width")
    height = video_stream.get("height")
    codec = video_stream.get("codec_name")
    avg_frame_rate = video_stream.get("avg_frame_rate")
    bit_rate = fmt.get("bit_rate")

    return FFProbeResult(
        duration_sec=float(duration) if duration else None,
        width=int(width) if width else None,
        height=int(height) if height else None,
        codec=str(codec) if codec else "",
        avg_frame_rate=str(avg_frame_rate) if avg_frame_rate else "",
        fps=_parse_frame_rate(avg_frame_rate),
        bit_rate=int(bit_rate) if bit_rate else None,
        has_audio=has_audio,
    )
