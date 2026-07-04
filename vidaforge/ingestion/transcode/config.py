from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vidaforge.index import DEFAULT_PARQUET_SIZE


@dataclass(slots=True)
class TranscodeConfig:
    input_path: Path
    output_data_path: Path
    output_meta_path: Path
    # Transcoded video targets. Low-resolution inputs are not upscaled.
    target_short_edge: int
    target_fps: int
    # x265 quality knob: lower values produce larger, higher-quality files.
    crf: int
    # Output pixel format. yuv420p is the compatibility default.
    pix_fmt: str
    # Output AAC audio bitrate.
    audio_bitrate: str
    input_run_id: str
    run_id: str
    source: str | None = None
    source_batch: str | None = None
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    parquet_size: int = DEFAULT_PARQUET_SIZE
    ray_address: str = "auto"
    ray_num_cpus: float = 4
    ffmpeg_threads: int | None = 4
    limit: int | None = None
    resume: bool = False


@dataclass(slots=True)
class TranscodeResult:
    input_path: Path
    output_data_path: Path
    output_meta_path: Path
    source_count: int
    input_count: int
    resumed_count: int
    output_count: int
    ok_count: int
    failed_count: int
    shard_count: int
    summary_path: Path
    elapsed_sec: float
