"""Stage 3 context preparation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from vidaforge.index import DEFAULT_PARQUET_SIZE


DEFAULT_CONTEXT_BATCH_SIZE = 4


@dataclass(slots=True)
class FrameContextConfig:
    sampled_fps: float = 2.0
    short_side: int = 384
    jpeg_qscale: int = 2


@dataclass(slots=True)
class AudioContextConfig:
    format: Literal["m4a", "wav"] = "m4a"
    sample_rate: int = 24000
    channels: int = 1


@dataclass(slots=True)
class ContextConfig:
    input_path: Path
    output_data_path: Path
    output_meta_path: Path
    run_id: str
    input_run_id: str
    source: str | None = None
    source_batch: str | None = None
    name: str = "step1_context"
    frame: FrameContextConfig = field(default_factory=FrameContextConfig)
    audio: AudioContextConfig = field(default_factory=AudioContextConfig)
    ffmpeg_bin: str = "ffmpeg"
    parquet_size: int = DEFAULT_PARQUET_SIZE
    batch_size: int = DEFAULT_CONTEXT_BATCH_SIZE
    ray_address: str = "auto"
    ray_num_cpus: float = 1.0
    limit: int | None = None
    resume: bool = False


@dataclass(slots=True)
class ContextResult:
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


__all__ = [
    "AudioContextConfig",
    "ContextConfig",
    "ContextResult",
    "DEFAULT_CONTEXT_BATCH_SIZE",
    "FrameContextConfig",
]
