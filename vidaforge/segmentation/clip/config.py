from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vidaforge.index import DEFAULT_PARQUET_SIZE


@dataclass(slots=True)
class ClipConfig:
    input_path: Path
    output_data_path: Path
    output_meta_path: Path
    source: str | None = None
    source_batch: str | None = None
    run_id: str = ""
    input_run_id: str = ""
    min_len_sec: float = 1.0
    max_len_sec: float = 10.0
    overlong_split_len_sec: float = 10.0
    boundary_trim_sec: float = 0.0
    ray_num_cpus: float = 1.0
    ffmpeg_bin: str = "ffmpeg"
    parquet_size: int = DEFAULT_PARQUET_SIZE
    ray_address: str = "auto"
    limit: int | None = None
    resume: bool = False


@dataclass(slots=True)
class ClipResult:
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
