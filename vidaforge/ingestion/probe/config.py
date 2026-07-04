from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vidaforge.index import DEFAULT_PARQUET_SIZE


DEFAULT_RAY_NUM_CPUS = 1.0
DEFAULT_BATCH_SIZE = 32


@dataclass(slots=True)
class ProbeConfig:
    input_path: Path
    output_path: Path
    source: str
    source_batch: str
    run_id: str
    ffprobe_bin: str = "ffprobe"
    temp_dir: Path | None = None
    parquet_size: int = DEFAULT_PARQUET_SIZE
    ray_address: str = "auto"
    ray_num_cpus: float = DEFAULT_RAY_NUM_CPUS
    batch_size: int = DEFAULT_BATCH_SIZE
    limit: int | None = None


@dataclass(slots=True)
class ProbeResult:
    input_path: Path
    output_path: Path
    source_count: int
    input_count: int
    resumed_count: int
    output_count: int
    ok_count: int
    failed_count: int
    shard_count: int
    summary_path: Path
    elapsed_sec: float
