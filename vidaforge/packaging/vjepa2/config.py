from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vidaforge.index import DEFAULT_PARQUET_SIZE


@dataclass(slots=True)
class VJEPA2PackConfig:
    input_path: Path
    output_path: Path
    run_id: str
    input_run_id: str
    source: str | None = None
    source_batch: str | None = None
    name: str = "vjepa2"
    select_pass: int | None = 1
    label: int = 0
    manifest_name: str = "train.csv"
    duration_min_sec: float | None = 4.0
    duration_max_sec: float | None = 10.0
    resolution_min: str | None = "480p"
    resolution_max: str | None = "720p"
    parquet_size: int = DEFAULT_PARQUET_SIZE
    limit: int | None = None


@dataclass(slots=True)
class VJEPA2PackResult:
    input_path: Path
    output_path: Path
    source_count: int
    input_count: int
    output_count: int
    ok_count: int
    rejected_count: int
    failed_count: int
    shard_count: int
    manifest_path: Path
    summary_path: Path
    elapsed_sec: float
