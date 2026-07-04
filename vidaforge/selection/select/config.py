from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vidaforge.index import DEFAULT_PARQUET_SIZE


@dataclass(slots=True)
class SelectConfig:
    input_path: Path
    output_path: Path
    run_id: str
    input_run_id: str
    source: str | None = None
    source_batch: str | None = None
    name: str = "step4_select"
    filter: dict[str, dict[str, object]] = field(default_factory=dict)
    dedup: dict[str, dict[str, object]] = field(default_factory=dict)
    parquet_size: int = DEFAULT_PARQUET_SIZE
    limit: int | None = None


@dataclass(slots=True)
class SelectResult:
    input_path: Path
    output_path: Path
    source_count: int
    input_count: int
    resumed_count: int
    output_count: int
    ok_count: int
    failed_count: int
    pass_count: int
    reject_count: int
    shard_count: int
    summary_path: Path
    elapsed_sec: float
