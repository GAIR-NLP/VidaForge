from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vidaforge.index import DEFAULT_PARQUET_SIZE


@dataclass(slots=True)
class AutoModelPackConfig:
    input_path: Path
    output_path: Path
    run_id: str
    input_run_id: str
    source: str | None = None
    source_batch: str | None = None
    name: str = "automodel"
    caption_field: str = "caption_level_3"
    select_pass: int | None = 1
    batch_size: int = 32
    dynamic_forward_batch_size: int = 4
    metadata_shard_size: int = 10_000
    parquet_size: int = DEFAULT_PARQUET_SIZE
    ray_address: str = "auto"
    replicas: int | str = "auto"
    ray_num_cpus: float = 1.0
    ray_num_gpus: float = 1.0
    bucket_resolution: str = "480p"
    bucket_upscale: bool = False
    bucket_durations_sec: list[float] = field(
        default_factory=lambda: [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    )
    limit: int | None = None
    resume: bool = False


@dataclass(slots=True)
class AutoModelPackResult:
    input_path: Path
    output_path: Path
    source_count: int
    input_count: int
    resumed_count: int
    output_count: int
    ok_count: int
    failed_count: int
    shard_count: int
    metadata_shard_count: int
    metadata_path: Path
    summary_path: Path
    elapsed_sec: float
