from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vidaforge.index import DEFAULT_PARQUET_SIZE
from vidaforge.serving.config import VLMInferenceConfig

from .schema import TAG_PROMPT_VERSION, TAG_SCHEMA_VERSION


@dataclass(slots=True)
class TagConfig:
    input_path: Path
    output_path: Path
    source: str
    source_batch: str
    run_id: str
    input_run_id: str
    name: str = "step3_tag"
    tag_schema_version: str = TAG_SCHEMA_VERSION
    tag_prompt_version: str = TAG_PROMPT_VERSION
    inference: VLMInferenceConfig = field(default_factory=VLMInferenceConfig)
    parquet_size: int = DEFAULT_PARQUET_SIZE
    batch_size: int = 128
    ray_num_cpus: float = 1.0
    ray_address: str = "auto"
    limit: int | None = None
    resume: bool = False


@dataclass(slots=True)
class TagResult:
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
