from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from vidaforge.index import DEFAULT_PARQUET_SIZE
from vidaforge.serving.config import VLMInferenceConfig

from .prompt import CAPTION_PROMPT_VERSION
from .schema import CAPTION_SCHEMA_VERSION


CaptionModeConfig = Literal["video", "video_audio"]


@dataclass(slots=True)
class CaptionConfig:
    input_path: Path
    output_path: Path
    source: str
    source_batch: str
    run_id: str
    input_run_id: str
    name: str = "step2_caption"
    schema_version: str = CAPTION_SCHEMA_VERSION
    prompt_version: str = CAPTION_PROMPT_VERSION
    mode: CaptionModeConfig = "video_audio"
    inference: VLMInferenceConfig = field(
        default_factory=lambda: VLMInferenceConfig(max_tokens=4096)
    )
    parquet_size: int = DEFAULT_PARQUET_SIZE
    batch_size: int = 128
    ray_num_cpus: float = 1.0
    ray_address: str = "auto"
    limit: int | None = None
    resume: bool = False


@dataclass(slots=True)
class CaptionResult:
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
