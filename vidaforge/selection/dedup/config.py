"""Stage 3 dedup configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vidaforge.index import DEFAULT_PARQUET_SIZE
from vidaforge.index.processing.ray_dedup import (
    DEFAULT_DEDUP_ACTOR_ROW_BATCH_SIZE,
    DEFAULT_DEDUP_MATCH_BATCH_SIZE,
)


class DedupConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deduplicator_name: ClassVar[str]


class PDQFeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_quality: float = 50.0

    @field_validator("min_quality")
    @classmethod
    def validate_min_quality(cls, value: float) -> float:
        if not 0.0 <= value <= 100.0:
            raise ValueError("feature.min_quality must be in [0, 100].")
        return value


class PDQMatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hamming_distance_threshold: int = 31
    min_similar_frame_ratio: float = 0.8
    top_k: int = 50
    index_backend: Literal["faiss_cpu"] = "faiss_cpu"

    @field_validator("hamming_distance_threshold")
    @classmethod
    def validate_hamming_distance_threshold(cls, value: int) -> int:
        if not 0 <= value <= 256:
            raise ValueError("hamming_distance_threshold must be in [0, 256].")
        return value

    @field_validator("min_similar_frame_ratio")
    @classmethod
    def validate_min_similar_frame_ratio(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("min_similar_frame_ratio must be in [0, 1].")
        return value

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("top_k must be > 0.")
        return value


class PDQDedupConfig(DedupConfigBase):
    deduplicator_name: ClassVar[str] = "pdq"

    version: str = "pdq_v1"
    feature: PDQFeatureConfig = Field(default_factory=PDQFeatureConfig)
    match: PDQMatchConfig = Field(default_factory=PDQMatchConfig)

    @model_validator(mode="after")
    def validate_version(self) -> "PDQDedupConfig":
        if self.version != "pdq_v1":
            raise ValueError("version must be pdq_v1.")
        return self


class CosmosFeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = "nvidia/Cosmos-Embed1-336p"
    forward_batch_size: int = 8
    frame_load_workers: int = 1
    prefetch_batches: int = 0

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("feature.model_name must not be empty.")
        return value

    @field_validator("forward_batch_size", mode="before")
    @classmethod
    def validate_forward_batch_size(cls, value: object) -> int:
        parsed = int(float(value))
        if parsed <= 0:
            raise ValueError("feature.forward_batch_size must be > 0.")
        return parsed

    @field_validator("frame_load_workers", mode="before")
    @classmethod
    def validate_frame_load_workers(cls, value: object) -> int:
        parsed = int(float(value))
        if parsed <= 0:
            raise ValueError("feature.frame_load_workers must be > 0.")
        return parsed

    @field_validator("prefetch_batches", mode="before")
    @classmethod
    def validate_prefetch_batches(cls, value: object) -> int:
        parsed = int(float(value))
        if parsed < 0:
            raise ValueError("feature.prefetch_batches must be >= 0.")
        return parsed


class CosmosMatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_cosine_similarity: float | None = None
    top_k: int = 50
    index_backend: Literal["faiss_auto", "faiss_cpu", "faiss_gpu", "gpu_cuvs"] = "faiss_auto"

    @field_validator("min_cosine_similarity")
    @classmethod
    def validate_min_cosine_similarity(cls, value: float | None) -> float | None:
        if value is not None and not -1.0 <= value <= 1.0:
            raise ValueError("match.min_cosine_similarity must be in [-1, 1].")
        return value

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("match.top_k must be > 0.")
        return value


class CosmosDedupConfig(DedupConfigBase):
    deduplicator_name: ClassVar[str] = "cosmos"

    version: str = "cosmos_embed1_v1"
    feature: CosmosFeatureConfig = Field(default_factory=CosmosFeatureConfig)
    match: CosmosMatchConfig = Field(default_factory=CosmosMatchConfig)

    @model_validator(mode="after")
    def validate_version(self) -> "CosmosDedupConfig":
        if self.version != "cosmos_embed1_v1":
            raise ValueError("version must be cosmos_embed1_v1.")
        return self


@dataclass(slots=True)
class DedupApplyConfig:
    enabled: bool = True
    replicas: int | str = "auto"
    ray_num_cpus: float = 1.0
    ray_num_gpus: float = 0.0
    batch_size: int = DEFAULT_DEDUP_ACTOR_ROW_BATCH_SIZE


@dataclass(slots=True)
class DedupMatchConfig:
    replicas: int | str = 1
    ray_num_cpus: float = 8.0
    ray_num_gpus: float = 0.0
    batch_size: int = DEFAULT_DEDUP_MATCH_BATCH_SIZE


@dataclass(slots=True)
class DedupConfig:
    input_path: Path
    output_path: Path
    run_id: str
    input_run_id: str
    source: str | None = None
    source_batch: str | None = None
    name: str = "step3_dedup"
    deduplicators: list[DedupConfigBase] = field(default_factory=list)
    parquet_size: int = DEFAULT_PARQUET_SIZE
    ray_address: str = "auto"
    apply: DedupApplyConfig = field(default_factory=DedupApplyConfig)
    match: DedupMatchConfig = field(default_factory=DedupMatchConfig)
    limit: int | None = None


@dataclass(slots=True)
class DedupResult:
    input_path: Path
    output_path: Path
    source_count: int
    input_count: int
    resumed_count: int
    output_count: int
    ok_count: int
    failed_count: int
    pair_count: int
    deduplicator_match_summary: dict[str, object]
    shard_count: int
    summary_path: Path
    elapsed_sec: float
