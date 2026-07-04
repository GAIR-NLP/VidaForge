"""Stage 3 filter configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vidaforge.index import DEFAULT_PARQUET_SIZE


class FilterConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filter_name: ClassVar[str]


class ExposureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_quantile: float = 0.20
    pixel_too_dark_threshold: float = 0.03
    pixel_too_bright_threshold: float = 0.97
    brightness_scale: tuple[float, float, float, float] = (0.05, 0.20, 0.80, 0.95)
    too_dark_ratio_scale: tuple[float, float] = (0.05, 0.50)
    too_bright_ratio_scale: tuple[float, float] = (0.05, 0.50)

    @field_validator(
        "score_quantile",
        "pixel_too_dark_threshold",
        "pixel_too_bright_threshold",
    )
    @classmethod
    def validate_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("value must be in [0, 1].")
        return value

    @field_validator(
        "brightness_scale",
        "too_dark_ratio_scale",
        "too_bright_ratio_scale",
    )
    @classmethod
    def validate_increasing_values(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not 0.0 <= item <= 1.0 for item in value):
            raise ValueError("scale values must be in [0, 1].")
        if any(left >= right for left, right in zip(value, value[1:])):
            raise ValueError("scale values must be strictly increasing.")
        return value

    @model_validator(mode="after")
    def validate_pixel_threshold_order(self) -> "ExposureConfig":
        if self.pixel_too_dark_threshold >= self.pixel_too_bright_threshold:
            raise ValueError(
                "pixel_too_dark_threshold must be smaller than "
                "pixel_too_bright_threshold."
            )
        return self


class ContrastConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_quantile: float = 0.20
    dynamic_range_percentiles: tuple[float, float] = (5.0, 95.0)
    dynamic_range_scale: tuple[float, float] = (0.05, 0.25)

    @field_validator("score_quantile")
    @classmethod
    def validate_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("value must be in [0, 1].")
        return value

    @field_validator("dynamic_range_percentiles")
    @classmethod
    def validate_percentiles(cls, value: tuple[float, float]) -> tuple[float, float]:
        if any(not 0.0 <= item <= 100.0 for item in value):
            raise ValueError("percentiles must be in [0, 100].")
        if value[0] >= value[1]:
            raise ValueError("percentiles must be strictly increasing.")
        return value

    @field_validator("dynamic_range_scale")
    @classmethod
    def validate_scale(cls, value: tuple[float, float]) -> tuple[float, float]:
        if any(not 0.0 <= item <= 1.0 for item in value):
            raise ValueError("scale values must be in [0, 1].")
        if value[0] >= value[1]:
            raise ValueError("scale values must be strictly increasing.")
        return value


class OpticalFilterConfig(FilterConfigBase):
    filter_name: ClassVar[str] = "optical"

    version: str
    exposure: ExposureConfig = Field(default_factory=ExposureConfig)
    contrast: ContrastConfig = Field(default_factory=ContrastConfig)

    @model_validator(mode="after")
    def validate_version(self) -> "OpticalFilterConfig":
        if self.version != "optical_v2":
            raise ValueError("version must be optical_v2.")
        return self


class VMAFMotionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean_at_score_0_5: float = 8.0

    @field_validator("mean_at_score_0_5")
    @classmethod
    def validate_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("mean_at_score_0_5 must be > 0.")
        return value


class SCDetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold: float = 10.0
    sample_fps: float = 8.0
    sample_width: int = 320
    cut_ratio_score_scale: tuple[float, float] = (0.0, 0.05)

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 100.0:
            raise ValueError("threshold must be in [0, 100].")
        return value

    @field_validator("sample_fps")
    @classmethod
    def validate_sample_fps(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("sample_fps must be > 0.")
        return value

    @field_validator("sample_width")
    @classmethod
    def validate_sample_width(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("sample_width must be > 0.")
        return value

    @field_validator("cut_ratio_score_scale")
    @classmethod
    def validate_cut_ratio_score_scale(
        cls,
        value: tuple[float, float],
    ) -> tuple[float, float]:
        if any(not 0.0 <= item <= 1.0 for item in value):
            raise ValueError("cut_ratio_score_scale values must be in [0, 1].")
        if value[0] >= value[1]:
            raise ValueError("cut_ratio_score_scale must be strictly increasing.")
        return value


class MotionFilterConfig(FilterConfigBase):
    filter_name: ClassVar[str] = "motion"

    version: str
    ffmpeg_bin: str = "ffmpeg"
    vmafmotion: VMAFMotionConfig = Field(default_factory=VMAFMotionConfig)
    scdet: SCDetConfig = Field(default_factory=SCDetConfig)

    @model_validator(mode="after")
    def validate_version(self) -> "MotionFilterConfig":
        if self.version != "motion_v1":
            raise ValueError("version must be motion_v1.")
        return self


class AestheticFilterConfig(FilterConfigBase):
    filter_name: ClassVar[str] = "aesthetic"

    version: str
    predictor_path: Path
    encoder_path: Path
    device: str = "cpu"
    dtype: str = "bfloat16"
    forward_batch_size: int = 512
    frame_load_workers: int = 1
    prefetch_batches: int = 0

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        if value not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda.")
        return value

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, value: str) -> str:
        if value not in {"float32", "float16", "bfloat16"}:
            raise ValueError("dtype must be float32, float16, or bfloat16.")
        return value

    @field_validator(
        "forward_batch_size",
        "frame_load_workers",
        "prefetch_batches",
        mode="before",
    )
    @classmethod
    def validate_non_negative_int(cls, value: object) -> int:
        parsed = int(float(value))
        if parsed < 0:
            raise ValueError("value must be >= 0.")
        return parsed

    @field_validator("forward_batch_size", "frame_load_workers")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be > 0.")
        return value

    @model_validator(mode="after")
    def validate_version(self) -> "AestheticFilterConfig":
        if self.version != "aesthetic_v1":
            raise ValueError("version must be aesthetic_v1.")
        return self


class TextFilterConfig(FilterConfigBase):
    filter_name: ClassVar[str] = "text"

    version: str
    model_path: str
    device: str = "cuda"
    text_min_confidence: float = 0.5
    text_ratio_quantile: float = 0.95
    forward_batch_size: int = 128
    frame_load_workers: int = 1
    prefetch_batches: int = 0

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        if value not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda.")
        return value

    @field_validator("text_min_confidence", "text_ratio_quantile")
    @classmethod
    def validate_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("value must be in [0, 1].")
        return value

    @field_validator(
        "forward_batch_size",
        "frame_load_workers",
        "prefetch_batches",
        mode="before",
    )
    @classmethod
    def validate_non_negative_int(cls, value: object) -> int:
        parsed = int(float(value))
        if parsed < 0:
            raise ValueError("value must be >= 0.")
        return parsed

    @field_validator("forward_batch_size", "frame_load_workers")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be > 0.")
        return value

    @model_validator(mode="after")
    def validate_version(self) -> "TextFilterConfig":
        if self.version != "text_v1":
            raise ValueError("version must be text_v1.")
        return self


@dataclass(slots=True)
class FilterConfig:
    input_path: Path
    output_path: Path
    run_id: str
    input_run_id: str
    source: str | None = None
    source_batch: str | None = None
    name: str = "step2_filter"
    filters: list[FilterConfigBase] = field(default_factory=list)
    batch_size: int = 128
    parquet_size: int = DEFAULT_PARQUET_SIZE
    ray_address: str = "auto"
    replicas: int | str = "auto"
    ray_num_cpus: float = 1.0
    ray_num_gpus: float = 0.0
    limit: int | None = None
    resume: bool = False


@dataclass(slots=True)
class FilterResult:
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
