from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scenedetect.detectors import (
    AdaptiveDetector,
    ContentDetector,
    HashDetector,
    HistogramDetector,
    ThresholdDetector,
)

from vidaforge.index import DEFAULT_PARQUET_SIZE

from .transnetv2 import TransNetV2Detector


DEFAULT_MIN_LEN_SEC = 2.0
DEFAULT_RAY_NUM_CPUS = 1.0


class DetectorConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    detector_name: ClassVar[str]

    def _seconds_to_frames(self, seconds: float, *, fps: float | None) -> int:
        fps = 24.0 if fps is None or fps <= 0 else fps
        return max(1, int(round(float(seconds) * fps)))

    def build(self, *, fps: float | None) -> object:
        if (detector_cls := getattr(self, "detector_cls", None)) is None:
            raise TypeError(f"{self.detector_name} detector config has no runtime detector.")
        if (min_len_sec := getattr(self, "min_len_sec", None)) is None:
            raise ValueError(f"{self.detector_name} requires min_len_sec.")
        kwargs = self.model_dump(exclude_none=True, by_alias=True)
        kwargs["min_scene_len"] = self._seconds_to_frames(float(min_len_sec), fps=fps)
        kwargs.pop("min_len_sec", None)
        return detector_cls(**kwargs)


class ContentDetectorConfig(DetectorConfigBase):
    detector_name: ClassVar[str] = "content"
    detector_cls: ClassVar[type] = ContentDetector
    min_len_sec: float | None = None

    threshold: float = 27.0
    weights: Any | None = None
    luma_only: bool = False
    kernel_size: int | None = None
    filter_mode: Any | None = None


class AdaptiveDetectorConfig(DetectorConfigBase):
    detector_name: ClassVar[str] = "adaptive"
    detector_cls: ClassVar[type] = AdaptiveDetector
    min_len_sec: float | None = None

    adaptive_threshold: float = 3.0
    window_width: int = 2
    min_content_val: float = 15.0
    weights: Any | None = None
    luma_only: bool = False
    kernel_size: int | None = None


class ThresholdDetectorConfig(DetectorConfigBase):
    detector_name: ClassVar[str] = "threshold"
    detector_cls: ClassVar[type] = ThresholdDetector
    min_len_sec: float | None = None

    threshold: float = 12.0
    fade_bias: float = 0.0
    add_final: bool = Field(False, serialization_alias="add_final_scene")
    method: Any | None = None
    block_size: Any | None = None


class HistogramDetectorConfig(DetectorConfigBase):
    detector_name: ClassVar[str] = "histogram"
    detector_cls: ClassVar[type] = HistogramDetector
    min_len_sec: float | None = None

    threshold: float = 0.05
    bins: int = 256


class HashDetectorConfig(DetectorConfigBase):
    detector_name: ClassVar[str] = "hash"
    detector_cls: ClassVar[type] = HashDetector
    min_len_sec: float | None = None

    threshold: float = 0.395
    size: int = 16
    lowpass: int = 2


class TransNetV2DetectorConfig(DetectorConfigBase):
    detector_name: ClassVar[str] = "transnetv2"
    detector_cls: ClassVar[type] = TransNetV2Detector
    min_len_sec: float | None = None

    weights_path: str | None = None
    threshold: float = 0.5
    prediction_mode: str = "single_frame"
    device: str | None = None

    @model_validator(mode="after")
    def validate_transnetv2_config(self) -> "TransNetV2DetectorConfig":
        if not self.weights_path:
            raise ValueError("TransNetV2DetectorConfig requires weights_path.")
        if self.device is None:
            self.device = "cpu"
            return self
        if self.device != "cpu":
            raise ValueError(
                "TransNetV2DetectorConfig currently only supports device='cpu'. "
                "GPU backend is not implemented yet."
            )
        return self


class UniformDetectorConfig(DetectorConfigBase):
    detector_name: ClassVar[str] = "uniform"

    len_sec: float = 5.0

    @field_validator("len_sec")
    @classmethod
    def validate_len_sec(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("UniformDetectorConfig requires len_sec > 0.")
        return value


class RandomDetectorConfig(DetectorConfigBase):
    detector_name: ClassVar[str] = "random"

    min_len_sec: float | None = None
    max_len_sec: float = 10.0
    len_step_sec: float = 1.0
    seed: int = 0

    @field_validator("min_len_sec")
    @classmethod
    def validate_min_len_sec(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("RandomDetectorConfig requires min_len_sec > 0.")
        return value

    @field_validator("max_len_sec")
    @classmethod
    def validate_max_len_sec(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("RandomDetectorConfig requires max_len_sec > 0.")
        return value

    @field_validator("len_step_sec")
    @classmethod
    def validate_len_step_sec(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("RandomDetectorConfig requires len_step_sec > 0.")
        return value

    @model_validator(mode="after")
    def validate_random_bounds(self) -> "RandomDetectorConfig":
        if (
            self.min_len_sec is not None
            and self.max_len_sec < self.min_len_sec
        ):
            raise ValueError(
                "RandomDetectorConfig requires max_len_sec >= min_len_sec."
            )
        return self

@dataclass(slots=True)
class DetectConfig:
    input_path: Path
    output_path: Path
    input_run_id: str
    run_id: str
    source: str | None = None
    source_batch: str | None = None
    detectors: list[DetectorConfigBase] = field(default_factory=list)
    min_len_sec: float = DEFAULT_MIN_LEN_SEC
    parquet_size: int = DEFAULT_PARQUET_SIZE
    ray_address: str = "auto"
    ray_num_cpus: float = DEFAULT_RAY_NUM_CPUS
    limit: int | None = None
    resume: bool = False


@dataclass(slots=True)
class DetectResult:
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
