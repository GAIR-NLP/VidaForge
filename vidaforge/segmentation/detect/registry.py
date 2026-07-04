from __future__ import annotations

from dataclasses import dataclass

from .config import (
    AdaptiveDetectorConfig,
    ContentDetectorConfig,
    DetectorConfigBase,
    HashDetectorConfig,
    HistogramDetectorConfig,
    RandomDetectorConfig,
    ThresholdDetectorConfig,
    TransNetV2DetectorConfig,
    UniformDetectorConfig,
)


@dataclass(frozen=True, slots=True)
class Detector:
    config_type: type[DetectorConfigBase]
    metadata_only: bool = False


DETECTORS: dict[str, Detector] = {
    "content": Detector(config_type=ContentDetectorConfig),
    "adaptive": Detector(config_type=AdaptiveDetectorConfig),
    "threshold": Detector(config_type=ThresholdDetectorConfig),
    "histogram": Detector(config_type=HistogramDetectorConfig),
    "hash": Detector(config_type=HashDetectorConfig),
    "transnetv2": Detector(config_type=TransNetV2DetectorConfig),
    "uniform": Detector(config_type=UniformDetectorConfig, metadata_only=True),
    "random": Detector(config_type=RandomDetectorConfig, metadata_only=True),
}
