"""Segmentation capabilities for boundary detection and clip asset generation."""

from vidaforge.segmentation.clip import (
    ClipConfig,
    ClipOrchestrator,
    ClipResult,
    ClipTiming,
    build_clip_timings_from_ticks,
    process_clip_row,
)
from vidaforge.segmentation.detect import (
    AdaptiveDetectorConfig,
    ContentDetectorConfig,
    DEFAULT_MIN_LEN_SEC,
    DEFAULT_RAY_NUM_CPUS,
    DetectConfig,
    DetectOrchestrator,
    DetectResult,
    DetectorConfigBase,
    HashDetectorConfig,
    HistogramDetectorConfig,
    RandomDetectorConfig,
    ThresholdDetectorConfig,
    TransNetV2DetectorConfig,
    UniformDetectorConfig,
)

__all__ = [
    "AdaptiveDetectorConfig",
    "ClipConfig",
    "ClipOrchestrator",
    "ClipResult",
    "ClipTiming",
    "ContentDetectorConfig",
    "DEFAULT_MIN_LEN_SEC",
    "DEFAULT_RAY_NUM_CPUS",
    "DetectConfig",
    "DetectResult",
    "DetectOrchestrator",
    "DetectorConfigBase",
    "HashDetectorConfig",
    "HistogramDetectorConfig",
    "RandomDetectorConfig",
    "ThresholdDetectorConfig",
    "TransNetV2DetectorConfig",
    "UniformDetectorConfig",
    "build_clip_timings_from_ticks",
    "process_clip_row",
]
