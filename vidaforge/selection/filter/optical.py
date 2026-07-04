from __future__ import annotations

from typing import ClassVar

import cv2
import numpy as np

from vidaforge.common import join_data_dir, parse_json_object
from vidaforge.filters import linear_decay, linear_growth, trapezoid_scale
from vidaforge.media.frames import load_bgr_image_arrays

from .base import FilterBase
from .config import ContrastConfig, ExposureConfig, OpticalFilterConfig


def compute_exposure(
    frames: list[np.ndarray],
    config: ExposureConfig,
) -> dict[str, float]:
    if not frames:
        raise ValueError("frames must not be empty")

    brightness_values: list[float] = []
    too_dark_ratios: list[float] = []
    too_bright_ratios: list[float] = []
    frame_scores: list[float] = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        brightness = float(np.median(gray))
        too_dark_ratio = float(np.mean(gray < config.pixel_too_dark_threshold))
        too_bright_ratio = float(np.mean(gray > config.pixel_too_bright_threshold))

        brightness_score = trapezoid_scale(
            brightness,
            a=config.brightness_scale[0],
            b=config.brightness_scale[1],
            c=config.brightness_scale[2],
            d=config.brightness_scale[3],
        )
        too_dark_score = linear_decay(
            too_dark_ratio,
            a=config.too_dark_ratio_scale[0],
            b=config.too_dark_ratio_scale[1],
        )
        too_bright_score = linear_decay(
            too_bright_ratio,
            a=config.too_bright_ratio_scale[0],
            b=config.too_bright_ratio_scale[1],
        )

        brightness_values.append(brightness)
        too_dark_ratios.append(too_dark_ratio)
        too_bright_ratios.append(too_bright_ratio)
        frame_scores.append(brightness_score * too_dark_score * too_bright_score)

    return {
        "score": round(
            float(np.percentile(frame_scores, config.score_quantile * 100.0)),
            6,
        ),
        "brightness": round(float(np.median(brightness_values)), 6),
        "too_dark_ratio": round(float(np.median(too_dark_ratios)), 6),
        "too_bright_ratio": round(float(np.median(too_bright_ratios)), 6),
    }


def compute_contrast(
    frames: list[np.ndarray],
    config: ContrastConfig,
) -> dict[str, float]:
    if not frames:
        raise ValueError("frames must not be empty")

    dynamic_ranges: list[float] = []
    frame_scores: list[float] = []
    low_percentile, high_percentile = config.dynamic_range_percentiles

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        low_value, high_value = np.percentile(
            gray,
            [low_percentile, high_percentile],
        )
        dynamic_range = float(high_value - low_value)
        score = linear_growth(
            dynamic_range,
            a=config.dynamic_range_scale[0],
            b=config.dynamic_range_scale[1],
        )

        dynamic_ranges.append(dynamic_range)
        frame_scores.append(score)

    return {
        "score": round(
            float(np.percentile(frame_scores, config.score_quantile * 100.0)),
            6,
        ),
        "dynamic_range": round(float(np.median(dynamic_ranges)), 6),
    }


class OpticalFilter(FilterBase[OpticalFilterConfig]):
    filter_name: ClassVar[str] = "optical"
    config_type: ClassVar[type[OpticalFilterConfig]] = OpticalFilterConfig

    def __init__(self, config: OpticalFilterConfig) -> None:
        if not isinstance(config, OpticalFilterConfig):
            raise TypeError(f"expected OpticalFilterConfig, got {type(config)!r}")
        self.config = config

    def apply(self, row: dict[str, object]) -> tuple[float, dict[str, object]]:
        frame_json = parse_json_object(row["frame_json"], description="frame_json")
        frame_paths = [join_data_dir(str(path)) for path in frame_json["frame_paths"]]
        frames = load_bgr_image_arrays(frame_paths)
        exposure = compute_exposure(frames, self.config.exposure)
        contrast = compute_contrast(frames, self.config.contrast)
        payload: dict[str, object] = {"exposure": exposure, "contrast": contrast}
        score = float(exposure["score"]) * float(contrast["score"])
        return round(score, 6), payload
