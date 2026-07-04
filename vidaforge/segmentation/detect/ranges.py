from __future__ import annotations

import hashlib
import random

from scenedetect import open_video
from scenedetect.scene_manager import SceneManager

from vidaforge.common import join_data_dir

from .config import (
    DetectorConfigBase,
    RandomDetectorConfig,
    UniformDetectorConfig,
)
from .registry import DETECTORS


def _build_frame_detect_ranges(
    row: dict[str, str | int | float | None],
    *,
    detectors: list[DetectorConfigBase],
) -> list[tuple[float, float]]:
    video_path = join_data_dir(str(row["video_path"]))
    fps_value = row["fps"]
    fps = None if fps_value is None else float(fps_value)
    built_detectors = [
        detector_config.build(fps=fps)
        for detector_config in detectors
    ]
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    for detector in built_detectors:
        scene_manager.add_detector(detector)
    scene_manager.detect_scenes(video=video, show_progress=False)
    detected_ranges = scene_manager.get_scene_list(start_in_scene=False)
    if detected_ranges:
        return [
            (float(start_time.get_seconds()), float(end_time.get_seconds()))
            for start_time, end_time in detected_ranges
        ]

    duration_sec = float(row["duration_sec"])
    if duration_sec <= 0:
        raise ValueError("detector returned no ranges and duration_sec <= 0")
    return [(0.0, duration_sec)]


def _build_uniform_detect_ranges(
    row: dict[str, str | int | float | None],
    detector_config: UniformDetectorConfig,
) -> list[tuple[float, float]]:
    duration_sec = float(row["duration_sec"])
    if duration_sec <= 0:
        raise ValueError("uniform detector requires duration_sec > 0")

    count = int((duration_sec + 1e-9) // detector_config.len_sec)
    return [
        (
            float(index) * detector_config.len_sec,
            float(index + 1) * detector_config.len_sec,
        )
        for index in range(count)
    ]


def _stable_random_seed(
    row: dict[str, str | int | float | None],
    *,
    seed: int,
) -> int:
    row_key = str(row["video_id"])
    digest = hashlib.blake2b(
        f"{seed}:{row_key}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _build_random_detect_ranges(
    row: dict[str, str | int | float | None],
    detector_config: RandomDetectorConfig,
) -> list[tuple[float, float]]:
    duration_sec = float(row["duration_sec"])
    if duration_sec <= 0:
        raise ValueError("random detector requires duration_sec > 0")
    if detector_config.min_len_sec is None:
        raise ValueError("random detector requires min_len_sec.")

    rng = random.Random(_stable_random_seed(row, seed=detector_config.seed))
    start_sec = 0.0
    detect_ranges: list[tuple[float, float]] = []

    while duration_sec - start_sec + 1e-9 >= detector_config.min_len_sec:
        remaining_sec = duration_sec - start_sec
        upper_sec = min(detector_config.max_len_sec, remaining_sec)
        step_count = int(
            (upper_sec - detector_config.min_len_sec + 1e-9)
            // detector_config.len_step_sec
        )
        candidates = [
            detector_config.min_len_sec
            + index * detector_config.len_step_sec
            for index in range(step_count + 1)
        ]
        len_sec = rng.choice(candidates)
        end_sec = start_sec + len_sec
        detect_ranges.append((start_sec, min(end_sec, duration_sec)))
        start_sec = end_sec

    return detect_ranges


def build_detect_ranges(
    row: dict[str, str | int | float | None],
    *,
    detectors: list[DetectorConfigBase],
) -> list[tuple[float, float]]:
    if len(detectors) == 1 and isinstance(detectors[0], UniformDetectorConfig):
        return _build_uniform_detect_ranges(row, detectors[0])
    if len(detectors) == 1 and isinstance(detectors[0], RandomDetectorConfig):
        return _build_random_detect_ranges(row, detectors[0])

    pyscenedetect_detectors = [
        detector
        for detector in detectors
        if detector.detector_name in DETECTORS
        and not DETECTORS[detector.detector_name].metadata_only
    ]
    if len(pyscenedetect_detectors) != len(detectors):
        detector_names = "+".join(detector.detector_name for detector in detectors)
        raise TypeError(f"{detector_names} contains non-frame detector configs.")
    return _build_frame_detect_ranges(
        row,
        detectors=pyscenedetect_detectors,
    )
