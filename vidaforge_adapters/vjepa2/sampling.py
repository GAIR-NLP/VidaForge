from __future__ import annotations

import math
from typing import Any

import numpy as np


def compute_vjepa_indices(
    *,
    total_frames: int,
    video_fps: float,
    frames_per_clip: int,
    frame_step: int | None,
    duration: float | None,
    fps: int | None,
    num_clips: int,
    random_clip_sampling: bool,
    allow_clip_overlap: bool,
    filter_short_videos: bool,
    seed: int | None = None,
    rng: Any | None = None,
) -> tuple[list[int], list[np.ndarray], dict[str, Any]]:
    """Compute the temporal indices used by the official V-JEPA VideoDataset.

    The formula intentionally mirrors `loadvideo_decord()` from V-JEPA2. The
    caller owns the `total_frames` source. For TorchCodec-backed loading this
    should be `len(VideoDecoder)`, not container `nb_frames`.
    """
    specified = sum(value is not None for value in (fps, duration, frame_step))
    if specified != 1:
        raise ValueError("exactly one of fps, duration, or frame_step must be set")
    if frames_per_clip <= 0:
        raise ValueError("frames_per_clip must be > 0")
    if num_clips <= 0:
        raise ValueError("num_clips must be > 0")
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")

    effective_frame_step = frame_step
    if duration is not None or fps is not None:
        if video_fps <= 0:
            raise ValueError(f"video_fps must be > 0 when using duration/fps sampling, got {video_fps}")
        rounded_video_fps = math.ceil(video_fps)
        if duration is not None:
            effective_frame_step = int(duration * rounded_video_fps / frames_per_clip)
        else:
            assert fps is not None
            effective_frame_step = rounded_video_fps // fps

    if effective_frame_step is None or effective_frame_step <= 0:
        raise ValueError(f"effective frame step must be > 0, got {effective_frame_step}")

    clip_len = int(frames_per_clip * effective_frame_step)
    if filter_short_videos and total_frames < clip_len:
        raise ValueError(
            f"video is shorter than requested clip length: total_frames={total_frames}, clip_len={clip_len}"
        )

    partition_len = total_frames // num_clips
    if partition_len <= 0:
        raise ValueError(f"partition_len must be > 0, got {partition_len}")

    random_source = rng
    if seed is not None:
        random_source = np.random.RandomState(seed)
    if random_source is None:
        random_source = np.random

    all_indices: list[int] = []
    clip_indices: list[np.ndarray] = []
    for clip_idx in range(num_clips):
        if partition_len > clip_len:
            end_index = clip_len
            if random_clip_sampling:
                end_index = int(random_source.randint(clip_len, partition_len))
            start_index = end_index - clip_len
            indices = np.linspace(start_index, end_index, num=frames_per_clip)
            indices = np.clip(indices, start_index, end_index - 1).astype(np.int64)
            indices = indices + clip_idx * partition_len
        elif not allow_clip_overlap:
            sampled_count = partition_len // effective_frame_step
            indices = np.linspace(0, partition_len, num=sampled_count)
            pad_count = frames_per_clip - sampled_count
            if pad_count > 0:
                indices = np.concatenate((indices, np.ones(pad_count) * partition_len))
            indices = np.clip(indices, 0, partition_len - 1).astype(np.int64)
            indices = indices + clip_idx * partition_len
        else:
            sample_len = min(clip_len, total_frames) - 1
            sampled_count = sample_len // effective_frame_step
            indices = np.linspace(0, sample_len, num=sampled_count)
            pad_count = frames_per_clip - sampled_count
            if pad_count > 0:
                indices = np.concatenate((indices, np.ones(pad_count) * sample_len))
            indices = np.clip(indices, 0, sample_len - 1).astype(np.int64)
            clip_step = 0
            if total_frames > clip_len and num_clips > 1:
                clip_step = (total_frames - clip_len) // (num_clips - 1)
            indices = indices + clip_idx * clip_step

        clip_indices.append(indices)
        all_indices.extend(int(value) for value in indices.tolist())

    stats = {
        "effective_frame_step": int(effective_frame_step),
        "clip_len": int(clip_len),
        "partition_len": int(partition_len),
    }
    return all_indices, clip_indices, stats
