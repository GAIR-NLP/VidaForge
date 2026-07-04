from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path

from vidaforge.common import join_data_dir

from .resolution import resolution_pixel_budget


_RESOLUTION_DIR_RE = re.compile(r"^\d+x\d+$")
_TEMPORAL_BUCKET_DIR_RE = re.compile(r"^\d+f$")


def prepare_output_path(output_path: str | Path, *, resume: bool) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)

    metadata_path = path / "metadata.json"
    metadata_path.unlink(missing_ok=True)
    shutil.rmtree(path / "shards", ignore_errors=True)

    if resume:
        return path

    for child in path.iterdir():
        if child.is_dir() and (
            _RESOLUTION_DIR_RE.match(child.name)
            or _TEMPORAL_BUCKET_DIR_RE.match(child.name)
        ):
            shutil.rmtree(child)
    for temp_path in path.rglob("*.meta.tmp"):
        temp_path.unlink(missing_ok=True)
    return path


def row_is_complete(row: dict[str, object]) -> bool:
    if int(row["automodel_ok"]) != 1:
        return False
    cache_file = str(row.get("automodel_cache_file", "")).strip()
    if not cache_file:
        return False
    cache_path = Path(cache_file).expanduser().resolve()
    try:
        return cache_path.is_file() and cache_path.stat().st_size > 0
    except OSError:
        return False


def write_metadata_files(
    *,
    output_path: str | Path,
    rows: list[dict[str, object]],
    metadata_shard_size: int,
) -> dict[str, object]:
    if metadata_shard_size <= 0:
        raise ValueError("metadata_shard_size must be > 0")

    path = Path(output_path).expanduser().resolve()
    shards_path = path / "shards"
    shutil.rmtree(shards_path, ignore_errors=True)
    shards_path.mkdir(parents=True, exist_ok=True)

    shard_files: list[str] = []
    item_count = 0
    for shard_index, start in enumerate(range(0, len(rows), metadata_shard_size)):
        shard_rows = rows[start : start + metadata_shard_size]
        shard_items: list[dict[str, object]] = []
        for row in shard_rows:
            width = int(row["automodel_bucket_width"])
            height = int(row["automodel_bucket_height"])
            bucket_frame_count = int(row["automodel_bucket_frame_count"])
            latent_shape = json.loads(str(row.get("automodel_latent_shape", "[]")))
            shard_items.append(
                {
                    "cache_file": str(row["automodel_cache_file"]),
                    "bucket_resolution": [width, height],
                    "bucket_frame_count": bucket_frame_count,
                    "latent_shape": latent_shape,
                    "aspect_ratio": float(row["automodel_aspect_ratio"]),
                    "clip_id": str(row["clip_id"]),
                    "video_id": str(row.get("video_id", "")),
                    "original_video_path": str(join_data_dir(str(row["clip_path"]))),
                    "source_resolution": [
                        int(row.get("automodel_source_width", 0) or 0),
                        int(row.get("automodel_source_height", 0) or 0),
                    ],
                    "caption_token_length": int(
                        row.get("automodel_caption_token_length", 0) or 0
                    ),
                    "caption_token_truncated": bool(
                        int(row.get("automodel_caption_token_truncated", 0) or 0)
                    ),
                    "caption_token_max_length": int(
                        row.get("automodel_caption_token_max_length", 0) or 0
                    ),
                }
            )
        shard_path = shards_path / f"metadata-{shard_index:06d}.json"
        shard_path.write_text(
            json.dumps(shard_items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        shard_files.append(str(shard_path.relative_to(path)))
        item_count += len(shard_items)

    metadata_path = path / "metadata.json"
    metadata_path.write_text(
        json.dumps({"shards": shard_files}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "metadata_path": str(metadata_path),
        "metadata_shard_count": len(shard_files),
        "metadata_row_count": item_count,
    }


def build_resolution_summary(
    rows: list[dict[str, object]],
    *,
    target_resolution: str,
) -> dict[str, object]:
    bucket_counter: Counter[str] = Counter()
    frame_counter: Counter[str] = Counter()
    temporal_spatial_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    caption_token_lengths: list[int] = []
    caption_token_truncated_count = 0
    below_target_count = 0
    target_pixels = resolution_pixel_budget(target_resolution)

    for row in rows:
        bucket_name = (
            f"{int(row['automodel_bucket_width'])}x{int(row['automodel_bucket_height'])}"
        )
        bucket_counter[bucket_name] += 1
        frame_count = int(row.get("automodel_bucket_frame_count", 0) or 0)
        if frame_count > 0:
            frame_counter[f"{frame_count}f"] += 1
            temporal_spatial_counter[f"{frame_count}f/{bucket_name}"] += 1

        caption_token_length = int(row.get("automodel_caption_token_length", 0) or 0)
        if caption_token_length > 0:
            caption_token_lengths.append(caption_token_length)
            caption_token_truncated_count += int(
                row.get("automodel_caption_token_truncated", 0) or 0
            )

        width = int(row.get("automodel_source_width", 0) or 0)
        height = int(row.get("automodel_source_height", 0) or 0)
        if width <= 0 or height <= 0:
            source_counter["unknown"] += 1
            continue

        if width * height < target_pixels:
            below_target_count += 1

        short_side = min(width, height)
        if short_side < 360:
            source_counter["short_side_<360"] += 1
        elif short_side < 480:
            source_counter["short_side_360_479"] += 1
        elif short_side < 720:
            source_counter["short_side_480_719"] += 1
        elif short_side < 1080:
            source_counter["short_side_720_1079"] += 1
        else:
            source_counter["short_side_>=1080"] += 1

    return {
        "bucket_resolution_distribution": _counter_distribution(bucket_counter),
        "bucket_frame_count_distribution": _counter_distribution(frame_counter),
        "bucket_temporal_spatial_distribution": _counter_distribution(
            temporal_spatial_counter
        ),
        "source_resolution_distribution": _counter_distribution(source_counter),
        "source_below_target_resolution_count": below_target_count,
        "source_below_target_resolution_ratio": (
            round(below_target_count / len(rows), 6) if rows else 0.0
        ),
        "caption_token_count": len(caption_token_lengths),
        "caption_token_truncated_count": caption_token_truncated_count,
        "caption_token_truncated_ratio": (
            round(caption_token_truncated_count / len(caption_token_lengths), 6)
            if caption_token_lengths
            else 0.0
        ),
        "caption_token_length_mean": _mean(caption_token_lengths),
        "caption_token_length_p95": _percentile(caption_token_lengths, 95),
        "caption_token_length_p99": _percentile(caption_token_lengths, 99),
        "caption_token_length_max": max(caption_token_lengths, default=0),
    }


def _counter_distribution(counter: Counter[str]) -> dict[str, dict[str, float | int]]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {
        key: {
            "count": count,
            "ratio": round(count / total, 6),
        }
        for key, count in sorted(counter.items())
    }


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    if percentile < 0 or percentile > 100:
        raise ValueError("percentile must be between 0 and 100")

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower_index = int(math.floor(rank))
    upper_index = int(math.ceil(rank))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = rank - lower_index
    return round(lower_value + (upper_value - lower_value) * weight, 3)
