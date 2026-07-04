from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from vidaforge.common.asset import hash_bucketed_path, safe_file_name
from vidaforge.common.paths import join_data_dir

from .encoder import AutoModelEncodedSample, AutoModelEncoder
from .resolution import resolution_pixel_budget, resolve_bucket_resolution
from .temporal import duration_bucket_frame_counts, select_bucket_frame_count


@dataclass(slots=True)
class AutoModelPackSample:
    row_index: int
    row: dict[str, object]
    video_path: Path
    caption: str
    bucket_frame_count: int
    bucket_resolution: tuple[int, int]


def meta_path_for_clip(
    output_path: str | Path,
    *,
    clip_id: str,
    bucket_frame_count: int,
    bucket_resolution: tuple[int, int],
) -> Path:
    frame_count = int(bucket_frame_count)
    if frame_count <= 0:
        raise ValueError("bucket_frame_count must be > 0")
    width, height = (int(bucket_resolution[0]), int(bucket_resolution[1]))
    if width <= 0 or height <= 0:
        raise ValueError("bucket_resolution must contain positive width and height")

    digest = hashlib.sha1(clip_id.encode("utf-8")).hexdigest()
    safe_clip_id = safe_file_name(clip_id, default="clip")
    file_name = f"{safe_clip_id}-{digest[:12]}.meta"
    return hash_bucketed_path(
        Path(output_path).expanduser().resolve() / f"{frame_count}f" / f"{width}x{height}",
        file_name,
    )


def scaled_forward_batch_size(
    *,
    dynamic_forward_batch_size: int,
    reference_frame_count: int,
    reference_pixels: int,
    bucket_frame_count: int,
    bucket_resolution: tuple[int, int],
) -> int:
    if dynamic_forward_batch_size <= 0:
        raise ValueError("dynamic_forward_batch_size must be > 0")
    if reference_frame_count <= 0:
        raise ValueError("reference_frame_count must be > 0")
    if reference_pixels <= 0:
        raise ValueError("reference_pixels must be > 0")
    if bucket_frame_count <= 0:
        raise ValueError("bucket_frame_count must be > 0")
    bucket_width, bucket_height = (
        int(bucket_resolution[0]),
        int(bucket_resolution[1]),
    )
    if bucket_width <= 0 or bucket_height <= 0:
        raise ValueError("bucket_resolution must contain positive width and height")

    reference_cost = reference_frame_count * reference_pixels
    bucket_cost = bucket_frame_count * bucket_width * bucket_height
    return max(
        1,
        int(math.floor(dynamic_forward_batch_size * reference_cost / bucket_cost)),
    )


def write_automodel_metafile(
    meta_path: str | Path,
    *,
    row: dict[str, object],
    video_path: Path,
    caption: str,
    caption_field: str,
    run_id: str,
    input_run_id: str,
    bucket_config: dict[str, object],
    sample: AutoModelEncodedSample,
) -> None:
    path = Path(meta_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = (int(sample.bucket_resolution[0]), int(sample.bucket_resolution[1]))
    bucket_frame_count = int(sample.num_frames)
    aspect_ratio = float(width) / float(height)
    metadata: dict[str, object] = {
        "clip_id": str(row["clip_id"]),
        "video_id": str(row.get("video_id", "")),
        "caption": caption,
        "caption_field": caption_field,
        "bucket_resolution": [width, height],
        "bucket_frame_count": bucket_frame_count,
        "aspect_ratio": aspect_ratio,
        "bucket_config": dict(bucket_config),
        "latent_shape": list(sample.video_latents.shape),
        "run_id": run_id,
        "input_run_id": input_run_id,
        "row": dict(row),
    }
    metadata.update(sample.metadata)

    payload: dict[str, object] = {
        "video_latents": sample.video_latents.detach().cpu(),
        "text_embeddings": sample.text_embeddings.detach().cpu(),
        "metadata": metadata,
        "original_filename": video_path.name,
        "original_video_path": str(video_path),
        "num_frames": int(sample.num_frames),
        "bucket_frame_count": bucket_frame_count,
    }
    for key, value in sample.extra_tensors.items():
        payload[key] = value.detach().cpu()

    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temp_path)
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


class AutoModelPackWorker:
    def __init__(
        self,
        *,
        output_path: str | Path,
        run_id: str,
        input_run_id: str,
        caption_field: str = "caption_level_3",
        dynamic_forward_batch_size: int = 4,
        bucket_resolution: str = "480p",
        bucket_upscale: bool = False,
        bucket_durations_sec: list[float] | None = None,
        encoder_cls: type | None = None,
        encoder_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if dynamic_forward_batch_size <= 0:
            raise ValueError("dynamic_forward_batch_size must be > 0")
        if not caption_field.strip():
            raise ValueError("caption_field must not be empty")
        if encoder_cls is None:
            raise RuntimeError("encoder_cls is required")

        self.output_path = Path(output_path).expanduser().resolve()
        self.run_id = run_id
        self.input_run_id = input_run_id
        self.caption_field = caption_field
        self.dynamic_forward_batch_size = dynamic_forward_batch_size
        self.bucket_resolution = bucket_resolution
        self.bucket_upscale = bucket_upscale
        self.bucket_durations_sec = list(
            bucket_durations_sec or [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        )
        self.reference_pixels = resolution_pixel_budget(self.bucket_resolution)
        self.encoder_kwargs = dict(encoder_kwargs or {})
        self.encoder: AutoModelEncoder = encoder_cls(**self.encoder_kwargs)
        self.encoder_input_size_multiple = int(
            getattr(self.encoder, "input_size_multiple", 0)
        )
        if self.encoder_input_size_multiple <= 0:
            raise ValueError("encoder.input_size_multiple must be > 0")
        self.encoder_temporal_stride = int(getattr(self.encoder, "temporal_stride", 0))
        if self.encoder_temporal_stride <= 0:
            raise ValueError("encoder.temporal_stride must be > 0")

    def build_pack_row(
        self,
        *,
        row: dict[str, object],
        ok: int,
        error: str,
        cache_file: str = "",
        bucket_resolution: tuple[int, int] | None = None,
        source_resolution: tuple[int, int] | None = None,
        bucket_frame_count: int = 0,
        input_frame_count: int = 0,
        num_frames: int = 0,
        latent_shape: list[int] | None = None,
        caption_token_length: int = 0,
        caption_token_truncated: int = 0,
        caption_token_max_length: int = 0,
    ) -> dict[str, object]:
        width = int(bucket_resolution[0]) if bucket_resolution is not None else 0
        height = int(bucket_resolution[1]) if bucket_resolution is not None else 0
        source_width = int(source_resolution[0]) if source_resolution is not None else 0
        source_height = int(source_resolution[1]) if source_resolution is not None else 0
        aspect_ratio = (
            float(width) / float(height)
            if width > 0 and height > 0
            else 0.0
        )
        payload = {
            "cache_file": cache_file,
            "bucket_resolution": [width, height],
            "bucket_frame_count": int(bucket_frame_count),
            "aspect_ratio": aspect_ratio,
            "num_frames": int(num_frames),
            "input_frame_count": int(input_frame_count),
            "latent_shape": list(latent_shape or []),
            "caption_field": self.caption_field,
            "caption_token_length": int(caption_token_length),
            "caption_token_truncated": int(caption_token_truncated),
            "caption_token_max_length": int(caption_token_max_length),
            "source_resolution": [source_width, source_height],
        }

        output_row = dict(row)
        output_row.update(
            {
                "automodel_ok": int(ok),
                "automodel_error": error,
                "automodel_cache_file": cache_file,
                "automodel_bucket_width": width,
                "automodel_bucket_height": height,
                "automodel_aspect_ratio": round(aspect_ratio, 10),
                "automodel_source_width": source_width,
                "automodel_source_height": source_height,
                "automodel_bucket_frame_count": int(bucket_frame_count),
                "automodel_input_frame_count": int(input_frame_count),
                "automodel_num_frames": int(num_frames),
                "automodel_latent_shape": json.dumps(
                    list(latent_shape or []),
                    ensure_ascii=False,
                ),
                "automodel_caption_field": self.caption_field,
                "automodel_caption_token_length": int(caption_token_length),
                "automodel_caption_token_truncated": int(caption_token_truncated),
                "automodel_caption_token_max_length": int(caption_token_max_length),
                "automodel_json": json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "input_run_id": self.input_run_id,
                "run_id": self.run_id,
            }
        )
        return output_row

    def process_batch(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        output_rows: list[dict[str, object] | None] = [None] * len(rows)
        samples_by_bucket: dict[tuple[int, int, int], list[AutoModelPackSample]] = {}

        for row_index, row in enumerate(rows):
            try:
                if int(row["caption_ok"]) != 1:
                    raise ValueError("caption_ok must be 1")

                clip_id = str(row["clip_id"])
                if not clip_id:
                    raise ValueError("clip_id must not be empty")

                caption = str(row[self.caption_field]).strip()
                if not caption:
                    raise ValueError(f"{self.caption_field} must not be empty")

                video_path = join_data_dir(str(row["clip_path"]))
                if not video_path.exists():
                    raise FileNotFoundError(f"clip_path does not exist: {video_path}")

                source_width = int(row["width"])
                source_height = int(row["height"])
                source_duration_sec = float(row["duration_sec"])
                fps = float(row["fps"])
                bucket_frame_count = select_bucket_frame_count(
                    source_duration_sec=source_duration_sec,
                    fps=fps,
                    durations_sec=self.bucket_durations_sec,
                    stride=self.encoder_temporal_stride,
                )
                bucket_resolution = resolve_bucket_resolution(
                    source_width=source_width,
                    source_height=source_height,
                    resolution=self.bucket_resolution,
                    size_multiple=self.encoder_input_size_multiple,
                    upscale=self.bucket_upscale,
                )
                pack_sample = AutoModelPackSample(
                    row_index=row_index,
                    row=row,
                    video_path=video_path,
                    caption=caption,
                    bucket_frame_count=bucket_frame_count,
                    bucket_resolution=bucket_resolution,
                )
                bucket_key = (
                    bucket_frame_count,
                    int(bucket_resolution[0]),
                    int(bucket_resolution[1]),
                )
                samples_by_bucket.setdefault(bucket_key, []).append(pack_sample)
            except Exception as exc:  # noqa: BLE001
                output_rows[row_index] = self.build_pack_row(
                    row=row,
                    ok=0,
                    error=str(exc),
                )

        all_pack_samples = [
            sample
            for pack_samples in samples_by_bucket.values()
            for sample in pack_samples
        ]
        reference_frame_count = 0
        if all_pack_samples:
            max_fps = max(float(sample.row["fps"]) for sample in all_pack_samples)
            reference_frame_count = max(
                duration_bucket_frame_counts(
                    self.bucket_durations_sec,
                    fps=max_fps,
                    stride=self.encoder_temporal_stride,
                )
            )

        for bucket_key, pack_samples in samples_by_bucket.items():
            bucket_frame_count, bucket_width, bucket_height = bucket_key
            bucket_resolution = (bucket_width, bucket_height)
            chunk_size = scaled_forward_batch_size(
                dynamic_forward_batch_size=self.dynamic_forward_batch_size,
                reference_frame_count=reference_frame_count,
                reference_pixels=self.reference_pixels,
                bucket_frame_count=bucket_frame_count,
                bucket_resolution=bucket_resolution,
            )
            for start in range(0, len(pack_samples), chunk_size):
                chunk = pack_samples[start : start + chunk_size]
                video_paths = [sample.video_path for sample in chunk]
                source_resolutions = [
                    (int(sample.row["width"]), int(sample.row["height"]))
                    for sample in chunk
                ]
                source_fps = [float(sample.row["fps"]) for sample in chunk]
                captions = [sample.caption for sample in chunk]

                try:
                    encoded_samples = self.encoder.encode_batch(
                        bucket_frame_count=bucket_frame_count,
                        bucket_resolution=bucket_resolution,
                        video_paths=video_paths,
                        source_resolutions=source_resolutions,
                        source_fps=source_fps,
                        captions=captions,
                    )
                    if len(encoded_samples) != len(chunk):
                        raise ValueError(
                            "encoder returned "
                            f"{len(encoded_samples)} samples for {len(chunk)} rows"
                        )
                except Exception as exc:  # noqa: BLE001
                    for pack_sample in chunk:
                        output_rows[pack_sample.row_index] = self.build_pack_row(
                            row=pack_sample.row,
                            ok=0,
                            error=str(exc),
                        )
                    continue

                for pack_sample, encoded_sample in zip(chunk, encoded_samples):
                    try:
                        encoded_bucket_resolution = (
                            int(encoded_sample.bucket_resolution[0]),
                            int(encoded_sample.bucket_resolution[1]),
                        )
                        if encoded_bucket_resolution != pack_sample.bucket_resolution:
                            raise ValueError(
                                "encoder returned unexpected bucket_resolution: "
                                f"expected={pack_sample.bucket_resolution}, "
                                f"actual={encoded_bucket_resolution}"
                            )
                        encoded_frame_count = int(encoded_sample.num_frames)
                        if encoded_frame_count != pack_sample.bucket_frame_count:
                            raise ValueError(
                                "encoder returned unexpected frame count: "
                                f"expected={pack_sample.bucket_frame_count}, "
                                f"actual={encoded_frame_count}"
                            )
                        source_resolution = (
                            int(pack_sample.row["width"]),
                            int(pack_sample.row["height"]),
                        )
                        meta_path = meta_path_for_clip(
                            self.output_path,
                            clip_id=str(pack_sample.row["clip_id"]),
                            bucket_frame_count=pack_sample.bucket_frame_count,
                            bucket_resolution=pack_sample.bucket_resolution,
                        )
                        write_automodel_metafile(
                            meta_path,
                            row=pack_sample.row,
                            video_path=pack_sample.video_path,
                            caption=pack_sample.caption,
                            caption_field=self.caption_field,
                            run_id=self.run_id,
                            input_run_id=self.input_run_id,
                            bucket_config={
                                "resolution": self.bucket_resolution,
                                "upscale": self.bucket_upscale,
                                "durations_sec": list(self.bucket_durations_sec),
                            },
                            sample=encoded_sample,
                        )
                        latent_shape = list(encoded_sample.video_latents.shape)
                        input_frame_count = int(
                            encoded_sample.metadata.get("input_frame_count", 0) or 0
                        )
                        caption_token_length = int(
                            encoded_sample.metadata.get("caption_token_length", 0) or 0
                        )
                        caption_token_truncated = int(
                            bool(
                                encoded_sample.metadata.get(
                                    "caption_token_truncated",
                                    False,
                                )
                            )
                        )
                        caption_token_max_length = int(
                            encoded_sample.metadata.get(
                                "caption_token_max_length",
                                encoded_sample.metadata.get("max_sequence_length", 0),
                            )
                            or 0
                        )
                        output_rows[pack_sample.row_index] = self.build_pack_row(
                            row=pack_sample.row,
                            ok=1,
                            error="",
                            cache_file=str(meta_path),
                            bucket_resolution=pack_sample.bucket_resolution,
                            source_resolution=source_resolution,
                            bucket_frame_count=pack_sample.bucket_frame_count,
                            input_frame_count=input_frame_count,
                            num_frames=encoded_frame_count,
                            latent_shape=latent_shape,
                            caption_token_length=caption_token_length,
                            caption_token_truncated=caption_token_truncated,
                            caption_token_max_length=caption_token_max_length,
                        )
                    except Exception as exc:  # noqa: BLE001
                        output_rows[pack_sample.row_index] = self.build_pack_row(
                            row=pack_sample.row,
                            ok=0,
                            error=str(exc),
                        )

        return [row for row in output_rows if row is not None]
