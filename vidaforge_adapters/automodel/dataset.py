from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


AutoModelBucketKey = tuple[int, int, int, tuple[int, ...]]

AUTOMODEL_OPTIONAL_TENSOR_FIELDS = (
    "text_mask",
    "text_embeddings_2",
    "text_mask_2",
    "image_embeds",
)


class AutoModelMetaDataset(Dataset):
    """Lazy dataset for Stage 5 AutoModel `.meta` cache outputs."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        device: str | torch.device = "cpu",
        limit: int | None = None,
    ) -> None:
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")

        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.map_location = map_location
        self.device = device
        self.metadata = self._read_metadata(limit=limit)
        if not self.metadata:
            raise ValueError(f"no AutoModel metadata items found under {self.cache_dir}")

        self._bucket_keys_by_index: list[AutoModelBucketKey] = []
        self.bucket_groups: dict[AutoModelBucketKey, dict[str, object]] = {}
        for index, item in enumerate(self.metadata):
            key = bucket_key_from_item(item)
            self._bucket_keys_by_index.append(key)
            group = self.bucket_groups.setdefault(
                key,
                {
                    "indices": [],
                    "bucket_frame_count": key[0],
                    "resolution": (key[1], key[2]),
                    "latent_shape": key[3],
                },
            )
            indices = group["indices"]
            if not isinstance(indices, list):
                raise TypeError("bucket group indices must be a list")
            indices.append(index)

        self.sorted_bucket_keys = sorted(self.bucket_groups)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = self.metadata[index]
        path = cache_path_from_item(item)
        payload = torch.load(path, map_location=self.map_location, weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError(f".meta payload must be a dict: {path}")

        video_latents = payload.get("video_latents")
        text_embeddings = payload.get("text_embeddings")
        metadata = payload.get("metadata")
        if not isinstance(video_latents, torch.Tensor):
            raise TypeError(f"video_latents must be a tensor: {path}")
        if not isinstance(text_embeddings, torch.Tensor):
            raise TypeError(f"text_embeddings must be a tensor: {path}")
        if not isinstance(metadata, dict):
            raise TypeError(f"metadata must be a dict: {path}")
        if video_latents.shape[0] != 1:
            raise ValueError(f"video_latents must contain one sample per .meta: {path}")
        if text_embeddings.shape[0] != 1:
            raise ValueError(
                f"text_embeddings must contain one sample per .meta: {path}"
            )

        key = self.bucket_key_for_index(index)
        expected_latent_shape = key[3]
        if tuple(int(value) for value in video_latents.shape) != expected_latent_shape:
            raise ValueError(
                f"latent shape mismatch for {path}: "
                f"metadata={expected_latent_shape}, "
                f"actual={tuple(video_latents.shape)}"
            )

        metadata_resolution = _read_resolution(metadata, field_name="metadata")
        item_resolution = (key[1], key[2])
        if metadata_resolution != item_resolution:
            raise ValueError(
                f"bucket_resolution mismatch for {path}: "
                f"metadata shard={item_resolution}, .meta={metadata_resolution}"
            )

        metadata_frame_count = int(metadata["bucket_frame_count"])
        payload_frame_count = int(payload["bucket_frame_count"])
        if key[0] != metadata_frame_count or key[0] != payload_frame_count:
            raise ValueError(
                f"bucket_frame_count mismatch for {path}: "
                f"metadata shard={key[0]}, .meta={metadata_frame_count}, "
                f"payload={payload_frame_count}"
            )

        result: dict[str, object] = {
            "text_embeddings": text_embeddings.to(self.device),
            "video_latents": video_latents.to(self.device),
            "metadata": metadata,
            "file_info": {
                "meta_filename": path.name,
                "original_filename": payload.get("original_filename", "unknown"),
                "original_video_path": payload.get("original_video_path", "unknown"),
                "deterministic_latents": payload.get(
                    "deterministic_latents", "unknown"
                ),
                "memory_optimization": payload.get("memory_optimization", "unknown"),
                "num_frames": payload.get("num_frames", "unknown"),
            },
            "bucket_frame_count": key[0],
            "bucket_resolution": torch.tensor((key[1], key[2]), dtype=torch.int64),
            "latent_shape": key[3],
        }

        for field in AUTOMODEL_OPTIONAL_TENSOR_FIELDS:
            value = payload.get(field)
            if value is not None:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"{field} must be a tensor when present: {path}")
                result[field] = value.to(self.device)

        return result

    def bucket_key_for_index(self, index: int) -> AutoModelBucketKey:
        return self._bucket_keys_by_index[index]

    def _read_metadata(self, *, limit: int | None) -> list[dict[str, Any]]:
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing metadata.json: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        shard_names = metadata.get("shards")
        if not isinstance(shard_names, list):
            raise ValueError(f"metadata.json must contain a shards list: {metadata_path}")

        metadata: list[dict[str, Any]] = []
        for shard_name in shard_names:
            shard_path = self.cache_dir / str(shard_name)
            if not shard_path.is_file():
                raise FileNotFoundError(f"missing metadata shard: {shard_path}")
            shard_items = json.loads(shard_path.read_text(encoding="utf-8"))
            if not isinstance(shard_items, list):
                raise ValueError(f"metadata shard must contain a list: {shard_path}")

            for item in shard_items:
                if not isinstance(item, dict):
                    raise ValueError(f"metadata item must be a dict in {shard_path}")
                bucket_key_from_item(item)
                cache_path_from_item(item)
                metadata.append(item)
                if limit is not None and len(metadata) >= limit:
                    return metadata
        return metadata


def bucket_key_from_item(item: dict[str, object]) -> AutoModelBucketKey:
    resolution = _read_resolution(item, field_name="metadata item")
    latent_shape = item.get("latent_shape")
    if not isinstance(latent_shape, list | tuple) or not latent_shape:
        raise ValueError(f"latent_shape must be a non-empty list: {item}")

    return (
        int(item["bucket_frame_count"]),
        resolution[0],
        resolution[1],
        tuple(int(value) for value in latent_shape),
    )


def cache_path_from_item(item: dict[str, object]) -> Path:
    cache_file = str(item.get("cache_file", "")).strip()
    if not cache_file:
        raise ValueError(f"metadata item missing cache_file: {item}")
    return Path(cache_file).expanduser().resolve()


def _read_resolution(
    item: dict[str, object],
    *,
    field_name: str,
) -> tuple[int, int]:
    resolution = item.get("bucket_resolution")
    if not isinstance(resolution, list | tuple) or len(resolution) != 2:
        raise ValueError(f"bucket_resolution must be [width, height] in {field_name}")
    width, height = (int(resolution[0]), int(resolution[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"bucket_resolution must contain positive values: {item}")
    return (width, height)
