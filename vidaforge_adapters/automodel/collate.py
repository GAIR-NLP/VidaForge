from __future__ import annotations

import torch

from .dataset import AUTOMODEL_OPTIONAL_TENSOR_FIELDS, AutoModelBucketKey


def collate_automodel_video(
    batch: list[dict[str, object]],
    *,
    model_type: str = "wan",
    include_metadata: bool = False,
) -> dict[str, object]:
    """Collate Stage 5 AutoModel `.meta` samples for video diffusion training."""
    if not batch:
        raise ValueError("cannot collate an empty AutoModel batch")

    bucket_keys = [_sample_bucket_key(item) for item in batch]
    if len(set(bucket_keys)) != 1:
        raise ValueError(f"mixed AutoModel bucket in one batch: {set(bucket_keys)}")

    video_latents = _cat_tensor_field(batch, "video_latents")
    text_embeddings = _cat_tensor_field(batch, "text_embeddings")
    result: dict[str, object] = {
        "video_latents": video_latents,
        "text_embeddings": text_embeddings,
        "data_type": "video",
    }

    for field in AUTOMODEL_OPTIONAL_TENSOR_FIELDS:
        present = [field in item for item in batch]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(f"{field} is present for only part of the batch")
        result[field] = _cat_tensor_field(batch, field)

    if include_metadata:
        result["metadata"] = [item["metadata"] for item in batch]
        result["file_info"] = [item["file_info"] for item in batch]

    return result


def collate_automodel_meta(batch: list[dict[str, object]]) -> dict[str, object]:
    """Debug collate that keeps metadata/file_info in the returned batch."""
    return collate_automodel_video(batch, include_metadata=True)


def _cat_tensor_field(batch: list[dict[str, object]], field: str) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for item in batch:
        value = item[field]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{field} must be a tensor")
        tensors.append(value)

    first_shape = tuple(tensors[0].shape)
    for tensor in tensors:
        if tuple(tensor.shape) != first_shape:
            raise ValueError(f"{field} shapes differ within one AutoModel batch")
    return torch.cat(tensors, dim=0)


def _sample_bucket_key(item: dict[str, object]) -> AutoModelBucketKey:
    resolution = _resolution_tuple(item["bucket_resolution"])
    latent_shape = item["latent_shape"]
    if not isinstance(latent_shape, tuple):
        raise TypeError("latent_shape must be a tuple")
    return (
        int(item["bucket_frame_count"]),
        resolution[0],
        resolution[1],
        tuple(int(value) for value in latent_shape),
    )


def _resolution_tuple(value: object) -> tuple[int, int]:
    if isinstance(value, torch.Tensor):
        values = value.tolist()
    else:
        values = value
    if not isinstance(values, list | tuple) or len(values) != 2:
        raise ValueError("bucket_resolution must be [width, height]")
    return (int(values[0]), int(values[1]))
