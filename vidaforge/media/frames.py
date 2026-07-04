"""Frame geometry helpers."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypeVar

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision.io import ImageReadMode, decode_image

ItemT = TypeVar("ItemT")


def build_short_side_scale_filter(
    short_side: int,
    *,
    allow_upscale: bool = True,
    use_gpu: bool = False,
) -> str:
    """Build an FFmpeg scale filter targeting a specific short side."""
    if short_side <= 0:
        raise ValueError("short_side must be > 0.")
    if not allow_upscale:
        width_expr = f"trunc(iw*min(1\\,{short_side}/min(iw\\,ih))/2)*2"
        height_expr = f"trunc(ih*min(1\\,{short_side}/min(iw\\,ih))/2)*2"
        if use_gpu:
            return (
                "scale_cuda="
                f"w='{width_expr}':"
                f"h='{height_expr}':"
                "format=nv12"
            )
        return f"scale={width_expr}:{height_expr}"
    if use_gpu:
        raise ValueError("GPU short-side scale currently requires allow_upscale=False.")
    return (
        f"scale='if(lte(iw,ih),{short_side},-2)':"
        f"'if(lte(iw,ih),-2,{short_side})'"
    )


def scaled_size(*, input_width: int, input_height: int, short_side: int) -> tuple[int, int]:
    if input_width <= 0 or input_height <= 0:
        raise ValueError("input_width and input_height must be > 0.")
    if short_side <= 0:
        raise ValueError("short_side must be > 0.")

    if input_width <= input_height:
        frame_width = short_side
        frame_height = int(round(input_height * short_side / input_width))
    else:
        frame_height = short_side
        frame_width = int(round(input_width * short_side / input_height))

    # Match ffmpeg scale=-2 behavior: keep dimensions divisible by 2.
    if frame_width % 2:
        frame_width += 1
    if frame_height % 2:
        frame_height += 1
    return frame_width, frame_height


def load_bgr_image_arrays(
    paths: list[Path],
    *,
    max_workers: int = 1,
) -> list[np.ndarray]:
    """Load images as OpenCV BGR ndarrays."""
    if not paths:
        raise ValueError("paths must not be empty")
    if max_workers <= 0:
        raise ValueError("max_workers must be > 0")

    def load(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {path}")
        return image

    if max_workers == 1:
        return [load(path) for path in paths]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(load, paths))


def load_rgb_image_arrays(
    paths: list[Path],
    *,
    max_workers: int = 1,
) -> list[np.ndarray]:
    """Load images as RGB ndarrays."""
    return [
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for image in load_bgr_image_arrays(paths, max_workers=max_workers)
    ]


def load_rgb_pil_images(paths: list[Path]) -> list[Image.Image]:
    """Load images as RGB PIL images."""
    if not paths:
        raise ValueError("paths must not be empty")

    images: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def load_rgb_frame_tensors(
    paths: list[Path],
    *,
    max_workers: int = 1,
) -> list[torch.Tensor]:
    """Load images as uint8 RGB tensors with shape [C, H, W]."""
    if not paths:
        raise ValueError("paths must not be empty")
    if max_workers <= 0:
        raise ValueError("max_workers must be > 0")

    if max_workers == 1:
        return [
            decode_image(str(path), mode=ImageReadMode.RGB)
            for path in paths
        ]

    def load(path: Path) -> torch.Tensor:
        return decode_image(str(path), mode=ImageReadMode.RGB)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(load, paths))


def iter_rgb_frame_tensor_batches(
    paths: list[Path],
    *,
    batch_size: int,
    max_workers: int = 1,
    prefetch_batches: int = 0,
) -> Iterator[list[torch.Tensor]]:
    """Yield RGB frame tensors loaded in bounded path batches."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if max_workers <= 0:
        raise ValueError("max_workers must be > 0")
    if prefetch_batches < 0:
        raise ValueError("prefetch_batches must be >= 0")
    if not paths:
        return

    path_batches = [
        paths[start : start + batch_size]
        for start in range(0, len(paths), batch_size)
    ]
    if prefetch_batches == 0:
        for path_batch in path_batches:
            yield load_rgb_frame_tensors(
                path_batch,
                max_workers=max_workers,
            )
        return

    max_pending = prefetch_batches
    with ThreadPoolExecutor(max_workers=max_pending) as prefetch_pool:
        pending: deque[Future[list[torch.Tensor]]] = deque()
        next_batch_index = 0

        def submit_until_full() -> None:
            nonlocal next_batch_index
            while (
                next_batch_index < len(path_batches)
                and len(pending) < max_pending
            ):
                path_batch = path_batches[next_batch_index]
                pending.append(
                    prefetch_pool.submit(
                        load_rgb_frame_tensors,
                        path_batch,
                        max_workers=max_workers,
                    )
                )
                next_batch_index += 1

        submit_until_full()
        while pending:
            frame_batch = pending.popleft().result()
            submit_until_full()
            yield frame_batch


def load_bgr_frame_tensors(
    paths: list[Path],
    *,
    max_workers: int = 1,
) -> list[torch.Tensor]:
    """Load images as uint8 BGR tensors with shape [C, H, W]."""
    if not paths:
        raise ValueError("paths must not be empty")
    if max_workers <= 0:
        raise ValueError("max_workers must be > 0")

    def load(path: Path) -> torch.Tensor:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {path}")
        return torch.from_numpy(image).permute(2, 0, 1).contiguous()

    if max_workers == 1:
        return [load(path) for path in paths]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(load, paths))


def select_uniform_items(
    items: Sequence[ItemT],
    *,
    target_count: int,
    description: str = "items",
) -> list[ItemT]:
    if target_count <= 0:
        raise ValueError("target_count must be > 0")
    if not items:
        raise ValueError(f"{description} must not be empty")
    if len(items) == target_count:
        return list(items)

    positions = np.linspace(0, len(items) - 1, target_count)
    indices = np.rint(positions).astype(np.int64)
    return [items[int(index)] for index in indices]
