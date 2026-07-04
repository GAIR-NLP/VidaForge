from __future__ import annotations

import math


def resolution_pixel_budget(resolution: str) -> int:
    text = str(resolution).strip().lower()
    if not text.endswith("p"):
        raise ValueError(f"bucket.resolution must look like '480p', got {resolution!r}")

    reference_height = int(text[:-1])
    if reference_height <= 0:
        raise ValueError("bucket.resolution height must be > 0")
    return int(round(reference_height * reference_height * 16 / 9))


def resolve_bucket_resolution(
    *,
    source_width: int,
    source_height: int,
    resolution: str,
    size_multiple: int,
    upscale: bool,
) -> tuple[int, int]:
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source width and height must be > 0")
    if size_multiple <= 0:
        raise ValueError("size_multiple must be > 0")

    max_pixels = resolution_pixel_budget(resolution)
    source_pixels = source_width * source_height
    scale = (
        math.sqrt(max_pixels / source_pixels)
        if upscale or source_pixels > max_pixels
        else 1.0
    )

    width = int(math.floor(source_width * scale / size_multiple)) * size_multiple
    height = int(math.floor(source_height * scale / size_multiple)) * size_multiple
    if width <= 0 or height <= 0:
        raise ValueError(
            "resolved bucket resolution is empty after size_multiple alignment: "
            f"source={source_width}x{source_height}, resolution={resolution!r}, "
            f"size_multiple={size_multiple}"
        )
    if width * height > max_pixels:
        raise ValueError(
            "resolved bucket resolution exceeds pixel budget: "
            f"resolved={width}x{height}, resolution={resolution!r}, "
            f"max_pixels={max_pixels}"
        )
    return width, height


__all__ = [
    "resolution_pixel_budget",
    "resolve_bucket_resolution",
]
