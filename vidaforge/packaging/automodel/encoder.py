from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch


@dataclass(slots=True)
class AutoModelEncodedSample:
    video_latents: torch.Tensor
    text_embeddings: torch.Tensor
    bucket_resolution: tuple[int, int]
    num_frames: int
    metadata: dict[str, object] = field(default_factory=dict)
    extra_tensors: dict[str, torch.Tensor] = field(default_factory=dict)


class AutoModelEncoder(Protocol):
    input_size_multiple: int

    def encode_batch(
        self,
        *,
        bucket_frame_count: int,
        bucket_resolution: tuple[int, int],
        video_paths: list[Path],
        source_resolutions: list[tuple[int, int]],
        source_fps: list[float],
        captions: list[str],
    ) -> list[AutoModelEncodedSample]:
        """Encode one model batch into per-sample AutoModel cache tensors."""
