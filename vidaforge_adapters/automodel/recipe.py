from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch

from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe


class VidaForgeTrainDiffusionRecipe(TrainDiffusionRecipe):
    """AutoModel diffusion recipe with a complete mixed-precision forward context.

    AutoModel keeps the model in FP32 while using BF16 for forward computation.
    FSDP normally casts managed parameters to the compute dtype, but it skips that
    path for a one-rank mesh and some model boundary modules can remain FP32. CUDA
    autocast handles those boundary operations while preserving FP32 master weights.
    """

    @contextmanager
    def _transformer_engine_fp8_context(self) -> Iterator[Any]:
        with super()._transformer_engine_fp8_context():
            with self._forward_autocast_context():
                yield

    def _forward_autocast_context(self) -> Any:
        enabled = (
            self.device.type == "cuda"
            and self.compute_dtype in (torch.float16, torch.bfloat16)
            and self.model_dtype != self.compute_dtype
        )
        return torch.autocast(
            device_type="cuda",
            dtype=self.compute_dtype,
            enabled=enabled,
        )
