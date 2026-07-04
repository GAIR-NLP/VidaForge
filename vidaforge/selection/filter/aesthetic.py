from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch

from vidaforge.common import join_data_dir, parse_json_object
from vidaforge.filters import linear_growth
from vidaforge.media.frames import (
    iter_rgb_frame_tensor_batches,
    load_rgb_frame_tensors,
)

from .base import FilterBase
from .config import AestheticFilterConfig

_RAW_SCORE_MIN = 1.0
_RAW_SCORE_MAX = 10.0
_SCORE_PERCENTILE = 20.0


class AestheticPredictor:
    def __init__(self, config: AestheticFilterConfig) -> None:
        import torch
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

        self.config = config.model_copy(deep=True)
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda requested but torch.cuda.is_available() is false"
            )

        dtype_by_name = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.device = torch.device(self.config.device)
        self.dtype = dtype_by_name[self.config.dtype]

        model, preprocessor = convert_v2_5_from_siglip(
            predictor_name_or_path=str(self.config.predictor_path.expanduser()),
            encoder_model_name=str(self.config.encoder_path.expanduser()),
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self._model = model.to(
            device=self.device, dtype=self.dtype
        ).eval()
        self._preprocessor = preprocessor

    def predict(
        self,
        frames: list[torch.Tensor],
    ) -> list[float]:
        if len(frames) == 0:
            raise ValueError("frames must not be empty")

        pixel_values = self._preprocessor(
            images=frames,
            return_tensors="pt",
            device=self.device,
        ).pixel_values
        pixel_values = pixel_values.to(dtype=self.dtype)

        with torch.inference_mode():
            logits = self._model(pixel_values).logits.reshape(-1)

        return [
            float(score)
            for score in logits.float().cpu().numpy().astype(np.float64).tolist()
        ]


class AestheticFilter(FilterBase[AestheticFilterConfig]):
    filter_name: ClassVar[str] = "aesthetic"
    config_type: ClassVar[type[AestheticFilterConfig]] = AestheticFilterConfig

    def __init__(self, config: AestheticFilterConfig) -> None:
        if not isinstance(config, AestheticFilterConfig):
            raise TypeError(f"expected AestheticFilterConfig, got {type(config)!r}")
        self.config = config
        self.predictor = AestheticPredictor(config)

    def score(
        self,
        raw_scores: list[float],
    ) -> tuple[float, dict[str, object]]:
        if not raw_scores:
            raise ValueError("raw_scores must not be empty")

        frame_scores = [
            linear_growth(raw_score, a=_RAW_SCORE_MIN, b=_RAW_SCORE_MAX)
            for raw_score in raw_scores
        ]
        p20 = round(float(np.percentile(frame_scores, _SCORE_PERCENTILE)), 6)
        payload: dict[str, object] = {
            "p20": p20,
            "mean": round(float(np.mean(frame_scores)), 6),
            "frame_count": len(frame_scores),
        }
        return p20, payload

    def apply(self, row: dict[str, object]) -> tuple[float, dict[str, object]]:
        frame_json = parse_json_object(row["frame_json"], description="frame_json")
        frame_paths = [join_data_dir(str(path)) for path in frame_json["frame_paths"]]
        frames = load_rgb_frame_tensors(frame_paths)
        raw_scores = self.predictor.predict(frames)
        return self.score(raw_scores)

    def apply_batch(
        self,
        rows: list[dict[str, object]],
    ) -> list[tuple[float, dict[str, object]]]:
        frame_counts: list[int] = []
        flat_frame_paths = []

        for row in rows:
            frame_json = parse_json_object(row["frame_json"], description="frame_json")
            frame_paths = [
                join_data_dir(str(path)) for path in frame_json["frame_paths"]
            ]
            if not frame_paths:
                raise ValueError("frame_json.frame_paths must not be empty")
            frame_counts.append(len(frame_paths))
            flat_frame_paths.extend(frame_paths)

        flat_raw_scores: list[float] = []
        for frame_batch in iter_rgb_frame_tensor_batches(
            flat_frame_paths,
            batch_size=self.config.forward_batch_size,
            max_workers=self.config.frame_load_workers,
            prefetch_batches=self.config.prefetch_batches,
        ):
            batch_scores = self.predictor.predict(frame_batch)
            if len(batch_scores) != len(frame_batch):
                raise RuntimeError(
                    f"aesthetic predictor returned {len(batch_scores)} scores "
                    f"for {len(frame_batch)} frames"
                )
            flat_raw_scores.extend(batch_scores)

        results: list[tuple[float, dict[str, object]]] = []
        offset = 0
        for frame_count in frame_counts:
            next_offset = offset + frame_count
            raw_scores = flat_raw_scores[offset:next_offset]
            offset = next_offset
            results.append(self.score(raw_scores))

        return results
