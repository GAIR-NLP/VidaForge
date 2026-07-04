from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import torch

from vidaforge.common import join_data_dir, parse_json_object
from vidaforge.media.frames import (
    iter_rgb_frame_tensor_batches,
    load_rgb_frame_tensors,
)

from .base import FilterBase
from .config import TextFilterConfig


def text_area_ratio(
    result: dict[str, Any],
    *,
    image_shape: tuple[int, int],
    text_min_confidence: float,
) -> tuple[float, int]:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    box_count = 0
    boxes = result["boxes"]
    scores = result["scores"]
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    else:
        boxes = np.asarray(boxes)
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    else:
        scores = np.asarray(scores)

    for box, score in zip(boxes, scores, strict=True):
        if float(score) < text_min_confidence:
            continue
        box_array = np.asarray(box, dtype=np.float32)
        if box_array.shape == (4,):
            x0, y0, x1, y1 = box_array.tolist()
            points = np.asarray(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                dtype=np.float32,
            )
        else:
            points = box_array.reshape(-1, 2)
        points = points.round().astype(np.int32)
        if len(points) < 3:
            continue
        cv2.fillPoly(mask, [points], 1)
        box_count += 1

    return float(mask.mean()), box_count


class TextDetector:
    def __init__(self, config: TextFilterConfig) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda requested but torch.cuda.is_available() is false"
            )

        self.device = torch.device(config.device)
        model_path = (
            str(Path(config.model_path).expanduser())
            if config.model_path.startswith("~")
            else config.model_path
        )
        self._model = AutoModelForObjectDetection.from_pretrained(
            model_path,
            trust_remote_code=True,
        ).to(self.device).eval()
        self._processor = AutoImageProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

    def predict(self, frames: list[torch.Tensor]) -> list[dict[str, Any]]:
        if not frames:
            raise ValueError("frames must not be empty")

        features = self._processor(
            images=frames,
            return_tensors=None,
            device=self.device,
        )
        pixel_values = features["pixel_values"]
        target_sizes = features["target_sizes"]
        if len(pixel_values) != len(frames):
            raise RuntimeError(
                f"text processor returned {len(pixel_values)} pixel values for "
                f"{len(frames)} frames"
            )
        if len(target_sizes) != len(frames):
            raise RuntimeError(
                f"text processor returned {len(target_sizes)} target sizes for "
                f"{len(frames)} frames"
            )

        # PP-OCRv5's HF processor unconditionally swaps BGR->RGB internally.
        # Our loader returns RGB tensors, so swap once more after preprocessing
        # to keep the tensor presented to the model in RGB order.
        pixel_values = [pixel_value[[2, 1, 0], :, :] for pixel_value in pixel_values]

        results: list[dict[str, Any] | None] = [None] * len(frames)
        items_by_shape: dict[
            tuple[int, int],
            list[tuple[int, torch.Tensor, torch.Tensor]],
        ] = {}
        for index, (pixel_value, target_size) in enumerate(
            zip(pixel_values, target_sizes, strict=True)
        ):
            shape = (int(pixel_value.shape[-2]), int(pixel_value.shape[-1]))
            items_by_shape.setdefault(shape, []).append(
                (index, pixel_value, target_size)
            )

        for grouped_items in items_by_shape.values():
            pixel_batch = torch.stack(
                [pixel_value for _, pixel_value, _ in grouped_items],
                dim=0,
            ).to(self.device)
            target_batch = torch.stack(
                [
                    target_size
                    if isinstance(target_size, torch.Tensor)
                    else torch.as_tensor(target_size)
                    for _, _, target_size in grouped_items
                ],
                dim=0,
            ).to(self.device)

            with torch.inference_mode():
                outputs = self._model(pixel_values=pixel_batch)

            group_results = self._processor.post_process_object_detection(
                outputs,
                target_sizes=target_batch,
            )

            if len(group_results) != len(grouped_items):
                raise RuntimeError(
                    f"text detector returned {len(group_results)} results for "
                    f"{len(grouped_items)} frames"
                )
            for (index, _, _), result in zip(
                grouped_items,
                group_results,
                strict=True,
            ):
                results[index] = result

        final_results: list[dict[str, Any]] = []
        for result in results:
            if result is None:
                raise RuntimeError("text detector did not produce a result for a frame")
            final_results.append(result)
        return final_results


class TextFilter(FilterBase[TextFilterConfig]):
    filter_name: ClassVar[str] = "text"
    config_type: ClassVar[type[TextFilterConfig]] = TextFilterConfig

    def __init__(self, config: TextFilterConfig) -> None:
        if not isinstance(config, TextFilterConfig):
            raise TypeError(f"expected TextFilterConfig, got {type(config)!r}")
        self.config = config
        self.detector = TextDetector(config)

    def score(
        self,
        results: list[dict[str, Any]],
        frames: list[torch.Tensor],
    ) -> tuple[float, dict[str, object]]:
        if not results:
            raise ValueError("results must not be empty")
        if len(results) != len(frames):
            raise ValueError("results and frames must have the same length")

        text_ratios: list[float] = []
        box_counts: list[int] = []
        for result, frame in zip(results, frames, strict=True):
            image_shape = (int(frame.shape[-2]), int(frame.shape[-1]))
            text_ratio, box_count = text_area_ratio(
                result,
                image_shape=image_shape,
                text_min_confidence=self.config.text_min_confidence,
            )
            text_ratios.append(text_ratio)
            box_counts.append(box_count)

        return self.aggregate_frame_metrics(
            text_ratios=text_ratios,
            box_counts=box_counts,
        )

    def aggregate_frame_metrics(
        self,
        *,
        text_ratios: list[float],
        box_counts: list[int],
    ) -> tuple[float, dict[str, object]]:
        if not text_ratios:
            raise ValueError("text_ratios must not be empty")
        if len(text_ratios) != len(box_counts):
            raise ValueError("text_ratios and box_counts must have the same length")

        text_ratio = float(
            np.percentile(
                np.asarray(text_ratios, dtype=np.float64),
                self.config.text_ratio_quantile * 100.0,
            )
        )
        score = max(0.0, min(1.0, 1.0 - text_ratio))
        payload: dict[str, object] = {
            "text_ratio": round(text_ratio, 6),
            "text_ratio_mean": round(float(np.mean(text_ratios)), 6),
            "detected_frame_ratio": round(
                float(np.mean([count > 0 for count in box_counts])),
                6,
            ),
            "box_count_mean": round(float(np.mean(box_counts)), 6),
            "frame_count": len(text_ratios),
        }
        return round(score, 6), payload

    def apply(self, row: dict[str, object]) -> tuple[float, dict[str, object]]:
        frame_json = parse_json_object(row["frame_json"], description="frame_json")
        frame_paths = [join_data_dir(str(path)) for path in frame_json["frame_paths"]]
        if not frame_paths:
            raise ValueError("frame_json.frame_paths must not be empty")

        frames = load_rgb_frame_tensors(
            frame_paths,
            max_workers=self.config.frame_load_workers,
        )
        results = self.detector.predict(frames)
        return self.score(results, frames)

    def apply_batch(
        self,
        rows: list[dict[str, object]],
    ) -> list[tuple[float, dict[str, object]]]:
        frame_counts: list[int] = []
        flat_frame_paths: list[Path] = []

        for row in rows:
            frame_json = parse_json_object(row["frame_json"], description="frame_json")
            frame_paths = [
                join_data_dir(str(path)) for path in frame_json["frame_paths"]
            ]
            if not frame_paths:
                raise ValueError("frame_json.frame_paths must not be empty")
            frame_counts.append(len(frame_paths))
            flat_frame_paths.extend(frame_paths)

        flat_text_ratios: list[float] = []
        flat_box_counts: list[int] = []
        for frame_batch in iter_rgb_frame_tensor_batches(
            flat_frame_paths,
            batch_size=self.config.forward_batch_size,
            max_workers=self.config.frame_load_workers,
            prefetch_batches=self.config.prefetch_batches,
        ):
            batch_results = self.detector.predict(frame_batch)
            if len(batch_results) != len(frame_batch):
                raise RuntimeError(
                    f"text detector returned {len(batch_results)} results for "
                    f"{len(frame_batch)} frames"
                )
            for result, frame in zip(batch_results, frame_batch, strict=True):
                image_shape = (int(frame.shape[-2]), int(frame.shape[-1]))
                text_ratio, box_count = text_area_ratio(
                    result,
                    image_shape=image_shape,
                    text_min_confidence=self.config.text_min_confidence,
                )
                flat_text_ratios.append(text_ratio)
                flat_box_counts.append(box_count)

        results: list[tuple[float, dict[str, object]]] = []
        offset = 0
        for frame_count in frame_counts:
            next_offset = offset + frame_count
            row_text_ratios = flat_text_ratios[offset:next_offset]
            row_box_counts = flat_box_counts[offset:next_offset]
            offset = next_offset
            results.append(
                self.aggregate_frame_metrics(
                    text_ratios=row_text_ratios,
                    box_counts=row_box_counts,
                )
            )
        return results
