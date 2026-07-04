from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scenedetect import FrameTimecode
from scenedetect.detector import SceneDetector

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None
    from PIL import Image
else:  # pragma: no cover
    Image = None

from vidaforge.models import TransNetV2

TRANSNETV2_INPUT_SHAPE = (27, 48, 3)
DEFAULT_TRANSNETV2_THRESHOLD = 0.5
DEFAULT_TRANSNETV2_PREDICTION_MODE: Literal["single_frame", "many_hot"] = "single_frame"

_MODEL_CACHE: dict[tuple[str, str], "_TransNetV2Predictor"] = {}


def _resolve_device(device: str | None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _resize_frame_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    if cv2 is not None:
        resized_bgr = cv2.resize(
            frame_bgr,
            (TRANSNETV2_INPUT_SHAPE[1], TRANSNETV2_INPUT_SHAPE[0]),
            interpolation=cv2.INTER_AREA,
        )
        return np.ascontiguousarray(resized_bgr[:, :, ::-1])

    if Image is None:  # pragma: no cover
        raise RuntimeError("Either opencv-python or Pillow is required for TransNetV2 frame resize.")

    pil_image = Image.fromarray(frame_bgr[:, :, ::-1])
    pil_image = pil_image.resize(
        (TRANSNETV2_INPUT_SHAPE[1], TRANSNETV2_INPUT_SHAPE[0]),
        Image.Resampling.BILINEAR,
    )
    return np.ascontiguousarray(np.asarray(pil_image, dtype=np.uint8))


def _predictions_to_scenes(predictions: np.ndarray, threshold: float) -> np.ndarray:
    binary = (predictions > threshold).astype(np.uint8)

    scenes: list[list[int]] = []
    t_prev = 0
    start = 0
    for frame_idx, value in enumerate(binary):
        if t_prev == 1 and value == 0:
            start = frame_idx
        if t_prev == 0 and value == 1 and frame_idx != 0:
            scenes.append([start, frame_idx])
        t_prev = int(value)

    if len(binary) > 0 and int(binary[-1]) == 0:
        scenes.append([start, len(binary) - 1])

    if not scenes:
        return np.array([[0, max(len(binary) - 1, 0)]], dtype=np.int32)
    return np.array(scenes, dtype=np.int32)


def _normalize_cut_frames(
    candidate_cuts: list[int],
    *,
    total_frames: int,
    min_scene_len_frames: int,
) -> list[int]:
    normalized: list[int] = []
    prev_boundary = 0
    for cut_frame in candidate_cuts:
        if cut_frame <= 0 or cut_frame >= total_frames:
            continue
        if cut_frame - prev_boundary < min_scene_len_frames:
            continue
        normalized.append(cut_frame)
        prev_boundary = cut_frame

    while normalized and (total_frames - normalized[-1]) < min_scene_len_frames:
        normalized.pop()

    return normalized


class _TransNetV2Predictor:
    def __init__(self, *, model_path: str | None, device: str | None) -> None:
        resolved_device = _resolve_device(device)
        weights_path = Path(model_path).expanduser().resolve() if model_path else (
            Path(__file__).resolve().with_name("transnetv2-pytorch-weights.pth")
        )
        if not weights_path.is_file():
            raise FileNotFoundError(f"TransNetV2 weights file not found at {weights_path}.")

        self.device = resolved_device
        self.weights_path = weights_path
        self.model = TransNetV2()
        state_dict = torch.load(weights_path, map_location=resolved_device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

    def predict_frames(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert len(frames.shape) == 4 and tuple(frames.shape[1:]) == TRANSNETV2_INPUT_SHAPE, (
            "TransNetV2 input must be [frames, 27, 48, 3]."
        )

        total_frames = len(frames)

        def input_iterator():
            pad_start = 25
            pad_end = 25 + 50 - (total_frames % 50 if total_frames % 50 != 0 else 50)

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * pad_start + [frames] + [end_frame] * pad_end,
                axis=0,
            )

            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                chunk = padded_inputs[ptr : ptr + 100]
                ptr += 50
                yield chunk[np.newaxis]

        single_predictions: list[np.ndarray] = []
        many_hot_predictions: list[np.ndarray] = []

        with torch.no_grad():
            for inputs in input_iterator():
                inputs_tensor = torch.from_numpy(inputs)
                one_hot_logits, many_hot_logits = self.model(inputs_tensor.to(self.device))
                one_hot = torch.sigmoid(one_hot_logits).cpu().numpy()
                many_hot = torch.sigmoid(many_hot_logits["many_hot"]).cpu().numpy()
                single_predictions.append(one_hot[0, 25:75, 0])
                many_hot_predictions.append(many_hot[0, 25:75, 0])

        single_frame_pred = np.concatenate(single_predictions, axis=0)[:total_frames]
        many_hot_pred = np.concatenate(many_hot_predictions, axis=0)[:total_frames]
        return single_frame_pred, many_hot_pred


def _get_predictor(*, model_path: str | None, device: str | None) -> _TransNetV2Predictor:
    resolved_device = _resolve_device(device)
    cache_key = (
        str(Path(model_path).expanduser().resolve()) if model_path else "__default__",
        resolved_device,
    )
    predictor = _MODEL_CACHE.get(cache_key)
    if predictor is None:
        predictor = _TransNetV2Predictor(model_path=model_path, device=resolved_device)
        _MODEL_CACHE[cache_key] = predictor
    return predictor


class TransNetV2Detector(SceneDetector):
    def __init__(
        self,
        threshold: float = DEFAULT_TRANSNETV2_THRESHOLD,
        min_scene_len: int = 15,
        model_path: str | None = None,
        weights_path: str | None = None,
        device: str | None = None,
        prediction_mode: Literal["single_frame", "many_hot"] = DEFAULT_TRANSNETV2_PREDICTION_MODE,
    ) -> None:
        super().__init__()
        if threshold <= 0 or threshold >= 1:
            raise ValueError("threshold must be between 0 and 1.")
        if min_scene_len <= 0:
            raise ValueError("min_scene_len must be > 0.")
        if prediction_mode not in {"single_frame", "many_hot"}:
            raise ValueError("prediction_mode must be 'single_frame' or 'many_hot'.")

        self.threshold = float(threshold)
        self.min_scene_len = int(min_scene_len)
        self.model_path = model_path or weights_path
        self.device = device
        self.prediction_mode = prediction_mode

        self._frames: list[np.ndarray] = []
        self._fps: float | None = None

    def process_frame(self, timecode: FrameTimecode, frame_img: np.ndarray) -> list[FrameTimecode]:
        if self._fps is None:
            self._fps = float(timecode.framerate)
        self._frames.append(_resize_frame_rgb(frame_img))
        return []

    def post_process(self, timecode: int) -> list[FrameTimecode]:
        if not self._frames:
            return []
        if self._fps is None:
            raise RuntimeError("TransNetV2Detector did not receive any frame timecode.")

        frames = np.stack(self._frames, axis=0).astype(np.uint8, copy=False)
        predictor = _get_predictor(model_path=self.model_path, device=self.device)
        single_frame_pred, many_hot_pred = predictor.predict_frames(frames)
        predictions = single_frame_pred if self.prediction_mode == "single_frame" else many_hot_pred

        scenes = _predictions_to_scenes(predictions, threshold=self.threshold)
        candidate_cuts = [int(scene_end) for _, scene_end in scenes[:-1]]
        cut_frames = _normalize_cut_frames(
            candidate_cuts,
            total_frames=len(frames),
            min_scene_len_frames=self.min_scene_len,
        )
        return [FrameTimecode(cut_frame, fps=self._fps) for cut_frame in cut_frames]
