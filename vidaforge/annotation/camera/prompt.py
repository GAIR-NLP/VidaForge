from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from vidaforge.common import join_data_dir
from vidaforge.common.asset import (
    frame_timeline_text,
    frame_timestamps_from_json,
)

from .schema import (
    CAMERA_LABEL_GUIDE_TEXT,
    CAMERA_LABEL_VERSION,
    CAMERA_PROMPT_VERSION,
    camera_structured_output_json_schema,
)


CAMERA_SYSTEM_PROMPT = """You are a camera-motion annotation model.
You must infer camera movement from the provided ordered frame sequence.
Return only valid JSON. Do not include markdown, comments, explanations, or confidence scores."""


@dataclass(frozen=True, slots=True)
class CameraPromptRequest:
    system_prompt: str
    user_prompt: str
    image_paths: tuple[str, ...]
    timestamps_sec: tuple[float, ...]
    label_version: str = CAMERA_LABEL_VERSION
    prompt_version: str = CAMERA_PROMPT_VERSION


def build_camera_prompt(
    *,
    frame_json: dict[str, Any],
    duration_sec: object,
) -> CameraPromptRequest:
    image_paths = tuple(str(join_data_dir(path)) for path in frame_json["frame_paths"])
    timestamps_sec = frame_timestamps_from_json(frame_json, expected_count=len(image_paths))
    sampled_fps = frame_json.get("sampled_fps", "")
    sampling_method = frame_json.get("sampling_method", "")
    user_prompt = f"""Task: annotate camera motion for this clip from the ordered frames.

You will receive {len(image_paths)} image frames in chronological order.
Clip duration: {float(duration_sec):.3f} seconds.
Frame sampling: {sampled_fps or "unknown"} fps, method={sampling_method or "unknown"}.

Frame timestamps:
{frame_timeline_text(len(image_paths), timestamps_sec)}

Important rules:
- Judge camera motion, not subject motion.
- Keep intentional camera movement separate from unintended shake.
- If the visual evidence is insufficient for a dimension, output "unknown".
- Use "no-*" labels only when the motion is explicitly absent.
- Do not output confidence scores.
- Return exactly one JSON object matching the schema below.

{CAMERA_LABEL_GUIDE_TEXT}

Allowed labels and expected JSON fields:
{json.dumps(camera_structured_output_json_schema(), ensure_ascii=False, indent=2)}"""
    return CameraPromptRequest(
        system_prompt=CAMERA_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        image_paths=image_paths,
        timestamps_sec=timestamps_sec,
    )
