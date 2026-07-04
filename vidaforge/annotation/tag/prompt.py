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
    TAG_LABEL_GUIDE_TEXT,
    TAG_PROMPT_VERSION,
    TAG_SCHEMA_VERSION,
    tag_structured_output_json_schema,
)


TAG_SYSTEM_PROMPT = """You are a video semantic tagging model.
You must assign stable low-cardinality tags from the provided ordered frame sequence.
Return only valid JSON. Do not include markdown, comments, explanations, or confidence scores."""


@dataclass(frozen=True, slots=True)
class TagPromptRequest:
    system_prompt: str
    user_prompt: str
    image_paths: tuple[str, ...]
    timestamps_sec: tuple[float, ...]
    schema_version: str = TAG_SCHEMA_VERSION
    prompt_version: str = TAG_PROMPT_VERSION


def _frame_context_text(
    *,
    image_count: int,
    timestamps_sec: tuple[float, ...],
    sampled_fps: object,
    sampling_method: object,
) -> str:
    return (
        f"Image frames: {image_count} frames in chronological order.\n"
        f"Frame sampling: {sampled_fps or 'unknown'} fps, "
        f"method={sampling_method or 'unknown'}.\n"
        "Frame timestamps:\n"
        f"{frame_timeline_text(image_count, timestamps_sec)}"
    )


def _duration_text(duration_sec: object) -> str:
    return f"{float(duration_sec):.3f}"


def build_tag_prompt(
    *,
    frame_json: dict[str, Any],
    duration_sec: object,
) -> TagPromptRequest:
    """Build a clip-level semantic tagging prompt from frame context."""
    image_paths = tuple(str(join_data_dir(path)) for path in frame_json["frame_paths"])
    timestamps_sec = frame_timestamps_from_json(frame_json, expected_count=len(image_paths))
    sampled_fps = frame_json.get("sampled_fps", "")
    sampling_method = frame_json.get("sampling_method", "")

    user_prompt = f"""Task: assign clip-level semantic tags for this video clip.

Clip context:
Clip duration: {_duration_text(duration_sec)} seconds.

Frame context:
{_frame_context_text(
    image_count=len(image_paths),
    timestamps_sec=timestamps_sec,
    sampled_fps=sampled_fps,
    sampling_method=sampling_method,
)}

Important rules:
- Assign tags for the whole clip, not for a single isolated frame.
- Do not output confidence scores.
- Return exactly one JSON object matching the schema below.

{TAG_LABEL_GUIDE_TEXT}

Allowed labels and expected JSON fields:
{json.dumps(tag_structured_output_json_schema(), ensure_ascii=False, indent=2)}"""
    return TagPromptRequest(
        system_prompt=TAG_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        image_paths=image_paths,
        timestamps_sec=timestamps_sec,
    )
