from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal

from vidaforge.common import join_data_dir
from vidaforge.common.asset import (
    frame_timeline_text,
    frame_timestamps_from_json,
)
from vidaforge.annotation.camera.schema import CAMERA_LABEL_FIELDS


CAPTION_PROMPT_VERSION = "caption_v1"
CAPTION_LEVELS = ("level_3", "level_2", "level_1", "level_0")

CaptionMode = Literal["video", "video_audio"]


CAPTION_SYSTEM_PROMPT = """You are a video captioning model.
You must describe only observable evidence from the provided media.
Return only valid JSON. Do not include markdown, comments, or explanations."""


@dataclass(frozen=True, slots=True)
class CaptionPromptRequest:
    system_prompt: str
    user_prompt: str
    image_paths: tuple[str, ...]
    timestamps_sec: tuple[float, ...]
    audio_paths: tuple[str, ...] = ()
    mode: CaptionMode = "video_audio"
    prompt_version: str = CAPTION_PROMPT_VERSION


def _camera_context_text(camera_json: dict[str, Any] | None) -> str:
    if not camera_json:
        return "No external camera context is provided."

    context = {key: camera_json[key] for key in CAMERA_LABEL_FIELDS if key in camera_json}
    return json.dumps(context, ensure_ascii=False, indent=2)


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


def _audio_context_text(*, mode: CaptionMode, audio_paths: tuple[str, ...]) -> str:
    if mode == "video" or not audio_paths:
        return "No audio is provided."
    return (
        "Use the provided audio as supporting evidence. "
        "If speech is present, describe who is speaking when visible, the speech context, "
        "and the semantic content of the speech. Level_3 must include the complete clearly "
        "audible speech content when speech is present, while still remaining a dense "
        "audio-visual caption rather than a standalone transcript. Put the spoken words in "
        "level_3 using a natural phrase such as 'He says: ...' or 'She says: ...'. Do not "
        "satisfy this requirement by only writing phrases such as 'he speaks', 'speech is "
        "clear', or 'speech is intelligible'. If speech is audible but you cannot reliably "
        "identify the words, write this exact sentence in level_3: 'The exact speech content "
        "is not clearly audible.' If speech is long, include as much as can be reliably heard "
        "without inventing missing words. For level_1 and level_2, summarize the speech "
        "content instead of transcribing it fully. Also mention music, sound effects, "
        "ambient sound, or silence when they are audible and relevant. "
        "If the audio is unclear, do not guess."
    )


def _caption_level_guide_text() -> str:
    return """Caption levels:
- level_0: Semantic gist, less than 30 words. Describe only the main subject, primary action, and scene. Do not mention camera motion or audio.
- level_1: Concise video caption, 50-100 words. Describe the main event flow, important subjects, actions, scene, and major changes. Mention obvious camera motion and summarize important audio when present.
- level_2: Detailed temporal caption, 100-200 words. Describe the initial state, action progression, subject/object relationships, position changes, relevant background, and main camera motion. Summarize speech content and relevant non-speech audio when useful.
- level_3: Dense reconstruction caption, 200-400 words. Describe the clip as completely as possible: subjects, attributes, actions, environment, lighting, color, style, composition, positions, temporal changes, camera motion, watermarks/logos, and relevant audio cues. In video_audio mode, include the complete clearly audible speech content when speech is present, using a phrase like 'He says: ...'."""


def _caption_rules_text() -> str:
    return """General rules:
- Focus on observable content. Do not infer hidden intent, symbolism, or backstory.
- Preserve temporal order. Describe what changes from the beginning to the middle and end.
- Describe subject motion and camera motion separately when possible.
- Use camera context from level_1 onward, but write it naturally. Never copy raw camera labels into the caption.
- For level_1, mention camera motion only if it is visually noticeable and relevant.
- For level_2, integrate the main camera motion into the action progression.
- For level_3, use all useful camera context to describe cinematography, framing, and motion dynamics.
- Avoid over-emphasizing tiny shake or small camera adjustments unless they affect the clip.
- Mention watermark/logo/text only if visible.
- The four captions should be different levels of detail, not near-duplicates.
- Do not include explicit frame IDs or timestamp numbers in the captions.
- Return exactly one JSON object with keys in this order: level_3, level_2, level_1, level_0.
- Each value must be a single string."""


def _duration_text(duration_sec: object) -> str:
    return f"{float(duration_sec):.3f}"


def build_caption_prompt(
    *,
    frame_json: dict[str, Any],
    audio_json: dict[str, Any] | None = None,
    camera_json: dict[str, Any] | None = None,
    duration_sec: object,
    mode: CaptionMode = "video_audio",
) -> CaptionPromptRequest:
    """Build a multi-level video caption prompt from frame/audio/camera context."""
    if mode not in {"video", "video_audio"}:
        raise ValueError(f"unsupported caption mode: {mode!r}")

    image_paths = tuple(str(join_data_dir(path)) for path in frame_json["frame_paths"])
    timestamps_sec = frame_timestamps_from_json(frame_json, expected_count=len(image_paths))
    sampled_fps = frame_json.get("sampled_fps", "")
    sampling_method = frame_json.get("sampling_method", "")

    audio_paths: tuple[str, ...] = ()
    if audio_json and audio_json.get("audio_paths"):
        audio_paths = tuple(str(join_data_dir(path)) for path in audio_json["audio_paths"])
    user_prompt = f"""Task: generate four multi-level captions for this video clip.

Clip context:
Clip duration: {_duration_text(duration_sec)} seconds.

Frame context:
{_frame_context_text(
    image_count=len(image_paths),
    timestamps_sec=timestamps_sec,
    sampled_fps=sampled_fps,
    sampling_method=sampling_method,
)}

Audio context:
{_audio_context_text(mode=mode, audio_paths=audio_paths)}

Camera context:
{_camera_context_text(camera_json)}

{_caption_level_guide_text()}

{_caption_rules_text()}"""
    return CaptionPromptRequest(
        system_prompt=CAPTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        image_paths=image_paths,
        timestamps_sec=timestamps_sec,
        audio_paths=audio_paths if mode == "video_audio" else (),
        mode=mode,
    )
