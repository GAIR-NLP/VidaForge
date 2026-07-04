from __future__ import annotations

from typing import Literal

from vidaforge.common import parse_json_object

from .prompt import CAPTION_LEVELS, CAPTION_PROMPT_VERSION
from .schema import (
    CAPTION_SCHEMA_VERSION,
    CaptionStructuredOutput,
)


CaptionMode = Literal["video", "video_audio"]


def parse_caption_response(
    text: str,
    *,
    mode: CaptionMode,
    model: str = "",
) -> dict[str, object]:
    parsed = CaptionStructuredOutput.model_validate(
        parse_json_object(
            text,
            description="caption response",
            allow_surrounding_text=True,
        )
    )
    captions = parsed.model_dump()
    return {
        "schema_version": CAPTION_SCHEMA_VERSION,
        "prompt_version": CAPTION_PROMPT_VERSION,
        "mode": mode,
        **{level: str(captions.get(level) or "") for level in CAPTION_LEVELS},
        "model": model,
        "ok": True,
        "error": "",
    }
