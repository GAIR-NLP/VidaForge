from __future__ import annotations

from vidaforge.common import parse_json_object

from .schema import (
    CAMERA_LABEL_VERSION,
    CAMERA_PROMPT_VERSION,
    CAMERA_LABEL_FIELDS,
    CameraStructuredOutput,
)


def parse_camera_response(text: str, *, model: str = "") -> dict[str, object]:
    parsed = CameraStructuredOutput.model_validate(
        parse_json_object(
            text,
            description="camera response",
            allow_surrounding_text=True,
        )
    )
    payload = parsed.model_dump()
    return {
        "label_version": CAMERA_LABEL_VERSION,
        "prompt_version": CAMERA_PROMPT_VERSION,
        **{key: payload[key] for key in CAMERA_LABEL_FIELDS},
        "model": model,
        "ok": True,
        "error": "",
    }
