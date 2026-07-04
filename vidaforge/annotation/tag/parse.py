from __future__ import annotations

from vidaforge.common import parse_json_object

from .schema import (
    TAG_LABEL_FIELDS,
    TAG_PROMPT_VERSION,
    TAG_SCHEMA_VERSION,
    TagStructuredOutput,
)


def parse_tag_response(text: str, *, model: str = "") -> dict[str, object]:
    parsed = TagStructuredOutput.model_validate(
        parse_json_object(
            text,
            description="tag response",
            allow_surrounding_text=True,
        )
    )
    payload = parsed.model_dump()
    return {
        "schema_version": TAG_SCHEMA_VERSION,
        "prompt_version": TAG_PROMPT_VERSION,
        **{key: payload[key] for key in TAG_LABEL_FIELDS},
        "model": model,
        "ok": True,
        "error": "",
    }

