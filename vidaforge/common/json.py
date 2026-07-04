from __future__ import annotations

import json


def parse_json_object(
    value: object,
    *,
    description: str = "json",
    allow_empty_value: bool = False,
    allow_surrounding_text: bool = False,
) -> dict[str, object] | None:
    """Parse a JSON object from a dict or JSON object string."""
    if value is None:
        if allow_empty_value:
            return None
        raise ValueError(f"missing {description}")

    if isinstance(value, dict):
        return dict(value)

    if isinstance(value, str):
        value = value.strip()
        if not value:
            if allow_empty_value:
                return None
            raise ValueError(f"empty {description}")

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            if not allow_surrounding_text:
                raise
            start = value.find("{")
            end = value.rfind("}")
            if start < 0 or end <= start:
                raise
            parsed = json.loads(value[start : end + 1])

        if not isinstance(parsed, dict):
            raise ValueError(f"{description} must be a JSON object")
        return parsed

    if allow_empty_value:
        return None
    raise ValueError(f"{description} must be a JSON object or JSON object string")
