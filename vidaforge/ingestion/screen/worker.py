from __future__ import annotations

import json
from typing import Any

from vidaforge.filters import check as check_filter_rules


def _get_filter_values(row: dict[str, object]) -> dict[str, object]:
    width = row["width"]
    height = row["height"]
    duration_sec = row["duration_sec"]
    return {
        "probe_ok": int(row["probe_ok"]),
        "short_side": (
            None if width is None or height is None else min(int(width), int(height))
        ),
        "fps": row["fps"],
        "duration_sec": None if duration_sec is None else float(duration_sec),
    }


def _build_screen_row(
    *,
    row: dict[str, object],
    input_run_id: str,
    run_id: str,
    screen_ok: int,
    screen_error: str,
    screen_pass: int,
    screen_reject_reason: str,
    screen_json: dict[str, object],
) -> dict[str, object]:
    output_row = dict(row)
    output_row.update(
        {
            "input_run_id": input_run_id,
            "run_id": run_id,
            "screen_ok": int(screen_ok),
            "screen_error": screen_error,
            "screen_pass": int(screen_pass),
            "screen_reject_reason": screen_reject_reason,
            "screen_json": json.dumps(screen_json, ensure_ascii=False),
        }
    )
    return output_row


def process_screen_row(
    *,
    row: dict[str, object],
    rules: dict[str, dict[str, object]],
    input_run_id: str,
    run_id: str,
) -> dict[str, object]:
    screen_pass = 0
    screen_reject_reason = ""
    screen_json: dict[str, Any] = {}
    screen_ok = 0
    screen_error = ""

    try:
        result = check_filter_rules(
            values=_get_filter_values(row),
            rules=rules,
        )
        screen_pass = int(result.passed)
        screen_reject_reason = result.reject_reason
        screen_json = result.json
        screen_ok = 1
    except Exception as exc:  # noqa: BLE001
        screen_error = str(exc)

    return _build_screen_row(
        row=row,
        input_run_id=input_run_id,
        run_id=run_id,
        screen_ok=screen_ok,
        screen_error=screen_error,
        screen_pass=screen_pass,
        screen_reject_reason=screen_reject_reason,
        screen_json=screen_json,
    )
