from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vidaforge.index import DEFAULT_PARQUET_SIZE


def default_screen_rules() -> dict[str, dict[str, object]]:
    return {
        "probe": {
            "field": "probe_ok",
            "equals": 1,
            "reject_reason": "probe_failed",
        },
        "short_side": {
            "field": "short_side",
            "min": 360,
            "reject_reason": "resolution_too_low",
        },
        "fps": {
            "field": "fps",
            "min": 20.0,
            "reject_reason": "fps_too_low",
        },
        "duration": {
            "field": "duration_sec",
            "min": 1.0,
            "max": 600.0,
            "min_reject_reason": "duration_too_short",
            "max_reject_reason": "duration_too_long",
        },
    }


@dataclass(slots=True)
class ScreenConfig:
    input_path: Path
    output_path: Path
    input_run_id: str
    run_id: str
    source: str | None = None
    source_batch: str | None = None
    parquet_size: int = DEFAULT_PARQUET_SIZE
    rules: dict[str, dict[str, object]] = field(default_factory=default_screen_rules)
    limit: int | None = None


@dataclass(slots=True)
class ScreenResult:
    input_path: Path
    output_path: Path
    source_count: int
    input_count: int
    resumed_count: int
    output_count: int
    ok_count: int
    failed_count: int
    pass_count: int
    reject_count: int
    shard_count: int
    summary_path: Path
    elapsed_sec: float
