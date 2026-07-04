from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def write_summary_json(summary: dict[str, object], output_path: str | Path) -> Path:
    summary_path = Path(output_path).expanduser().resolve() / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def finalize_summary_file(
    summary_path: str | Path,
    *,
    elapsed_sec: float | None = None,
    finished_at: str | None = None,
    updates: dict[str, object] | None = None,
) -> dict[str, object]:
    path = Path(summary_path).expanduser().resolve()
    summary = json.loads(path.read_text(encoding="utf-8"))
    if updates:
        summary.update(updates)
    if finished_at is not None:
        summary["finished_at"] = finished_at
    if elapsed_sec is not None:
        summary["elapsed_sec"] = elapsed_sec
    path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
