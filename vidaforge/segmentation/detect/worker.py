from __future__ import annotations

from .config import DetectorConfigBase
from .ranges import build_detect_ranges


def _detect_ranges_to_ticks(detect_ranges: list[tuple[float, float]]) -> list[float]:
    return [round(float(start_sec), 6) for start_sec, _ in detect_ranges] + (
        [round(float(detect_ranges[-1][1]), 6)] if detect_ranges else []
    )


def _build_detect_row(
    *,
    row: dict[str, object],
    run_id: str,
    input_run_id: str,
    detectors: list[str],
    detect_ranges: list[tuple[float, float]],
    detect_ok: int,
    detect_error: str,
) -> dict[str, object]:
    detect_row: dict[str, object] = dict(row)
    detect_row.update(
        {
            "ticks_sec": _detect_ranges_to_ticks(detect_ranges),
            "detectors": list(detectors),
            "detect_ok": int(detect_ok),
            "detect_error": detect_error,
            "input_run_id": input_run_id,
            "run_id": run_id,
        }
    )
    return detect_row


def process_detect_row(
    *,
    row: dict[str, object],
    detectors: list[DetectorConfigBase],
    run_id: str,
    input_run_id: str,
) -> dict[str, object]:
    """Process one finalized Stage 1 video row into one Detect video row."""
    detector_names = [
        detector_config.detector_name
        for detector_config in detectors
    ]
    detect_ranges: list[tuple[float, float]] = []
    detect_ok = 0
    detect_error = ""

    if int(row["transcode_ok"]) != 1:
        detect_error = f"transcode_ok != 1: {row['transcode_error']}"
    else:
        try:
            detect_ranges = build_detect_ranges(
                row,
                detectors=detectors,
            )
            detect_ok = 1
        except Exception as exc:  # noqa: BLE001
            detect_error = str(exc)

    return _build_detect_row(
        row=row,
        run_id=run_id,
        input_run_id=input_run_id,
        detectors=detector_names,
        detect_ranges=detect_ranges,
        detect_ok=detect_ok,
        detect_error=detect_error,
    )
