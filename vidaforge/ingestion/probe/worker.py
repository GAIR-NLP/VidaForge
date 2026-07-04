from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

from vidaforge.common import get_raw_video_path, video_id_from_raw_path
from vidaforge.media import FFProbeResult, run_ffprobe


_FFPROBE_TIMEOUT_SEC = 30


def _build_probe_row(
    *,
    row: dict[str, object],
    source: str,
    source_batch: str,
    run_id: str,
    raw_path: str,
    raw_type: str,
    raw_member_path: str,
    filesize_bytes: int | None,
    ffprobe_result: FFProbeResult,
    probe_ok: int,
    probe_error: str,
    probe_elapsed_ms: int,
) -> dict[str, object]:
    probe_row = dict(row)
    probe_row.update(
        {
            "video_id": video_id_from_raw_path(
                source=source,
                source_batch=source_batch,
                raw_path=raw_path,
                raw_member_path=raw_member_path,
            ),
            "raw_type": raw_type,
            "raw_path": raw_path,
            "raw_member_path": raw_member_path,
            "video_path": raw_path,
            "source": source,
            "source_batch": source_batch,
            "run_id": run_id,
            "filesize_bytes": filesize_bytes,
            **asdict(ffprobe_result),
            "probe_ok": int(probe_ok),
            "probe_error": probe_error,
            "probe_elapsed_ms": int(probe_elapsed_ms),
        }
    )
    return probe_row


def process_probe_row(
    *,
    row: dict[str, object],
    source: str,
    source_batch: str,
    run_id: str,
    ffprobe_bin: str,
    temp_dir: str | Path | None = None,
) -> dict[str, object]:
    started_at = time.perf_counter()
    raw_path_text = str(row["raw_path"])
    raw_type = str(row.get("raw_type", "file"))
    raw_member_path = str(row.get("raw_member_path", ""))
    filesize_bytes: int | None = None
    ffprobe_result = FFProbeResult()
    probe_ok = 0
    probe_error = ""

    try:
        with get_raw_video_path(row, temp_dir=temp_dir) as raw_video_path:
            filesize_bytes = int(raw_video_path.stat().st_size)
            ffprobe_result = run_ffprobe(
                raw_video_path,
                ffprobe_bin=ffprobe_bin,
                timeout_sec=_FFPROBE_TIMEOUT_SEC,
            )
        probe_ok = 1
    except Exception as exc:  # noqa: BLE001
        probe_error = str(exc)

    return _build_probe_row(
        row=row,
        source=source,
        source_batch=source_batch,
        run_id=run_id,
        raw_path=raw_path_text,
        raw_type=raw_type,
        raw_member_path=raw_member_path,
        filesize_bytes=filesize_bytes,
        ffprobe_result=ffprobe_result,
        probe_ok=probe_ok,
        probe_error=probe_error,
        probe_elapsed_ms=int((time.perf_counter() - started_at) * 1000),
    )
