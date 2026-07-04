from __future__ import annotations

from pathlib import Path

from vidaforge.common import hash_bucketed_path, join_data_dir, strip_data_dir
from vidaforge.media import (
    CLIP_ENCODE_BACKEND_CPU,
    run_ffmpeg_clip,
)
from .timing import ClipTiming, build_clip_timings_from_ticks


def _build_clip_row(
    row: dict[str, object],
    *,
    clip_id: str,
    clip_path: Path,
    clip_timing: ClipTiming,
    run_id: str,
    input_run_id: str,
    clip_ok: int,
    clip_error: str,
    ffmpeg_elapsed_sec: float = 0.0,
    filesize_bytes: int = 0,
) -> dict[str, object]:
    video_id = str(row["video_id"])
    clip_row: dict[str, object] = dict(row)
    clip_row.update(
        {
            "clip_id": clip_id,
            "video_id": video_id,
            "clip_path": strip_data_dir(clip_path),
            **clip_timing.to_dict(),
            "ffmpeg_elapsed_sec": round(float(ffmpeg_elapsed_sec), 3),
            "filesize_bytes": int(filesize_bytes),
            "input_run_id": input_run_id,
            "run_id": run_id,
            "clip_ok": int(clip_ok),
            "clip_error": clip_error,
        }
    )
    return clip_row


def process_clip_row(
    *,
    row: dict[str, object],
    output_data_path: str,
    run_id: str,
    input_run_id: str,
    min_len_sec: float = 1.0,
    max_len_sec: float = 10.0,
    overlong_split_len_sec: float = 10.0,
    boundary_trim_sec: float = 0.0,
    ffmpeg_bin: str = "ffmpeg",
) -> list[dict[str, object]]:
    if int(row["detect_ok"]) != 1:
        return []

    output_root = Path(output_data_path).expanduser().resolve()
    video_duration_sec = float(row["duration_sec"])
    if video_duration_sec <= 0:
        raise ValueError("duration_sec must be > 0")

    clip_timings = build_clip_timings_from_ticks(
        row["ticks_sec"],
        video_duration_sec=video_duration_sec,
        min_len_sec=min_len_sec,
        max_len_sec=max_len_sec,
        overlong_split_len_sec=overlong_split_len_sec,
        boundary_trim_sec=boundary_trim_sec,
    )

    output_rows: list[dict[str, object]] = []
    video_id = str(row["video_id"])
    video_path = join_data_dir(str(row["video_path"]))
    for clip_timing in clip_timings:
        clip_id = (
            f"{video_id}:clip:"
            f"{clip_timing.clip_index:05d}:{clip_timing.split_index:02d}"
        )
        clip_path = hash_bucketed_path(output_root, clip_id).with_suffix(".mp4")
        try:
            ffmpeg_result = run_ffmpeg_clip(
                video_path,
                clip_path,
                start_sec=clip_timing.start_sec,
                duration_sec=clip_timing.duration_sec,
                stream_copy=False,
                encode_backend=CLIP_ENCODE_BACKEND_CPU,
                ffmpeg_bin=ffmpeg_bin,
                overwrite=True,
            )
            output_rows.append(
                _build_clip_row(
                    row,
                    clip_id=clip_id,
                    clip_path=clip_path,
                    clip_timing=clip_timing,
                    run_id=run_id,
                    input_run_id=input_run_id,
                    clip_ok=1,
                    clip_error="",
                    ffmpeg_elapsed_sec=ffmpeg_result.elapsed_sec,
                    filesize_bytes=ffmpeg_result.filesize_bytes,
                )
            )
        except Exception as exc:  # noqa: BLE001
            output_rows.append(
                _build_clip_row(
                    row,
                    clip_id=clip_id,
                    clip_path=clip_path,
                    clip_timing=clip_timing,
                    run_id=run_id,
                    input_run_id=input_run_id,
                    clip_ok=0,
                    clip_error=str(exc),
                )
            )
    return output_rows
