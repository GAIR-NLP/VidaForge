from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from vidaforge.common import get_raw_video_path, hash_bucketed_path, strip_data_dir
from vidaforge.media import FFProbeResult, run_ffmpeg_transcode, run_ffprobe


_FFPROBE_TIMEOUT_SEC = 30


def _build_transcode_row(
    *,
    row: dict[str, object],
    input_run_id: str,
    run_id: str,
    video_path: Path,
    ffprobe_result: FFProbeResult,
    filesize_bytes: int | None,
    transcode_ok: int,
    transcode_error: str,
    transcode_mode: str,
    transcode_elapsed_sec: float,
) -> dict[str, object]:
    transcode_row = dict(row)
    # Always overwrite media metadata so failed rows do not keep stale raw-video values.
    transcode_row.update(
        {
            "video_path": strip_data_dir(video_path),
            **asdict(ffprobe_result),
            "filesize_bytes": filesize_bytes,
            "input_run_id": input_run_id,
            "run_id": run_id,
            "transcode_ok": int(transcode_ok),
            "transcode_error": transcode_error,
            "transcode_mode": transcode_mode,
            "transcode_elapsed_sec": transcode_elapsed_sec,
        }
    )
    return transcode_row


def process_transcode_row(
    *,
    row: dict[str, object],
    output_data_path: str,
    input_run_id: str,
    run_id: str,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    target_short_edge: int,
    target_fps: int,
    crf: int,
    pix_fmt: str,
    audio_bitrate: str,
    ffmpeg_threads: int | None,
) -> dict[str, object]:
    video_path = hash_bucketed_path(
        Path(output_data_path).expanduser().resolve(),
        str(row["video_id"]),
    ).with_suffix(".mp4")
    filesize_bytes: int | None = None
    ffprobe_result = FFProbeResult()
    transcode_ok = 0
    transcode_error = ""
    resolved_transcode_mode = ""
    transcode_elapsed_sec = 0.0

    try:
        with get_raw_video_path(row) as raw_video_path:
            ffmpeg_result = run_ffmpeg_transcode(
                raw_video_path,
                video_path,
                min_edge=target_short_edge,
                fps=target_fps,
                crf=crf,
                pix_fmt=pix_fmt,
                audio_bitrate=audio_bitrate,
                ffmpeg_threads=ffmpeg_threads,
                ffmpeg_bin=ffmpeg_bin,
            )
        resolved_transcode_mode = ffmpeg_result.transcode_mode
        transcode_elapsed_sec = ffmpeg_result.elapsed_sec
        filesize_bytes = int(video_path.stat().st_size)
        try:
            ffprobe_result = run_ffprobe(
                video_path,
                ffprobe_bin=ffprobe_bin,
                timeout_sec=_FFPROBE_TIMEOUT_SEC,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"ffprobe: {exc}") from exc
        transcode_ok = 1
    except Exception as exc:  # noqa: BLE001
        transcode_error = str(exc)

    return _build_transcode_row(
        row=row,
        input_run_id=input_run_id,
        run_id=run_id,
        video_path=video_path,
        ffprobe_result=ffprobe_result,
        filesize_bytes=filesize_bytes,
        transcode_ok=transcode_ok,
        transcode_error=transcode_error,
        transcode_mode=resolved_transcode_mode,
        transcode_elapsed_sec=transcode_elapsed_sec,
    )
