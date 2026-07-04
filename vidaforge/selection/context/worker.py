from __future__ import annotations

import json
from pathlib import Path

from vidaforge.common import hash_bucketed_path, join_data_dir, strip_data_dir
from vidaforge.media import (
    FRAME_SAMPLING_METHOD_UNIFORM,
    run_ffmpeg_extract_frames_audio,
)

from .config import AudioContextConfig, FrameContextConfig


def _build_context_row(
    *,
    row: dict[str, object],
    run_id: str,
    input_run_id: str,
    context_ok: int,
    context_error: str,
    frame_ok: int,
    frame_error: str,
    frame_json: dict[str, object],
    audio_json: dict[str, object],
    audio_ok: int = 0,
    audio_error: str = "",
) -> dict[str, object]:
    output_row = dict(row)
    output_row.update(
        {
            "frame_json": json.dumps(frame_json, ensure_ascii=False, sort_keys=True),
            "audio_json": json.dumps(audio_json, ensure_ascii=False, sort_keys=True),
            "audio_ok": audio_ok,
            "audio_error": audio_error,
            "input_run_id": input_run_id,
            "run_id": run_id,
            "context_ok": context_ok,
            "context_error": context_error,
            "frame_ok": frame_ok,
            "frame_error": frame_error,
        }
    )
    return output_row


def process_context_row(
    *,
    row: dict[str, object],
    output_data_path: str,
    run_id: str,
    input_run_id: str,
    frame_config: FrameContextConfig,
    audio_config: AudioContextConfig,
    ffmpeg_bin: str = "ffmpeg",
) -> dict[str, object]:
    """Build Stage 3 Step 1 context metadata for one clip row."""
    output_root = Path(output_data_path).expanduser().resolve()

    if int(row["clip_ok"]) != 1:
        context_error = f"clip_ok != 1: {row['clip_error']}"
        return _build_context_row(
            row=row,
            run_id=run_id,
            input_run_id=input_run_id,
            context_ok=0,
            context_error=context_error,
            frame_ok=0,
            frame_error=context_error,
            frame_json={},
            audio_ok=0,
            audio_error="",
            audio_json={},
        )

    duration_sec = float(row["duration_sec"])
    clip_width = int(row["width"])
    clip_height = int(row["height"])
    has_audio = bool(row["has_audio"])
    clip_path = join_data_dir(str(row["clip_path"]))
    clip_id = str(row["clip_id"])
    output_path = hash_bucketed_path(output_root, clip_id)

    try:
        extract_result = run_ffmpeg_extract_frames_audio(
            clip_path,
            output_path,
            frame_sampled_fps=frame_config.sampled_fps,
            frame_short_side=frame_config.short_side,
            frame_jpeg_qscale=frame_config.jpeg_qscale,
            frame_sampling_method=FRAME_SAMPLING_METHOD_UNIFORM,
            has_audio=has_audio,
            audio_format=audio_config.format,
            audio_sample_rate=audio_config.sample_rate,
            audio_channels=audio_config.channels,
            duration_sec=duration_sec,
            input_width=clip_width,
            input_height=clip_height,
            ffmpeg_bin=ffmpeg_bin,
        )
    except Exception as exc:  # noqa: BLE001
        context_error = str(exc)
        return _build_context_row(
            row=row,
            run_id=run_id,
            input_run_id=input_run_id,
            context_ok=0,
            context_error=context_error,
            frame_ok=0,
            frame_error=context_error,
            frame_json={},
            audio_json={
                "has_audio": has_audio,
                "audio_paths": [],
                "audio_format": "",
            },
            audio_ok=0,
            audio_error=context_error if has_audio else "",
        )

    frame_paths = [strip_data_dir(path) for path in extract_result.frame_paths]
    audio_paths = [strip_data_dir(path) for path in extract_result.audio_paths]
    audio_ok = int(has_audio and bool(audio_paths))
    audio_json: dict[str, object] = {
        "has_audio": has_audio,
        "audio_paths": audio_paths,
        "audio_format": extract_result.audio_format,
    }
    frame_json: dict[str, object] = {
        "sampling_method": extract_result.sampling_method,
        "sampled_fps": float(frame_config.sampled_fps),
        "sampled_frame_count": len(extract_result.frame_paths),
        "timestamps_sec": extract_result.timestamps_sec,
        "frame_width": extract_result.frame_width,
        "frame_height": extract_result.frame_height,
        "frame_paths": frame_paths,
    }

    return _build_context_row(
        row=row,
        run_id=run_id,
        input_run_id=input_run_id,
        context_ok=1,
        context_error="",
        frame_ok=1,
        frame_error="",
        frame_json=frame_json,
        audio_json=audio_json,
        audio_ok=audio_ok,
        audio_error="",
    )
