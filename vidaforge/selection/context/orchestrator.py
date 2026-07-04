"""Stage 3 context preparation orchestration."""

from __future__ import annotations

from functools import partial
import shutil
import time

from vidaforge.common import join_data_dir, parse_json_object, utc_now_iso, write_summary_json
from vidaforge.index import (
    count_parquet,
    run_ray_task_processing,
)
from vidaforge.media import FRAME_SAMPLING_METHOD_UNIFORM

from .config import ContextConfig, ContextResult
from .worker import process_context_row


def _existing_asset_files(paths: object) -> bool:
    if not isinstance(paths, list) or not paths:
        return False
    for path in paths:
        asset_path = join_data_dir(str(path))
        try:
            if not asset_path.is_file() or asset_path.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def is_context_row_complete(row: dict[str, object]) -> bool:
    try:
        frame_json = parse_json_object(row["frame_json"], description="frame_json")
        if frame_json is None:
            return False
        if not _existing_asset_files(frame_json["frame_paths"]):
            return False
        if int(row["audio_ok"]) == 1:
            audio_json = parse_json_object(row["audio_json"], description="audio_json")
            if audio_json is None:
                return False
            return _existing_asset_files(audio_json["audio_paths"])
        return True
    except Exception:  # noqa: BLE001
        return False


def _validate_config(config: ContextConfig) -> None:
    if not config.run_id.strip():
        raise ValueError("run_id must be set")
    if not config.input_run_id.strip():
        raise ValueError("input_run_id must be set")
    if config.frame.sampled_fps <= 0:
        raise ValueError("frame.sampled_fps must be > 0")
    if config.frame.short_side <= 0:
        raise ValueError("frame.short_side must be > 0")
    if not 2 <= config.frame.jpeg_qscale <= 31:
        raise ValueError("frame.jpeg_qscale must be in [2, 31]")
    if config.audio.format not in {"m4a", "wav"}:
        raise ValueError("audio.format currently only supports: m4a, wav")
    if config.audio.sample_rate <= 0:
        raise ValueError("audio.sample_rate must be > 0")
    if config.audio.channels <= 0:
        raise ValueError("audio.channels must be > 0")
    if config.parquet_size <= 0:
        raise ValueError("parquet_size must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.ray_num_cpus <= 0:
        raise ValueError("ray_num_cpus must be > 0")


class ContextOrchestrator:
    """Orchestrate Stage 3 Step 1 context preparation."""

    def __init__(
        self,
        stage_name: str = "stage3_selection",
        step_name: str = "step1_context",
    ) -> None:
        self.stage_name = stage_name
        self.step_name = step_name

    def build_context(self, config: ContextConfig) -> ContextResult:
        _validate_config(config)

        input_path = config.input_path.expanduser().resolve()
        output_data_path = config.output_data_path.expanduser().resolve()
        output_meta_path = config.output_meta_path.expanduser().resolve()
        source_count = count_parquet(input_path, unit="clip")

        started_at = utc_now_iso()
        started_perf = time.perf_counter()
        if not config.resume:
            shutil.rmtree(output_data_path, ignore_errors=True)
        output_data_path.mkdir(parents=True, exist_ok=True)
        output_meta_path.mkdir(parents=True, exist_ok=True)

        context_worker = partial(
            process_context_row,
            output_data_path=str(output_data_path),
            run_id=config.run_id,
            input_run_id=config.input_run_id,
            frame_config=config.frame,
            audio_config=config.audio,
            ffmpeg_bin=config.ffmpeg_bin,
        )
        stats, writer_summary = run_ray_task_processing(
            input_path=input_path,
            output_path=output_meta_path,
            parquet_size=config.parquet_size,
            input_unit="clip",
            output_unit="clip",
            step="context",
            ray_address=config.ray_address,
            ray_num_cpus=config.ray_num_cpus,
            worker=context_worker,
            task_batch_size=config.batch_size,
            limit=config.limit,
            resume=config.resume,
            is_complete=is_context_row_complete,
            desc="stage3 context",
        )
        elapsed_sec = round(time.perf_counter() - started_perf, 3)

        summary = {
            "created_at": utc_now_iso(),
            **writer_summary,
            "stage": self.stage_name,
            "step": self.step_name,
            "frame_sampled_fps": config.frame.sampled_fps,
            "frame_short_side": config.frame.short_side,
            "frame_jpeg_qscale": config.frame.jpeg_qscale,
            "frame_sampling_method": FRAME_SAMPLING_METHOD_UNIFORM,
            "audio_format": config.audio.format,
            "audio_sample_rate": config.audio.sample_rate,
            "audio_channels": config.audio.channels,
            "ffmpeg_bin": config.ffmpeg_bin,
            "ray_address": config.ray_address,
            "ray_num_cpus": config.ray_num_cpus,
            "batch_size": config.batch_size,
            "resume": config.resume,
            "parquet_size": config.parquet_size,
            "source_count": source_count,
            "input_count": stats.input_count,
            "resumed_count": stats.resumed_count,
            "output_count": stats.output_count,
            "ok_count": stats.ok_count,
            "failed_count": stats.failed_count,
            "shard_count": int(writer_summary["shard_count"]),
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "elapsed_sec": elapsed_sec,
            "failed_examples": stats.failed_examples,
            "input_path": str(input_path),
            "output_data_path": str(output_data_path),
            "output_meta_path": str(output_meta_path),
            "run_id": config.run_id,
            "input_run_id": config.input_run_id,
            "source": config.source or "",
            "source_batch": config.source_batch or "",
            "limit": config.limit,
        }
        summary_path = write_summary_json(summary, output_meta_path)

        return ContextResult(
            input_path=input_path,
            output_data_path=output_data_path,
            output_meta_path=output_meta_path,
            source_count=source_count,
            input_count=stats.input_count,
            resumed_count=stats.resumed_count,
            output_count=stats.output_count,
            ok_count=stats.ok_count,
            failed_count=stats.failed_count,
            shard_count=int(writer_summary["shard_count"]),
            summary_path=summary_path,
            elapsed_sec=elapsed_sec,
        )
