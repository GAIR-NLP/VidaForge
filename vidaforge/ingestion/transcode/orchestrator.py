from __future__ import annotations

from functools import partial
import shutil
import time

from vidaforge.common import utc_now_iso, write_summary_json
from vidaforge.index import (
    count_parquet,
    run_ray_task_processing,
)

from .config import TranscodeConfig, TranscodeResult
from .worker import process_transcode_row


def _validate_config(config: TranscodeConfig) -> None:
    if not config.input_run_id.strip():
        raise ValueError("input_run_id must be set")
    if not config.run_id.strip():
        raise ValueError("run_id must be set")
    if config.parquet_size <= 0:
        raise ValueError("parquet_size must be > 0")
    if config.ray_num_cpus <= 0:
        raise ValueError("ray_num_cpus must be > 0")
    if config.ffmpeg_threads is not None and config.ffmpeg_threads <= 0:
        raise ValueError("ffmpeg_threads must be > 0 when provided")


class TranscodeOrchestrator:
    """Run Stage 1 transcode and emit standardized video assets."""

    def __init__(
        self,
        stage_name: str = "stage1_ingestion",
        step_name: str = "step3_transcode",
    ) -> None:
        self.stage_name = stage_name
        self.step_name = step_name

    def transcode(self, config: TranscodeConfig) -> TranscodeResult:
        _validate_config(config)

        input_path = config.input_path.expanduser().resolve()
        output_data_path = config.output_data_path.expanduser().resolve()
        output_meta_path = config.output_meta_path.expanduser().resolve()
        source_count = count_parquet(input_path, unit="video")

        started_at = utc_now_iso()
        started_perf = time.perf_counter()
        if not config.resume:
            shutil.rmtree(output_data_path, ignore_errors=True)
        output_data_path.mkdir(parents=True, exist_ok=True)
        output_meta_path.mkdir(parents=True, exist_ok=True)

        transcode_worker = partial(
            process_transcode_row,
            output_data_path=str(output_data_path),
            input_run_id=config.input_run_id,
            run_id=config.run_id,
            ffmpeg_bin=config.ffmpeg_bin,
            ffprobe_bin=config.ffprobe_bin,
            target_short_edge=config.target_short_edge,
            target_fps=config.target_fps,
            crf=config.crf,
            pix_fmt=config.pix_fmt,
            audio_bitrate=config.audio_bitrate,
            ffmpeg_threads=config.ffmpeg_threads,
        )

        stats, writer_summary = run_ray_task_processing(
            input_path=input_path,
            output_path=output_meta_path,
            parquet_size=config.parquet_size,
            input_unit="video",
            output_unit="video",
            step="transcode",
            ray_address=config.ray_address,
            ray_num_cpus=config.ray_num_cpus,
            worker=transcode_worker,
            limit=config.limit,
            resume=config.resume,
            desc="stage1 transcode",
        )
        elapsed_sec = round(time.perf_counter() - started_perf, 3)

        summary = {
            "created_at": utc_now_iso(),
            **writer_summary,
            "stage": self.stage_name,
            "step": self.step_name,
            "ffmpeg_bin": config.ffmpeg_bin,
            "ffprobe_bin": config.ffprobe_bin,
            "target_short_edge": config.target_short_edge,
            "target_fps": config.target_fps,
            "crf": config.crf,
            "pix_fmt": config.pix_fmt,
            "audio_bitrate": config.audio_bitrate,
            "ray_address": config.ray_address,
            "ray_num_cpus": config.ray_num_cpus,
            "ffmpeg_threads": config.ffmpeg_threads,
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
            "input_run_id": config.input_run_id,
            "run_id": config.run_id,
            "source": config.source or "",
            "source_batch": config.source_batch or "",
            "limit": config.limit,
        }
        summary_path = write_summary_json(summary, output_meta_path)

        return TranscodeResult(
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
