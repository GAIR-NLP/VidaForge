from __future__ import annotations

from functools import partial
import shutil
import time

from vidaforge.common import utc_now_iso, write_summary_json
from vidaforge.index import (
    count_parquet,
    run_ray_task_processing,
)

from .config import ClipConfig, ClipResult
from .worker import process_clip_row


def _validate_config(config: ClipConfig) -> None:
    if not config.run_id.strip():
        raise ValueError("run_id must be set")
    if not config.input_run_id.strip():
        raise ValueError("input_run_id must be set")
    if config.min_len_sec <= 0:
        raise ValueError("min_len_sec must be > 0")
    if config.overlong_split_len_sec > config.max_len_sec:
        raise ValueError("overlong_split_len_sec must be <= max_len_sec")
    if config.boundary_trim_sec < 0:
        raise ValueError("boundary_trim_sec must be >= 0")
    if config.parquet_size <= 0:
        raise ValueError("parquet_size must be > 0")
    if config.ray_num_cpus <= 0:
        raise ValueError("ray_num_cpus must be > 0")


class ClipOrchestrator:
    """Orchestrate Stage 2 Step 2 clip asset generation."""

    def __init__(
        self,
        stage_name: str = "stage2_segmentation",
        step_name: str = "step2_clip",
    ) -> None:
        self.stage_name = stage_name
        self.step_name = step_name

    def clip(self, config: ClipConfig) -> ClipResult:
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
        clip_worker = partial(
            process_clip_row,
            output_data_path=str(output_data_path),
            run_id=config.run_id,
            input_run_id=config.input_run_id,
            min_len_sec=config.min_len_sec,
            max_len_sec=config.max_len_sec,
            overlong_split_len_sec=config.overlong_split_len_sec,
            boundary_trim_sec=config.boundary_trim_sec,
            ffmpeg_bin=config.ffmpeg_bin,
        )

        stats, writer_summary = run_ray_task_processing(
            input_path=input_path,
            output_path=output_meta_path,
            parquet_size=config.parquet_size,
            input_unit="video",
            output_unit="clip",
            step="clip",
            ray_address=config.ray_address,
            ray_num_cpus=config.ray_num_cpus,
            worker=clip_worker,
            limit=config.limit,
            resume=config.resume,
            desc="stage2 clip",
        )
        elapsed_sec = round(time.perf_counter() - started_perf, 3)

        summary = {
            "created_at": utc_now_iso(),
            **writer_summary,
            "stage": self.stage_name,
            "step": self.step_name,
            "min_len_sec": config.min_len_sec,
            "max_len_sec": config.max_len_sec,
            "overlong_split_len_sec": config.overlong_split_len_sec,
            "boundary_trim_sec": config.boundary_trim_sec,
            "ffmpeg_bin": config.ffmpeg_bin,
            "ray_address": config.ray_address,
            "ray_num_cpus": config.ray_num_cpus,
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

        return ClipResult(
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
