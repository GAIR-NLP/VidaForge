from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from vidaforge.common import utc_now_iso, write_summary_json
from vidaforge.common.ray import validate_replicas
from vidaforge.index import count_parquet, iter_parquet, run_ray_actor_processing

from .config import AutoModelPackConfig, AutoModelPackResult
from .output import (
    build_resolution_summary,
    prepare_output_path,
    row_is_complete,
    write_metadata_files,
)
from .resolution import resolution_pixel_budget
from .worker import AutoModelPackWorker


def validate_automodel_pack_config(config: AutoModelPackConfig) -> None:
    if not config.run_id.strip():
        raise ValueError("run_id must be set")
    if not config.input_run_id.strip():
        raise ValueError("input_run_id must be set")
    if not config.caption_field.strip():
        raise ValueError("caption_field must not be empty")
    if config.select_pass not in {None, 0, 1}:
        raise ValueError("select_pass must be one of: null, 0, 1")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if config.dynamic_forward_batch_size <= 0:
        raise ValueError("dynamic_forward_batch_size must be > 0")
    if config.metadata_shard_size <= 0:
        raise ValueError("metadata_shard_size must be > 0")
    if config.parquet_size <= 0:
        raise ValueError("parquet_size must be > 0")
    validate_replicas(config.replicas)
    if config.ray_num_cpus <= 0:
        raise ValueError("ray_num_cpus must be > 0")
    if config.ray_num_gpus < 0:
        raise ValueError("ray_num_gpus must be >= 0")
    resolution_pixel_budget(config.bucket_resolution)
    if not config.bucket_durations_sec:
        raise ValueError("bucket.durations_sec must not be empty")
    for duration_sec in config.bucket_durations_sec:
        if float(duration_sec) <= 0:
            raise ValueError("bucket.durations_sec values must be > 0")


def automodel_input_row_filter(
    row: dict[str, object],
    *,
    select_pass: int | None,
) -> bool:
    if int(row["caption_ok"]) != 1:
        return False
    if select_pass is None:
        return True
    return int(row["select_pass"]) == int(select_pass)


class AutoModelPackOrchestrator:
    """Orchestrate Stage 5 AutoModel .meta dataset export."""

    def __init__(
        self,
        stage_name: str = "stage5_packaging",
        step_name: str = "automodel",
    ) -> None:
        self.stage_name = stage_name
        self.step_name = step_name

    def pack(
        self,
        config: AutoModelPackConfig,
        *,
        encoder_cls: type,
        encoder_kwargs: dict[str, Any] | None = None,
    ) -> AutoModelPackResult:
        validate_automodel_pack_config(config)

        input_path = config.input_path.expanduser().resolve()
        output_path = prepare_output_path(
            config.output_path,
            resume=config.resume,
        )
        source_count = count_parquet(input_path, unit="clip")

        started_at = utc_now_iso()
        started_perf = time.perf_counter()
        row_filter = lambda row: automodel_input_row_filter(
            row,
            select_pass=config.select_pass,
        )
        stats, writer_summary = run_ray_actor_processing(
            input_path=input_path,
            output_path=output_path,
            parquet_size=config.parquet_size,
            input_unit="clip",
            output_unit="clip",
            step="automodel",
            ray_address=config.ray_address,
            actor_cls=AutoModelPackWorker,
            actor_count=config.replicas,
            actor_options={
                "num_cpus": config.ray_num_cpus,
                "num_gpus": config.ray_num_gpus,
            },
            actor_kwargs={
                "output_path": output_path,
                "run_id": config.run_id,
                "input_run_id": config.input_run_id,
                "caption_field": config.caption_field,
                "dynamic_forward_batch_size": config.dynamic_forward_batch_size,
                "bucket_resolution": config.bucket_resolution,
                "bucket_upscale": config.bucket_upscale,
                "bucket_durations_sec": list(config.bucket_durations_sec),
                "encoder_cls": encoder_cls,
                "encoder_kwargs": dict(encoder_kwargs or {}),
            },
            batch_size=config.batch_size,
            limit=config.limit,
            filter=row_filter,
            resume=config.resume,
            is_complete=row_is_complete,
            desc="automodel pack",
        )
        packed_rows = [
            dict(row)
            for row in iter_parquet(output_path, unit="clip")
            if row_is_complete(row)
        ]
        packed_rows.sort(key=lambda row: str(row["clip_id"]))
        metadata_summary = write_metadata_files(
            output_path=output_path,
            rows=packed_rows,
            metadata_shard_size=config.metadata_shard_size,
        )
        resolution_summary = build_resolution_summary(
            packed_rows,
            target_resolution=config.bucket_resolution,
        )
        elapsed_sec = round(time.perf_counter() - started_perf, 3)

        summary = {
            "created_at": utc_now_iso(),
            **writer_summary,
            **metadata_summary,
            "stage": self.stage_name,
            "step": self.step_name,
            "ray_address": config.ray_address,
            "replicas": writer_summary.get("actor_count", config.replicas),
            "replicas_requested": config.replicas,
            "ray_num_cpus": config.ray_num_cpus,
            "ray_num_gpus": config.ray_num_gpus,
            "resume": config.resume,
            "parquet_size": config.parquet_size,
            "caption_field": config.caption_field,
            "select_pass": config.select_pass,
            "batch_size": config.batch_size,
            "dynamic_forward_batch_size": config.dynamic_forward_batch_size,
            "metadata_shard_size": config.metadata_shard_size,
            "bucket": {
                "resolution": config.bucket_resolution,
                "upscale": config.bucket_upscale,
                "durations_sec": list(config.bucket_durations_sec),
            },
            **resolution_summary,
            "source_count": source_count,
            "input_count": stats.input_count,
            "resumed_count": stats.resumed_count,
            "output_count": stats.output_count,
            "ok_count": stats.ok_count,
            "failed_count": stats.failed_count,
            "packed_count": len(packed_rows),
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "elapsed_sec": elapsed_sec,
            "failed_examples": stats.failed_examples,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "input_run_id": config.input_run_id,
            "run_id": config.run_id,
            "source": config.source or "",
            "source_batch": config.source_batch or "",
            "limit": config.limit,
        }
        summary_path = write_summary_json(summary, output_path)

        return AutoModelPackResult(
            input_path=input_path,
            output_path=output_path,
            source_count=source_count,
            input_count=stats.input_count,
            resumed_count=stats.resumed_count,
            output_count=stats.output_count,
            ok_count=stats.ok_count,
            failed_count=stats.failed_count,
            shard_count=int(writer_summary["shard_count"]),
            metadata_shard_count=int(metadata_summary["metadata_shard_count"]),
            metadata_path=Path(str(metadata_summary["metadata_path"])),
            summary_path=summary_path,
            elapsed_sec=elapsed_sec,
        )
