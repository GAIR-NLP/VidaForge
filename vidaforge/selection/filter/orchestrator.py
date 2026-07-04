"""Stage 3 filter orchestration."""

from __future__ import annotations

import time

from vidaforge.common import utc_now_iso, write_summary_json
from vidaforge.common.ray import validate_replicas
from vidaforge.index import count_parquet, run_ray_actor_processing

from .config import FilterConfig, FilterConfigBase, FilterResult
from .registry import FILTERS
from .worker import FilterWorker


def _validate_config(config: FilterConfig) -> None:
    if not config.run_id.strip():
        raise ValueError("run_id must be set")
    if not config.input_run_id.strip():
        raise ValueError("input_run_id must be set")
    if config.parquet_size <= 0:
        raise ValueError("parquet_size must be > 0")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    validate_replicas(config.replicas)
    if config.ray_num_cpus <= 0:
        raise ValueError("ray_num_cpus must be > 0")
    if config.ray_num_gpus < 0:
        raise ValueError("ray_num_gpus must be >= 0")
    filter_specs = config.filters
    if not filter_specs:
        raise ValueError("filters must not be empty")
    for filter_config in filter_specs:
        if not isinstance(filter_config, FilterConfigBase):
            raise TypeError(
                "filters must contain FilterConfigBase instances; "
                f"got {type(filter_config)!r}"
            )
    filter_names = [filter_config.filter_name for filter_config in filter_specs]
    if len(set(filter_names)) != len(filter_names):
        raise ValueError(f"filters must be unique: {filter_names}")
    unsupported = sorted(set(filter_names) - set(FILTERS))
    if unsupported:
        raise ValueError(f"unsupported filters: {unsupported}")


class FilterOrchestrator:
    """Orchestrate Stage 3 Step 2 clip filtering."""

    def __init__(
        self,
        stage_name: str = "stage3_selection",
        step_name: str = "step2_filter",
    ) -> None:
        self.stage_name = stage_name
        self.step_name = step_name

    def filter(self, config: FilterConfig) -> FilterResult:
        _validate_config(config)

        input_path = config.input_path.expanduser().resolve()
        output_path = config.output_path.expanduser().resolve()
        source_count = count_parquet(input_path, unit="clip")

        started_at = utc_now_iso()
        started_perf = time.perf_counter()
        output_path.mkdir(parents=True, exist_ok=True)
        filter_specs = config.filters
        filter_names = [filter_config.filter_name for filter_config in filter_specs]

        stats, writer_summary = run_ray_actor_processing(
            input_path=input_path,
            output_path=output_path,
            parquet_size=config.parquet_size,
            input_unit="clip",
            output_unit="clip",
            step="filter",
            ray_address=config.ray_address,
            actor_cls=FilterWorker,
            actor_count=config.replicas,
            actor_options={
                "num_cpus": config.ray_num_cpus,
                "num_gpus": config.ray_num_gpus,
            },
            actor_kwargs={
                "filters": filter_specs,
                "run_id": config.run_id,
                "input_run_id": config.input_run_id,
            },
            batch_size=config.batch_size,
            limit=config.limit,
            filter=lambda row: int(row["frame_ok"]) == 1,
            resume=config.resume,
            desc="stage3 filter",
        )
        elapsed_sec = round(time.perf_counter() - started_perf, 3)

        summary = {
            "created_at": utc_now_iso(),
            **writer_summary,
            "stage": self.stage_name,
            "step": self.step_name,
            "filters": filter_names,
            "filter_configs": {
                filter_config.filter_name: filter_config.model_dump(
                    mode="json",
                    exclude_none=True,
                )
                for filter_config in filter_specs
            },
            "ray_address": config.ray_address,
            "replicas": writer_summary.get("actor_count", config.replicas),
            "replicas_requested": config.replicas,
            "ray_num_cpus": config.ray_num_cpus,
            "ray_num_gpus": config.ray_num_gpus,
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
            "output_path": str(output_path),
            "input_run_id": config.input_run_id,
            "run_id": config.run_id,
            "source": config.source or "",
            "source_batch": config.source_batch or "",
            "limit": config.limit,
        }
        summary_path = write_summary_json(summary, output_path)

        return FilterResult(
            input_path=input_path,
            output_path=output_path,
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
