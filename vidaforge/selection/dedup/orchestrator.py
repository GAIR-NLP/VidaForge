"""Stage 3 dedup orchestration."""

from __future__ import annotations

import time

from vidaforge.common import utc_now_iso, write_summary_json
from vidaforge.common.ray import validate_replicas
from vidaforge.index import (
    count_parquet,
    run_ray_actor_deduping,
)

from .config import DedupConfig, DedupConfigBase, DedupResult
from .registry import DEDUPLICATORS
from .worker import DedupWorker


def _validate_config(config: DedupConfig) -> None:
    if not config.run_id.strip():
        raise ValueError("run_id must be set")
    if not config.input_run_id.strip():
        raise ValueError("input_run_id must be set")
    if config.parquet_size <= 0:
        raise ValueError("parquet_size must be > 0")
    if not isinstance(config.apply.enabled, bool):
        raise TypeError("apply.enabled must be a bool")
    if config.apply.batch_size <= 0:
        raise ValueError("apply.batch_size must be > 0")
    if config.match.batch_size <= 0:
        raise ValueError("match.batch_size must be > 0")
    validate_replicas(config.apply.replicas)
    validate_replicas(config.match.replicas)
    if config.apply.ray_num_cpus <= 0:
        raise ValueError("apply.ray_num_cpus must be > 0")
    if config.apply.ray_num_gpus < 0:
        raise ValueError("apply.ray_num_gpus must be >= 0")
    if config.match.ray_num_cpus <= 0:
        raise ValueError("match.ray_num_cpus must be > 0")
    if config.match.ray_num_gpus < 0:
        raise ValueError("match.ray_num_gpus must be >= 0")
    if not config.deduplicators:
        raise ValueError("deduplicators must not be empty")

    for dedup_config in config.deduplicators:
        if not isinstance(dedup_config, DedupConfigBase):
            raise TypeError(
                "deduplicators must contain DedupConfigBase instances; "
                f"got {type(dedup_config)!r}"
            )
    dedup_names = [
        dedup_config.deduplicator_name for dedup_config in config.deduplicators
    ]
    if len(set(dedup_names)) != len(dedup_names):
        raise ValueError(f"deduplicators must be unique: {dedup_names}")
    unsupported = sorted(set(dedup_names) - set(DEDUPLICATORS))
    if unsupported:
        raise ValueError(f"unsupported deduplicators: {unsupported}")


class DedupOrchestrator:
    """Run global clip deduplication."""

    def __init__(
        self,
        stage_name: str = "stage3_selection",
        step_name: str = "step3_dedup",
    ) -> None:
        self.stage_name = stage_name
        self.step_name = step_name

    def dedup(self, config: DedupConfig) -> DedupResult:
        _validate_config(config)

        input_path = config.input_path.expanduser().resolve()
        output_path = config.output_path.expanduser().resolve()
        source_count = count_parquet(input_path, unit="clip")

        started_at = utc_now_iso()
        started_perf = time.perf_counter()
        output_path.mkdir(parents=True, exist_ok=True)

        row_filter = lambda row: int(row["filter_ok"]) == 1
        use_gpu_faiss = config.match.ray_num_gpus > 0
        match_faiss_num_threads = max(1, int(config.match.ray_num_cpus))
        stats, writer_summary, match_summary = run_ray_actor_deduping(
            input_path=input_path,
            output_path=output_path,
            input_unit="clip",
            output_unit="clip",
            step="dedup",
            ray_address=config.ray_address,
            actor_cls=DedupWorker,
            apply_actor_count=config.apply.replicas,
            apply_actor_options={
                "num_cpus": config.apply.ray_num_cpus,
                "num_gpus": config.apply.ray_num_gpus,
            },
            apply_actor_kwargs={
                "deduplicators": config.deduplicators,
                "run_id": config.run_id,
                "input_run_id": config.input_run_id,
                "use_gpu_faiss": False,
                "faiss_num_threads": None,
            },
            match_actor_count=config.match.replicas,
            match_actor_options={
                "num_cpus": config.match.ray_num_cpus,
                "num_gpus": config.match.ray_num_gpus,
            },
            match_actor_kwargs={
                "deduplicators": config.deduplicators,
                "run_id": config.run_id,
                "input_run_id": config.input_run_id,
                "use_gpu_faiss": use_gpu_faiss,
                "faiss_num_threads": match_faiss_num_threads,
            },
            parquet_size=config.parquet_size,
            apply_enabled=config.apply.enabled,
            apply_batch_size=config.apply.batch_size,
            match_batch_size=config.match.batch_size,
            limit=config.limit,
            filter=row_filter,
            desc="dedup",
        )

        elapsed_sec = round(time.perf_counter() - started_perf, 3)
        summary = {
            "created_at": utc_now_iso(),
            **writer_summary,
            "stage": self.stage_name,
            "step": self.step_name,
            "deduplicators": [dedup_config.deduplicator_name for dedup_config in config.deduplicators],
            "deduplicator": {
                dedup_config.deduplicator_name: dedup_config.model_dump(
                    mode="json",
                    exclude_none=True,
                )
                for dedup_config in config.deduplicators
            },
            "ray_address": config.ray_address,
            "apply": {
                "enabled": config.apply.enabled,
                "replicas": writer_summary.get(
                    "apply_actor_count",
                    config.apply.replicas,
                ),
                "replicas_requested": config.apply.replicas,
                "ray_num_cpus": config.apply.ray_num_cpus,
                "ray_num_gpus": config.apply.ray_num_gpus,
                "batch_size": config.apply.batch_size,
            },
            "match": {
                "replicas": writer_summary.get(
                    "match_actor_count",
                    config.match.replicas,
                ),
                "replicas_requested": config.match.replicas,
                "ray_num_cpus": config.match.ray_num_cpus,
                "ray_num_gpus": config.match.ray_num_gpus,
                "batch_size": config.match.batch_size,
                "faiss_num_threads": match_faiss_num_threads,
            },
            "use_gpu_faiss": use_gpu_faiss,
            "match_faiss_num_threads": match_faiss_num_threads,
            "parquet_size": config.parquet_size,
            "pair_count": int(match_summary.get("pair_count", 0)),
            "source_count": source_count,
            "input_count": stats.input_count,
            "resumed_count": stats.resumed_count,
            "output_count": stats.output_count,
            "ok_count": stats.ok_count,
            "failed_count": stats.failed_count,
            "deduplicator_match_summary": match_summary.get(
                "deduplicator_match_summary",
                {},
            ),
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

        return DedupResult(
            input_path=input_path,
            output_path=output_path,
            source_count=source_count,
            input_count=stats.input_count,
            resumed_count=stats.resumed_count,
            output_count=stats.output_count,
            ok_count=stats.ok_count,
            failed_count=stats.failed_count,
            pair_count=int(match_summary.get("pair_count", 0)),
            deduplicator_match_summary=dict(
                match_summary.get("deduplicator_match_summary", {})
            ),
            shard_count=int(writer_summary["shard_count"]),
            summary_path=summary_path,
            elapsed_sec=elapsed_sec,
        )
