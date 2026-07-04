from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
import shutil
from typing import Any

from tqdm import tqdm

from vidaforge.common.ray import (
    alive_node_resources,
    ray_runtime_env,
    resolve_replicas,
)

from ..parquet import StreamingParquetShardWriter, count_parquet, iter_parquet
from .base import ProcessingStats


DEFAULT_DEDUP_ACTOR_ROW_BATCH_SIZE = 128
DEFAULT_DEDUP_MATCH_BATCH_SIZE = 10_000


def _iter_ranges(count: int, batch_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, count, batch_size):
        yield start, min(start + batch_size, count)


def _resolve_match_batch_size(unit_count: int, batch_size: int) -> int:
    if unit_count <= 0:
        return 1
    if batch_size <= 0:
        raise ValueError("match_batch_size must be > 0")
    return min(int(batch_size), unit_count)


def run_ray_actor_deduping(
    *,
    input_path: Path,
    output_path: Path,
    parquet_size: int,
    input_unit: str,
    output_unit: str,
    step: str,
    ray_address: str,
    actor_cls: type,
    apply_actor_count: int | str,
    apply_actor_options: dict[str, Any],
    match_actor_count: int | str,
    match_actor_options: dict[str, Any],
    apply_actor_kwargs: dict[str, Any] | None = None,
    match_actor_kwargs: dict[str, Any] | None = None,
    apply_enabled: bool = True,
    apply_batch_size: int = DEFAULT_DEDUP_ACTOR_ROW_BATCH_SIZE,
    match_batch_size: int = DEFAULT_DEDUP_MATCH_BATCH_SIZE,
    limit: int | None = None,
    filter: Callable[[dict[str, object]], bool] | None = None,
    desc: str | None = None,
    failed_examples_limit: int = 1000,
) -> tuple[ProcessingStats, dict[str, object], dict[str, object]]:
    if apply_batch_size <= 0:
        raise ValueError("apply_batch_size must be > 0")
    if match_batch_size <= 0:
        raise ValueError("match_batch_size must be > 0")

    input_count = count_parquet(
        input_path,
        unit=input_unit,
        limit=limit,
        filter=filter,
    )
    stats = ProcessingStats(input_count=input_count)
    output_path.mkdir(parents=True, exist_ok=True)

    if input_count == 0:
        writer = StreamingParquetShardWriter(
            output_path,
            unit=output_unit,
            parquet_size=parquet_size,
            reset=True,
        )
        writer.close()
        writer_summary = writer.summary()
        writer_summary["apply_enabled"] = apply_enabled
        writer_summary["apply_actor_count"] = 0
        writer_summary["apply_actor_count_requested"] = apply_actor_count
        writer_summary["match_actor_count"] = 0
        writer_summary["match_actor_count_requested"] = match_actor_count
        writer_summary["apply_batch_size"] = apply_batch_size
        writer_summary["match_batch_size"] = match_batch_size
        return stats, writer_summary, {
            "pair_count": 0,
            "deduplicator_match_summary": {},
        }

    try:
        import ray
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Ray is required for this pipeline step. Install ray first.") from exc

    output_id_field = f"{output_unit}_id"
    input_id_field = f"{input_unit}_id"
    path_field = f"{output_unit}_path"
    error_field = f"{step}_error"
    ok_field = f"{step}_ok"
    feature_root = output_path / "features"
    if apply_enabled and feature_root.exists():
        shutil.rmtree(feature_root)
    if not apply_enabled and not feature_root.is_dir():
        raise RuntimeError(
            "feature store does not exist for apply_enabled=false: "
            f"{feature_root}"
        )

    ray_was_initialized = False
    try:
        ray_was_initialized = ray.is_initialized()
        if not ray_was_initialized:
            ray.init(address=ray_address, ignore_reinit_error=True)

        def kill_actors(actors: Iterable[object]) -> None:
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:  # noqa: BLE001
                    pass

        def run_apply(actors: list[object]) -> None:
            pending_refs: dict[object, tuple[object, int]] = {}
            submitted_count = 0

            def submit_next(
                actor: object,
                row_iter: Iterator[dict[str, object]],
            ) -> bool:
                nonlocal submitted_count
                row_batch: list[dict[str, object]] = []
                while (
                    submitted_count < input_count
                    and len(row_batch) < apply_batch_size
                ):
                    try:
                        row = next(row_iter)
                    except StopIteration:
                        break
                    submitted_count += 1
                    row_batch.append(dict(row))
                if not row_batch:
                    return False

                result_ref = actor.apply_batch.remote(rows=row_batch)
                pending_refs[result_ref] = (actor, len(row_batch))
                return True

            def collect_one(
                row_iter: Iterator[dict[str, object]],
                progress_bar: tqdm | None = None,
            ) -> None:
                if not pending_refs:
                    return
                done_refs, _ = ray.wait(list(pending_refs), num_returns=1)
                done_ref = done_refs[0]
                actor, input_batch_size = pending_refs.pop(done_ref)
                ray.get(done_ref)
                if progress_bar is not None:
                    progress_bar.update(input_batch_size)
                    submit_next(actor, row_iter)

            with tqdm(
                total=input_count,
                desc=f"{step} apply",
                unit=input_unit,
            ) as progress_bar:
                row_iter = iter(
                    iter_parquet(
                        input_path,
                        unit=input_unit,
                        limit=limit,
                        filter=filter,
                    )
                )
                for actor in actors:
                    submit_next(actor, row_iter)
                while pending_refs:
                    collect_one(row_iter, progress_bar)

        resolved_apply_actor_count = 0
        if apply_enabled:
            resolved_apply_actor_count = resolve_replicas(
                apply_actor_count,
                cluster_resources=ray.cluster_resources(),
                node_resources=alive_node_resources(ray.nodes()),
                cpus_per_replica=float(apply_actor_options.get("num_cpus", 1.0)),
                gpus_per_replica=float(apply_actor_options.get("num_gpus", 0.0)),
                max_replicas=input_count,
            )
            ApplyActor = ray.remote(
                **apply_actor_options,
                runtime_env=ray_runtime_env(),
            )(actor_cls)
            apply_actor_common_kwargs = dict(apply_actor_kwargs or {})
            apply_actors = [
                ApplyActor.remote(**apply_actor_common_kwargs)
                for _ in range(resolved_apply_actor_count)
            ]

            try:
                run_apply(apply_actors)

                ray.get(
                    [
                        actor.save_features.remote(
                            feature_root=feature_root,
                            shard_name=f"actor-{actor_index:05d}",
                        )
                        for actor_index, actor in enumerate(apply_actors)
                    ]
                )
            finally:
                kill_actors(apply_actors)

        resolved_match_actor_count = resolve_replicas(
            match_actor_count,
            cluster_resources=ray.cluster_resources(),
            node_resources=alive_node_resources(ray.nodes()),
            cpus_per_replica=float(match_actor_options.get("num_cpus", 1.0)),
            gpus_per_replica=float(match_actor_options.get("num_gpus", 0.0)),
            max_replicas=input_count,
        )
        MatchActor = ray.remote(
            **match_actor_options,
            runtime_env=ray_runtime_env(),
        )(actor_cls)
        match_actor_common_kwargs = dict(match_actor_kwargs or {})
        match_actors = [
            MatchActor.remote(**match_actor_common_kwargs)
            for _ in range(resolved_match_actor_count)
        ]

        def run_match(
            actors: list[object],
            unit_count: int,
        ) -> tuple[list[dict[str, object]], dict[str, int]]:
            resolved_match_batch_size = _resolve_match_batch_size(
                unit_count,
                match_batch_size,
            )
            ranges = list(_iter_ranges(unit_count, resolved_match_batch_size))
            duplicate_pairs: list[dict[str, object]] = []
            pending_refs: dict[object, object] = {}

            def submit_next(
                actor: object,
                range_iter: Iterator[tuple[int, int]],
            ) -> bool:
                try:
                    start, end = next(range_iter)
                except StopIteration:
                    return False
                result_ref = actor.find_duplicate_pairs.remote(start, end)
                pending_refs[result_ref] = actor
                return True

            def collect_one(
                range_iter: Iterator[tuple[int, int]],
                progress_bar: tqdm | None = None,
            ) -> None:
                if not pending_refs:
                    return
                done_refs, _ = ray.wait(list(pending_refs), num_returns=1)
                done_ref = done_refs[0]
                actor = pending_refs.pop(done_ref)
                duplicate_pairs.extend(ray.get(done_ref))
                if progress_bar is not None:
                    progress_bar.update(1)
                    submit_next(actor, range_iter)

            with tqdm(
                total=len(ranges),
                desc=f"{step} match",
                unit="batch",
            ) as progress_bar:
                range_iter = iter(ranges)
                for actor in actors:
                    submit_next(actor, range_iter)
                while pending_refs:
                    collect_one(range_iter, progress_bar)

            return duplicate_pairs, {
                "match_batch_size": resolved_match_batch_size,
            }

        try:
            ray.get(
                [
                    actor.load_features.remote(feature_root=feature_root)
                    for actor in match_actors
                ]
            )

            unit_count = int(ray.get(match_actors[0].unit_count.remote()))
            if unit_count != input_count:
                raise RuntimeError(
                    "feature count does not match current input count "
                    f"(feature_count={unit_count}, input_count={input_count})"
                )
            duplicate_pairs, match_writer_summary = run_match(match_actors, unit_count)

            group_result = ray.get(
                match_actors[0].build_duplicate_rows.remote(duplicate_pairs)
            )
        finally:
            kill_actors(match_actors)
        match_summary = dict(group_result["summary"])

        writer = StreamingParquetShardWriter(
            output_path,
            unit=output_unit,
            parquet_size=parquet_size,
            reset=True,
        )
        try:
            for row in tqdm(
                iter_parquet(
                    input_path,
                    unit=input_unit,
                    limit=limit,
                    filter=filter,
                ),
                total=input_count,
                desc=desc or step,
                unit=output_unit,
            ):
                unit_id = str(row[input_id_field])
                dedup_row = group_result["rows"][unit_id]
                output_row = actor_cls.merge_rows(row, dedup_row)
                writer.write(output_row)
                stats.output_count += 1
                if int(output_row[ok_field]) == 1:
                    stats.ok_count += 1
                else:
                    stats.failed_count += 1
                    if len(stats.failed_examples) < failed_examples_limit:
                        stats.failed_examples.append(
                            {
                                "unit": output_unit,
                                "step": step,
                                "id": str(output_row[output_id_field]),
                                "path": str(output_row.get(path_field, "")),
                                "error": str(output_row[error_field]),
                            }
                        )
        finally:
            writer.close()

        writer_summary = writer.summary()
        writer_summary["apply_enabled"] = apply_enabled
        writer_summary["apply_actor_count"] = resolved_apply_actor_count
        writer_summary["apply_actor_count_requested"] = apply_actor_count
        writer_summary["match_actor_count"] = resolved_match_actor_count
        writer_summary["match_actor_count_requested"] = match_actor_count
        writer_summary.update(match_writer_summary)
        writer_summary["apply_batch_size"] = apply_batch_size
        writer_summary.setdefault("match_batch_size", match_batch_size)
        return stats, writer_summary, match_summary
    finally:
        if not ray_was_initialized and "ray" in locals() and ray.is_initialized():
            ray.shutdown()
