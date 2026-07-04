from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from tqdm import tqdm

from vidaforge.common.ray import ray_runtime_env

from ..parquet import StreamingParquetShardWriter
from .base import ProcessingStats, prepare_parquet_rows


def _run_ray_async_actor_rows(
    *,
    rows: Iterable[dict[str, object]],
    stats: ProcessingStats,
    output_path: Path,
    parquet_size: int,
    output_unit: str,
    step: str,
    ray_address: str,
    actor_cls: type,
    actor_options: dict[str, Any],
    actor_kwargs: list[dict[str, Any]],
    batch_size: int = 128,
    reset: bool = True,
    desc: str | None = None,
    failed_examples_limit: int = 1000,
) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not actor_kwargs:
        raise ValueError("actor_kwargs must be non-empty")

    output_id_field = f"{output_unit}_id"
    path_field = f"{output_unit}_path"
    error_field = f"{step}_error"
    ok_field = f"{step}_ok"
    writer = StreamingParquetShardWriter(
        output_path,
        unit=output_unit,
        parquet_size=parquet_size,
        reset=reset,
    )
    if stats.input_count == 0:
        writer.close()
        writer_summary = writer.summary()
        writer_summary["actor_count"] = 0
        writer_summary["actor_count_requested"] = len(actor_kwargs)
        writer_summary["batch_size"] = batch_size
        return writer_summary

    try:
        import ray
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Ray is required for this pipeline step. Install ray first.") from exc

    pending_refs: dict[object, tuple[object, int]] = {}
    ray_was_initialized = False
    submitted_count = 0

    def submit_next(
        actor: object,
        row_iter: Iterator[dict[str, object]],
    ) -> bool:
        nonlocal submitted_count
        row_batch: list[dict[str, object]] = []
        while submitted_count < stats.input_count and len(row_batch) < batch_size:
            try:
                row = next(row_iter)
            except StopIteration:
                break
            submitted_count += 1
            row_batch.append(dict(row))
        if not row_batch:
            return False

        result_ref = actor.process_batch.remote(rows=row_batch)
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
        actor_result = ray.get(done_ref)
        if isinstance(actor_result, dict):
            output_rows = [actor_result]
        elif isinstance(actor_result, list):
            output_rows = actor_result
        else:
            raise TypeError(
                "Ray async actor process_batch must return a dict row or a list "
                f"of dict rows; got {type(actor_result)!r}"
            )
        for output_row in output_rows:
            if not isinstance(output_row, dict):
                raise TypeError(
                    "Ray async actor process_batch list result must contain dict rows"
                )
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
        if progress_bar is not None:
            progress_bar.update(input_batch_size)
            submit_next(actor, row_iter)

    try:
        ray_was_initialized = ray.is_initialized()
        if not ray_was_initialized:
            ray.init(address=ray_address, ignore_reinit_error=True)

        RemoteActor = ray.remote(
            **actor_options,
            runtime_env=ray_runtime_env(),
        )(actor_cls)
        actors = [RemoteActor.remote(**kwargs) for kwargs in actor_kwargs]
        row_iter = iter(rows)

        with tqdm(
            total=stats.input_count,
            desc=desc or step,
            unit=output_unit,
        ) as progress_bar:
            for actor in actors:
                submit_next(actor, row_iter)
            while pending_refs:
                collect_one(row_iter, progress_bar)
    finally:
        writer.close()
        if not ray_was_initialized and "ray" in locals() and ray.is_initialized():
            ray.shutdown()

    writer_summary = writer.summary()
    writer_summary["actor_count"] = len(actor_kwargs)
    writer_summary["actor_count_requested"] = len(actor_kwargs)
    writer_summary["batch_size"] = batch_size
    return writer_summary


def run_ray_async_actor_processing(
    *,
    input_path: Path,
    output_path: Path,
    parquet_size: int,
    input_unit: str,
    output_unit: str,
    step: str,
    ray_address: str,
    actor_cls: type,
    actor_options: dict[str, Any],
    actor_kwargs: list[dict[str, Any]],
    batch_size: int = 128,
    limit: int | None = None,
    filter: Callable[[dict[str, object]], bool] | None = None,
    resume: bool = False,
    is_complete: Callable[[dict[str, object]], bool] | None = None,
    desc: str | None = None,
    failed_examples_limit: int = 1000,
) -> tuple[ProcessingStats, dict[str, object]]:
    rows, stats = prepare_parquet_rows(
        input_path=input_path,
        input_unit=input_unit,
        output_path=output_path,
        output_unit=output_unit,
        step=step,
        limit=limit,
        filter=filter,
        resume=resume,
        is_complete=is_complete,
    )
    writer_summary = _run_ray_async_actor_rows(
        rows=rows,
        stats=stats,
        output_path=output_path,
        parquet_size=parquet_size,
        output_unit=output_unit,
        step=step,
        ray_address=ray_address,
        actor_cls=actor_cls,
        actor_options=actor_options,
        actor_kwargs=actor_kwargs,
        batch_size=batch_size,
        reset=not resume,
        desc=desc,
        failed_examples_limit=failed_examples_limit,
    )
    return stats, writer_summary
