from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from tqdm import tqdm

from vidaforge.common.ray import ray_runtime_env, resolve_max_pending_tasks
from ..parquet import StreamingParquetShardWriter
from .base import ProcessingStats, prepare_parquet_rows


DEFAULT_COLLECT_TASK_BATCH_SIZE = 256


def iter_row_batches(
    rows: Iterable[dict[str, object]],
    batch_size: int,
) -> Iterable[list[dict[str, object]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    batch: list[dict[str, object]] = []
    for row in rows:
        batch.append(dict(row))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _run_ray_task_rows(
    *,
    rows: Iterable[dict[str, object]],
    stats: ProcessingStats,
    output_path: Path,
    parquet_size: int,
    output_unit: str,
    step: str,
    ray_address: str,
    ray_num_cpus: float,
    task_batch_size: int,
    worker: Callable[..., dict[str, object] | list[dict[str, object]]],
    reset: bool = True,
    desc: str | None = None,
    failed_examples_limit: int = 1000,
) -> dict[str, object]:
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
        return writer.summary()

    try:
        import ray
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Ray is required for this pipeline step. Install ray first.") from exc

    pending_refs: dict[object, int] = {}
    ray_was_initialized = False

    def submit_task(task_remote: object, row_batch: list[dict[str, object]]) -> None:
        pending_refs[task_remote.remote(rows=row_batch)] = len(row_batch)

    def write_task_result(
        task_result: dict[str, object] | list[dict[str, object]],
    ) -> None:
        if isinstance(task_result, dict):
            output_rows = [task_result]
        elif isinstance(task_result, list):
            output_rows = task_result
        else:
            raise TypeError(
                "Ray row task must return a dict row or a list of dict rows; "
                f"got {type(task_result)!r}"
            )
        for output_row in output_rows:
            if not isinstance(output_row, dict):
                raise TypeError("Ray row task list result must contain dict rows")
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
                            "path": str(output_row[path_field]),
                            "error": str(output_row[error_field]),
                        }
                    )

    def collect_ready_tasks(progress_bar: tqdm | None = None) -> None:
        if not pending_refs:
            return

        max_returns = min(DEFAULT_COLLECT_TASK_BATCH_SIZE, len(pending_refs))
        done_refs, _ = ray.wait(
            list(pending_refs),
            num_returns=max_returns,
            timeout=0,
        )
        if not done_refs:
            done_refs, _ = ray.wait(list(pending_refs), num_returns=1)

        completed_row_count = 0
        for done_ref in done_refs:
            completed_row_count += pending_refs.pop(done_ref)
        for task_result in ray.get(done_refs):
            write_task_result(task_result)
        if progress_bar is not None:
            progress_bar.update(completed_row_count)

    try:
        ray_was_initialized = ray.is_initialized()
        if not ray_was_initialized:
            ray.init(address=ray_address, ignore_reinit_error=True)

        cluster_cpus = float(ray.cluster_resources().get("CPU", 0.0))
        effective_max_pending_tasks = resolve_max_pending_tasks(
            total_cpus=int(cluster_cpus),
            cpus_per_task=ray_num_cpus,
        )

        def run_worker_batch(
            rows: list[dict[str, object]],
        ) -> dict[str, object] | list[dict[str, object]]:
            output_rows: list[dict[str, object]] = []
            for row in rows:
                result = worker(row=row)
                if isinstance(result, dict):
                    output_rows.append(result)
                elif isinstance(result, list):
                    output_rows.extend(result)
                else:
                    raise TypeError(
                        "Ray row task worker must return a dict row or a list of "
                        f"dict rows; got {type(result)!r}"
                    )
            return output_rows

        task_remote = ray.remote(
            num_cpus=ray_num_cpus,
            runtime_env=ray_runtime_env(),
        )(run_worker_batch)
        with tqdm(
            total=stats.input_count,
            desc=desc or step,
            unit=output_unit,
        ) as progress_bar:
            for row_batch in iter_row_batches(rows, task_batch_size):
                submit_task(task_remote, row_batch)
                if len(pending_refs) >= effective_max_pending_tasks:
                    collect_ready_tasks(progress_bar)
            while pending_refs:
                collect_ready_tasks(progress_bar)
    finally:
        writer.close()
        if not ray_was_initialized and "ray" in locals() and ray.is_initialized():
            ray.shutdown()

    return writer.summary()


def run_ray_task_processing(
    *,
    output_path: Path,
    parquet_size: int,
    output_unit: str,
    step: str,
    ray_address: str,
    ray_num_cpus: float,
    worker: Callable[..., dict[str, object] | list[dict[str, object]]],
    task_batch_size: int = 1,
    input_path: Path | None = None,
    input_unit: str | None = None,
    rows: Iterable[dict[str, object]] | None = None,
    row_count: int | None = None,
    limit: int | None = None,
    filter: Callable[[dict[str, object]], bool] | None = None,
    resume: bool = False,
    is_complete: Callable[[dict[str, object]], bool] | None = None,
    desc: str | None = None,
    failed_examples_limit: int = 1000,
) -> tuple[ProcessingStats, dict[str, object]]:
    if task_batch_size <= 0:
        raise ValueError("task_batch_size must be > 0")

    if rows is None:
        if input_path is None:
            raise ValueError("input_path is required when rows is not provided")
        if input_unit is None:
            raise ValueError("input_unit is required when rows is not provided")
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
    else:
        if input_path is not None:
            raise ValueError("input_path and rows are mutually exclusive")
        if input_unit is not None:
            raise ValueError("input_unit is only valid with input_path")
        if row_count is None:
            raise ValueError("row_count is required when rows is provided")
        if limit is not None or filter is not None or resume or is_complete is not None:
            raise ValueError(
                "limit, filter, resume, and is_complete are only valid with input_path"
            )
        stats = ProcessingStats(input_count=row_count)

    writer_summary = _run_ray_task_rows(
        rows=rows,
        stats=stats,
        output_path=output_path,
        parquet_size=parquet_size,
        output_unit=output_unit,
        step=step,
        ray_address=ray_address,
        ray_num_cpus=ray_num_cpus,
        task_batch_size=task_batch_size,
        worker=worker,
        reset=not resume,
        desc=desc,
        failed_examples_limit=failed_examples_limit,
    )
    return stats, writer_summary
