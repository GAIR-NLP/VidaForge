from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

from tqdm import tqdm

from ..parquet import StreamingParquetShardWriter
from .base import ProcessingStats, prepare_parquet_rows


async def _run_async_rows(
    *,
    rows: Iterable[dict[str, object]],
    stats: ProcessingStats,
    output_path: Path,
    parquet_size: int,
    concurrency: int,
    output_unit: str,
    step: str,
    worker: Callable[
        [int, dict[str, object]],
        Awaitable[dict[str, object] | list[dict[str, object]]],
    ],
    reset: bool = True,
    desc: str | None = None,
    failed_examples_limit: int = 1000,
) -> dict[str, object]:
    output_id_field = f"{output_unit}_id"
    path_field = f"{output_unit}_path"
    error_field = f"{step}_error"
    ok_field = f"{step}_ok"
    queue_size = max(1, concurrency * 2)
    input_queue: asyncio.Queue[tuple[int, dict[str, object]] | None] = asyncio.Queue(
        maxsize=queue_size
    )
    output_queue: asyncio.Queue[
        dict[str, object] | list[dict[str, object]] | None
    ] = asyncio.Queue(maxsize=queue_size)
    parquet_writer = StreamingParquetShardWriter(
        output_path,
        unit=output_unit,
        parquet_size=parquet_size,
        reset=reset,
    )
    if stats.input_count == 0:
        parquet_writer.close()
        return parquet_writer.summary()
    progress_bar = tqdm(
        total=stats.input_count,
        desc=desc or step,
        unit=output_unit,
    )

    async def reader() -> None:
        try:
            row_index = 0
            for row in rows:
                await input_queue.put((row_index, row))
                row_index += 1
        finally:
            for _ in range(concurrency):
                await input_queue.put(None)

    async def run_worker() -> None:
        while True:
            item = await input_queue.get()
            if item is None:
                await output_queue.put(None)
                return

            row_index, row = item
            await output_queue.put(await worker(row_index, row))

    async def writer() -> None:
        finished_workers = 0
        try:
            while finished_workers < concurrency:
                item = await output_queue.get()
                if item is None:
                    finished_workers += 1
                    continue

                if isinstance(item, dict):
                    output_rows = [item]
                elif isinstance(item, list):
                    output_rows = item
                else:
                    raise TypeError(
                        "async row worker must return a dict row or a list of dict rows; "
                        f"got {type(item)!r}"
                    )
                for output_row in output_rows:
                    if not isinstance(output_row, dict):
                        raise TypeError("async row worker list result must contain dict rows")
                    parquet_writer.write(output_row)
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
                progress_bar.update(1)
        finally:
            parquet_writer.close()

    try:
        await asyncio.gather(
            reader(),
            writer(),
            *(run_worker() for _ in range(concurrency)),
        )
    finally:
        progress_bar.close()
    return parquet_writer.summary()


async def run_async_processing(
    *,
    input_path: Path,
    output_path: Path,
    parquet_size: int,
    concurrency: int,
    input_unit: str,
    output_unit: str,
    step: str,
    worker: Callable[
        [int, dict[str, object]],
        Awaitable[dict[str, object] | list[dict[str, object]]],
    ],
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
    writer_summary = await _run_async_rows(
        rows=rows,
        stats=stats,
        output_path=output_path,
        parquet_size=parquet_size,
        concurrency=concurrency,
        output_unit=output_unit,
        step=step,
        worker=worker,
        reset=not resume,
        desc=desc,
        failed_examples_limit=failed_examples_limit,
    )
    return stats, writer_summary
