from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path

from ..parquet import StreamingParquetShardWriter
from .base import PassRejectProcessingStats, prepare_parquet_rows


def run_pass_reject_processing(
    *,
    input_path: Path,
    output_path: Path,
    parquet_size: int,
    input_unit: str,
    output_unit: str,
    step: str,
    worker: Callable[..., dict[str, object]],
    limit: int | None = None,
    filter: Callable[[dict[str, object]], bool] | None = None,
    failed_examples_limit: int = 1000,
) -> tuple[PassRejectProcessingStats, dict[str, object]]:
    output_id_field = f"{output_unit}_id"
    path_field = f"{output_unit}_path"
    ok_field = f"{step}_ok"
    error_field = f"{step}_error"
    pass_field = f"{step}_pass"
    reject_reason_field = f"{step}_reject_reason"

    writer = StreamingParquetShardWriter(
        output_path,
        unit=output_unit,
        parquet_size=parquet_size,
        reset=True,
    )
    pass_writer = StreamingParquetShardWriter(
        output_path / "pass",
        unit=output_unit,
        parquet_size=parquet_size,
        reset=True,
    )
    reject_writer = StreamingParquetShardWriter(
        output_path / "reject",
        unit=output_unit,
        parquet_size=parquet_size,
        reset=True,
    )
    reject_reason_counts: Counter[str] = Counter()
    rows, processing_stats = prepare_parquet_rows(
        input_path=input_path,
        input_unit=input_unit,
        output_path=output_path,
        output_unit=output_unit,
        step=step,
        limit=limit,
        filter=filter,
        resume=False,
    )
    stats = PassRejectProcessingStats(input_count=processing_stats.input_count)
    if stats.input_count == 0:
        writer.close()
        pass_writer.close()
        reject_writer.close()
        summary = writer.summary()
        summary["pass_output"] = pass_writer.summary()
        summary["reject_output"] = reject_writer.summary()
        return stats, summary

    try:
        for row in rows:
            output_row = worker(row=dict(row))
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

            if int(output_row[pass_field]) == 1:
                stats.pass_count += 1
                pass_writer.write(output_row)
            else:
                stats.reject_count += 1
                reject_writer.write(output_row)
                reject_reason = str(output_row[reject_reason_field])
                if reject_reason:
                    reject_reason_counts[reject_reason] += 1
    finally:
        writer.close()
        pass_writer.close()
        reject_writer.close()

    stats.reject_reason_counts = dict(sorted(reject_reason_counts.items()))
    summary = writer.summary()
    summary["pass_output"] = pass_writer.summary()
    summary["reject_output"] = reject_writer.summary()
    return stats, summary
