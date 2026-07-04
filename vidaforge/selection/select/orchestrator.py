from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import time

from tqdm import tqdm

from vidaforge.common import utc_now_iso, write_summary_json
from vidaforge.index import (
    PassRejectProcessingStats,
    StreamingParquetShardWriter,
    count_parquet,
    iter_parquet,
)

from .config import SelectConfig, SelectResult
from .worker import SelectWorker


def _validate_config(config: SelectConfig) -> None:
    if not config.input_run_id.strip():
        raise ValueError("input_run_id must be set")
    if not config.run_id.strip():
        raise ValueError("run_id must be set")
    if config.parquet_size <= 0:
        raise ValueError("parquet_size must be > 0")
    if not config.filter and not config.dedup:
        raise ValueError("select filter or dedup config must not be empty")


class SelectOrchestrator:
    """Run Stage 3 select rules over filtered clip metadata."""

    def __init__(
        self,
        stage_name: str = "stage3_selection",
        step_name: str = "step4_select",
    ) -> None:
        self.stage_name = stage_name
        self.step_name = step_name

    def select(self, config: SelectConfig) -> SelectResult:
        _validate_config(config)

        input_path = config.input_path.expanduser().resolve()
        output_path = config.output_path.expanduser().resolve()
        source_count = count_parquet(input_path, unit="clip")
        input_count = count_parquet(input_path, unit="clip", limit=config.limit)

        started_at = utc_now_iso()
        started_perf = time.perf_counter()
        worker = SelectWorker(
            filter_config=config.filter,
            dedup_config=config.dedup,
            input_run_id=config.input_run_id,
            run_id=config.run_id,
        )

        def filter_pass_rows() -> Iterable[dict[str, object]]:
            for row in tqdm(
                iter_parquet(input_path, unit="clip", limit=config.limit),
                total=input_count,
                desc="select build dedup groups",
                unit="clip",
            ):
                row = dict(row)
                filter_result = worker.build_filter_result(row)
                if int(filter_result["ok"]) == 1 and bool(filter_result["passed"]):
                    yield row

        dedup_result = worker.build_dedup_result(filter_pass_rows())

        stats = PassRejectProcessingStats(input_count=input_count)
        reject_reason_counts: Counter[str] = Counter()
        writer = StreamingParquetShardWriter(
            output_path,
            unit="clip",
            parquet_size=config.parquet_size,
            reset=True,
        )
        pass_writer = StreamingParquetShardWriter(
            output_path / "pass",
            unit="clip",
            parquet_size=config.parquet_size,
            reset=True,
        )
        reject_writer = StreamingParquetShardWriter(
            output_path / "reject",
            unit="clip",
            parquet_size=config.parquet_size,
            reset=True,
        )

        try:
            for row in tqdm(
                iter_parquet(input_path, unit="clip", limit=config.limit),
                total=input_count,
                desc="select",
                unit="clip",
            ):
                row = dict(row)
                filter_result = worker.build_filter_result(row)
                output_row = worker.build_select_row(
                    row,
                    filter_result=filter_result,
                    dedup_result=dedup_result,
                )

                writer.write(output_row)
                stats.output_count += 1

                if int(output_row["select_ok"]) == 1:
                    stats.ok_count += 1
                else:
                    stats.failed_count += 1
                    if len(stats.failed_examples) < 1000:
                        stats.failed_examples.append(
                            {
                                "unit": "clip",
                                "step": "select",
                                "id": str(output_row["clip_id"]),
                                "path": str(output_row["clip_path"]),
                                "error": str(output_row["select_error"]),
                            }
                        )

                if int(output_row["select_pass"]) == 1:
                    stats.pass_count += 1
                    pass_writer.write(output_row)
                else:
                    stats.reject_count += 1
                    reject_writer.write(output_row)
                    reject_reason = str(output_row["select_reject_reason"])
                    if reject_reason:
                        reject_reason_counts[reject_reason] += 1
        finally:
            writer.close()
            pass_writer.close()
            reject_writer.close()

        stats.reject_reason_counts = dict(sorted(reject_reason_counts.items()))
        writer_summary = writer.summary()
        writer_summary["pass_output"] = pass_writer.summary()
        writer_summary["reject_output"] = reject_writer.summary()

        elapsed_sec = round(time.perf_counter() - started_perf, 3)
        summary = {
            "created_at": utc_now_iso(),
            **writer_summary,
            "stage": self.stage_name,
            "step": self.step_name,
            "filter": config.filter,
            "dedup": config.dedup,
            "dedup_summary": dedup_result["summary"],
            "source_count": source_count,
            "input_count": stats.input_count,
            "resumed_count": 0,
            "output_count": stats.output_count,
            "ok_count": stats.ok_count,
            "failed_count": stats.failed_count,
            "pass_count": stats.pass_count,
            "reject_count": stats.reject_count,
            "reject_reason_counts": stats.reject_reason_counts,
            "failed_examples": stats.failed_examples,
            "shard_count": int(writer_summary["shard_count"]),
            "started_at": started_at,
            "finished_at": utc_now_iso(),
            "elapsed_sec": elapsed_sec,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "input_run_id": config.input_run_id,
            "run_id": config.run_id,
            "source": config.source or "",
            "source_batch": config.source_batch or "",
            "limit": config.limit,
        }
        summary_path = write_summary_json(summary, output_path)

        return SelectResult(
            input_path=input_path,
            output_path=output_path,
            source_count=source_count,
            input_count=stats.input_count,
            resumed_count=0,
            output_count=stats.output_count,
            ok_count=stats.ok_count,
            failed_count=stats.failed_count,
            pass_count=stats.pass_count,
            reject_count=stats.reject_count,
            shard_count=int(writer_summary["shard_count"]),
            summary_path=summary_path,
            elapsed_sec=elapsed_sec,
        )
