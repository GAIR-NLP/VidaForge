from __future__ import annotations

from functools import partial
import time

from vidaforge.common import utc_now_iso, write_summary_json
from vidaforge.index import count_parquet, run_pass_reject_processing

from .config import ScreenConfig, ScreenResult
from .worker import process_screen_row


def _validate_config(config: ScreenConfig) -> None:
    if not config.input_run_id.strip():
        raise ValueError("input_run_id must be set")
    if not config.run_id.strip():
        raise ValueError("run_id must be set")
    if config.parquet_size <= 0:
        raise ValueError("parquet_size must be > 0")
    if not config.rules:
        raise ValueError("screen rules must not be empty")

    for rule_name, rule in config.rules.items():
        if not str(rule_name).strip():
            raise ValueError("screen rule name must not be empty")
        if "field" not in rule:
            raise ValueError(f"screen rule {rule_name!r} is missing field")
        if not any(key in rule for key in ("equals", "min", "max")):
            raise ValueError(
                f"screen rule {rule_name!r} must define equals, min, or max"
            )
        if "equals" in rule and "reject_reason" not in rule:
            raise ValueError(
                f"screen rule {rule_name!r} with equals must define reject_reason"
            )
        if (
            "min" in rule
            and "reject_reason" not in rule
            and "min_reject_reason" not in rule
        ):
            raise ValueError(
                f"screen rule {rule_name!r} with min must define a reject reason"
            )
        if (
            "max" in rule
            and "reject_reason" not in rule
            and "max_reject_reason" not in rule
        ):
            raise ValueError(
                f"screen rule {rule_name!r} with max must define a reject reason"
            )


class ScreenOrchestrator:
    """Run Stage 1 screen rules over probed video metadata."""

    def __init__(
        self,
        stage_name: str = "stage1_ingestion",
        step_name: str = "step2_screen",
    ) -> None:
        self.stage_name = stage_name
        self.step_name = step_name

    def screen(self, config: ScreenConfig) -> ScreenResult:
        _validate_config(config)

        input_path = config.input_path.expanduser().resolve()
        output_path = config.output_path.expanduser().resolve()
        source_count = count_parquet(input_path, unit="video")

        started_at = utc_now_iso()
        started_perf = time.perf_counter()
        screen_worker = partial(
            process_screen_row,
            rules=config.rules,
            input_run_id=config.input_run_id,
            run_id=config.run_id,
        )
        stats, writer_summary = run_pass_reject_processing(
            input_path=input_path,
            output_path=output_path,
            parquet_size=config.parquet_size,
            input_unit="video",
            output_unit="video",
            step="screen",
            worker=screen_worker,
            limit=config.limit,
        )
        elapsed_sec = round(time.perf_counter() - started_perf, 3)
        summary = {
            "created_at": utc_now_iso(),
            **writer_summary,
            "stage": self.stage_name,
            "step": self.step_name,
            "rules": config.rules,
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

        return ScreenResult(
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
