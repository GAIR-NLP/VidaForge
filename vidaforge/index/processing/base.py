from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ..parquet import count_parquet, iter_parquet
from ..resume import load_completed_ids


@dataclass(slots=True)
class ProcessingStats:
    input_count: int = 0
    resumed_count: int = 0
    output_count: int = 0
    ok_count: int = 0
    failed_count: int = 0
    failed_examples: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class PassRejectProcessingStats:
    input_count: int = 0
    output_count: int = 0
    ok_count: int = 0
    failed_count: int = 0
    pass_count: int = 0
    reject_count: int = 0
    reject_reason_counts: dict[str, int] = field(default_factory=dict)
    failed_examples: list[dict[str, object]] = field(default_factory=list)


def prepare_parquet_rows(
    *,
    input_path: Path,
    input_unit: str,
    output_path: Path,
    output_unit: str,
    step: str,
    limit: int | None = None,
    filter: Callable[[dict[str, object]], bool] | None = None,
    resume: bool = False,
    is_complete: Callable[[dict[str, object]], bool] | None = None,
) -> tuple[Iterable[dict[str, object]], ProcessingStats]:
    input_count = count_parquet(
        input_path,
        unit=input_unit,
        limit=limit,
        filter=filter,
    )
    resumed_count = 0
    rows: Iterable[dict[str, object]] = iter_parquet(
        input_path,
        unit=input_unit,
        limit=limit,
        filter=filter,
    )

    if resume:
        completed_ids = load_completed_ids(
            output_path,
            input_unit=input_unit,
            output_unit=output_unit,
            step=step,
            is_complete=is_complete,
        )
        id_field = f"{input_unit}_id"

        def is_resumed(row: dict[str, object]) -> bool:
            return str(row[id_field]) in completed_ids

        resumed_count = sum(
            1
            for row in iter_parquet(
                input_path,
                unit=input_unit,
                limit=limit,
                filter=filter,
            )
            if is_resumed(row)
        )
        input_count -= resumed_count
        rows = (
            row
            for row in iter_parquet(
                input_path,
                unit=input_unit,
                limit=limit,
                filter=filter,
            )
            if not is_resumed(row)
        )

    return rows, ProcessingStats(
        input_count=input_count,
        resumed_count=resumed_count,
    )
