from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Callable, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from vidaforge.common import write_summary_json

DEFAULT_PARQUET_SIZE = 500_000
_PARQUET_BATCH_SIZE = 10_000


def _next_shard_index(output_path: Path, *, unit: str) -> int:
    pattern = re.compile(rf"^{re.escape(unit)}-(\d+)\.parquet$")
    max_index = -1
    for path in output_path.glob(f"{unit}-*.parquet"):
        match = pattern.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def resolve_parquet_paths(
    input_path: str | Path,
    *,
    unit: str | None = None,
) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    pattern = "*.parquet" if not unit else f"{unit}-*.parquet"
    return sorted(child for child in path.glob(pattern) if child.is_file())


def write_parquet(
    rows: Iterable[dict[str, str | int | float | None]],
    output_parquet: str | Path,
) -> None:
    output_path = Path(output_parquet).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    materialized_rows = list(rows)
    if materialized_rows:
        table = pa.Table.from_pylist(materialized_rows)
    else:
        table = pa.Table.from_pylist(
            [],
            schema=pa.schema(
                [
                    ("video_path", pa.string()),
                    ("filesize_bytes", pa.int64()),
                ]
            ),
        )

    pq.write_table(table, output_path)


def load_parquet(
    input_parquet: str | Path,
    *,
    unit: str | None = None,
    columns: list[str] | None = None,
) -> list[dict[str, str | int | float | None]]:
    paths = resolve_parquet_paths(input_parquet, unit=unit)
    if not paths:
        return []

    rows: list[dict[str, str | int | float | None]] = []
    for path in paths:
        table = pq.read_table(path, columns=columns)
        rows.extend(table.to_pylist())
    return rows


def iter_parquet(
    input_parquet: str | Path,
    batch_size: int = _PARQUET_BATCH_SIZE,
    *,
    unit: str | None = None,
    columns: list[str] | None = None,
    limit: int | None = None,
    filter: Callable[[dict[str, object]], bool] | None = None,
) -> Iterator[dict[str, str | int | float | None]]:
    if limit is not None and limit <= 0:
        return
    yielded_rows = 0
    for path in resolve_parquet_paths(input_parquet, unit=unit):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                if filter is not None and not filter(row):
                    continue
                yield row
                yielded_rows += 1
                if limit is not None and yielded_rows >= limit:
                    return


def count_parquet(
    input_parquet: str | Path,
    *,
    unit: str | None = None,
    limit: int | None = None,
    filter: Callable[[dict[str, object]], bool] | None = None,
) -> int:
    if limit is not None and limit <= 0:
        return 0
    if filter is not None or limit is not None:
        return sum(
            1
            for _ in iter_parquet(
                input_parquet,
                unit=unit,
                limit=limit,
                filter=filter,
            )
        )

    total_rows = 0
    for path in resolve_parquet_paths(input_parquet, unit=unit):
        total_rows += pq.ParquetFile(path).metadata.num_rows
    return total_rows


class StreamingParquetShardWriter:
    """Streaming writer for ``<unit>-*.parquet`` shard directories."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        unit: str,
        parquet_size: int = DEFAULT_PARQUET_SIZE,
        reset: bool = True,
    ) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.unit = unit
        self.parquet_size = parquet_size
        if not self.unit.strip():
            raise ValueError("unit is required")
        if self.parquet_size <= 0:
            raise ValueError("parquet_size must be > 0")

        self.output_path.mkdir(parents=True, exist_ok=True)
        if reset:
            for stale_path in self.output_path.glob(f"{self.unit}-*.parquet"):
                stale_path.unlink()
            summary_path = self.output_path / "summary.json"
            if summary_path.exists():
                summary_path.unlink()

        self._writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None
        self._current_path: Path | None = None
        self._buffer: list[dict[str, object]] = []
        self._rows_in_current_file = 0
        self._shard_index = (
            0
            if reset
            else _next_shard_index(self.output_path, unit=self.unit)
        )
        self._shard_paths: list[Path] = []
        self._shard_rows: list[int] = []
        self.total_rows = 0

    def write(self, row: dict[str, object]) -> None:
        self.total_rows += 1
        self._buffer.append(row)
        if (
            self._writer is not None
            and self._rows_in_current_file + len(self._buffer) > self.parquet_size
        ):
            self._close_writer()
        if len(self._buffer) >= min(_PARQUET_BATCH_SIZE, self.parquet_size):
            self._flush_buffer()

    def close(self) -> None:
        self._flush_buffer()
        self._close_writer()

    def summary(self) -> dict[str, object]:
        shards = [
            {
                "path": path.name,
                "rows": rows,
                "size_bytes": path.stat().st_size,
            }
            for path, rows in zip(self._shard_paths, self._shard_rows)
        ]
        schema = self._schema or pa.schema([])
        return {
            "output_path": str(self.output_path),
            "unit": self.unit,
            "total_rows": self.total_rows,
            "shard_count": len(self._shard_paths),
            "parquet_size": self.parquet_size,
            "schema_fields": [
                {"name": field.name, "type": str(field.type)}
                for field in schema
            ],
            "shards": shards,
        }

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return

        if self._writer is None or self._current_path is None:
            self._current_path = (
                self.output_path / f"{self.unit}-{self._shard_index:05d}.parquet"
            )
            if self._schema is None:
                table = pa.Table.from_pylist(self._buffer)
                self._schema = table.schema
            else:
                table = pa.Table.from_pylist(self._buffer, schema=self._schema)
            self._writer = pq.ParquetWriter(self._current_path, self._schema)
            self._shard_paths.append(self._current_path)
            self._shard_rows.append(0)
            self._shard_index += 1
        else:
            table = pa.Table.from_pylist(self._buffer, schema=self._schema)

        self._writer.write_table(table)
        self._rows_in_current_file += len(self._buffer)
        self._shard_rows[-1] += len(self._buffer)
        self._buffer.clear()

    def _close_writer(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
        self._current_path = None
        self._rows_in_current_file = 0


def write_parquet_shards(
    rows: Iterable[dict[str, str | int | float | None]],
    output_path: str | Path,
    *,
    unit: str,
    parquet_size: int = DEFAULT_PARQUET_SIZE,
    summary_extra: dict[str, object] | None = None,
    write_summary: bool = True,
) -> dict[str, object]:
    if not unit.strip():
        raise ValueError("unit is required")
    output_root = Path(output_path).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for stale_path in output_root.glob(f"{unit}-*.parquet"):
        stale_path.unlink()
    summary_path = output_root / "summary.json"
    if write_summary and summary_path.exists():
        summary_path.unlink()

    writer: pq.ParquetWriter | None = None
    writer_schema: pa.Schema | None = None
    current_path: Path | None = None
    shard_paths: list[Path] = []
    shard_rows: list[int] = []
    buffer: list[dict[str, str | int | float | None]] = []
    rows_in_current_file = 0
    shard_index = 0
    total_rows = 0

    def flush_buffer() -> None:
        nonlocal writer
        nonlocal writer_schema
        nonlocal current_path
        nonlocal rows_in_current_file
        nonlocal shard_index
        if not buffer:
            return

        if writer is None or current_path is None:
            current_path = output_root / f"{unit}-{shard_index:05d}.parquet"
            table = pa.Table.from_pylist(buffer)
            writer_schema = table.schema
            writer = pq.ParquetWriter(current_path, writer_schema)
            shard_paths.append(current_path)
            shard_rows.append(0)
            shard_index += 1
        else:
            table = pa.Table.from_pylist(buffer, schema=writer_schema)

        writer.write_table(table)
        rows_in_current_file += len(buffer)
        shard_rows[-1] += len(buffer)
        buffer.clear()

    def close_writer() -> None:
        nonlocal writer
        nonlocal writer_schema
        nonlocal current_path
        nonlocal rows_in_current_file
        if writer is not None:
            writer.close()
            writer = None
            writer_schema = None
            current_path = None
            rows_in_current_file = 0

    for row in rows:
        total_rows += 1
        buffer.append(row)
        if writer is not None and rows_in_current_file + len(buffer) > parquet_size:
            close_writer()
        if len(buffer) >= _PARQUET_BATCH_SIZE:
            flush_buffer()

    flush_buffer()
    close_writer()
    shards = [
        {
            "path": path.name,
            "rows": rows,
            "size_bytes": path.stat().st_size,
        }
        for path, rows in zip(shard_paths, shard_rows)
    ]
    summary = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "output_path": str(output_root),
        "unit": unit,
        "total_rows": total_rows,
        "shard_count": len(shard_paths),
        "parquet_size": parquet_size,
        "schema_fields": [
            {"name": field.name, "type": str(field.type)}
            for field in (writer_schema or pa.schema([]))
        ],
        "shards": shards,
    }
    if summary_extra:
        summary.update(summary_extra)
    if write_summary:
        summary_file = write_summary_json(summary, output_root)
        summary["summary_path"] = str(summary_file)
    else:
        summary["summary_path"] = ""
    return summary
