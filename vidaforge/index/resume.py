from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from vidaforge.common import join_data_dir

from .parquet import iter_parquet


def load_completed_ids(
    output_path: Path,
    *,
    input_unit: str,
    output_unit: str,
    step: str,
    is_complete: Callable[[dict[str, object]], bool] | None = None,
) -> set[str]:
    id_field, ok_field = f"{input_unit}_id", f"{step}_ok"
    asset_field = f"{output_unit}_path"

    completed_ids: set[str] = set()
    for row in iter_parquet(output_path, unit=output_unit):
        if int(row[ok_field]) != 1:
            continue
        if is_complete is not None:
            if not is_complete(row):
                continue
        elif asset_field in row:
            asset_path = str(row[asset_field]).strip()
            if not asset_path:
                continue
            resolved_asset_path = join_data_dir(asset_path)
            try:
                if not resolved_asset_path.is_file() or resolved_asset_path.stat().st_size <= 0:
                    continue
            except OSError:
                continue
        completed_ids.add(str(row[id_field]))
    return completed_ids
