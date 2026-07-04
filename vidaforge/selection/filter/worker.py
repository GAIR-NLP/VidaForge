from __future__ import annotations

import json

from .config import FilterConfigBase
from .registry import FILTERS


def _build_filter_results(
    *,
    name: str,
    ok: int,
    error: str,
    score: float,
    payload: dict[str, object],
) -> dict[str, object]:
    return {
        f"{name}_ok": int(ok),
        f"{name}_error": error,
        f"{name}_score": round(float(score), 6),
        f"{name}_json": json.dumps(payload, ensure_ascii=False, sort_keys=True),
    }


def merge_filter_fields(
    row: dict[str, object],
    *,
    current_filters: list[str],
    current_errors: list[str],
) -> dict[str, object]:
    previous_filters = row.get("filters") or []
    if isinstance(previous_filters, list):
        filter_names = [str(item) for item in previous_filters]
    else:
        filter_names = [str(previous_filters)]

    previous_error = row.get("filter_error") or ""
    current_error = "; ".join(current_errors)
    merged_error = "; ".join(
        error for error in [str(previous_error), current_error] if error
    )

    return {
        "filters": list(dict.fromkeys([*filter_names, *current_filters])),
        "filter_ok": int(int(row.get("filter_ok", 1)) == 1 and not current_errors),
        "filter_error": merged_error,
    }


class FilterWorker:
    def __init__(
        self,
        *,
        filters: list[FilterConfigBase],
        run_id: str,
        input_run_id: str,
    ) -> None:
        self.run_id = run_id
        self.input_run_id = input_run_id
        self.filters = [FILTERS[config.filter_name](config) for config in filters]
        self.filter_names = [
            filter_instance.filter_name for filter_instance in self.filters
        ]

    def process_batch(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        output_rows = [dict(row) for row in rows]
        errors_by_row: list[list[str]] = [[] for _ in rows]

        for filter_instance in self.filters:
            filter_name = filter_instance.filter_name
            try:
                filter_outputs = filter_instance.apply_batch(rows)
                if len(filter_outputs) != len(rows):
                    raise RuntimeError(
                        f"{filter_name}.apply_batch returned {len(filter_outputs)} "
                        f"results for {len(rows)} rows"
                    )
                row_errors: list[str] = ["" for _ in rows]
            except Exception:
                filter_outputs = []
                row_errors = []
                for row in rows:
                    try:
                        filter_outputs.append(filter_instance.apply(row))
                        row_errors.append("")
                    except Exception as exc:  # noqa: BLE001
                        filter_outputs.append((0.0, {}))
                        row_errors.append(str(exc))

            for index, (score, payload) in enumerate(filter_outputs):
                error = row_errors[index]
                ok = int(not error)
                if error:
                    errors_by_row[index].append(f"{filter_name}: {error}")
                output_rows[index].update(
                    _build_filter_results(
                        name=filter_name,
                        ok=ok,
                        error=error,
                        score=float(score),
                        payload=dict(payload),
                    )
                )

        for index, output_row in enumerate(output_rows):
            output_row.update(
                merge_filter_fields(
                    rows[index],
                    current_filters=list(self.filter_names),
                    current_errors=errors_by_row[index],
                )
            )
            output_row.update(
                {
                    "input_run_id": self.input_run_id,
                    "run_id": self.run_id,
                }
            )
        return output_rows
