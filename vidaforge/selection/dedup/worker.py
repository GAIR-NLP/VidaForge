from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from .config import DedupConfigBase
from .registry import DEDUPLICATORS


class DedupWorker:
    def __init__(
        self,
        *,
        deduplicators: list[DedupConfigBase],
        run_id: str,
        input_run_id: str,
        use_gpu_faiss: bool = False,
        faiss_num_threads: int | None = None,
    ) -> None:
        self.run_id = run_id
        self.input_run_id = input_run_id
        self.use_gpu_faiss = use_gpu_faiss
        self.faiss_num_threads = (
            max(1, int(faiss_num_threads))
            if faiss_num_threads is not None
            else None
        )
        if self.faiss_num_threads is not None:
            import faiss

            faiss.omp_set_num_threads(self.faiss_num_threads)
        self.deduplicators = [
            DEDUPLICATORS[config.deduplicator_name](
                config,
                use_gpu_faiss=use_gpu_faiss,
            )
            for config in deduplicators
        ]
        self.deduplicator_names = [
            deduplicator.deduplicator_name for deduplicator in self.deduplicators
        ]

    @staticmethod
    def merge_rows(
        row: dict[str, object],
        dedup_row: dict[str, object],
    ) -> dict[str, object]:
        previous_error = str(row.get("dedup_error") or "")
        current_error = str(dedup_row.get("dedup_error") or "")

        output_row = dict(row)
        output_row.update(dedup_row)
        output_row.update(
            {
                "deduplicators": list(
                    dict.fromkeys(
                        [
                            *list(row.get("deduplicators") or []),
                            *list(dedup_row.get("deduplicators") or []),
                        ]
                    )
                ),
                "dedup_ok": int(
                    int(row.get("dedup_ok", 1)) == 1
                    and int(dedup_row["dedup_ok"]) == 1
                ),
                "dedup_error": "; ".join(
                    error for error in [previous_error, current_error] if error
                ),
            }
        )
        return output_row

    def apply(self, row: dict[str, object]) -> None:
        for deduplicator in self.deduplicators:
            deduplicator.apply(row)

    def apply_batch(self, rows: list[dict[str, object]]) -> None:
        for deduplicator in self.deduplicators:
            deduplicator.apply_batch(rows)

    def save_features(self, *, feature_root: Path, shard_name: str) -> None:
        for deduplicator in self.deduplicators:
            deduplicator.save_features(
                feature_root / deduplicator.deduplicator_name / shard_name
            )

    def load_features(self, *, feature_root: Path) -> None:
        for deduplicator in self.deduplicators:
            deduplicator.load_features(
                feature_root / deduplicator.deduplicator_name
            )

    def unit_count(self) -> int:
        if not self.deduplicators:
            return 0
        return self.deduplicators[0].unit_count()

    def find_duplicate_pairs(self, start: int, end: int) -> list[dict[str, object]]:
        duplicate_pairs: list[dict[str, object]] = []
        for deduplicator in self.deduplicators:
            duplicate_pairs.extend(deduplicator.find_duplicate_pairs(start, end))
        return duplicate_pairs

    def build_duplicate_rows(
        self,
        duplicate_pairs: list[dict[str, object]],
    ) -> dict[str, object]:
        rows_by_deduplicator: dict[str, dict[str, dict[str, object]]] = {}
        summary_by_deduplicator: dict[str, dict[str, int]] = {}

        for deduplicator in tqdm(
            self.deduplicators,
            desc="dedup build deduplicators",
            unit="deduplicator",
            mininterval=5.0,
        ):
            name = deduplicator.deduplicator_name
            deduplicator_output = deduplicator.build_rows(duplicate_pairs)
            rows_by_deduplicator[name] = deduplicator_output["rows"]
            summary_by_deduplicator[name] = deduplicator_output["summary"]

        rows: dict[str, dict[str, object]] = {}
        for clip_id in tqdm(
            self.deduplicators[0].unit_ids(),
            desc="dedup merge rows",
            unit="clip",
            mininterval=5.0,
        ):
            rows[clip_id] = self._build_dedup_row(
                {
                    name: rows_by_deduplicator[name][clip_id]
                    for name in self.deduplicator_names
                }
            )

        return {
            "rows": rows,
            "summary": {
                "pair_count": sum(
                    int(summary["pair_count"])
                    for summary in summary_by_deduplicator.values()
                ),
                "deduplicator_match_summary": summary_by_deduplicator,
            },
        }

    def _build_dedup_row(
        self,
        deduplicator_rows: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        dedup_errors = [
            f"{name}: {deduplicator_row['error']}"
            for name, deduplicator_row in deduplicator_rows.items()
            if int(deduplicator_row["ok"]) != 1
        ]
        dedup_json = {
            name: deduplicator_row["json"]
            for name, deduplicator_row in deduplicator_rows.items()
        }

        dedup_row: dict[str, object] = {
            "deduplicators": list(self.deduplicator_names),
            "dedup_ok": int(not dedup_errors),
            "dedup_error": "; ".join(dedup_errors),
            "dedup_json": json.dumps(
                dedup_json,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "input_run_id": self.input_run_id,
            "run_id": self.run_id,
        }
        for name, deduplicator_row in deduplicator_rows.items():
            dedup_row.update(
                {
                    f"{name}_ok": int(deduplicator_row["ok"]),
                    f"{name}_error": str(deduplicator_row["error"]),
                    f"{name}_group_id": str(
                        deduplicator_row["json"].get("group_id") or ""
                    ),
                    f"{name}_group_size": int(
                        deduplicator_row["json"].get("group_size", 1)
                    ),
                    f"{name}_is_best_clip_in_group": int(
                        deduplicator_row["json"].get("is_best_clip_in_group", 1)
                    ),
                    f"{name}_best_clip_id_in_group": str(
                        deduplicator_row["json"].get("best_clip_id_in_group") or ""
                    ),
                    f"{name}_json": json.dumps(
                        deduplicator_row["json"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
        return dedup_row
