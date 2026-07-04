from __future__ import annotations

from collections.abc import Iterable
import json
import math

from vidaforge.filters import check as check_filter_rules
from vidaforge.filters import resolve_field

_FILTER_RULE_FIELDS = {
    "filter_ok": "filter_ok",
    "optical": "optical_score",
    "motion": "motion_score",
    "aesthetic": "aesthetic_score",
    "text": "text_score",
}
_SCORE_FILTERS = ("aesthetic", "text", "optical", "motion")


class SelectWorker:
    def __init__(
        self,
        *,
        filter_config: dict[str, dict[str, object]],
        dedup_config: dict[str, dict[str, object]],
        input_run_id: str,
        run_id: str,
    ) -> None:
        self.filter_config = filter_config
        self.dedup_config = dedup_config
        self.input_run_id = input_run_id
        self.run_id = run_id

        unknown_filters = sorted(set(filter_config) - set(_FILTER_RULE_FIELDS))
        if unknown_filters:
            raise ValueError(f"unknown select filter rules: {unknown_filters}")

        self.filter_rules: dict[str, dict[str, object]] = {}
        for name, field in _FILTER_RULE_FIELDS.items():
            if name not in filter_config:
                continue
            rule = dict(filter_config[name])
            rule["field"] = field
            self.filter_rules[name] = rule

        if "dedup_ok" in dedup_config:
            rule = dict(dedup_config["dedup_ok"])
            rule["field"] = "dedup_ok"
            self.filter_rules["dedup_ok"] = rule

        for name, rule in self.filter_rules.items():
            if not any(key in rule for key in ("equals", "min", "max")):
                raise ValueError(
                    f"select filter rule {name!r} must define equals, min, or max"
                )
            if "equals" in rule and "reject_reason" not in rule:
                raise ValueError(
                    f"select filter rule {name!r} with equals must define "
                    "reject_reason"
                )
            if (
                "min" in rule
                and "reject_reason" not in rule
                and "min_reject_reason" not in rule
            ):
                raise ValueError(
                    f"select filter rule {name!r} with min must define a reject "
                    "reason"
                )
            if (
                "max" in rule
                and "reject_reason" not in rule
                and "max_reject_reason" not in rule
            ):
                raise ValueError(
                    f"select filter rule {name!r} with max must define a reject "
                    "reason"
                )

        self.dedup_methods: dict[str, dict[str, object]] = {}
        for name, config in dedup_config.items():
            if name == "dedup_ok":
                continue
            method_config = dict(config)
            if "keep_ratio" not in method_config:
                raise ValueError(f"select dedup method {name!r} is missing keep_ratio")
            keep_ratio = float(method_config["keep_ratio"])
            if keep_ratio <= 0.0 or keep_ratio > 1.0:
                raise ValueError(
                    f"select dedup method {name!r} keep_ratio must be in (0, 1]"
                )
            min_keep = int(method_config.get("min_keep", 1))
            if min_keep < 1:
                raise ValueError(
                    f"select dedup method {name!r} min_keep must be >= 1"
                )
            max_keep = method_config.get("max_keep")
            if isinstance(max_keep, str) and max_keep.strip().lower() == "null":
                max_keep = None
            if max_keep is not None:
                max_keep = int(max_keep)
                if max_keep < min_keep:
                    raise ValueError(
                        f"select dedup method {name!r} max_keep must be >= min_keep"
                    )
            if "reject_reason" not in method_config:
                raise ValueError(
                    f"select dedup method {name!r} must define reject_reason"
                )

            method_config["keep_ratio"] = keep_ratio
            method_config["min_keep"] = min_keep
            method_config["max_keep"] = max_keep
            self.dedup_methods[name] = method_config

        self.score_fields = [
            _FILTER_RULE_FIELDS[name]
            for name in _SCORE_FILTERS
            if name in self.filter_config
        ]

    def build_filter_result(self, row: dict[str, object]) -> dict[str, object]:
        try:
            values = {
                str(rule["field"]): resolve_field(row, str(rule["field"]))
                for rule in self.filter_rules.values()
            }
            result = check_filter_rules(values=values, rules=self.filter_rules)
            return {
                "ok": 1,
                "error": "",
                "passed": bool(result.passed),
                "reject_reason": result.reject_reason,
                "json": result.json,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": 0,
                "error": str(exc),
                "passed": False,
                "reject_reason": "",
                "json": {},
            }

    def score_row(self, row: dict[str, object]) -> float:
        score = 0.0
        for field in self.score_fields:
            value = resolve_field(row, field)
            if value is None:
                raise ValueError(f"missing score field for select dedup: {field}")
            score += float(value)
        return score

    def build_dedup_result(
        self,
        rows: Iterable[dict[str, object]],
    ) -> dict[str, object]:
        candidates: list[dict[str, object]] = []
        for row in rows:
            clip_id = str(row["clip_id"])
            candidate: dict[str, object] = {
                "clip_id": clip_id,
                "score": self.score_row(row),
            }
            for method in self.dedup_methods:
                group_field = f"{method}_group_id"
                if group_field not in row:
                    raise ValueError(
                        f"select dedup method {method!r} requires field "
                        f"{group_field!r}"
                    )
                candidate[f"{method}_group_id"] = str(row.get(group_field) or "")
            candidates.append(candidate)

        active_candidates = list(candidates)
        keep_ids_by_method: dict[str, set[str]] = {}
        summary_by_method: dict[str, dict[str, object]] = {}

        for method, config in self.dedup_methods.items():
            group_key = f"{method}_group_id"
            keep_ids: set[str] = set()
            groups: dict[str, list[dict[str, object]]] = {}
            singleton_count = 0

            for candidate in active_candidates:
                group_id = str(candidate.get(group_key) or "")
                if not group_id:
                    keep_ids.add(str(candidate["clip_id"]))
                    singleton_count += 1
                    continue
                groups.setdefault(group_id, []).append(candidate)

            for group_candidates in groups.values():
                group_candidates = sorted(
                    group_candidates,
                    key=lambda item: (-float(item["score"]), str(item["clip_id"])),
                )
                keep_count = math.ceil(
                    len(group_candidates) * float(config["keep_ratio"])
                )
                keep_count = max(int(config["min_keep"]), keep_count)
                max_keep = config.get("max_keep")
                if max_keep is not None:
                    keep_count = min(int(max_keep), keep_count)
                keep_count = min(len(group_candidates), keep_count)
                keep_ids.update(
                    str(candidate["clip_id"])
                    for candidate in group_candidates[:keep_count]
                )

            active_ids = {str(candidate["clip_id"]) for candidate in active_candidates}
            rejected_ids = active_ids - keep_ids
            keep_ids_by_method[method] = keep_ids
            summary_by_method[method] = {
                "candidate_count": len(active_candidates),
                "singleton_count": singleton_count,
                "group_count": len(groups),
                "kept_count": len(keep_ids),
                "rejected_count": len(rejected_ids),
                "keep_ratio": config["keep_ratio"],
                "min_keep": config["min_keep"],
                "max_keep": config.get("max_keep"),
            }
            active_candidates = [
                candidate
                for candidate in active_candidates
                if str(candidate["clip_id"]) in keep_ids
            ]

        return {
            "keep_ids_by_method": keep_ids_by_method,
            "summary": summary_by_method,
            "final_keep_ids": {
                str(candidate["clip_id"]) for candidate in active_candidates
            },
        }

    def build_select_row(
        self,
        row: dict[str, object],
        *,
        filter_result: dict[str, object],
        dedup_result: dict[str, object],
    ) -> dict[str, object]:
        select_ok = int(filter_result["ok"])
        select_error = str(filter_result["error"])
        select_json = dict(filter_result["json"])
        select_pass = 0
        select_reject_reason = ""

        if select_ok != 1:
            pass
        elif not bool(filter_result["passed"]):
            select_reject_reason = str(filter_result["reject_reason"])
        else:
            select_pass = 1
            clip_id = str(row["clip_id"])
            keep_ids_by_method = dedup_result["keep_ids_by_method"]
            for method, config in self.dedup_methods.items():
                group_field = f"{method}_group_id"
                group_id = str(row.get(group_field) or "")
                method_pass = clip_id in keep_ids_by_method[method]
                method_json: dict[str, object] = {
                    "pass": method_pass,
                    "field": group_field,
                    "group_id": group_id,
                    "keep_ratio": config["keep_ratio"],
                    "min_keep": config["min_keep"],
                    "max_keep": config.get("max_keep"),
                }
                group_size_field = f"{method}_group_size"
                if group_size_field in row:
                    method_json["group_size"] = int(row.get(group_size_field) or 1)
                if not method_pass:
                    select_pass = 0
                    select_reject_reason = str(config["reject_reason"])
                    method_json["reject_reason"] = select_reject_reason
                    select_json[method] = method_json
                    break
                select_json[method] = method_json

        output_row = dict(row)
        output_row.update(
            {
                "input_run_id": self.input_run_id,
                "run_id": self.run_id,
                "select_ok": select_ok,
                "select_error": select_error,
                "select_pass": int(select_pass),
                "select_reject_reason": select_reject_reason,
                "select_json": json.dumps(
                    select_json,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
        )
        return output_row
