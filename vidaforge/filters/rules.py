from __future__ import annotations

from dataclasses import dataclass

from vidaforge.common import parse_json_object


@dataclass(slots=True)
class RuleCheckResult:
    passed: bool
    reject_reason: str
    json: dict[str, object]


def _missing_reject_reason(field: str) -> str:
    return f"missing_{field}"


def resolve_field(row: dict[str, object], field: str) -> object:
    parts = field.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid field path: {field!r}")

    if parts[0] not in row:
        return None
    value: object = row[parts[0]]
    for index, part in enumerate(parts[1:], start=1):
        if value is None:
            return None
        if isinstance(value, str):
            value = parse_json_object(value, description=".".join(parts[:index]))
        if not isinstance(value, dict):
            return None
        if part not in value:
            return None
        value = value[part]
    return value


def _check_one(
    *,
    value: object,
    rule: dict[str, object],
) -> dict[str, object]:
    field = str(rule["field"])
    result: dict[str, object] = {
        "pass": True,
        "field": field,
        "value": value,
    }

    if value is None:
        result["pass"] = False
        result["reject_reason"] = str(
            rule.get("missing_reject_reason") or _missing_reject_reason(field)
        )
        return result

    if "equals" in rule:
        expected = rule["equals"]
        result["equals"] = expected
        if value != expected:
            result["pass"] = False
            result["reject_reason"] = str(rule["reject_reason"])
            return result

    if "min" in rule:
        min_value = float(rule["min"])
        result["min"] = rule["min"]
        if float(value) < min_value:
            result["pass"] = False
            result["reject_reason"] = str(
                rule.get("min_reject_reason") or rule["reject_reason"]
            )
            return result

    if "max" in rule:
        max_value = float(rule["max"])
        result["max"] = rule["max"]
        if float(value) > max_value:
            result["pass"] = False
            result["reject_reason"] = str(
                rule.get("max_reject_reason") or rule["reject_reason"]
            )
            return result

    return result


def check(
    *,
    values: dict[str, object],
    rules: dict[str, dict[str, object]],
) -> RuleCheckResult:
    result_json: dict[str, object] = {}
    reject_reasons: list[str] = []
    for rule_name, rule in rules.items():
        field = str(rule["field"])
        result = _check_one(value=values.get(field), rule=rule)
        result_json[rule_name] = result
        if not bool(result["pass"]):
            reject_reasons.append(str(result["reject_reason"]))
    return RuleCheckResult(
        passed=not reject_reasons,
        reject_reason=";".join(reject_reasons),
        json=result_json,
    )
