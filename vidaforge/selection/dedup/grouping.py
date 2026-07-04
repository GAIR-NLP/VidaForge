from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


def merge_duplicate_pairs(
    unit_ids: list[str],
    duplicate_pairs: Iterable[tuple[str, str]],
) -> list[list[str]]:
    parent = {unit_id: unit_id for unit_id in unit_ids}

    def find(unit_id: str) -> str:
        while parent[unit_id] != unit_id:
            parent[unit_id] = parent[parent[unit_id]]
            unit_id = parent[unit_id]
        return unit_id

    for left, right in duplicate_pairs:
        if left not in parent:
            raise ValueError(f"duplicate pair contains unknown unit: {left}")
        if right not in parent:
            raise ValueError(f"duplicate pair contains unknown unit: {right}")
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    groups: dict[str, list[str]] = defaultdict(list)
    for unit_id in unit_ids:
        groups[find(unit_id)].append(unit_id)
    return sorted(
        [sorted(group) for group in groups.values()],
        key=lambda group: group[0],
    )
