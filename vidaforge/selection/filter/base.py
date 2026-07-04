from __future__ import annotations

from typing import ClassVar, Generic, TypeVar

from .config import FilterConfigBase


ConfigT = TypeVar("ConfigT", bound=FilterConfigBase)


class FilterBase(Generic[ConfigT]):
    filter_name: ClassVar[str]
    config_type: ClassVar[type[ConfigT]]

    def __init__(self, config: ConfigT) -> None:
        _ = config

    def apply(self, row: dict[str, object]) -> tuple[float, dict[str, object]]:
        raise NotImplementedError

    def apply_batch(
        self,
        rows: list[dict[str, object]],
    ) -> list[tuple[float, dict[str, object]]]:
        return [self.apply(row) for row in rows]
