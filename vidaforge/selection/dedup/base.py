from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Generic, TypeVar

from .config import DedupConfigBase

ConfigT = TypeVar("ConfigT", bound=DedupConfigBase)


class DeduplicatorBase(Generic[ConfigT]):
    deduplicator_name: ClassVar[str]
    config_type: ClassVar[type[ConfigT]]

    def __init__(self, config: ConfigT, *, use_gpu_faiss: bool = False) -> None:
        self.config = config
        self.use_gpu_faiss = use_gpu_faiss

    def apply(self, row: dict[str, object]) -> None:
        _ = row
        raise NotImplementedError

    def apply_batch(self, rows: list[dict[str, object]]) -> None:
        for row in rows:
            self.apply(row)

    def save_features(self, output_path: Path) -> None:
        _ = output_path
        raise NotImplementedError

    def load_features(self, feature_root: Path) -> None:
        _ = feature_root
        raise NotImplementedError

    def unit_count(self) -> int:
        raise NotImplementedError

    def unit_ids(self) -> list[str]:
        raise NotImplementedError

    def find_duplicate_pairs(self, start: int, end: int) -> list[dict[str, object]]:
        _ = start
        _ = end
        raise NotImplementedError

    def build_rows(
        self,
        duplicate_pairs: list[dict[str, object]],
    ) -> dict[str, object]:
        _ = duplicate_pairs
        raise NotImplementedError
