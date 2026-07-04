from __future__ import annotations

from typing import Any

from .base import DeduplicatorBase
from .cosmos import CosmosDeduplicator
from .pdq import PDQDeduplicator


DEDUPLICATORS: dict[str, type[DeduplicatorBase[Any]]] = {
    "cosmos": CosmosDeduplicator,
    "pdq": PDQDeduplicator,
}
