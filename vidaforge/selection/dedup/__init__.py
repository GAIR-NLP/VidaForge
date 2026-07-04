"""Stage 3 Step 3 clip deduplication."""

from .config import DedupApplyConfig, DedupConfig, DedupMatchConfig, DedupResult
from .orchestrator import DedupOrchestrator

__all__ = [
    "DedupApplyConfig",
    "DedupConfig",
    "DedupMatchConfig",
    "DedupOrchestrator",
    "DedupResult",
]
