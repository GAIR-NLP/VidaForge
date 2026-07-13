"""V-JEPA2 training-side adapters and tools."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

_EXPORTS = {
    "DistributedWeightedSampler": (".dataset", "DistributedWeightedSampler"),
    "TorchCodecVideoDataset": (".dataset", "TorchCodecVideoDataset"),
    "compute_vjepa_indices": (".sampling", "compute_vjepa_indices"),
    "make_videodataset": (".dataset", "make_videodataset"),
    "read_vjepa_manifest": (".dataset", "read_vjepa_manifest"),
}

__all__ = [
    "DistributedWeightedSampler",
    "TorchCodecVideoDataset",
    "compute_vjepa_indices",
    "make_videodataset",
    "read_vjepa_manifest",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


if TYPE_CHECKING:
    from .dataset import (
        DistributedWeightedSampler,
        TorchCodecVideoDataset,
        make_videodataset,
        read_vjepa_manifest,
    )
    from .sampling import compute_vjepa_indices
