"""VidaForge core package."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vidaforge")
except PackageNotFoundError:  # pragma: no cover - local source tree without install
    __version__ = "0+unknown"

__all__ = ["__version__"]
