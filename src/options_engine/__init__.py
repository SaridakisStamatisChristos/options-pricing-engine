"""Options pricing engine public package metadata."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("options-pricing-engine")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "2.1.0.dev0"

__all__ = ["__version__"]
