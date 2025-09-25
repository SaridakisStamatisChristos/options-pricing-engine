from __future__ import annotations

import sys
from importlib import util
from pathlib import Path
from typing import Any


_MODULE_NAME = "options_engine.api._minimal_routes"
_module = sys.modules.get(_MODULE_NAME)
if _module is None:
    module_path = Path(__file__).resolve().parent.parent / "routes.py"
    spec = util.spec_from_file_location(_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("Unable to load minimal routes module")
    _module = util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = _module
    spec.loader.exec_module(_module)

BUILD_ID = _module.__dict__.get("_DEFAULT_BUILD_ID", "")


def __getattr__(name: str) -> Any:
    if name in {"register_routes", "BUILD_ID"}:
        return getattr(_module, name)
    raise AttributeError(name)


def __setattr__(name: str, value: Any) -> None:  # pragma: no cover - setters only used in tests
    if name == "BUILD_ID":
        setattr(_module, name, value)
    else:
        raise AttributeError(name)


__all__ = ["register_routes", "BUILD_ID"]
