"""Runtime manager package with API, models, and orchestration service."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runtime.manager.api import app, create_app

__all__ = ["app", "create_app"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module("runtime.manager.api"), name)
    globals()[name] = value
    return value
