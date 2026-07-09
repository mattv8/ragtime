"""Core utilities module."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragtime.core import llama_cpp, lmstudio, model_limits, omlx, vision_models
    from ragtime.core.auth import get_external_origin
    from ragtime.core.logging import (
        BLUE,
        BOLD_RED,
        GREEN,
        NOTICE,
        RED,
        RESET,
        YELLOW,
        get_logger,
        get_ui_width,
        setup_logging,
    )
    from ragtime.core.security import validate_odoo_code, validate_sql_query

_AUTH_EXPORTS = {"get_external_origin"}
_LOGGING_EXPORTS = {
    "BLUE",
    "BOLD_RED",
    "GREEN",
    "NOTICE",
    "RED",
    "RESET",
    "YELLOW",
    "get_logger",
    "get_ui_width",
    "setup_logging",
}
_SECURITY_EXPORTS = {"validate_odoo_code", "validate_sql_query"}
# Keep this in sync with the TYPE_CHECKING imports and __all__ for pyright.
_SUBMODULE_EXPORTS = {"llama_cpp", "lmstudio", "model_limits", "omlx", "vision_models"}

__all__ = (
    "setup_logging",
    "get_logger",
    "get_ui_width",
    "NOTICE",
    "BLUE",
    "GREEN",
    "YELLOW",
    "RED",
    "BOLD_RED",
    "RESET",
    "get_external_origin",
    "validate_sql_query",
    "validate_odoo_code",
    "llama_cpp",
    "lmstudio",
    "model_limits",
    "omlx",
    "vision_models",
)


def __getattr__(name: str) -> Any:
    if name in _AUTH_EXPORTS:
        module = import_module("ragtime.core.auth")
    elif name in _LOGGING_EXPORTS:
        module = import_module("ragtime.core.logging")
    elif name in _SECURITY_EXPORTS:
        module = import_module("ragtime.core.security")
    elif name in _SUBMODULE_EXPORTS:
        module = import_module(f"ragtime.core.{name}")
        globals()[name] = module
        return module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(module, name)
    globals()[name] = value
    return value
