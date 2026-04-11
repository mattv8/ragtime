"""Core utilities module."""

from .auth import get_external_origin
from .logging import (
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
from .security import validate_odoo_code, validate_sql_query

__all__ = [
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
]
