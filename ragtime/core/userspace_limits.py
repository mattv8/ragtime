"""Shared limits for User Space operations."""

from __future__ import annotations

import math
from collections.abc import Mapping

USERSPACE_SQLITE_IMPORT_MIN_BYTES = 100 * 1024 * 1024
USERSPACE_SQLITE_IMPORT_DEFAULT_MAX_BYTES = USERSPACE_SQLITE_IMPORT_MIN_BYTES
USERSPACE_SQLITE_IMPORT_MAX_BYTES = 100 * 1024 * 1024 * 1024

USERSPACE_PRIMITIVE_UPLOAD_MIN_BYTES = 1024 * 1024
USERSPACE_PRIMITIVE_UPLOAD_DEFAULT_MAX_BYTES = 100 * 1024 * 1024
USERSPACE_PRIMITIVE_UPLOAD_MAX_BYTES = 1024 * 1024 * 1024

USERSPACE_PRIMITIVE_ARCHIVE_MIN_ENTRIES = 1
USERSPACE_PRIMITIVE_ARCHIVE_DEFAULT_MAX_ENTRIES = 500
USERSPACE_PRIMITIVE_ARCHIVE_MAX_ENTRIES = 10000

USERSPACE_EXEC_TIMEOUT_HARD_CAP_SECONDS = 3600
USERSPACE_EXEC_TIMEOUT_MAX_FLOOR_SECONDS = 30
USERSPACE_EXEC_TIMEOUT_DEFAULT_SECONDS = 120
USERSPACE_EXEC_TIMEOUT_MAX_SECONDS = 600


def _userspace_timeout_setting(settings: object, snake_name: str, camel_name: str) -> object:
    if isinstance(settings, Mapping):
        return settings.get(snake_name, settings.get(camel_name))
    return getattr(settings, snake_name, getattr(settings, camel_name, None))


def _valid_settings_integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def resolve_userspace_exec_timeout_bounds(settings: object) -> tuple[int, int]:
    """Return safe User Space command timeout bounds from settings-like input."""
    raw_maximum = _valid_settings_integer(
        _userspace_timeout_setting(
            settings,
            "userspace_exec_timeout_max_seconds",
            "userspaceExecTimeoutMaxSeconds",
        )
    )
    maximum = USERSPACE_EXEC_TIMEOUT_MAX_SECONDS if raw_maximum is None else raw_maximum
    maximum = max(USERSPACE_EXEC_TIMEOUT_MAX_FLOOR_SECONDS, min(USERSPACE_EXEC_TIMEOUT_HARD_CAP_SECONDS, maximum))

    raw_default = _valid_settings_integer(
        _userspace_timeout_setting(
            settings,
            "userspace_exec_timeout_default_seconds",
            "userspaceExecTimeoutDefaultSeconds",
        )
    )
    default = USERSPACE_EXEC_TIMEOUT_DEFAULT_SECONDS if raw_default is None else raw_default
    default = max(1, min(maximum, default))
    return default, maximum


def _coerce_requested_timeout(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("timeout_seconds must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return int(value)
        raise ValueError("timeout_seconds must be an integer")
    if isinstance(value, str):
        normalized = value.strip()
        digits = normalized[1:] if normalized[:1] in {"+", "-"} else normalized
        if digits and digits.isdecimal():
            try:
                return int(normalized)
            except (OverflowError, ValueError):
                raise ValueError("timeout_seconds must be an integer") from None
    raise ValueError("timeout_seconds must be an integer")


def resolve_userspace_exec_timeout(settings: object, requested_timeout: object = None) -> int:
    """Resolve an optional command timeout without silently widening explicit input."""
    default, maximum = resolve_userspace_exec_timeout_bounds(settings)
    if requested_timeout is None:
        return default
    timeout = _coerce_requested_timeout(requested_timeout)
    if not 1 <= timeout <= maximum:
        raise ValueError(f"timeout_seconds must be between 1 and {maximum}")
    return timeout


# Index archive extraction limits
ARCHIVE_MAX_TOTAL_SIZE_MIN_BYTES = 100 * 1024 * 1024  # 100 MB
ARCHIVE_MAX_TOTAL_SIZE_DEFAULT_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
ARCHIVE_MAX_TOTAL_SIZE_MAX_BYTES = 500 * 1024 * 1024 * 1024  # 500 GB

ARCHIVE_MAX_FILE_COUNT_MIN = 100
ARCHIVE_MAX_FILE_COUNT_DEFAULT = 100000
ARCHIVE_MAX_FILE_COUNT_MAX = 500000


def clamp_userspace_sqlite_import_max_bytes(value: int | None) -> int:
    if value is None:
        return USERSPACE_SQLITE_IMPORT_DEFAULT_MAX_BYTES
    return max(
        USERSPACE_SQLITE_IMPORT_MIN_BYTES,
        min(USERSPACE_SQLITE_IMPORT_MAX_BYTES, int(value)),
    )


def clamp_userspace_primitive_upload_max_bytes(value: int | None) -> int:
    if value is None:
        return USERSPACE_PRIMITIVE_UPLOAD_DEFAULT_MAX_BYTES
    return max(
        USERSPACE_PRIMITIVE_UPLOAD_MIN_BYTES,
        min(USERSPACE_PRIMITIVE_UPLOAD_MAX_BYTES, int(value)),
    )


def clamp_userspace_primitive_archive_max_entries(value: int | None) -> int:
    if value is None:
        return USERSPACE_PRIMITIVE_ARCHIVE_DEFAULT_MAX_ENTRIES
    return max(
        USERSPACE_PRIMITIVE_ARCHIVE_MIN_ENTRIES,
        min(USERSPACE_PRIMITIVE_ARCHIVE_MAX_ENTRIES, int(value)),
    )


def format_userspace_sqlite_import_limit(value: int) -> str:
    if value % (1024 * 1024 * 1024) == 0:
        return f"{value // (1024 * 1024 * 1024)} GB"
    return f"{value // (1024 * 1024)} MB"


def clamp_archive_max_total_size_bytes(value: int | None) -> int:
    """Clamp archive extraction size to valid range."""
    if value is None:
        return ARCHIVE_MAX_TOTAL_SIZE_DEFAULT_BYTES
    return max(
        ARCHIVE_MAX_TOTAL_SIZE_MIN_BYTES,
        min(ARCHIVE_MAX_TOTAL_SIZE_MAX_BYTES, int(value)),
    )


def clamp_archive_max_file_count(value: int | None) -> int:
    """Clamp archive extraction file count to valid range."""
    if value is None:
        return ARCHIVE_MAX_FILE_COUNT_DEFAULT
    return max(
        ARCHIVE_MAX_FILE_COUNT_MIN,
        min(ARCHIVE_MAX_FILE_COUNT_MAX, int(value)),
    )
