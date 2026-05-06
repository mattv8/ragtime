"""Shared limits for User Space operations."""

from __future__ import annotations

USERSPACE_SQLITE_IMPORT_MIN_BYTES = 100 * 1024 * 1024
USERSPACE_SQLITE_IMPORT_DEFAULT_MAX_BYTES = USERSPACE_SQLITE_IMPORT_MIN_BYTES
USERSPACE_SQLITE_IMPORT_MAX_BYTES = 100 * 1024 * 1024 * 1024


def clamp_userspace_sqlite_import_max_bytes(value: int | None) -> int:
    if value is None:
        return USERSPACE_SQLITE_IMPORT_DEFAULT_MAX_BYTES
    return max(
        USERSPACE_SQLITE_IMPORT_MIN_BYTES,
        min(USERSPACE_SQLITE_IMPORT_MAX_BYTES, int(value)),
    )


def format_userspace_sqlite_import_limit(value: int) -> str:
    if value % (1024 * 1024 * 1024) == 0:
        return f"{value // (1024 * 1024 * 1024)} GB"
    return f"{value // (1024 * 1024)} MB"
