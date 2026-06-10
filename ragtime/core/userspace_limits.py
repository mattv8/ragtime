"""Shared limits for User Space operations."""

from __future__ import annotations

USERSPACE_SQLITE_IMPORT_MIN_BYTES = 100 * 1024 * 1024
USERSPACE_SQLITE_IMPORT_DEFAULT_MAX_BYTES = USERSPACE_SQLITE_IMPORT_MIN_BYTES
USERSPACE_SQLITE_IMPORT_MAX_BYTES = 100 * 1024 * 1024 * 1024

USERSPACE_PRIMITIVE_UPLOAD_MIN_BYTES = 1024 * 1024
USERSPACE_PRIMITIVE_UPLOAD_DEFAULT_MAX_BYTES = 100 * 1024 * 1024
USERSPACE_PRIMITIVE_UPLOAD_MAX_BYTES = 1024 * 1024 * 1024

USERSPACE_PRIMITIVE_ARCHIVE_MIN_ENTRIES = 1
USERSPACE_PRIMITIVE_ARCHIVE_DEFAULT_MAX_ENTRIES = 500
USERSPACE_PRIMITIVE_ARCHIVE_MAX_ENTRIES = 10000


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
