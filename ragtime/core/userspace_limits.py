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
