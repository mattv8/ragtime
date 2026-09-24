"""Typed private storage references for runtime SQLite history."""

from __future__ import annotations

from dataclasses import dataclass

RESTIC_IMAGE_PATH = "/database.sqlite3"
HISTORY_RECEIPT_RETENTION_SECONDS = 30 * 24 * 60 * 60
HISTORY_EXPORT_RETENTION_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class ResticArtifact:
    """A validated immutable image stored in the runtime Restic repository."""

    repository_id: str
    snapshot_id: str
    path: str
    size_bytes: int
    sha256: str
