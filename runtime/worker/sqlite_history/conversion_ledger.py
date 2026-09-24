"""Pure validation helpers for durable legacy-conversion ledgers."""

from __future__ import annotations

import hashlib
from typing import Any

LEDGER_STAGES = frozenset({"prepared", "repository_verified", "catalog_published", "source_removed"})


def operation_id(workspace_id: str, blob: str, digest: str) -> str:
    """Return the stable Restic operation identity for one legacy blob."""
    identity = hashlib.sha256(f"{workspace_id}\0{blob}\0{digest}".encode()).hexdigest()
    return f"legacy-v1-{identity}"


def ledger_filename(operation: str) -> str:
    return hashlib.sha256(operation.encode()).hexdigest() + ".json"


def is_direct_blob(blob: Any) -> bool:
    return isinstance(blob, str) and blob.startswith("blobs/") and "/" not in blob[6:] and blob[6:] not in {"", ".", ".."}


def valid_metadata(record: dict[str, Any]) -> tuple[str, int] | None:
    digest, size_bytes = record.get("sha256"), record.get("size_bytes")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
    ):
        return None
    return digest, size_bytes


def valid_ledger(record: Any, workspace_id: str, filename: str) -> bool:
    if not isinstance(record, dict) or record.get("version") != 1:
        return False
    if record.get("workspace_id") != workspace_id or record.get("stage") not in LEDGER_STAGES:
        return False
    if not is_direct_blob(record.get("legacy_blob")) or valid_metadata(record) is None:
        return False
    digest, _ = valid_metadata(record) or ("", 0)
    operation = record.get("operation_id")
    backup_ids = record.get("backup_ids")
    return (
        operation == operation_id(workspace_id, record["legacy_blob"], digest)
        and filename == ledger_filename(str(operation))
        and isinstance(backup_ids, list)
        and all(isinstance(backup_id, str) and backup_id for backup_id in backup_ids)
    )
