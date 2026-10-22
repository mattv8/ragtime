"""Protected completion evidence for account-originated workspace creation."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from ragtime.core.datetimes import utc_now

MAX_CREATION_RECEIPT_BYTES = 2048
RECEIPT_FILENAME = ".development-creation-complete.json"
RECEIPT_VERSION = 1


class CompletionUnconfirmed(RuntimeError):
    """Provisioning completed, but durable completion evidence could not be recorded."""


def receipt_path(workspace_dir: Path) -> Path:
    return workspace_dir / RECEIPT_FILENAME


def read_completion_receipt(workspace_dir: Path, *, workspace_id: str, request_id: str, user_id: str) -> dict[str, Any] | None:
    """Return only a small regular-file receipt that proves this exact completion."""
    path = receipt_path(workspace_dir)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_CREATION_RECEIPT_BYTES:
            return None
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read(MAX_CREATION_RECEIPT_BYTES + 1)
    except OSError:
        return None
    finally:
        os.close(descriptor)
    if len(raw) > MAX_CREATION_RECEIPT_BYTES:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    expected = {
        "version": RECEIPT_VERSION,
        "workspace_id": workspace_id,
        "request_id": request_id,
        "user_id": user_id,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    if not isinstance(payload.get("completed_at"), str):
        return None
    return payload


def publish_completion_receipt(workspace_dir: Path, *, workspace_id: str, request_id: str, user_id: str, completed_at: datetime | None = None) -> dict[str, Any]:
    """Atomically publish the completion receipt after provisioning is complete."""
    payload = {
        "version": RECEIPT_VERSION,
        "workspace_id": workspace_id,
        "request_id": request_id,
        "user_id": user_id,
        "completed_at": (completed_at or utc_now()).isoformat(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_CREATION_RECEIPT_BYTES:
        raise ValueError("Creation receipt exceeds maximum size")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".tmp-development-creation-", dir=workspace_dir)
    temp_name = Path(temporary).name
    try:
        try:
            written = 0
            while written < len(encoded):
                written += os.write(descriptor, encoded[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(workspace_dir / temp_name, receipt_path(workspace_dir))
        directory_descriptor = os.open(workspace_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(workspace_dir / temp_name)
        except OSError:
            pass
        raise
    return payload
