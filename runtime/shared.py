"""Shared types and constants used across runtime manager, worker, and ragtime userspace."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import HTTPException

RuntimeSessionState = Literal["starting", "running", "stopping", "stopped", "error"]

VALID_SESSION_STATES: set[str] = {"starting", "running", "stopping", "stopped", "error"}

RUNTIME_BOOTSTRAP_CONFIG_PATH = ".ragtime/runtime-bootstrap.json"
RUNTIME_BOOTSTRAP_STAMP_PATH = ".ragtime/.runtime-bootstrap.done"


def normalize_file_path(file_path: str, *, check_reserved: bool = False) -> str:
    """Normalize a workspace-relative file path and reject traversal attempts.

    Args:
        file_path: Raw file path string.
        check_reserved: When True, also reject paths under ``.ragtime/``.

    Returns:
        A clean, forward-slash-joined relative path.

    Raises:
        HTTPException: On invalid or traversal-attempting paths.
    """
    normalized = file_path.replace("\\", "/").strip().lstrip("/")
    path = Path(normalized)
    if not normalized or normalized == ".." or any(part == ".." for part in path.parts):
        raise HTTPException(status_code=400, detail="Invalid file path")
    clean = "/".join(path.parts)
    if not clean or clean == ".":
        raise HTTPException(status_code=400, detail="Invalid file path")
    if check_reserved and clean.startswith(".ragtime/"):
        raise HTTPException(status_code=400, detail="Invalid file path")
    return clean
