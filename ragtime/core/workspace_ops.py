from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Collection

import posixpath
from fastapi import HTTPException

SQLITE_MANAGED_DIR_PREFIX = ".ragtime/db/"
SQLITE_FILE_EXTENSIONS = frozenset({".sqlite", ".sqlite3", ".db", ".db3"})
WORKSPACE_DEFAULT_GITIGNORE_PATTERNS = (
    "node_modules/",
    "dist/",
    "__pycache__/",
)
PLATFORM_MANAGED_GITIGNORE_PATTERNS = (
    ".ragtime/runtime-bootstrap.json",
    ".ragtime/.runtime-bootstrap.done",
)


def normalize_relative_file_path(file_path: str) -> str:
    normalized = file_path.replace("\\", "/").strip().lstrip("/")
    path = Path(normalized)
    if not normalized or normalized == ".." or any(part == ".." for part in path.parts):
        raise HTTPException(status_code=400, detail="Invalid file path")
    clean = "/".join(path.parts)
    if not clean or clean == ".":
        raise HTTPException(status_code=400, detail="Invalid file path")
    return clean


def enforce_sqlite_managed_path(clean_path: str) -> None:
    suffix = Path(clean_path).suffix.lower()
    if suffix in SQLITE_FILE_EXTENSIONS and not clean_path.startswith(
        SQLITE_MANAGED_DIR_PREFIX
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "SQLite persistence files must be managed under .ragtime/db/. "
                "Use paths like .ragtime/db/app.sqlite3."
            ),
        )


def normalize_runtime_file_path(
    file_path: str,
    *,
    check_reserved: bool = False,
    enforce_sqlite_managed: bool = False,
    is_reserved_path: Callable[[str], bool] | None = None,
) -> str:
    clean = normalize_relative_file_path(file_path)
    if check_reserved and clean.startswith(".ragtime/"):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if is_reserved_path is not None and is_reserved_path(clean):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if enforce_sqlite_managed:
        enforce_sqlite_managed_path(clean)
    return clean


def compute_file_hash(file_path: Path, hash_algorithm: str = "sha256") -> str:
    hasher = hashlib.new(hash_algorithm)
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def workspace_mount_target_repo_relative_path(target_path: str) -> str | None:
    raw = (target_path or "").strip()
    if not raw or "\x00" in raw:
        return None
    normalized_target = posixpath.normpath(raw)
    if not normalized_target.startswith("/workspace/"):
        return None
    relative = normalized_target[len("/workspace/") :].strip("/")
    if not relative or relative == ".":
        return None
    parts = relative.split("/")
    if any(part in ("..", ".", "") for part in parts):
        return None
    return relative


def workspace_path_matches_mount_prefix(path: str, prefix: str) -> bool:
    normalized_path = (path or "").strip().replace("\\", "/").lstrip("/")
    normalized_prefix = (prefix or "").strip().replace("\\", "/").lstrip("/")
    if not normalized_path or not normalized_prefix:
        return False
    return normalized_path == normalized_prefix or normalized_path.startswith(
        normalized_prefix + "/"
    )


def deduplicate_ancestor_paths(paths: list[str]) -> list[str]:
    if len(paths) <= 1:
        return list(paths)
    sorted_paths = sorted(paths)
    result: list[str] = []
    for path in sorted_paths:
        if result and (path == result[-1] or path.startswith(result[-1] + "/")):
            continue
        result.append(path)
    return result


def sync_scope_relative_paths(
    root: Path,
    *,
    ignored_relative_paths: Collection[str] | None = None,
) -> dict[str, Path]:
    results: dict[str, Path] = {}
    ignored = set(ignored_relative_paths or ())
    if not root.exists():
        return results
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith(".git/") or relative == ".git":
            continue
        if relative in ignored:
            continue
        results[relative] = path
    return results
