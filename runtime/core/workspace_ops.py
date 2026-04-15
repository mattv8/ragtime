from __future__ import annotations

import hashlib
import posixpath
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Literal

from fastapi import HTTPException

SQLITE_MANAGED_DIR_PREFIX = ".ragtime/db/"
SQLITE_FILE_EXTENSIONS = frozenset({".sqlite", ".sqlite3", ".db", ".db3"})
PLATFORM_MANAGED_GITIGNORE_PATTERNS = (
    ".ragtime/runtime-bootstrap.json",
    ".ragtime/.runtime-bootstrap.done",
)


@dataclass(frozen=True, slots=True)
class WorkspaceTreeEntry:
    path: str
    size_bytes: int
    updated_at: datetime
    entry_type: Literal["file", "directory"]


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


def list_workspace_tree_entries(
    files_dir: Path,
    *,
    include_dirs: bool = False,
) -> list[WorkspaceTreeEntry]:
    entries: list[WorkspaceTreeEntry] = []
    if not files_dir.exists():
        return entries
    for path in files_dir.rglob("*"):
        is_file = path.is_file()
        is_dir = path.is_dir()
        if not is_file and not (include_dirs and is_dir):
            continue
        stat = path.stat()
        entries.append(
            WorkspaceTreeEntry(
                path=str(path.relative_to(files_dir)),
                size_bytes=stat.st_size if is_file else 0,
                updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                entry_type="directory" if is_dir else "file",
            )
        )
    entries.sort(key=lambda item: item.path)
    return entries


def list_mount_source_tree_entries(
    mount_specs: list[dict[str, Any]],
    *,
    include_dirs: bool = False,
) -> list[WorkspaceTreeEntry]:
    entries_by_path: dict[str, WorkspaceTreeEntry] = {}
    for spec in mount_specs:
        source_local_path = str(spec.get("source_local_path", "") or "")
        target_path = str(spec.get("target_path", "") or "")
        repo_rel = workspace_mount_target_repo_relative_path(target_path)
        if not repo_rel or not source_local_path:
            continue

        source_dir = Path(source_local_path)
        if not source_dir.is_dir():
            continue

        if include_dirs:
            try:
                stat = source_dir.stat()
                entries_by_path.setdefault(
                    repo_rel,
                    WorkspaceTreeEntry(
                        path=repo_rel,
                        size_bytes=0,
                        updated_at=datetime.fromtimestamp(
                            stat.st_mtime, tz=timezone.utc
                        ),
                        entry_type="directory",
                    ),
                )
            except OSError:
                pass

        for path in source_dir.rglob("*"):
            is_file = path.is_file()
            is_dir = path.is_dir()
            if not is_file and not (include_dirs and is_dir):
                continue
            try:
                relative = path.relative_to(source_dir).as_posix()
                mapped_path = f"{repo_rel}/{relative}"
                stat = path.stat()
            except (OSError, ValueError):
                continue
            entries_by_path.setdefault(
                mapped_path,
                WorkspaceTreeEntry(
                    path=mapped_path,
                    size_bytes=stat.st_size if is_file else 0,
                    updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                    entry_type="directory" if is_dir else "file",
                ),
            )

    return sorted(entries_by_path.values(), key=lambda item: item.path)
