from __future__ import annotations

import hashlib
import os
import posixpath
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Collection, Iterator, Literal

from fastapi import HTTPException

SQLITE_MANAGED_DIR_PREFIX = ".ragtime/db/"
SQLITE_FILE_EXTENSIONS = frozenset({".sqlite", ".sqlite3", ".db", ".db3"})

# Directory names hidden by the downstream filtering layer in
# ragtime/userspace/service.py (_HIDDEN_DIRS), minus `.ragtime`: parts of
# `.ragtime` are surfaced intentionally (e.g. `.ragtime/db/migrations/`,
# `.ragtime/scripts/`, `.ragtime/runtime-entrypoint.json`). Keep this aligned
# with downstream hide rules while preserving that exception.
_HIDDEN_DIR_NAMES = frozenset({".git", "node_modules", "__pycache__", "dist"})

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
    if suffix in SQLITE_FILE_EXTENSIONS and not clean_path.startswith(SQLITE_MANAGED_DIR_PREFIX):
        raise HTTPException(
            status_code=400,
            detail=("SQLite persistence files must be managed under .ragtime/db/. Use paths like .ragtime/db/app.sqlite3."),
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
    return normalized_path == normalized_prefix or normalized_path.startswith(normalized_prefix + "/")


def _is_path_contained_under(child: Path, parent: Path) -> bool:
    """Return True if ``child`` is contained within ``parent`` after resolving symlinks.

    ``Path.resolve(strict=False)`` follows any existing symlinks but still keeps
    non-existent suffixes, so the same check works for existing files and paths
    that are about to be written.
    """
    try:
        resolved_parent = parent.resolve(strict=False)
        resolved_child = child.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return False
    return resolved_child.is_relative_to(resolved_parent)


def resolve_workspace_mount_source_path(
    mounts: list[dict[str, Any]],
    rel_path: str,
) -> tuple[Path, bool] | None:
    """Map a workspace-relative path to its source-side filesystem path.

    Iterates ``mounts`` looking for the deepest target prefix that contains
    ``rel_path`` and returns the corresponding source-side absolute path plus
    the mount's read-only flag. Returns None if no mount matches.

    Symlink containment is enforced: if the resolved target path escapes the
    mount source root (e.g. via a symlink planted inside the mount), this
    function returns None rather than exposing the out-of-bounds path.
    """
    normalized = rel_path.strip().replace("\\", "/").lstrip("/")
    if not normalized:
        return None
    candidates: list[tuple[str, Path, Path, bool]] = []
    for mount in mounts:
        repo_rel = workspace_mount_target_repo_relative_path(str(mount.get("target_path", "") or ""))
        source_local_path = str(mount.get("source_local_path", "") or "").strip()
        if not repo_rel or not source_local_path:
            continue
        if not workspace_path_matches_mount_prefix(normalized, repo_rel):
            continue
        source_root = Path(source_local_path)
        suffix = normalized[len(repo_rel) :].lstrip("/")
        source_file = source_root if not suffix else source_root.joinpath(*suffix.split("/"))
        candidates.append((repo_rel, source_file, source_root, bool(mount.get("read_only", True))))
    if not candidates:
        return None
    _, source_file, source_root, read_only = max(candidates, key=lambda item: len(item[0]))
    if not _is_path_contained_under(source_file, source_root):
        return None
    return source_file, read_only


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


def _iter_tree_entries(
    root: Path,
    include_dirs: bool,
) -> Iterator[WorkspaceTreeEntry]:
    """Yield tree entries under ``root`` while pruning hidden directories.

    This mirrors the downstream filtering in ``ragtime/userspace/service.py``:
    entries inside ``_HIDDEN_DIR_NAMES`` are skipped at any depth.  We never
    prune ``.ragtime`` because parts of it are surfaced to callers.

    Symlink semantics (workspace filesystem mounts are by design and must
    appear in the tree UI):

    - Symlinks whose resolved target stays inside ``root`` are listed and,
      for directories, traversed.  This keeps contained mount/volume layouts
      (e.g. ``current -> releases/v2``) browsable.
    - Symlinks escaping ``root`` are neither listed nor followed, matching the
      access-time containment in ``resolve_workspace_mount_source_path`` and
      the repo symlink-safety guidance (never wander into host filesystem
      areas the walk root does not own).
    - Loops are broken with per-branch ancestor-chain cycle detection using
      ``(st_dev, st_ino)`` identities: a link back to an ancestor is shown as
      a directory entry but never descended.  Sibling links to the same
      target do not shadow the real directory.
    """
    try:
        root_resolved = root.resolve()
        root_stat = root.stat()
    except OSError:
        return

    # Stack of (directory path, relative posix path, ancestor identities).
    stack: list[tuple[Path, str, frozenset[tuple[int, int]]]] = [
        (root, "", frozenset({(root_stat.st_dev, root_stat.st_ino)})),
    ]

    while stack:
        dirpath, rel_prefix, ancestor_ids = stack.pop()
        try:
            with os.scandir(dirpath) as scan:
                children = sorted(scan, key=lambda item: item.name)
        except OSError:
            continue

        for child in children:
            rel_path = f"{rel_prefix}/{child.name}" if rel_prefix else child.name
            try:
                is_symlink = child.is_symlink()
                is_dir = child.is_dir(follow_symlinks=True)
            except OSError:
                continue

            if is_symlink:
                # Dangling links fail resolution; escaping links are hidden to
                # match access-time containment.
                try:
                    resolved_target = Path(child.path).resolve(strict=True)
                except (OSError, RuntimeError):
                    continue
                if not resolved_target.is_relative_to(root_resolved):
                    continue

            if is_dir:
                if child.name in _HIDDEN_DIR_NAMES:
                    continue
                try:
                    dir_stat = os.stat(child.path)
                except OSError:
                    continue
                if include_dirs:
                    yield WorkspaceTreeEntry(
                        path=rel_path,
                        size_bytes=0,
                        updated_at=datetime.fromtimestamp(dir_stat.st_mtime, tz=timezone.utc),
                        entry_type="directory",
                    )
                dir_id = (dir_stat.st_dev, dir_stat.st_ino)
                if dir_id in ancestor_ids:
                    # Symlink loop back to an ancestor: entry stays visible,
                    # but we never descend.
                    continue
                stack.append((Path(child.path), rel_path, ancestor_ids | {dir_id}))
                continue

            try:
                file_stat = child.stat(follow_symlinks=True)
            except OSError:
                continue
            yield WorkspaceTreeEntry(
                path=rel_path,
                size_bytes=file_stat.st_size,
                updated_at=datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc),
                entry_type="file",
            )


def list_workspace_tree_entries(
    files_dir: Path,
    *,
    include_dirs: bool = False,
) -> list[WorkspaceTreeEntry]:
    if not files_dir.exists():
        return []
    entries = list(_iter_tree_entries(files_dir, include_dirs))
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
                        updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                        entry_type="directory",
                    ),
                )
            except OSError:
                pass

        for entry in _iter_tree_entries(source_dir, include_dirs):
            relative = Path(entry.path).as_posix()
            mapped_path = f"{repo_rel}/{relative}"
            entries_by_path.setdefault(
                mapped_path,
                WorkspaceTreeEntry(
                    path=mapped_path,
                    size_bytes=entry.size_bytes,
                    updated_at=entry.updated_at,
                    entry_type=entry.entry_type,
                ),
            )

    return sorted(entries_by_path.values(), key=lambda item: item.path)
