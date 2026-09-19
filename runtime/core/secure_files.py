"""Descriptor-relative file operations for untrusted workspace paths.

Callers must supply a lexical, workspace-relative path.  This module never
resolves a path and then reopens it by name: every component is opened from an
already trusted directory descriptor with ``O_NOFOLLOW``.
"""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class SecureFileError(Exception):
    """An unsafe or unsupported file operation below a trusted root."""


def _flags(*, directory: bool = False) -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW") if directory else ("O_NOFOLLOW",)
    if any(not hasattr(os, name) for name in required):
        raise SecureFileError("Secure descriptor-relative file operations are unavailable")
    return os.O_RDONLY | (os.O_DIRECTORY if directory else os.O_NONBLOCK) | os.O_NOFOLLOW


def _components(relative_path: str) -> list[str]:
    parts = relative_path.split("/")
    if not parts or any(not part or part in {".", ".."} for part in parts):
        raise SecureFileError("Invalid workspace file path")
    return parts


def _open_root(root: Path) -> int:
    try:
        return os.open(os.fspath(root), _flags(directory=True))
    except (OSError, ValueError) as exc:
        raise SecureFileError("Workspace file root is unavailable or unsafe") from exc


def _open_parent(root_fd: int, parts: list[str], *, create: bool) -> tuple[int, str]:
    current_fd = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            try:
                next_fd = os.open(component, _flags(directory=True), dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o755, dir_fd=current_fd)
                except FileExistsError:
                    pass
                next_fd = os.open(component, _flags(directory=True), dir_fd=current_fd)
            except OSError as exc:
                raise SecureFileError("Workspace file path contains an unsafe directory") from exc
            os.close(current_fd)
            current_fd = next_fd
        return current_fd, parts[-1]
    except Exception:
        os.close(current_fd)
        raise


def _require_regular(fd: int) -> None:
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        raise SecureFileError("Workspace file target is not a regular file")


def read_text(root: Path, relative_path: str, *, encoding: str = "utf-8") -> str | None:
    """Read a regular file, returning None for absent or unsafe targets.

    Reads intentionally do not distinguish unsafe/missing targets, preserving
    callers' non-disclosure response contract.
    """
    try:
        parts = _components(relative_path)
        root_fd = _open_root(root)
        try:
            parent_fd, leaf = _open_parent(root_fd, parts, create=False)
            try:
                fd = os.open(leaf, _flags(), dir_fd=parent_fd)
                try:
                    _require_regular(fd)
                    chunks: list[bytes] = []
                    while chunk := os.read(fd, 1024 * 1024):
                        chunks.append(chunk)
                    return b"".join(chunks).decode(encoding)
                finally:
                    os.close(fd)
            finally:
                os.close(parent_fd)
        finally:
            os.close(root_fd)
    except (OSError, SecureFileError, UnicodeError, ValueError):
        return None


def write_text(root: Path, relative_path: str, content: str, *, encoding: str = "utf-8") -> None:
    """Create or replace a regular file without following any symlink."""
    payload = content.encode(encoding)
    parts = _components(relative_path)
    root_fd = _open_root(root)
    try:
        parent_fd, leaf = _open_parent(root_fd, parts, create=True)
        try:
            try:
                fd = os.open(
                    leaf,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK,
                    0o644,
                    dir_fd=parent_fd,
                )
            except FileExistsError:
                try:
                    fd = os.open(leaf, os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
                except OSError as exc:
                    raise SecureFileError("Workspace file target is unsafe") from exc
            try:
                # Validate before destructive truncation; an attacker cannot
                # turn a symlink/special file into a write target here.
                _require_regular(fd)
                os.ftruncate(fd, 0)
                offset = 0
                while offset < len(payload):
                    offset += os.write(fd, payload[offset:])
            finally:
                os.close(fd)
        finally:
            os.close(parent_fd)
    finally:
        os.close(root_fd)


def delete_file(root: Path, relative_path: str) -> bool:
    """Delete a regular file or final symlink; missing files are a no-op."""
    parts = _components(relative_path)
    root_fd = _open_root(root)
    try:
        try:
            parent_fd, leaf = _open_parent(root_fd, parts, create=False)
        except FileNotFoundError:
            return False
        try:
            try:
                mode = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False).st_mode
            except FileNotFoundError:
                return False
            if not (stat.S_ISREG(mode) or stat.S_ISLNK(mode)):
                return False
            os.unlink(leaf, dir_fd=parent_fd)
            return True
        finally:
            os.close(parent_fd)
    finally:
        os.close(root_fd)


@contextmanager
def open_directory(root: Path, relative_path: str, *, create: bool = False) -> Iterator[int]:
    """Yield a pinned directory below ``root`` without following components.

    The returned descriptor, rather than a subsequently reopened pathname, is
    the capability callers must pass to untrusted-file helpers/children.
    """
    parts = _components(relative_path)
    root_fd = _open_root(root)
    try:
        parent_fd, leaf = _open_parent(root_fd, [*parts, ".placeholder"], create=create)
        try:
            # _open_parent opens every actual directory component above its
            # final leaf; using a harmless synthetic leaf lets it return the
            # requested directory descriptor without a path re-open.
            if leaf != ".placeholder":  # defensive; _components guarantees it.
                raise SecureFileError("Invalid workspace directory path")
            yield parent_fd
        finally:
            os.close(parent_fd)
    finally:
        os.close(root_fd)


def ensure_directory(root: Path, relative_path: str) -> None:
    """Create a directory chain through no-follow descriptors if needed."""
    with open_directory(root, relative_path, create=True):
        pass


def publish_regular_file(
    source_root: Path,
    source_relative: str,
    target_root: Path,
    target_relative: str,
) -> None:
    """Atomically publish a regular source file using pinned parent FDs.

    Both paths are opened component-by-component with ``O_NOFOLLOW``.  This
    deliberately avoids the unsafe ``resolve()``-then-``replace()`` pattern
    when a workspace directory can be replaced between validation and publish.
    """
    source_parts = _components(source_relative)
    target_parts = _components(target_relative)
    source_root_fd = _open_root(source_root)
    target_root_fd = _open_root(target_root)
    try:
        source_parent_fd, source_leaf = _open_parent(source_root_fd, source_parts, create=False)
        try:
            source_fd = os.open(source_leaf, _flags(), dir_fd=source_parent_fd)
            try:
                _require_regular(source_fd)
                target_parent_fd, target_leaf = _open_parent(target_root_fd, target_parts, create=False)
                try:
                    temporary = f".{target_leaf}.{uuid.uuid4().hex}.tmp"
                    target_fd = os.open(
                        temporary,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=target_parent_fd,
                    )
                    try:
                        while block := os.read(source_fd, 1024 * 1024):
                            offset = 0
                            while offset < len(block):
                                offset += os.write(target_fd, block[offset:])
                        os.fsync(target_fd)
                    except Exception:
                        os.unlink(temporary, dir_fd=target_parent_fd)
                        raise
                    finally:
                        os.close(target_fd)
                    os.replace(temporary, target_leaf, src_dir_fd=target_parent_fd, dst_dir_fd=target_parent_fd)
                    os.fsync(target_parent_fd)
                finally:
                    os.close(target_parent_fd)
            finally:
                os.close(source_fd)
        finally:
            os.close(source_parent_fd)
    finally:
        os.close(source_root_fd)
        os.close(target_root_fd)


def sha256_regular_file(root: Path, relative_path: str) -> str:
    """Hash a regular file through a no-follow descriptor."""
    parts = _components(relative_path)
    root_fd = _open_root(root)
    try:
        parent_fd, leaf = _open_parent(root_fd, parts, create=False)
        try:
            fd = os.open(leaf, _flags(), dir_fd=parent_fd)
            try:
                _require_regular(fd)
                digest = hashlib.sha256()
                while block := os.read(fd, 1024 * 1024):
                    digest.update(block)
                return digest.hexdigest()
            finally:
                os.close(fd)
        finally:
            os.close(parent_fd)
    finally:
        os.close(root_fd)


def stat_regular_file(root: Path, relative_path: str) -> os.stat_result:
    """Return metadata for a regular file through a no-follow descriptor."""
    parts = _components(relative_path)
    root_fd = _open_root(root)
    try:
        parent_fd, leaf = _open_parent(root_fd, parts, create=False)
        try:
            fd = os.open(leaf, _flags(), dir_fd=parent_fd)
            try:
                _require_regular(fd)
                return os.fstat(fd)
            finally:
                os.close(fd)
        finally:
            os.close(parent_fd)
    finally:
        os.close(root_fd)
