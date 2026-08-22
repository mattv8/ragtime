"""
Shared utilities for computing file/directory digest fingerprints.

Provides consistent hashing for file metadata to detect changes across
SSH mounts and cloud storage backends.
"""

from __future__ import annotations

import hashlib


def build_tree_state_fingerprint(
    remote_files: dict[str, tuple[int, int]],
    remote_dirs: set[str],
    local_files: dict[str, tuple[int, int]],
    local_dirs: set[str],
) -> str:
    """
    Build a SHA-256 digest fingerprint of file/directory tree metadata.

    Combines file and directory state from both remote and local sources
    into a single deterministic hash to detect changes.

    Args:
        remote_files: dict mapping file path to (size_bytes, mtime_seconds)
        remote_dirs: set of remote directory relative paths
        local_files: dict mapping file path to (size_bytes, mtime_seconds)
        local_dirs: set of local directory relative paths

    Returns:
        Hex digest of SHA-256 hash combining all metadata
    """
    digest = hashlib.sha256()
    for prefix, files, directories in (
        ("remote", remote_files, remote_dirs),
        ("local", local_files, local_dirs),
    ):
        digest.update(prefix.encode("ascii"))
        digest.update(b"\0")
        for directory in sorted(directories):
            digest.update(b"d\0")
            digest.update(directory.encode("utf-8", errors="ignore"))
            digest.update(b"\0")
        for path, (size_bytes, mtime_seconds) in sorted(files.items()):
            digest.update(b"f\0")
            digest.update(path.encode("utf-8", errors="ignore"))
            digest.update(b"\0")
            digest.update(f"{size_bytes}:{mtime_seconds}".encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()
