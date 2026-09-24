"""Strictly read-only legacy SQLite history inventory.

This module deliberately uses ``lstat`` and direct-child paths throughout.  An
inventory is a reporting operation, so an unsafe or concurrently removed path
is reported as an issue rather than followed or recreated.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
from pathlib import Path
from typing import Any

from .conversion_ledger import is_direct_blob, valid_ledger, valid_metadata

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_OVERHEAD = 64 * 1024 * 1024


def _directory(path: Path) -> bool:
    """Return whether *path* is a real directory, never following a symlink."""
    try:
        details = path.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(details.st_mode) and not stat.S_ISLNK(details.st_mode)


def _regular_file(path: Path) -> os.stat_result | None:
    """Return direct-file metadata, or ``None`` for absent/unsafe paths."""
    try:
        details = path.lstat()
    except OSError:
        return None
    if stat.S_ISREG(details.st_mode) and not stat.S_ISLNK(details.st_mode):
        return details
    return None


def _safe_directory(root: Path, *parts: str) -> Path | None:
    """Build a real-directory chain below root without traversing symlinks."""
    current = root
    if not _directory(current):
        return None
    for part in parts:
        current = current / part
        if not _directory(current):
            return None
    return current


def _safe_file(root: Path, *parts: str) -> tuple[Path, os.stat_result] | None:
    if not parts:
        return None
    parent = _safe_directory(root, *parts[:-1])
    if parent is None:
        return None
    path = parent / parts[-1]
    details = _regular_file(path)
    return (path, details) if details is not None else None


def _tree_bytes(root: Path) -> int:
    """Count physical regular-file bytes beneath a previously safe directory."""
    seen: set[tuple[int, int]] = set()

    def walk(directory: Path) -> int:
        total = 0
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        details = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    key = (details.st_dev, details.st_ino)
                    if stat.S_ISREG(details.st_mode):
                        if key not in seen:
                            seen.add(key)
                            total += details.st_size
                    elif stat.S_ISDIR(details.st_mode) and not stat.S_ISLNK(details.st_mode):
                        total += walk(Path(entry.path))
        except OSError:
            return total
        return total

    return walk(root)


def _blank_report(workspace_id: str, issues: list[str], *, verify_integrity: bool) -> dict[str, Any]:
    return {
        "workspace_id": workspace_id,
        "ready_records": 0,
        "legacy_records": 0,
        "unique_legacy_blobs": 0,
        "legacy_bytes": 0,
        "logical_legacy_bytes": 0,
        "unique_legacy_contents": 0 if verify_integrity else None,
        "restic_records": 0,
        "ledger_pending": 0,
        "ledger_pending_bytes": 0,
        "issues": issues,
    }


def _file_sha256(path: Path) -> str | None:
    """Hash an already validated direct file without following a leaf symlink."""
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _legacy_blob(storage: Any, row: dict[str, Any]) -> str | None:
    if isinstance(storage, dict):
        return storage.get("blob") if storage.get("kind") == "legacy_file" else None
    return row.get("blob")


def _read_manifest(root: Path, workspace_id: str) -> tuple[dict[str, Any] | None, str | None]:
    safe_file = _safe_file(root, "workspaces", workspace_id, "sqlite_backups", "manifest-v1.json")
    if safe_file is None:
        return None, "history catalog directory is missing or unsafe"
    path, _ = safe_file
    try:
        with path.open(encoding="utf-8") as source:
            manifest = json.load(source)
    except (OSError, ValueError, json.JSONDecodeError):
        return None, "history catalog is unreadable"
    if not isinstance(manifest, dict) or manifest.get("workspace_id") != workspace_id or not isinstance(manifest.get("backups"), list):
        return None, "history catalog is unreadable"
    return manifest, None


def _discover_workspaces(root: Path) -> list[str]:
    workspaces = _safe_directory(root, "workspaces")
    if workspaces is None:
        return []
    discovered: list[str] = []
    try:
        with os.scandir(workspaces) as entries:
            names = sorted(entry.name for entry in entries if _ID.fullmatch(entry.name))
    except OSError:
        return []
    for workspace_id in names:
        if _safe_directory(root, "workspaces", workspace_id, "files") is None:
            continue
        history = root / "workspaces" / workspace_id / "sqlite_backups"
        manifest = history / "manifest-v1.json"
        if (_safe_directory(root, "workspaces", workspace_id, "sqlite_backups") is None and os.path.lexists(history)) or os.path.lexists(manifest):
            discovered.append(workspace_id)
    return discovered


def _inventory_workspace(
    root: Path, workspace_id: str, *, verify_integrity: bool
) -> tuple[dict[str, Any], dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    issues: list[str] = []
    if _safe_directory(root, "workspaces", workspace_id, "files") is None:
        issues.append("workspace files directory is missing or unsafe")
    manifest, manifest_issue = _read_manifest(root, workspace_id)
    if manifest_issue is not None:
        issues.append(manifest_issue)
        return _blank_report(workspace_id, sorted(set(issues)), verify_integrity=verify_integrity), {}, {}

    report = _blank_report(workspace_id, issues, verify_integrity=verify_integrity)
    images: dict[tuple[int, int], int] = {}
    pending_images: dict[tuple[int, int], int] = {}
    logical_digests: set[str] = set()
    actual_digests: dict[tuple[int, int], str | None] = {}
    legacy_blobs: set[str] = set()
    metadata_by_blob: dict[str, tuple[str, int]] = {}
    assert manifest is not None
    for row in manifest["backups"]:
        if not isinstance(row, dict) or row.get("status") != "ready":
            continue
        report["ready_records"] += 1
        storage = row.get("storage")
        if isinstance(storage, dict) and storage.get("kind") == "restic":
            report["restic_records"] += 1
            if valid_metadata(row) is None:
                issues.append("ready backup metadata is invalid")
            continue

        report["legacy_records"] += 1
        blob = _legacy_blob(storage, row)
        metadata = valid_metadata(row)
        if metadata is None:
            issues.append("ready backup metadata is invalid")
        if not is_direct_blob(blob):
            issues.append("legacy blob path is missing or unsafe")
            continue
        assert isinstance(blob, str)
        direct_file = _safe_file(
            root,
            "workspaces",
            workspace_id,
            "sqlite_backups",
            *Path(blob).parts,
        )
        if direct_file is None:
            issues.append("legacy blob path is missing or unsafe")
            continue
        _, details = direct_file
        key = (details.st_dev, details.st_ino)
        images[key] = details.st_size
        legacy_blobs.add(blob)
        if verify_integrity and key not in actual_digests:
            actual_digests[key] = _file_sha256(direct_file[0])
        actual_digest = actual_digests.get(key)
        if verify_integrity:
            if actual_digest is None:
                issues.append("legacy blob is unreadable")
            else:
                logical_digests.add(actual_digest)
        if metadata is not None:
            digest, expected_size = metadata
            report["logical_legacy_bytes"] += expected_size
            if expected_size != details.st_size:
                issues.append("legacy blob size does not match catalog metadata")
            if verify_integrity and actual_digest is not None and digest != actual_digest:
                issues.append("legacy blob digest does not match catalog metadata")
            previous = metadata_by_blob.setdefault(blob, metadata)
            if previous != metadata:
                issues.append("legacy blob metadata is inconsistent")

    ledger = _safe_directory(root, "workspaces", workspace_id, "sqlite_backups", "conversion-ledger")
    ledger_path = root / "workspaces" / workspace_id / "sqlite_backups" / "conversion-ledger"
    if os.path.lexists(ledger_path) and ledger is None:
        issues.append("conversion ledger is unsafe")
    elif ledger is not None:
        try:
            with os.scandir(ledger) as entries:
                ledger_entries = sorted(entries, key=lambda item: item.name)
        except OSError:
            ledger_entries = []
            issues.append("conversion ledger is unreadable")
        for item in ledger_entries:
            path = Path(item.path)
            try:
                if not item.is_file(follow_symlinks=False) or path.suffix != ".json":
                    raise ValueError
                with path.open(encoding="utf-8") as source:
                    record = json.load(source)
                if not valid_ledger(record, workspace_id, path.name):
                    raise ValueError
            except (OSError, ValueError, json.JSONDecodeError):
                issues.append("conversion ledger is unreadable")
                continue
            if record["stage"] == "source_removed":
                continue
            report["ledger_pending"] += 1
            if record["stage"] != "catalog_published":
                continue
            blob = record["legacy_blob"]
            if blob in legacy_blobs:
                continue
            direct_file = _safe_file(
                root,
                "workspaces",
                workspace_id,
                "sqlite_backups",
                *Path(blob).parts,
            )
            if direct_file is None:
                continue
            _, details = direct_file
            if details.st_size != record["size_bytes"]:
                issues.append("conversion ledger source size is invalid")
                continue
            pending_images[(details.st_dev, details.st_ino)] = details.st_size

    report["unique_legacy_blobs"] = len(images)
    report["legacy_bytes"] = sum(images.values())
    report["unique_legacy_contents"] = len(logical_digests) if verify_integrity else None
    report["ledger_pending_bytes"] = sum(pending_images.values())
    report["issues"] = sorted(set(issues))
    return report, images, pending_images


def inventory_legacy_history(root: Path, workspace_ids: list[str] | None = None, *, verify_integrity: bool = True) -> dict[str, Any]:
    """Enumerate validated retained catalogs without creating files or repositories.

    ``None`` discovers retained catalogs.  A supplied list, including ``[]``, is
    an explicit scope and is never widened by discovery.
    """
    root = Path(root)
    requested = None if workspace_ids is None else sorted(set(workspace_ids))
    if requested is not None and any(not isinstance(item, str) or not _ID.fullmatch(item) for item in requested):
        raise ValueError("Invalid workspace ID")
    entries = _discover_workspaces(root) if requested is None else requested
    reports: list[dict[str, Any]] = []
    all_images: dict[tuple[int, int], int] = {}
    all_pending_images: dict[tuple[int, int], int] = {}
    ingest_reserve = 0
    verification_reserve = 0
    for workspace_id in entries:
        report, images, pending_images = _inventory_workspace(root, workspace_id, verify_integrity=verify_integrity)
        reports.append(report)
        all_images.update(images)
        all_pending_images.update(pending_images)
        if images:
            ingest_reserve = max(ingest_reserve, *(4 * size + _OVERHEAD for size in images.values()))
        if pending_images:
            verification_reserve = max(verification_reserve, *(size + _OVERHEAD for size in pending_images.values()))

    repository = _safe_directory(root, "_sqlite_history", "restic")
    repository_bytes = _tree_bytes(repository) if repository is not None else 0
    try:
        free_bytes = shutil.disk_usage(root if _directory(root) else root.parent).free
    except OSError:
        free_bytes = 0
    totals = {
        "ready_records": sum(report["ready_records"] for report in reports),
        "legacy_records": sum(report["legacy_records"] for report in reports),
        "unique_legacy_blobs": len(all_images),
        "legacy_bytes": sum(all_images.values()),
        "logical_legacy_bytes": sum(report["logical_legacy_bytes"] for report in reports),
        "unique_legacy_contents": (
            None if any(report["unique_legacy_contents"] is None for report in reports) else sum(report["unique_legacy_contents"] for report in reports)
        ),
        "restic_records": sum(report["restic_records"] for report in reports),
        "ledger_pending": sum(report["ledger_pending"] for report in reports),
        "ledger_pending_bytes": sum(all_pending_images.values()),
        "repository_physical_bytes": repository_bytes,
    }
    required_headroom = max(ingest_reserve, verification_reserve)
    return {
        "version": 1,
        "active": False,
        "workspaces": reports,
        "totals": totals,
        "free_bytes": free_bytes,
        "required_headroom_bytes": required_headroom,
        # Ingestion needs the conversion protocol's 4D reserve.  Once the
        # catalog is published, only a bounded restore verification remains.
        "admission_estimate": {
            "legacy_ingest_reserve_bytes": ingest_reserve,
            "published_cleanup_verification_reserve_bytes": verification_reserve,
            "required_headroom_bytes": required_headroom,
        },
    }
