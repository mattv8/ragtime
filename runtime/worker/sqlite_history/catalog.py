"""Atomic manifest-v1 history catalog shared with legacy history readers."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class CatalogError(RuntimeError):
    pass


class HistoryCatalog:
    """Only writer for v1 manifests; v2 is deliberately never synthesized."""

    def __init__(self, root: Path, workspace_id: str) -> None:
        if not workspace_id or "/" in workspace_id or "\\" in workspace_id:
            raise ValueError("Invalid workspace ID")
        self.root, self.workspace_id = root, workspace_id

    @property
    def path(self) -> Path:
        return self.root / "manifest-v1.json"

    @contextmanager
    def locked(self) -> Iterator[dict[str, Any]]:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise CatalogError("History catalog root is invalid")
        fd = os.open(self.root / ".lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise CatalogError("History catalog lock is invalid")
            fcntl.flock(fd, fcntl.LOCK_EX)
            manifest = self.load()
            yield manifest
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "version": 1,
                "workspace_id": self.workspace_id,
                "backups": [],
                "previews": {},
                "operations": {},
                "last_scheduled_at": None,
                "next_scheduled_at": None,
            }
        if self.path.is_symlink() or not self.path.is_file():
            raise CatalogError("History catalog is invalid")
        try:
            manifest = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CatalogError("History catalog is unreadable") from exc
        if not isinstance(manifest, dict) or manifest.get("workspace_id") != self.workspace_id:
            raise CatalogError("History catalog belongs to another workspace")
        for key, value in (("backups", []), ("previews", {}), ("operations", {}), ("next_scheduled_at", None)):
            manifest.setdefault(key, value)
        if not isinstance(manifest["backups"], list) or not isinstance(manifest["previews"], dict) or not isinstance(manifest["operations"], dict):
            raise CatalogError("History catalog is invalid")
        return manifest

    def save(self, manifest: dict[str, Any]) -> None:
        fd, name = tempfile.mkstemp(prefix="manifest-", suffix=".json", dir=self.root)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                json.dump(manifest, out, sort_keys=True, separators=(",", ":"))
                out.flush()
                os.fsync(out.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def storage(row: dict[str, Any]) -> dict[str, str]:
        value = row.get("storage")
        if isinstance(value, dict) and value.get("kind") in {"legacy_file", "restic"}:
            return dict(value)
        blob = row.get("blob")
        if not isinstance(blob, str):
            raise CatalogError("backup storage is invalid")
        return {"kind": "legacy_file", "blob": blob}

    def list_ready(self) -> list[dict[str, Any]]:
        with self.locked() as manifest:
            return sorted((dict(row) for row in manifest["backups"] if row.get("status") == "ready"), key=lambda row: str(row.get("created_at")), reverse=True)

    def add_ready(self, row: dict[str, Any], storage: dict[str, Any]) -> None:
        if storage.get("kind") == "legacy_file" and isinstance(storage.get("blob"), str):
            row = {**row, "blob": storage["blob"], "storage": dict(storage)}
        elif storage.get("kind") == "restic" and all(isinstance(storage.get(x), str) for x in ("repository_id", "snapshot_id", "path")):
            row = {**row, "storage": dict(storage)}
        else:
            raise CatalogError("Invalid history storage reference")
        if not row.get("id"):
            raise CatalogError("Logical backup ID is required")
        with self.locked() as manifest:
            if any(item.get("id") == row["id"] for item in manifest["backups"]):
                raise CatalogError("Logical backup ID already exists")
            manifest["backups"].append({**row, "status": "ready"})
            self.save(manifest)

    @staticmethod
    def logical_charge(manifest: dict[str, Any]) -> int:
        """Per-workspace charge; aliases count once, global dedup does not."""
        images: set[tuple[str, int]] = set()
        for row in manifest.get("backups", []):
            if row.get("status") != "ready":
                continue
            digest, size = row.get("sha256"), row.get("size_bytes")
            if isinstance(digest, str) and isinstance(size, int) and size >= 0:
                images.add((digest, size))
        return sum(size for _, size in images)
