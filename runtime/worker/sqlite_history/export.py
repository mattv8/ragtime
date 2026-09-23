"""Consistent, private runtime-history export staging.

Callers stream only the returned staging directory/archive; they never receive a
live repository path.  ``include_repository_key`` is deliberately explicit.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any

from runtime.core.sqlite_history_scratch import is_managed_scratch

from .storage import repository_gate


class RuntimeHistoryExporter:
    def __init__(self, service: Any) -> None:
        self.service = service

    async def stage_server_export(self, destination_root: Path, *, include_repository_key: bool) -> Path:
        """Stage only the portable runtime-history state, off the event loop.

        In particular, cache and scratch are not backup inputs and the repository
        password is copied only when the caller selected an encrypted envelope.
        The flock is acquired and released by the same worker thread.
        """
        return await asyncio.to_thread(self._stage_server_export_sync, destination_root, include_repository_key)

    def _stage_server_export_sync(self, destination_root: Path, include_repository_key: bool) -> Path:
        runtime_root = self.service._runtime.root
        destination_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="sqlite-history-export-", dir=destination_root))
        repository = self.service.repository
        try:
            with repository_gate(runtime_root, exclusive=True):
                history_root = runtime_root / "_sqlite_history"
                if os.path.lexists(history_root / "transfers" / "import-journal.json"):
                    raise RuntimeError("runtime history import recovery is required")
                if repository is None:
                    raise RuntimeError("runtime history repository is unavailable")
                repository_id = repository._initialize(threading.Event(), ())
                exported_history = staging / "_sqlite_history"
                # This is an allowlist.  Never turn this into a history-root
                # copy: it contains password material, cache, and scratch.
                for name in ("restic", "operations", "activation-v1.json"):
                    source = history_root / name
                    if source.exists() and not source.is_symlink():
                        self._copy_regular_tree(source, exported_history / name)
                # Catalog copy must stay within exclusive barrier for consistency
                workspaces = runtime_root / "workspaces"
                catalogs = staging / "workspaces"
                if workspaces.is_dir():
                    for workspace in workspaces.iterdir():
                        source = workspace / "sqlite_backups"
                        if source.is_dir() and not source.is_symlink():
                            self._copy_regular_tree(source, catalogs / workspace.name / "sqlite_backups")
                password = getattr(repository, "_password_path", None)
                if include_repository_key:
                    if not isinstance(password, Path) or password.is_symlink() or not password.is_file():
                        raise RuntimeError("runtime history repository key is unavailable")
                    key_dir = staging / "secrets"
                    key_dir.mkdir(mode=0o700)
                    shutil.copy2(password, key_dir / "repository-password")
            metadata: dict[str, object] = {"version": 1, "includes_repository_key": include_repository_key, "repository_id": repository_id}
            (staging / "history-export.json").write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
            return staging
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    @staticmethod
    def _copy_regular_tree(source: Path, destination: Path) -> None:
        """Copy a trusted runtime subtree without preserving executable links."""
        if source.is_symlink():
            raise RuntimeError("runtime history export source is unsafe")
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination, follow_symlinks=False)
            return
        destination.mkdir(parents=True, exist_ok=True)
        for entry in source.rglob("*"):
            if entry.is_dir() and is_managed_scratch(entry):
                # rglob has already discovered this direct child, but pruning
                # descendants below prevents private temp contents from export.
                continue
            if any(is_managed_scratch(parent) for parent in entry.parents if parent != source.parent):
                continue
            if entry.is_symlink():
                raise RuntimeError("runtime history export source contains a link")
            target = destination / entry.relative_to(source)
            mode = entry.stat().st_mode
            if stat.S_ISDIR(mode):
                target.mkdir(parents=True, exist_ok=True)
            elif stat.S_ISREG(mode):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(entry, target, follow_symlinks=False)
            else:
                raise RuntimeError("runtime history export source contains a special file")

    async def stage_workspace_export(self, workspace_id: str, destination: Path) -> Path:
        """Materialize workspace images in legacy-compatible portable layout."""
        destination.mkdir(parents=True, exist_ok=True)
        root = self.service._root(workspace_id)
        manifest = await self.service.list_backups(workspace_id)
        blobs = destination / "blobs"
        blobs.mkdir()
        exported: list[dict[str, Any]] = []
        for row in manifest:
            if row.get("status") != "ready":
                continue
            image = blobs / f"{row['id']}.sqlite3"
            await self.service.download_to_path(workspace_id, str(row["id"]), image)
            copy = dict(row)
            copy.pop("storage", None)
            copy["blob"] = f"blobs/{image.name}"
            exported.append(copy)
        (destination / "manifest-v1.json").write_text(
            json.dumps({"version": 1, "workspace_id": workspace_id, "backups": exported, "previews": {}, "operations": {}}, sort_keys=True), encoding="utf-8"
        )
        return destination
