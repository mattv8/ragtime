"""Consistent, private runtime-history export staging.

Callers stream only the returned staging directory/archive; they never receive a
live repository path.  ``include_repository_key`` is deliberately explicit.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import shutil
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException

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
                size_estimate = self._portable_size(runtime_root, destination_root)
                assert isinstance(size_estimate, tuple)
                portable_size, linked_pack_size = size_estimate
                # Same-filesystem immutable packs are hardlinked, so they do
                # not consume a second staging copy.  The archive always does.
                # If linking later falls back to EXDEV, the callback below
                # re-admits the additional copy before writing it.
                required = portable_size * 2 - linked_pack_size + 64 * 1024 * 1024
                if shutil.disk_usage(destination_root).free < required:
                    raise HTTPException(status_code=507, detail="SQLite history export is blocked by insufficient disk space")
                exported_history = staging / "_sqlite_history"
                # This is an allowlist.  Never turn this into a history-root
                # copy: it contains password material, cache, and scratch.
                for name in ("restic", "operations", "activation-v1.json"):
                    source = history_root / name
                    if source.exists() and not source.is_symlink():
                        self._copy_regular_tree(
                            source,
                            exported_history / name,
                            immutable_packs=name == "restic",
                            copy_headroom=lambda size: self._require_copy_headroom(destination_root, portable_size, size),
                        )
                # Catalog copy must stay within exclusive barrier for consistency
                workspaces = runtime_root / "workspaces"
                catalogs = staging / "workspaces"
                if workspaces.is_dir():
                    for workspace in workspaces.iterdir():
                        source = workspace / "sqlite_backups"
                        if source.is_dir() and not source.is_symlink():
                            self._copy_regular_tree(source, catalogs / workspace.name / "sqlite_backups", exclude_transient=True)
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
    def _portable_size(runtime_root: Path, destination_root: Path | None = None) -> tuple[int, int] | int:
        """Estimate only portable inputs, never scratch/cache/transfers."""
        total = 0
        linked_packs = 0
        history = runtime_root / "_sqlite_history"
        for name in ("restic", "operations", "activation-v1.json"):
            source = history / name
            if source.is_file() and not source.is_symlink():
                total += source.stat().st_size
            elif source.is_dir() and not source.is_symlink():
                for entry in source.rglob("*"):
                    if entry.is_file() and not entry.is_symlink() and not is_managed_scratch(entry):
                        size = entry.stat().st_size
                        total += size
                        if destination_root is not None and entry.relative_to(source).parts[:1] == ("data",):
                            try:
                                if entry.stat().st_dev == destination_root.stat().st_dev:
                                    linked_packs += size
                            except OSError:
                                pass
        workspaces = runtime_root / "workspaces"
        if workspaces.is_dir() and not workspaces.is_symlink():
            for workspace in workspaces.iterdir():
                catalog = workspace / "sqlite_backups"
                if catalog.is_dir() and not catalog.is_symlink():
                    total += sum(
                        entry.stat().st_size
                        for entry in catalog.rglob("*")
                        if entry.is_file() and not entry.is_symlink() and not RuntimeHistoryExporter._is_transient_export_path(entry.relative_to(catalog))
                    )
        return (total, linked_packs) if destination_root is not None else total

    @staticmethod
    def _require_copy_headroom(destination_root: Path, archive_size: int, copy_size: int) -> None:
        """Re-admit capacity when a predicted hardlink must become a copy."""
        if shutil.disk_usage(destination_root).free < archive_size + copy_size + 64 * 1024 * 1024:
            raise HTTPException(status_code=507, detail="SQLite history export is blocked by insufficient disk space")

    @staticmethod
    def _is_transient_export_path(relative: Path) -> bool:
        """Never package disposable transfer/download scratch from a catalog."""
        return any(part in {"downloads", "candidates", "imports"} for part in relative.parts)

    @staticmethod
    def _copy_regular_tree(
        source: Path,
        destination: Path,
        *,
        immutable_packs: bool = False,
        exclude_transient: bool = False,
        copy_headroom: Callable[[int], None] | None = None,
    ) -> None:
        """Copy a trusted runtime subtree without preserving executable links."""
        if source.is_symlink():
            raise RuntimeError("runtime history export source is unsafe")
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination, follow_symlinks=False)
            return
        destination.mkdir(parents=True, exist_ok=True)
        for entry in source.rglob("*"):
            relative = entry.relative_to(source)
            if exclude_transient and RuntimeHistoryExporter._is_transient_export_path(relative):
                continue
            if entry.is_dir() and is_managed_scratch(entry):
                # rglob has already discovered this direct child, but pruning
                # descendants below prevents private temp contents from export.
                continue
            if any(is_managed_scratch(parent) for parent in entry.parents if parent != source.parent):
                continue
            if entry.is_symlink():
                raise RuntimeError("runtime history export source contains a link")
            target = destination / relative
            mode = entry.stat().st_mode
            if stat.S_ISDIR(mode):
                target.mkdir(parents=True, exist_ok=True)
            elif stat.S_ISREG(mode):
                target.parent.mkdir(parents=True, exist_ok=True)
                if immutable_packs and entry.relative_to(source).parts[:1] == ("data",):
                    try:
                        os.link(entry, target)
                    except OSError as exc:
                        if exc.errno != errno.EXDEV:
                            raise
                        if copy_headroom is not None:
                            copy_headroom(entry.stat().st_size)
                        shutil.copy2(entry, target, follow_symlinks=False)
                else:
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
