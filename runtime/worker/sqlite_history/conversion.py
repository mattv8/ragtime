"""Crash-resumable conversion of legacy SQLite-history blobs to Restic."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import HTTPException

from .models import RESTIC_IMAGE_PATH, ResticArtifact


class LegacyHistoryConverter:
    """Publish one verified Restic image before removing its legacy source."""

    _HEADROOM = 64 * 1024 * 1024

    def __init__(self, service: Any, workspace_id: str, *, pass_fds: tuple[int, ...]) -> None:
        self.service = service
        self.workspace_id = workspace_id
        self.root = service._root(workspace_id)
        self.pass_fds = pass_fds

    @staticmethod
    def _operation_id(workspace_id: str, blob: str, digest: str) -> str:
        # This deliberately does not expose a user controlled path in a tag/name.
        identity = hashlib.sha256(f"{workspace_id}\0{blob}\0{digest}".encode()).hexdigest()
        return f"legacy-v1-{identity}"

    def _ledger_path(self, operation_id: str) -> Path:
        directory = self.root / "conversion-ledger"
        self.service._require_legacy_directory(directory)
        return directory / f"{hashlib.sha256(operation_id.encode()).hexdigest()}.json"

    def _read_ledger(self, operation_id: str) -> dict[str, Any] | None:
        path = self._ledger_path(operation_id)
        try:
            details = path.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode):
            raise HTTPException(status_code=409, detail="SQLite history conversion journal is invalid")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=409, detail="SQLite history conversion journal is unreadable") from exc
        if not isinstance(value, dict) or value.get("operation_id") != operation_id:
            raise HTTPException(status_code=409, detail="SQLite history conversion journal is invalid")
        self._validate_ledger(value, path)
        return value

    def _unfinished_ledger(self) -> dict[str, Any] | None:
        directory = self.root / "conversion-ledger"
        if not directory.exists():
            return None
        self.service._require_legacy_directory(directory)
        for path in sorted(directory.iterdir()):
            if path.suffix != ".json":
                continue
            if path.is_symlink() or not path.is_file():
                raise HTTPException(status_code=409, detail="SQLite history conversion journal is invalid")
            try:
                ledger = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=409, detail="SQLite history conversion journal is unreadable") from exc
            if not isinstance(ledger, dict):
                raise HTTPException(status_code=409, detail="SQLite history conversion journal is invalid")
            self._validate_ledger(ledger, path)
            if ledger.get("stage") != "source_removed":
                return ledger
        return None

    def _validate_ledger(self, ledger: dict[str, Any], path: Path) -> None:
        blob, digest, operation_id = ledger.get("legacy_blob"), ledger.get("sha256"), ledger.get("operation_id")
        if (
            ledger.get("version") != 1
            or ledger.get("workspace_id") != self.workspace_id
            or not isinstance(blob, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or not isinstance(ledger.get("size_bytes"), int)
            or ledger["size_bytes"] < 0
            or ledger.get("stage") not in {"prepared", "repository_verified", "catalog_published", "source_removed"}
            or operation_id != self._operation_id(self.workspace_id, blob, digest)
            or path.name != hashlib.sha256(str(operation_id).encode()).hexdigest() + ".json"
            or not isinstance(ledger.get("backup_ids"), list)
            or not all(isinstance(value, str) and value for value in ledger["backup_ids"])
        ):
            raise HTTPException(status_code=409, detail="SQLite history conversion journal is invalid")
        # This confirms direct-child format and no-follow safety before any unlink.
        self.service._protected_path(self.root, blob, "blobs")

    def _save_ledger(self, ledger: dict[str, Any]) -> None:
        path = self._ledger_path(str(ledger["operation_id"]))
        fd, name = tempfile.mkstemp(prefix="conversion-", suffix=".json", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(ledger, output, sort_keys=True, separators=(",", ":"))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _storage(artifact: ResticArtifact) -> dict[str, str]:
        return {"kind": "restic", "repository_id": artifact.repository_id, "snapshot_id": artifact.snapshot_id, "path": artifact.path}

    @staticmethod
    def _artifact(ledger: dict[str, Any]) -> ResticArtifact:
        storage = ledger.get("storage")
        if not isinstance(storage, dict):
            raise HTTPException(status_code=409, detail="SQLite history conversion journal is invalid")
        try:
            return ResticArtifact(
                str(storage["repository_id"]), str(storage["snapshot_id"]), str(storage["path"]), int(ledger["size_bytes"]), str(ledger["sha256"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail="SQLite history conversion journal is invalid") from exc

    def _rows(self, manifest: dict[str, Any], blob: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in manifest["backups"]:
            if row.get("status") != "ready":
                continue
            storage = row.get("storage")
            if isinstance(storage, dict):
                legacy_blob = storage.get("blob") if storage.get("kind") == "legacy_file" else None
            else:
                legacy_blob = row.get("blob")
            if legacy_blob == blob:
                rows.append(row)
        return rows

    async def _adopt(self, operation_id: str, digest: str, size_bytes: int, legacy_operation_id: str | None) -> ResticArtifact | None:
        repository = self.service.repository
        # The adapter owns tag filtering.  Do not inspect or sweep untagged snapshots.
        if not hasattr(repository, "list_operation_snapshot_ids") or not hasattr(repository, "initialize"):
            return None
        repository_path = getattr(repository, "_repository_path", None)
        if not isinstance(repository_path, Path) or not self._existing_repository_is_safe(repository_path):
            return None
        repository_id = await repository.initialize(pass_fds=self.pass_fds)
        candidates = await repository.list_operation_snapshot_ids(workspace_id=self.workspace_id, operation_id=operation_id, pass_fds=self.pass_fds)
        # Older interrupted versions used the primary logical row ID.  It remains
        # discovery-only and is still byte verified before it can be adopted.
        if legacy_operation_id:
            candidates += await repository.list_operation_snapshot_ids(workspace_id=self.workspace_id, operation_id=legacy_operation_id, pass_fds=self.pass_fds)
        for snapshot_id in candidates:
            artifact = ResticArtifact(repository_id, snapshot_id, RESTIC_IMAGE_PATH, size_bytes, digest)
            await self._verify(artifact)
            return artifact
        return None

    @staticmethod
    def _existing_repository_is_safe(repository_path: Path) -> bool:
        try:
            details, config = repository_path.lstat(), repository_path / "config"
            config_details = config.lstat()
        except OSError:
            return False
        return (
            stat.S_ISDIR(details.st_mode)
            and not stat.S_ISLNK(details.st_mode)
            and stat.S_ISREG(config_details.st_mode)
            and not stat.S_ISLNK(config_details.st_mode)
        )

    async def _verify(self, artifact: ResticArtifact) -> None:
        repository = self.service.repository
        if hasattr(repository, "verify"):
            await repository.verify(artifact, pass_fds=self.pass_fds)
            return
        # Compatibility with narrow test adapters: still prove restored bytes.
        directory = self.root / "downloads"
        self.service._require_legacy_directory(directory)
        destination = directory / f".conversion-verify-{hashlib.sha256(artifact.snapshot_id.encode()).hexdigest()}.sqlite3"
        try:
            await repository.materialize(artifact, destination, pass_fds=self.pass_fds)
            if destination.stat().st_size != artifact.size_bytes or self.service._legacy_sha256(destination) != artifact.sha256:
                raise HTTPException(status_code=409, detail="SQLite history conversion repository verification failed")
        finally:
            destination.unlink(missing_ok=True)

    def _assert_headroom(self, source: Path, size_bytes: int) -> None:
        required = 4 * size_bytes + self._HEADROOM
        targets = [
            Path(getattr(self.service.repository, "_repository_path", source.parent)),
            Path(getattr(self.service.repository, "_scratch_path", source.parent)),
            Path(getattr(self.service.repository, "_cache_path", source.parent)),
        ]
        for target in targets:
            while not target.exists() and target != target.parent:
                target = target.parent
            filesystem = os.statvfs(target)
            if filesystem.f_bavail * filesystem.f_frsize < required:
                raise HTTPException(status_code=507, detail="SQLite history conversion is blocked by insufficient disk space")

    async def migrate(self, cancel_check: Callable[[], Awaitable[bool]] | None = None) -> int:
        from .service import run_sqlite_blocking

        converted = 0
        while True:
            prepared = await run_sqlite_blocking(self._prepare_next)
            if prepared is None:
                return converted
            ledger, blob, digest, size_bytes, source = prepared
            operation_id = str(ledger["operation_id"])
            if cancel_check is not None and await cancel_check():
                return converted
            if ledger["stage"] == "prepared":
                artifact = await self._adopt(operation_id, digest, size_bytes, ledger.get("legacy_operation_id"))
                if artifact is None:
                    await run_sqlite_blocking(self._assert_headroom, source, size_bytes)
                    artifact = await self.service.repository.ingest(
                        source,
                        workspace_id=self.workspace_id,
                        operation_id=operation_id,
                        sha256=digest,
                        size_bytes=size_bytes,
                        pass_fds=self.pass_fds,
                    )
                await self._verify(artifact)
                ledger.update(stage="repository_verified", storage=self._storage(artifact))
                await run_sqlite_blocking(self._save_ledger, ledger)
            if ledger["stage"] == "repository_verified":
                await self._verify(self._artifact(ledger))
                affected = await run_sqlite_blocking(self._publish, blob, ledger)
                ledger["stage"] = "catalog_published"
                await run_sqlite_blocking(self._save_ledger, ledger)
                converted += len(affected)
            if ledger["stage"] == "catalog_published":
                await self._verify(self._artifact(ledger))
                await run_sqlite_blocking(self._remove_published_source, blob, digest, size_bytes, ledger)
                ledger["stage"] = "source_removed"
                await run_sqlite_blocking(self._save_ledger, ledger)

    def _prepare_next(self) -> tuple[dict[str, Any], str, str, int, Path] | None:
        with self.service._catalog_lock_for_conversion(self.root):
            manifest = self.service._load(self.root, self.workspace_id)
            ledger = self._unfinished_ledger()
            pending = None if ledger is not None else next(iter(self._legacy_groups(manifest).items()), None)
            if ledger is None and pending is None:
                return None
            if ledger is None:
                assert pending is not None
                blob, rows = pending
                primary = rows[0]
                digest = str(primary.get("sha256") or "")
                size_bytes = int(primary.get("size_bytes") or -1)
                if (
                    len(digest) != 64
                    or size_bytes < 0
                    or any(str(row.get("sha256")) != digest or int(row.get("size_bytes") or -1) != size_bytes for row in rows)
                ):
                    raise HTTPException(status_code=409, detail="SQLite legacy backup integrity verification failed")
                operation_id = self._operation_id(self.workspace_id, blob, digest)
                ledger = self._read_ledger(operation_id) or {
                    "version": 1,
                    "workspace_id": self.workspace_id,
                    "legacy_blob": blob,
                    "sha256": digest,
                    "size_bytes": size_bytes,
                    "operation_id": operation_id,
                    "legacy_operation_id": f"legacy-{primary['id']}",
                    "backup_ids": [str(row["id"]) for row in rows],
                    "stage": "prepared",
                }
                self._save_ledger(ledger)
            blob, digest, size_bytes = str(ledger["legacy_blob"]), str(ledger["sha256"]), int(ledger["size_bytes"])
            if ledger["stage"] == "source_removed" and self._legacy_groups(manifest).get(blob):
                raise HTTPException(status_code=409, detail="SQLite history conversion journal conflicts with new legacy references")
            source = self.service._protected_path(self.root, blob, "blobs")
            if (
                ledger["stage"] not in {"catalog_published", "source_removed"}
                and not self._catalog_has_expected_storage(manifest, ledger)
                and (not source.is_file() or source.stat().st_size != size_bytes or self.service._legacy_sha256(source) != digest)
            ):
                raise HTTPException(status_code=409, detail="SQLite legacy backup integrity verification failed")
            return ledger, blob, digest, size_bytes, source

    def _publish(self, blob: str, ledger: dict[str, Any]) -> list[dict[str, Any]]:
        with self.service._catalog_lock_for_conversion(self.root):
            manifest = self.service._load(self.root, self.workspace_id)
            affected = self._rows(manifest, blob)
            for row in affected:
                row["storage"] = dict(ledger["storage"])
                row["blob"] = None
            self.service._save(self.root, manifest)
            return affected

    def _remove_published_source(self, blob: str, digest: str, size_bytes: int, ledger: dict[str, Any]) -> None:
        with self.service._catalog_lock_for_conversion(self.root):
            manifest = self.service._load(self.root, self.workspace_id)
            if self._rows(manifest, blob) or not self._catalog_has_expected_storage(manifest, ledger):
                raise HTTPException(status_code=409, detail="SQLite history conversion catalog publication is incomplete")
            source = self.service._protected_path(self.root, blob, "blobs")
            source.unlink(missing_ok=True)
            directory_fd = os.open(source.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

    def _legacy_groups(self, manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in manifest["backups"]:
            if row.get("status") != "ready":
                continue
            storage = row.get("storage")
            if isinstance(storage, dict):
                if storage.get("kind") != "legacy_file":
                    continue
                blob = storage.get("blob")
            else:
                blob = row.get("blob")
            if isinstance(blob, str):
                groups.setdefault(blob, []).append(row)
        return groups

    @staticmethod
    def _catalog_has_expected_storage(manifest: dict[str, Any], ledger: dict[str, Any]) -> bool:
        storage = ledger.get("storage")
        if not isinstance(storage, dict):
            return False
        rows = {str(row.get("id")): row for row in manifest.get("backups", [])}
        return all(
            backup_id in rows
            and rows[backup_id].get("storage") == storage
            and str(rows[backup_id].get("sha256")) == ledger.get("sha256")
            and int(rows[backup_id].get("size_bytes") or -1) == ledger.get("size_bytes")
            for backup_id in ledger.get("backup_ids", [])
        )
