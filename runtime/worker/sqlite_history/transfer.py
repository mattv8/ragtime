"""Portable, receipt-first server backup transfers for runtime history."""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import fcntl
import hashlib
import hmac
import json
import logging
import os
import shutil
import tarfile
import tempfile
import threading
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID, uuid4

from fastapi import HTTPException

from runtime.core.sqlite_workspace_state import SqliteWorkspaceStateError, validate_workspace_id

from . import models as history_models
from .export import RuntimeHistoryExporter
from .repository import ResticRepository
from .storage import repository_gate

_logger = logging.getLogger(__name__)

_MAX_BUNDLE_BYTES = 64 * 1024 * 1024 * 1024
_MAX_MEMBERS = 100_000
_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024 * 1024
_EXTRACTION_FREE_SPACE_MARGIN = 64 * 1024 * 1024
_TERMINAL_STATUSES = frozenset({"completed", "failed"})
_JOURNAL_VERSION = 1


class RuntimeHistoryTransfers:
    """Own staged bundles and keep imports non-destructive until verified.

    One instance per coordinator (``SqliteHistoryCoordinator
    .get_history_transfers``) so the background task registry and per-transfer
    liveness flocks survive individual HTTP requests.  ``start`` recovers the
    durable install journal and orphaned receipts on process start;
    ``shutdown`` drains accepted work without cancelling a mutating install.
    """

    def __init__(self, coordinator: Any) -> None:
        self._coordinator = coordinator
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def _service(self) -> Any:
        return self._coordinator._service()  # activation/capability gate

    def _root(self) -> Path:
        # Transfer recovery must run before activation is evaluated.  In
        # particular, do not obtain the history service here: doing so makes a
        # perfectly recoverable activation rename crash unrecoverable.
        return self._runtime_root() / "_sqlite_history" / "transfers"

    def _runtime_root(self) -> Path:
        root = getattr(self._coordinator, "_root", None)
        if root is None:  # Compatibility for isolated legacy test adapters.
            root = getattr(self._coordinator, "_runtime_root", None)
        if root is None:
            root = self._coordinator._runtime.root
        return Path(root)

    @staticmethod
    def _now() -> str:
        return datetime.datetime.now(datetime.UTC).isoformat()

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Recover durable transfer state before accepting new work."""
        await asyncio.to_thread(self._recover_sync)

    def _require_recovered(self) -> None:
        if os.path.lexists(self._journal_path()):
            raise HTTPException(status_code=503, detail="Runtime history import recovery is required")

    async def shutdown(self) -> None:
        """Drain in-flight transfer tasks; never cancel a mutating install."""
        tasks = tuple(self._tasks.values())
        if tasks:
            await asyncio.gather(*(asyncio.shield(task) for task in tasks), return_exceptions=True)

    def _recover_sync(self) -> None:
        try:
            self._recover_import_journal()
        except Exception:
            _logger.exception("Runtime history import journal recovery failed")
        try:
            self._fail_orphaned_receipts()
        except Exception:
            _logger.exception("Runtime history transfer receipt recovery failed")

    def _recover_import_journal(self) -> None:
        """Roll back an incomplete install or finish a committed cleanup.

        The journal is written before the first destructive action, so its
        absence means the destination was never touched.  ``installing`` means
        the install may be partial and every action must be undone;
        ``committed`` means only backup cleanup may still be pending.  A
        rollback that cannot finish keeps the journal for the next start.
        """
        root = self._runtime_root()
        path = self._journal_path()
        path.with_suffix(".tmp").unlink(missing_ok=True)
        if path.is_symlink() or not path.is_file():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("version") != _JOURNAL_VERSION
            or payload.get("phase") not in {"installing", "committed"}
            or not isinstance(payload.get("actions"), list)
        ):
            raise RuntimeError("runtime history import journal is invalid")
        undo = [self._journal_action(root, entry) for entry in payload["actions"]]
        with repository_gate(root, exclusive=True):
            if payload["phase"] == "committed":
                self._cleanup_backups(undo)
                self._clear_journal()
                _logger.info("Finished committed history import cleanup for transfer %s", payload.get("transfer_id"))
            elif self._rollback_actions(undo):
                self._clear_journal()
                _logger.warning("Rolled back incomplete history import for transfer %s", payload.get("transfer_id"))
            else:
                raise RuntimeError("runtime history import rollback is incomplete")

    def _fail_orphaned_receipts(self) -> None:
        """Fail accepted receipts whose owner no longer holds the liveness flock.

        Liveness is decided only by the per-transfer flock, never elapsed
        time, so an active transfer owned by another live process is left
        untouched.
        """
        for kind, identifier_key in (("exports", "export_id"), ("imports", "import_id")):
            base = self._root() / kind
            if base.is_symlink() or not base.is_dir():
                continue
            for entry in sorted(base.iterdir()):
                if entry.is_symlink() or not entry.is_dir():
                    continue
                try:
                    UUID(entry.name)
                except ValueError:
                    continue
                try:
                    receipt = json.loads((entry / "receipt.json").read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not isinstance(receipt, dict) or receipt.get("status") in _TERMINAL_STATUSES:
                    continue
                try:
                    lock_fd = self._acquire_transfer_lock(kind, entry.name)
                except (BlockingIOError, OSError):
                    continue
                try:
                    self._write_receipt(
                        kind,
                        entry.name,
                        {**receipt, identifier_key: entry.name, "status": "failed", "completed_at": self._now()},
                    )
                    for pattern in ("unpacked-*", "sqlite-history-export-*"):
                        for leftover in entry.glob(pattern):
                            shutil.rmtree(leftover, ignore_errors=True)
                    # An interrupted import can never be resumed without its owner.
                    self._remove_path(entry / "import.bundle")
                finally:
                    self._release_transfer_lock(lock_fd)
                _logger.warning("Failed orphaned runtime history transfer %s/%s on startup", kind, entry.name)

    async def cleanup(self) -> None:
        """Remove only terminal, unreferenced transfer payloads.

        A live flock is the authority for a download or a worker still using a
        receipt; age is merely the retention policy.  Older receipts without a
        timestamp are retained rather than guessed away.
        """
        await asyncio.to_thread(self._cleanup_sync)

    def _cleanup_sync(self) -> None:
        export_age = history_models.HISTORY_EXPORT_RETENTION_SECONDS
        receipt_age = history_models.HISTORY_RECEIPT_RETENTION_SECONDS
        now = datetime.datetime.now(datetime.UTC)
        for kind, identifier_key in (("exports", "export_id"), ("imports", "import_id")):
            base = self._root() / kind
            if base.is_symlink() or not base.is_dir():
                continue
            for entry in base.iterdir():
                try:
                    UUID(entry.name)
                except ValueError:
                    continue
                if entry.is_symlink() or not entry.is_dir():
                    continue
                try:
                    fd = self._acquire_transfer_lock(kind, entry.name)
                except BlockingIOError:
                    continue
                try:
                    self._cleanup_terminal_entry(kind, entry, now, export_age, receipt_age)
                finally:
                    self._release_transfer_lock(fd)

    def _cleanup_terminal_entry(self, kind: str, entry: Path, now: datetime.datetime, export_age: int, receipt_age: int) -> None:
        """Delete only while holding the same exclusive flock cleanup probed."""
        try:
            receipt = json.loads((entry / "receipt.json").read_text(encoding="utf-8"))
            created = datetime.datetime.fromisoformat(receipt["completed_at"])
        except (KeyError, OSError, TypeError, ValueError):
            return
        if receipt.get("status") not in _TERMINAL_STATUSES:
            return
        age = (now - created.astimezone(datetime.UTC)).total_seconds()
        payload = entry / ("export.bundle" if kind == "exports" else "import.bundle")
        if kind == "imports" and payload.exists():
            self._remove_path(payload)
        if kind == "exports" and age >= export_age and payload.exists():
            self._remove_path(payload)
        if age >= receipt_age and not payload.exists() and not self._journal_references(entry.name):
            self._remove_path(entry)

    def _journal_references(self, transfer_id: str) -> bool:
        try:
            payload = json.loads(self._journal_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return isinstance(payload, dict) and payload.get("transfer_id") == transfer_id

    # -- per-transfer liveness ----------------------------------------------

    def _lock_path(self, kind: str, transfer_id: str) -> Path:
        return self._receipt_path(kind, transfer_id).parent / "transfer.lock"

    def _acquire_transfer_lock(self, kind: str, transfer_id: str, *, shared: bool = False) -> int:
        """Hold the per-transfer liveness flock for the accepted task's lifetime."""
        lock_path = self._lock_path(kind, transfer_id)
        lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            return fd
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _release_transfer_lock(fd: int) -> None:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)

    def _transfer_is_live(self, kind: str, transfer_id: str) -> bool:
        """Report liveness from the actual flock, never wall-clock age."""
        try:
            fd = os.open(self._lock_path(kind, transfer_id), os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return False
        except OSError:
            return True  # fail safe: never fail a transfer we cannot probe
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)

    # -- receipts ------------------------------------------------------------

    def _receipt_path(self, kind: str, transfer_id: str) -> Path:
        # Validate transfer_id is a valid UUID to prevent path traversal
        try:
            UUID(transfer_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid transfer ID format") from exc
        return self._root() / kind / transfer_id / "receipt.json"

    def _write_receipt(self, kind: str, transfer_id: str, payload: dict[str, Any]) -> None:
        path = self._receipt_path(kind, transfer_id)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        self._fsync_directory(path.parent)

    def _receipt(self, kind: str, transfer_id: str) -> dict[str, Any]:
        try:
            return json.loads(self._receipt_path(kind, transfer_id).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="SQLite history transfer not found") from exc

    async def status(self) -> dict[str, Any]:
        self._require_recovered()
        service = self._service()
        repository = service.repository
        if repository is None:
            raise HTTPException(status_code=503, detail="Runtime SQLite history repository is unavailable")
        repository_id = await repository.initialize()
        return {"active": True, "repository_id": repository_id, "export_version": 1}

    # -- exports -------------------------------------------------------------

    async def accept_export(self, *, include_repository_key: bool) -> dict[str, Any]:
        self._require_recovered()
        transfer_id = str(uuid4())
        receipt = {
            "export_id": transfer_id,
            "status": "accepted",
            "accepted_at": self._now(),
            "export_version": 1,
            "includes_repository_key": include_repository_key,
        }
        lock_fd = self._acquire_transfer_lock("exports", transfer_id)
        try:
            self._write_receipt("exports", transfer_id, receipt)
        except BaseException:
            self._release_transfer_lock(lock_fd)
            raise
        self._spawn(transfer_id, self._run_export(transfer_id, include_repository_key, lock_fd))
        return receipt

    def _spawn(self, transfer_id: str, coroutine: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coroutine)
        self._tasks[transfer_id] = task
        task.add_done_callback(lambda _task: self._tasks.pop(transfer_id, None))

    async def _run_export(self, transfer_id: str, include_repository_key: bool, lock_fd: int) -> None:
        try:
            receipt_dir = self._receipt_path("exports", transfer_id).parent
            staged: Path | None = None
            try:
                staged = await RuntimeHistoryExporter(self._service()).stage_server_export(receipt_dir, include_repository_key=include_repository_key)
                bundle = receipt_dir / "export.bundle"
                await asyncio.to_thread(self._archive, staged, bundle)
                await asyncio.to_thread(self._fsync_file, bundle)
                bundle_size_bytes, bundle_sha256 = await asyncio.to_thread(self._bundle_digest, bundle)
                metadata = json.loads((staged / "history-export.json").read_text(encoding="utf-8"))
                self._write_receipt(
                    "exports",
                    transfer_id,
                    {
                        "export_id": transfer_id,
                        "status": "completed",
                        "completed_at": self._now(),
                        "export_version": 1,
                        "repository_id": metadata["repository_id"],
                        "includes_repository_key": include_repository_key,
                        "bundle_size_bytes": bundle_size_bytes,
                        "bundle_sha256": bundle_sha256,
                    },
                )
            except Exception:
                _logger.exception("Export failed for %s", transfer_id)
                self._write_receipt(
                    "exports",
                    transfer_id,
                    {
                        "export_id": transfer_id,
                        "status": "failed",
                        "completed_at": self._now(),
                        "export_version": 1,
                        "includes_repository_key": include_repository_key,
                    },
                )
            finally:
                if staged is not None:
                    shutil.rmtree(staged, ignore_errors=True)
        finally:
            self._release_transfer_lock(lock_fd)

    def export_receipt(self, transfer_id: str) -> dict[str, Any]:
        return self._receipt("exports", transfer_id)

    def export_bundle(self, transfer_id: str) -> Path:
        receipt = self.export_receipt(transfer_id)
        bundle = self._receipt_path("exports", transfer_id).parent / "export.bundle"
        if receipt.get("status") != "completed" or not bundle.is_file() or bundle.is_symlink():
            raise HTTPException(status_code=409, detail="SQLite history export is not ready")
        return bundle

    @contextlib.asynccontextmanager
    async def export_download_lifetime(self, transfer_id: str) -> AsyncIterator[None]:
        """Pin a bundle only while its ASGI response actually runs."""
        try:
            lock_fd = await asyncio.to_thread(self._acquire_transfer_lock, "exports", transfer_id, shared=True)
        except (BlockingIOError, OSError) as exc:
            raise HTTPException(status_code=409, detail="SQLite history export is busy") from exc
        try:
            self.export_bundle(transfer_id)
            yield
        finally:
            await asyncio.to_thread(self._release_transfer_lock, lock_fd)

    # -- imports -------------------------------------------------------------

    async def accept_import(self, source: Path, metadata: dict[str, Any]) -> dict[str, Any]:
        self._require_recovered()
        transfer_id = str(uuid4())
        receipt_path = self._receipt_path("imports", transfer_id)
        receipt = {"import_id": transfer_id, "status": "accepted", "accepted_at": self._now()}
        lock_fd = self._acquire_transfer_lock("imports", transfer_id)
        try:
            target = receipt_path.parent / "import.bundle"
            await asyncio.to_thread(self._move_and_fsync, source, target)
            self._write_receipt("imports", transfer_id, receipt)
        except BaseException:
            self._release_transfer_lock(lock_fd)
            raise
        self._spawn(transfer_id, self._run_import(transfer_id, metadata, lock_fd))
        return receipt

    async def _run_import(self, transfer_id: str, metadata: dict[str, Any], lock_fd: int) -> None:
        try:
            try:
                activate = await asyncio.to_thread(self._import_sync, transfer_id, metadata)
                if activate:
                    # Coordinator activation schedules maintenance and must run
                    # on its owning asyncio loop, never the install worker.
                    self._coordinator.activate()
                self._write_receipt("imports", transfer_id, {"import_id": transfer_id, "status": "completed", "completed_at": self._now()})
            except Exception:
                _logger.exception("Import failed for %s", transfer_id)
                self._write_receipt("imports", transfer_id, {"import_id": transfer_id, "status": "failed", "completed_at": self._now()})
        finally:
            self._remove_path(self._receipt_path("imports", transfer_id).parent / "import.bundle")
            self._release_transfer_lock(lock_fd)

    def import_receipt(self, transfer_id: str) -> dict[str, Any]:
        return self._receipt("imports", transfer_id)

    @staticmethod
    def _archive(source: Path, destination: Path) -> None:
        # The bundle is already an internal transport artifact.  Plain tar
        # avoids spending the exclusive repository interval compressing it;
        # import retains r:* compatibility with historic gzip bundles.
        with tarfile.open(destination, "w") as archive:
            for entry in sorted(source.rglob("*")):
                archive.add(entry, arcname=entry.relative_to(source).as_posix(), recursive=False)

    def _import_sync(self, transfer_id: str, metadata: dict[str, Any]) -> bool:
        root = self._runtime_root()
        receipt_dir = self._receipt_path("imports", transfer_id).parent
        unpacked = Path(tempfile.mkdtemp(prefix="unpacked-", dir=receipt_dir))
        try:
            bundle_path = receipt_dir / "import.bundle"
            expected_size = metadata.get("bundle_size_bytes")
            expected_digest = metadata.get("bundle_sha256")
            if expected_size is not None or expected_digest is not None:
                if not isinstance(expected_size, int) or expected_size <= 0 or not isinstance(expected_digest, str) or len(expected_digest) != 64:
                    raise RuntimeError("import bundle integrity metadata is invalid")
                actual_size, actual_digest = self._bundle_digest(bundle_path)
                if actual_size != expected_size or not hmac.compare_digest(actual_digest, expected_digest):
                    raise RuntimeError("import bundle integrity verification failed")

            # Extract and validate bundle structure
            self._safe_extract(bundle_path, unpacked)
            export_metadata_path = unpacked / "history-export.json"
            if not export_metadata_path.is_file():
                raise RuntimeError("import bundle missing history-export.json")
            export_metadata = json.loads(export_metadata_path.read_text(encoding="utf-8"))

            # Validate incoming metadata matches declared state
            includes_key = bool(export_metadata.get("includes_repository_key"))
            expected_id = metadata.get("repository_id")
            if expected_id != export_metadata.get("repository_id") or not isinstance(expected_id, str):
                raise RuntimeError("repository identity does not match bundle")

            # Validate secret files match key-inclusion declaration
            secrets_dir = unpacked / "secrets"
            has_secrets = secrets_dir.is_dir() and not secrets_dir.is_symlink()
            if includes_key and not has_secrets:
                raise RuntimeError("bundle declared key inclusion but secrets directory missing")
            if not includes_key and has_secrets:
                raise RuntimeError("bundle declared no key but secrets directory present")

            # Validate private unpacked state before taking the live repository
            # gate.  The gate is for the short recheck/install transaction.
            reachable_snapshots = self._validate_import(unpacked, expected_id, includes_key)
            with repository_gate(root, exclusive=True):
                self._require_recovered()
                self._validate_destination_key(unpacked, expected_id, includes_key)
                self._validate_destination_catalogs(root, expected_id, unpacked, reachable_snapshots)
                self._install_verified(root, unpacked, transfer_id)
                return getattr(self._coordinator, "activate", None) is not None
        finally:
            shutil.rmtree(unpacked, ignore_errors=True)

    def _validate_import(self, unpacked: Path, repository_id: str, includes_key: bool) -> set[str]:
        """Validate that the imported bundle matches its declared identity and key state.

        For keyless imports, use the existing destination key rather than one in the bundle.
        For key-inclusive imports, validate the portable key matches the repository.
        Preserves original destination until all checks pass.
        """
        root = self._runtime_root()
        repo_root_import = unpacked / "_sqlite_history"
        repo_root_dest = root / "_sqlite_history"

        # Determine which key to use for validation; validate early
        if includes_key:
            # Portable key must be present in the bundle
            key_path = unpacked / "secrets" / "repository-password"
            if not key_path.is_file() or key_path.is_symlink():
                raise RuntimeError("portable repository key is unavailable in bundle")
            _logger.info("Import validates portable key from bundle for repository %s", repository_id)
        else:
            # Keyless: use existing destination key for validation
            key_path = repo_root_dest / "secrets" / "repository-password"
            if not key_path.is_file() or key_path.is_symlink():
                raise RuntimeError("keyless import requires existing repository key at destination")
            _logger.info("Import validates with existing destination key for repository %s", repository_id)

        # Validate bundle metadata has correct key declaration
        export_metadata_path = unpacked / "history-export.json"
        if not export_metadata_path.is_file():
            raise RuntimeError("import bundle missing history-export.json")
        export_metadata = json.loads(export_metadata_path.read_text(encoding="utf-8"))
        if bool(export_metadata.get("includes_repository_key")) != includes_key:
            raise RuntimeError("metadata key declaration does not match bundle contents")

        # Validate restic repository in bundle using the appropriate key
        repository = ResticRepository(
            repository_path=repo_root_import / "restic",
            cache_path=repo_root_import / "cache",
            password_path=key_path,
            scratch_path=repo_root_import / "scratch",
        )
        actual = repository._initialize(threading.Event(), ())
        if actual != repository_id:
            raise RuntimeError("repository key does not match repository")
        repository._check(True, threading.Event(), ())
        return self._validate_catalogs(unpacked, repository_id, repository)

    def _validate_destination_key(self, unpacked: Path, repository_id: str, includes_key: bool) -> None:
        """Recheck key-dependent validation after waiting for the live gate."""
        if includes_key:
            return
        key_path = self._runtime_root() / "_sqlite_history" / "secrets" / "repository-password"
        if key_path.is_symlink() or not key_path.is_file():
            raise RuntimeError("keyless import requires existing repository key at destination")
        repository = ResticRepository(
            repository_path=unpacked / "_sqlite_history" / "restic",
            cache_path=unpacked / "_sqlite_history" / "cache",
            password_path=key_path,
            scratch_path=unpacked / "_sqlite_history" / "scratch",
        )
        if repository._initialize(threading.Event(), ()) != repository_id:
            raise RuntimeError("destination repository key no longer matches import")

    @staticmethod
    def _valid_workspace_id(value: str) -> bool:
        try:
            return validate_workspace_id(value) == value
        except SqliteWorkspaceStateError:
            return False

    def _validate_catalogs(self, unpacked: Path, repository_id: str, repository: ResticRepository) -> set[str]:
        snapshots = json.loads(repository._run("snapshots", "--json", cancelled=threading.Event(), pass_fds=()))
        reachable = {str(item["id"]) for item in snapshots if isinstance(item, dict) and isinstance(item.get("id"), str)}
        workspaces = unpacked / "workspaces"
        if not workspaces.exists():
            return reachable
        if workspaces.is_symlink() or not workspaces.is_dir():
            raise RuntimeError("import bundle workspace catalogs are invalid")
        for workspace in workspaces.iterdir():
            if workspace.is_symlink() or not workspace.is_dir() or not self._valid_workspace_id(workspace.name):
                raise RuntimeError("import bundle workspace ID is invalid")
            catalog = workspace / "sqlite_backups" / "manifest-v1.json"
            catalog_dir = catalog.parent
            if not catalog.exists() and catalog_dir.is_dir() and not catalog_dir.is_symlink() and all(entry.name == ".lock" for entry in catalog_dir.iterdir()):
                continue
            if catalog.is_symlink() or not catalog.is_file():
                raise RuntimeError("import bundle history catalog is invalid")
            try:
                manifest = json.loads(catalog.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RuntimeError("import bundle history catalog is unreadable") from exc
            if not isinstance(manifest, dict) or manifest.get("workspace_id") != workspace.name or not isinstance(manifest.get("backups"), list):
                raise RuntimeError("import bundle history catalog is invalid")
            for row in manifest["backups"]:
                if not isinstance(row, dict) or row.get("status") != "ready":
                    continue
                storage = row.get("storage") if isinstance(row, dict) else None
                # Old verified local rows predate Restic storage references.
                # They are safe to retain: this install only replaces Restic
                # state, and a later migration can handle their local payload.
                if storage is None and isinstance(row.get("sha256"), str) and isinstance(row.get("size_bytes"), int):
                    continue
                if not isinstance(storage, dict) or storage.get("kind") != "restic":
                    raise RuntimeError("import bundle history storage reference is invalid")
                if storage.get("repository_id") != repository_id or storage.get("snapshot_id") not in reachable:
                    raise RuntimeError("import bundle history snapshot reference is unreachable")
        return reachable

    def _validate_destination_catalogs(self, root: Path, repository_id: str, unpacked: Path, reachable_snapshots: set[str]) -> None:
        """Fail before mutation when a destination-only catalog would be stranded."""
        incoming = unpacked / "workspaces"
        incoming_catalog_workspaces = (
            {entry.name for entry in incoming.iterdir() if (entry / "sqlite_backups").is_dir() and not (entry / "sqlite_backups").is_symlink()}
            if incoming.is_dir() and not incoming.is_symlink()
            else set()
        )
        workspaces = root / "workspaces"
        if not workspaces.is_dir() or workspaces.is_symlink():
            return
        for workspace in workspaces.iterdir():
            if workspace.name in incoming_catalog_workspaces:
                continue
            catalog = workspace / "sqlite_backups" / "manifest-v1.json"
            if not catalog.exists():
                continue
            if catalog.is_symlink() or not catalog.is_file():
                raise HTTPException(status_code=409, detail="Destination SQLite history catalog is invalid")
            try:
                manifest = json.loads(catalog.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise HTTPException(status_code=409, detail="Destination SQLite history catalog is unreadable") from exc
            for row in manifest.get("backups", []) if isinstance(manifest, dict) else []:
                storage = row.get("storage") if isinstance(row, dict) else None
                if isinstance(storage, dict) and storage.get("kind") == "restic":
                    if storage.get("repository_id") != repository_id or storage.get("snapshot_id") not in reachable_snapshots:
                        raise HTTPException(status_code=409, detail="Import would strand destination SQLite history catalog")

    # -- durable install ------------------------------------------------------

    def _journal_path(self) -> Path:
        return self._root() / "import-journal.json"

    def _write_journal(self, payload: dict[str, Any]) -> None:
        path = self._journal_path()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        self._fsync_directory(path.parent)

    def _clear_journal(self) -> None:
        path = self._journal_path()
        path.unlink(missing_ok=True)
        path.with_suffix(".tmp").unlink(missing_ok=True)
        self._fsync_directory(path.parent)

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as source:
            os.fsync(source.fileno())

    @staticmethod
    def _move_and_fsync(source: Path, target: Path) -> None:
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        RuntimeHistoryTransfers._fsync_file(target)
        RuntimeHistoryTransfers._fsync_directory(target.parent)

    @staticmethod
    def _journal_payload(root: Path, transfer_id: str, phase: str, undo: list[tuple[Path, Path, bool]]) -> dict[str, Any]:
        return {
            "version": _JOURNAL_VERSION,
            "transfer_id": transfer_id,
            "phase": phase,
            "actions": [
                {
                    "target": target.relative_to(root).as_posix(),
                    "backup": backup.relative_to(root).as_posix(),
                    "had_existing": had_existing,
                }
                for target, backup, had_existing in undo
            ],
        }

    @staticmethod
    def _journal_action(root: Path, entry: Any) -> tuple[Path, Path, bool]:
        if not isinstance(entry, dict):
            raise RuntimeError("runtime history import journal is invalid")
        return (
            RuntimeHistoryTransfers._resolve_root_relative(root, entry.get("target")),
            RuntimeHistoryTransfers._resolve_root_relative(root, entry.get("backup")),
            bool(entry.get("had_existing")),
        )

    @staticmethod
    def _resolve_root_relative(root: Path, value: Any) -> Path:
        """Resolve a journaled path strictly as a validated root-relative path."""
        if not isinstance(value, str) or not value:
            raise RuntimeError("runtime history import journal path is invalid")
        relative = PurePosixPath(value)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise RuntimeError("runtime history import journal path is invalid")
        return root.joinpath(*relative.parts)

    def _install_verified(self, root: Path, unpacked: Path, transfer_id: str) -> None:
        """Install a verified bundle under a durable write-ahead journal.

        The journal (root-relative paths only) is fsynced before the first
        destructive action, so a crash at any later point is recoverable at
        startup.  Repository, operations, activation, workspace catalogs and
        the portable key are installed under one journal so they change
        together or not at all.  Success flips the journal to ``committed``
        before this transfer's backups (and only this transfer's) are
        removed; failure rolls every action back and keeps the journal when
        the rollback itself cannot finish.
        """
        live = root / "_sqlite_history"
        incoming = unpacked / "_sqlite_history"
        with repository_gate(root, exclusive=True):
            live.mkdir(parents=True, exist_ok=True)
            actions: list[tuple[Path, Path, Path, bool]] = []  # source, target, backup, had_existing
            for name in ("restic", "operations", "activation-v1.json"):
                candidate = incoming / name
                if not candidate.exists():
                    continue
                target = live / name
                actions.append((candidate, target, live / f".{name}.before-import-{transfer_id}", target.exists()))
            workspaces_dir = unpacked / "workspaces"
            if workspaces_dir.is_dir():
                for catalog in sorted(workspaces_dir.glob("*/sqlite_backups")):
                    workspace = catalog.parent.name
                    target = root / "workspaces" / workspace / "sqlite_backups"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    backup = target.parent / f".sqlite_backups.before-import-{transfer_id}"
                    actions.append((catalog, target, backup, target.exists()))
            portable_key = unpacked / "secrets" / "repository-password"
            if portable_key.is_file() and not portable_key.is_symlink():
                secrets_dir = live / "secrets"
                secrets_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
                target = secrets_dir / "repository-password"
                backup = secrets_dir / f".repository-password.before-import-{transfer_id}"
                actions.append((portable_key, target, backup, target.exists()))
            if not actions:
                return
            undo = [(target, backup, had_existing) for _source, target, backup, had_existing in actions]
            self._write_journal(self._journal_payload(root, transfer_id, "installing", undo))
            try:
                for source, target, backup, had_existing in actions:
                    if had_existing:
                        os.replace(target, backup)
                    os.replace(source, target)
            except BaseException:
                if self._rollback_actions(undo):
                    self._clear_journal()
                else:
                    _logger.error(
                        "Runtime history import %s rollback is incomplete; journal retained for startup recovery",
                        transfer_id,
                    )
                raise
            self._write_journal(self._journal_payload(root, transfer_id, "committed", undo))
            self._cleanup_backups(undo)
            self._clear_journal()

    def _rollback_actions(self, actions: list[tuple[Path, Path, bool]]) -> bool:
        """Restore originals in reverse order; report success only when complete.

        ``os.replace`` cannot replace a nonempty directory, so any installed
        target is removed before its backup is renamed into place.  Targets
        that never existed before the install are simply removed.
        """
        restored = True
        for target, backup, had_existing in reversed(actions):
            try:
                if backup.is_symlink() or backup.exists():
                    self._remove_path(target)
                    os.replace(backup, target)
                elif not had_existing:
                    self._remove_path(target)
            except OSError:
                _logger.exception("Runtime history import rollback failed for %s", target)
                restored = False
        return restored

    def _cleanup_backups(self, actions: list[tuple[Path, Path, bool]]) -> None:
        """Delete exactly this transfer's journaled backups, nothing else."""
        for _target, backup, _had_existing in actions:
            with contextlib.suppress(OSError):
                self._remove_path(backup)

    @staticmethod
    def _remove_path(path: Path) -> None:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

    @staticmethod
    def _bundle_digest(path: Path) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        return size, digest.hexdigest()

    @staticmethod
    def _safe_extract(bundle: Path, destination: Path) -> None:
        """Safely extract tar bundle with size/entry/duplicate bounds.

        Validates:
        - Bundle is a regular file (not symlink)
        - Compressed size <= _MAX_BUNDLE_BYTES
        - Total uncompressed size <= _MAX_UNCOMPRESSED_BYTES
        - Number of entries <= _MAX_MEMBERS
        - No duplicates, absolute paths, traversal, or special files
        Permission bits (masked to 0o777) are preserved so the repository key
        keeps its required 0600 mode; setuid/setgid/sticky bits are dropped.
        """
        if bundle.is_symlink() or not bundle.is_file() or bundle.stat().st_size > _MAX_BUNDLE_BYTES:
            raise RuntimeError("history transfer bundle is invalid")

        with tarfile.open(bundle, "r:*") as archive:
            members = archive.getmembers()
            if len(members) > _MAX_MEMBERS:
                raise RuntimeError("history transfer bundle has too many entries")

            seen_paths: set[str] = set()
            total_uncompressed = 0

            for member in members:
                path = PurePosixPath(member.name)

                # Check for duplicates
                path_str = path.as_posix()
                if path_str in seen_paths:
                    raise RuntimeError("history transfer bundle has duplicate entries")
                seen_paths.add(path_str)

                # Check for unsafe path patterns
                if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise RuntimeError("history transfer bundle has unsafe entries")

                # Accumulate uncompressed size
                if member.isfile():
                    total_uncompressed += member.size
                    if total_uncompressed > _MAX_UNCOMPRESSED_BYTES:
                        raise RuntimeError("history transfer bundle uncompressed size exceeds limit")

            required = total_uncompressed + _EXTRACTION_FREE_SPACE_MARGIN
            if shutil.disk_usage(destination).free < required:
                raise HTTPException(status_code=507, detail="SQLite history import is blocked by insufficient disk space")

            extracted_bytes = 0
            for member in members:
                path = PurePosixPath(member.name)
                # Extract safely
                target = destination.joinpath(*path.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    os.chmod(target, member.mode & 0o777)
                    continue

                # Ensure parent exists and extract file
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError("history transfer bundle is invalid")
                with source, target.open("xb") as output:
                    remaining = member.size
                    while remaining:
                        block = source.read(min(1024 * 1024, remaining))
                        if not block:
                            raise RuntimeError("history transfer bundle member is truncated")
                        output.write(block)
                        remaining -= len(block)
                        extracted_bytes += len(block)
                        if extracted_bytes > total_uncompressed:
                            raise RuntimeError("history transfer bundle uncompressed size exceeds limit")
                    if source.read(1):
                        raise RuntimeError("history transfer bundle member size is invalid")
                os.chmod(target, member.mode & 0o777)
