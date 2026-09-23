"""Durable, idempotent runtime SQLite-history operation receipts.

Receipts are intentionally a small filesystem-only primitive.  They establish
idempotency and retain enough request data for a later runtime process to
reconcile work; scheduling and network observation belong to the service layer.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import stat
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID, uuid5


class OperationConflict(RuntimeError):
    """An operation receipt is malformed or conflicts with a request."""


class OperationNotFound(RuntimeError):
    """No receipt exists for the requested operation."""


class OperationStore:
    """Atomic, no-follow receipt store rooted in runtime-private storage.

    ``receipt.lock`` is held only while reading or replacing a receipt.
    ``operation.lock`` is a distinct, long-lived flock that callers retain
    while a child can mutate history; pass its yielded fd through ``pass_fds``
    when spawning that child.
    """

    PHASES = frozenset(
        {
            "accepted",
            "running",
            "repository_committed",
            "catalog_committed",
            "completed",
            "failed",
            "cancelling",
            "cancelled",
            "interrupted",
            "reconciling",
        }
    )
    _TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})
    _TRANSITIONS = {
        "accepted": frozenset({"running", "cancelled", "failed", "interrupted", "reconciling"}),
        "running": frozenset({"repository_committed", "cancelling", "failed", "interrupted", "reconciling"}),
        "repository_committed": frozenset({"catalog_committed", "cancelling", "failed", "interrupted", "reconciling"}),
        "catalog_committed": frozenset({"completed", "failed", "interrupted", "reconciling"}),
        "cancelling": frozenset({"cancelled", "failed", "interrupted", "reconciling"}),
        # A restart may establish which durable boundary was reached, then
        # resume that boundary without treating a controller timeout as death.
        "reconciling": frozenset({"running", "repository_committed", "catalog_committed", "completed", "failed", "cancelled", "interrupted"}),
    }
    _PROGRESS_FIELDS = {
        "accepted": frozenset(),
        "running": frozenset({"database_outcomes", "error"}),
        "repository_committed": frozenset({"database_outcomes", "repository_refs", "error"}),
        "catalog_committed": frozenset({"database_outcomes", "repository_refs", "error"}),
        "completed": frozenset({"acknowledged_at"}),
        "failed": frozenset({"acknowledged_at", "error"}),
        "cancelling": frozenset({"database_outcomes", "error"}),
        "cancelled": frozenset({"acknowledged_at", "error"}),
        "interrupted": frozenset({"acknowledged_at", "error", "database_outcomes", "repository_refs"}),
        "reconciling": frozenset({"database_outcomes", "repository_refs", "error"}),
    }
    _DATABASE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
    _IDENTITY_FIELDS = frozenset({"operation_id", "workspace_id", "creator_id", "request_digest", "kind", "accepted_payload", "created_at"})

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @staticmethod
    def suboperation_id(parent_operation_id: str, database_name: str) -> str:
        """Return UUIDv5(parent UUID, validated database name)."""
        parent = UUID(parent_operation_id)
        if not isinstance(database_name, str) or not OperationStore._DATABASE_NAME.fullmatch(database_name):
            raise ValueError("Invalid database name")
        return str(uuid5(parent, database_name))

    def accept(
        self,
        *,
        operation_id: str,
        workspace_id: str,
        creator_id: str,
        request_digest: str,
        kind: str,
        accepted_payload: Any = None,
    ) -> dict[str, Any]:
        """Durably accept an immutable request, or return its exact replay."""
        operation_id = self._operation_id(operation_id)
        self._validate_identity(workspace_id, creator_id, request_digest, kind, accepted_payload)
        with self._receipt_lock(operation_id, create=True):
            existing = self._read(operation_id)
            identity = (workspace_id, creator_id, request_digest, kind)
            if existing is not None:
                if tuple(existing.get(key) for key in ("workspace_id", "creator_id", "request_digest", "kind")) != identity:
                    raise OperationConflict("Operation ID conflicts with an existing request")
                return existing
            now = self._now()
            receipt = {
                "operation_id": operation_id,
                "workspace_id": workspace_id,
                "creator_id": creator_id,
                "request_digest": request_digest,
                "kind": kind,
                "accepted_payload": accepted_payload,
                "phase": "accepted",
                "database_outcomes": {},
                "repository_refs": [],
                "error": None,
                "acknowledged_at": None,
                "retired_at": None,
                "created_at": now,
                "updated_at": now,
            }
            self._write(operation_id, receipt)
            return receipt

    def get(self, operation_id: str) -> dict[str, Any]:
        operation_id = self._operation_id(operation_id)
        with self._receipt_lock(operation_id, create=False):
            receipt = self._read(operation_id)
            if receipt is None:
                raise OperationNotFound("Operation receipt is unavailable")
            return receipt

    def get_for_workspace(self, operation_id: str, workspace_id: str) -> dict[str, Any]:
        """Return a receipt only when it belongs to the trusted workspace."""
        receipt = self.get(operation_id)
        if receipt["workspace_id"] != workspace_id:
            raise OperationNotFound("Operation receipt is unavailable")
        return receipt

    def transition(self, operation_id: str, phase: str, **updates: Any) -> dict[str, Any]:
        """Atomically make one explicit durable phase transition.

        Repeating a phase is allowed only for that phase's documented progress
        fields.  Identity and accepted payload are never mutable.
        """
        if phase not in self.PHASES:
            raise ValueError("Invalid runtime operation phase")
        operation_id = self._operation_id(operation_id)
        with self._receipt_lock(operation_id, create=False):
            receipt = self._require(operation_id)
            current = receipt["phase"]
            if receipt.get("retired_at") is not None:
                raise OperationConflict("Operation has been retired")
            if current == phase:
                self._apply_updates(receipt, updates, self._PROGRESS_FIELDS[current])
            else:
                if phase not in self._TRANSITIONS.get(current, frozenset()):
                    raise OperationConflict("Invalid runtime operation transition")
                self._apply_updates(receipt, updates, self._mutable_fields_for_transition(phase))
                receipt["phase"] = phase
            receipt["updated_at"] = self._now()
            self._write(operation_id, receipt)
            return receipt

    def update_progress(self, operation_id: str, **updates: Any) -> dict[str, Any]:
        """Persist authorised progress without advancing the phase."""
        operation_id = self._operation_id(operation_id)
        with self._receipt_lock(operation_id, create=False):
            receipt = self._require(operation_id)
            if receipt.get("retired_at") is not None:
                raise OperationConflict("Operation has been retired")
            self._apply_updates(receipt, updates, self._PROGRESS_FIELDS[receipt["phase"]])
            receipt["updated_at"] = self._now()
            self._write(operation_id, receipt)
            return receipt

    def request_cancel(self, operation_id: str) -> dict[str, Any]:
        """Idempotently request cancellation; draining remains service-owned."""
        receipt = self.get(operation_id)
        if receipt["phase"] in self._TERMINAL or receipt["phase"] == "cancelling":
            return receipt
        if receipt["phase"] == "accepted":
            return self.transition(operation_id, "cancelled")
        return self.transition(operation_id, "cancelling")

    def acknowledge(self, operation_id: str, *, acknowledged_at: str | None = None) -> dict[str, Any]:
        return self.update_progress(operation_id, acknowledged_at=acknowledged_at or self._now())

    def list_active(self, *, workspace_id: str | None = None) -> list[dict[str, Any]]:
        """List nonterminal receipts without creating directories for a GET."""
        return [receipt for receipt in self.list_workspace(workspace_id) if receipt["phase"] not in self._TERMINAL]

    def list_workspace(self, workspace_id: str | None) -> list[dict[str, Any]]:
        operations = self._operations_directory(create=False)
        if operations is None:
            return []
        results: list[dict[str, Any]] = []
        with os.scandir(operations) as entries:
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False):
                    raise OperationConflict("Operation directory is invalid")
                try:
                    operation_id = self._operation_id(entry.name)
                except ValueError as exc:
                    raise OperationConflict("Operation directory is invalid") from exc
                with self._receipt_lock(operation_id, create=False):
                    receipt = self._read(operation_id)
                    if receipt is not None and (workspace_id is None or receipt["workspace_id"] == workspace_id):
                        results.append(receipt)
        return sorted(results, key=lambda item: (item["created_at"], item["operation_id"]))

    def retire(self, operation_id: str) -> dict[str, Any]:
        """Write a compact durable tombstone; it can never be accepted anew."""
        operation_id = self._operation_id(operation_id)
        with self._receipt_lock(operation_id, create=False):
            receipt = self._require(operation_id)
            if receipt["phase"] not in self._TERMINAL:
                raise OperationConflict("Only terminal operations may be retired")
            if receipt.get("retired_at") is None:
                receipt["accepted_payload"] = None
                receipt["database_outcomes"] = {}
                receipt["repository_refs"] = []
                receipt["error"] = None
                receipt["retired_at"] = self._now()
                receipt["updated_at"] = receipt["retired_at"]
                self._write(operation_id, receipt)
            return receipt

    def clear_for_workspace_deletion(self, operation_id: str, *, workspace_id: str) -> None:
        """Remove a tombstone only as part of durable workspace deletion.

        Normal retention must use :meth:`retire`; removing a tombstone earlier
        would allow a delayed idempotent replay to become new work.
        """
        operation_id = self._operation_id(operation_id)
        with self._receipt_lock(operation_id, create=False):
            receipt = self._require(operation_id)
            if receipt["workspace_id"] != workspace_id or receipt.get("retired_at") is None:
                raise OperationConflict("Only this workspace's tombstone may be cleared")
            directory = self._operation_directory(operation_id, create=False)
            assert directory is not None
            for name in ("receipt.json", "receipt.lock", "operation.lock"):
                try:
                    os.unlink(directory / name)
                except FileNotFoundError:
                    continue
            os.rmdir(directory)

    @contextmanager
    def hold_liveness(self, operation_id: str, *, blocking: bool = False) -> Iterator[int]:
        """Hold the operation liveness flock and yield its inheritable fd.

        ``blocking=False`` is the admission default.  A caller spawning a
        mutating child must use ``pass_fds=(fd,)`` so its lifetime remains
        observable if the parent dies.
        """
        operation_id = self._operation_id(operation_id)
        with self._receipt_lock(operation_id, create=False):
            self._require(operation_id)
        fd = self._open_leaf(operation_id, "operation.lock", os.O_CREAT | os.O_RDWR, create=True)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OperationConflict("Operation liveness lock is invalid")
            flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            try:
                fcntl.flock(fd, flags)
            except BlockingIOError as exc:
                raise OperationConflict("Operation is already live") from exc
            os.set_inheritable(fd, True)
            yield fd
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def is_live(self, operation_id: str) -> bool:
        """Return liveness from an actual flock, never elapsed wall-clock time."""
        operation_id = self._operation_id(operation_id)
        with self._receipt_lock(operation_id, create=False):
            self._require(operation_id)
        fd = self._open_leaf(operation_id, "operation.lock", os.O_RDWR, create=False)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)

    @contextmanager
    def _receipt_lock(self, operation_id: str, *, create: bool) -> Iterator[None]:
        if create:
            self._operation_directory(operation_id, create=True)
        elif self._operation_directory(operation_id, create=False) is None:
            raise OperationNotFound("Operation receipt is unavailable")
        fd = self._open_leaf(operation_id, "receipt.lock", os.O_CREAT | os.O_RDWR, create=create)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OperationConflict("Operation receipt lock is invalid")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _require(self, operation_id: str) -> dict[str, Any]:
        receipt = self._read(operation_id)
        if receipt is None:
            raise OperationNotFound("Operation receipt is unavailable")
        return receipt

    def _read(self, operation_id: str) -> dict[str, Any] | None:
        try:
            fd = self._open_leaf(operation_id, "receipt.json", os.O_RDONLY, create=False)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise OperationConflict("Operation receipt is invalid") from exc
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OperationConflict("Operation receipt is invalid")
            with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as source:
                payload = json.load(source)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise OperationConflict("Operation receipt is unreadable") from exc
        finally:
            os.close(fd)
        if not isinstance(payload, dict) or payload.get("operation_id") != operation_id or payload.get("phase") not in self.PHASES:
            raise OperationConflict("Operation receipt is invalid")
        return payload

    def _write(self, operation_id: str, receipt: dict[str, Any]) -> None:
        directory = self._operation_directory(operation_id, create=False)
        assert directory is not None
        name = f".receipt-{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        directory_fd = os.open(directory, directory_flags)
        try:
            fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as output:
                    json.dump(receipt, output, sort_keys=True, separators=(",", ":"))
                    output.flush()
                    os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(name, "receipt.json", src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            os.fsync(directory_fd)
        finally:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            os.close(directory_fd)

    def _operations_directory(self, *, create: bool) -> Path | None:
        if create:
            self._root.mkdir(mode=0o700, parents=True, exist_ok=True)
            (self._root / "operations").mkdir(mode=0o700, exist_ok=True)
        try:
            root_details = self._root.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(root_details.st_mode) or not stat.S_ISDIR(root_details.st_mode):
            raise OperationConflict("Operation store root is invalid")
        path = self._root / "operations"
        try:
            details = path.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise OperationConflict("Operation store directory is invalid")
        return path

    def _operation_directory(self, operation_id: str, *, create: bool) -> Path | None:
        operations = self._operations_directory(create=create)
        if operations is None:
            return None
        path = operations / operation_id
        if create:
            path.mkdir(mode=0o700, exist_ok=True)
        try:
            details = path.lstat()
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise OperationConflict("Operation directory is invalid")
        return path

    def _open_leaf(self, operation_id: str, name: str, flags: int, *, create: bool) -> int:
        directory = self._operation_directory(operation_id, create=create)
        if directory is None:
            raise FileNotFoundError(name)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        directory_fd = os.open(directory, directory_flags)
        try:
            return os.open(name, flags, 0o600, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)

    @classmethod
    def _validate_identity(cls, workspace_id: str, creator_id: str, request_digest: str, kind: str, payload: Any) -> None:
        if not all(isinstance(value, str) and value for value in (workspace_id, creator_id, request_digest, kind)):
            raise ValueError("Invalid operation receipt fields")
        try:
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Accepted payload must be JSON-compatible") from exc

    @classmethod
    def _apply_updates(cls, receipt: dict[str, Any], updates: dict[str, Any], allowed: frozenset[str]) -> None:
        forbidden = set(updates) - allowed
        if forbidden or cls._IDENTITY_FIELDS.intersection(updates):
            raise OperationConflict("Operation update includes immutable or unauthorised fields")
        receipt.update(updates)

    @staticmethod
    def _mutable_fields_for_transition(phase: str) -> frozenset[str]:
        return OperationStore._PROGRESS_FIELDS[phase]

    @staticmethod
    def _operation_id(value: str) -> str:
        return str(UUID(value))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
