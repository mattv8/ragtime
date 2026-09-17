"""Durable, descriptor-safe staging of the retired workspace S3 tree."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import stat
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator

from fastapi import HTTPException

from ragtime.core.server_backup import locked_operation
from ragtime.userspace.object_storage import control

_GENERATION = re.compile(r"^[0-9a-f]{32}$")
_WORKSPACE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CHUNK_SIZE = 1024 * 1024
_workspace_fences: dict[str, asyncio.Lock] = {}
_logger = logging.getLogger(__name__)


def workspace_gc_fence(workspace_id: str) -> asyncio.Lock:
    """Shared in-process admission fence for runtime starts and source GC."""
    return _workspace_fences.setdefault(workspace_id, asyncio.Lock())


class LegacyMigrationError(RuntimeError):
    """A legacy source or durable migration record is unsafe."""


def _safe_relative(value: str) -> PurePosixPath:
    if not value or value.startswith("/") or value.endswith("/") or "//" in value:
        raise LegacyMigrationError("unsafe legacy manifest path")
    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        raise LegacyMigrationError("unsafe legacy manifest path")
    return PurePosixPath(*parts)


def _fsync(fd: int) -> None:
    os.fsync(fd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short filesystem write")
        view = view[written:]


@contextmanager
def _open_absolute_dir(path: Path, *, create: bool = False) -> Iterator[int]:
    """Walk an absolute directory path without ever following a symlink."""
    if not path.is_absolute():
        path = path.absolute()
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            try:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, dir_fd=fd)
                _fsync(fd)
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _open_relative_dir(root_fd: int, parts: tuple[str, ...], *, create: bool = False) -> Iterator[int]:
    fd = os.dup(root_fd)
    try:
        for part in parts:
            try:
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, dir_fd=fd)
                _fsync(fd)
                next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


def _directory_identity(fd: int) -> dict[str, int]:
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode):
        raise LegacyMigrationError("legacy source root is not a directory")
    return {"device": info.st_dev, "inode": info.st_ino}


def _fd_identity(fd: int) -> dict[str, Any]:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise LegacyMigrationError("legacy source is not a regular file")
    hasher = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while chunk := os.read(fd, _CHUNK_SIZE):
        hasher.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return {"device": info.st_dev, "inode": info.st_ino, "size": info.st_size, "mtime_ns": info.st_mtime_ns, "sha256": hasher.hexdigest()}


@contextmanager
def _open_source_file(root_fd: int, relative: PurePosixPath) -> Iterator[tuple[int, int, str]]:
    with _open_relative_dir(root_fd, relative.parts[:-1]) as parent_fd:
        file_fd = os.open(relative.parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
        try:
            yield parent_fd, file_fd, relative.parts[-1]
        finally:
            os.close(file_fd)


def _remove_tree(parent_fd: int, name: str) -> None:
    """Remove only a known, descriptor-pinned staging directory."""
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        for child in os.listdir(fd):
            info = os.stat(child, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                _remove_tree(fd, child)
            elif stat.S_ISREG(info.st_mode):
                os.unlink(child, dir_fd=fd)
            else:
                raise LegacyMigrationError("unsafe staging generation entry")
    finally:
        os.close(fd)
    os.rmdir(name, dir_fd=parent_fd)


class LegacyObjectStorageMigrator:
    def __init__(self, storage_root: Path, workspace_root: Callable[[str], Path]) -> None:
        self.storage_root = storage_root
        self.workspace_root = workspace_root

    def source_buckets(self, workspace_id: str) -> Path:
        self._validate_workspace_id(workspace_id)
        return self.workspace_root(workspace_id) / "s3" / "buckets"

    def receipt_dir(self, workspace_id: str) -> Path:
        self._validate_workspace_id(workspace_id)
        return self.storage_root / "_legacy_imports" / workspace_id

    @staticmethod
    def _validate_workspace_id(workspace_id: str) -> None:
        if not _WORKSPACE_ID.fullmatch(workspace_id):
            raise LegacyMigrationError("unsafe workspace id")

    def _receipt_path(self, workspace_id: str, generation: str) -> Path:
        if not _GENERATION.fullmatch(generation):
            raise LegacyMigrationError("unsafe staging generation")
        return self.receipt_dir(workspace_id) / f"receipt-{generation}.json"

    @contextmanager
    def _source_root(self, workspace_id: str) -> Iterator[int]:
        try:
            with _open_absolute_dir(self.source_buckets(workspace_id)) as fd:
                yield fd
        except OSError as exc:
            raise LegacyMigrationError("legacy object-storage buckets directory is unsafe or missing") from exc

    def _receipt_root(self, workspace_id: str, *, create: bool = False) -> AbstractContextManager[int]:
        return _open_absolute_dir(self.receipt_dir(workspace_id), create=create)

    def _write_json_atomic(self, path: Path, value: dict[str, Any]) -> None:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        with _open_absolute_dir(path.parent, create=True) as parent_fd:
            tmp = f".{path.name}.{secrets.token_hex(8)}.tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent_fd)
            try:
                _write_all(fd, payload)
                _fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            _fsync(parent_fd)

    def _validate_receipt(self, workspace_id: str, receipt: dict[str, Any]) -> None:
        generation = receipt.get("generation")
        manifest = receipt.get("manifest")
        if (
            receipt.get("version") != 1
            or receipt.get("workspace_id") != workspace_id
            or not isinstance(generation, str)
            or not _GENERATION.fullmatch(generation)
            or not isinstance(manifest, dict)
            or manifest.get("version") != 1
            or manifest.get("workspace_id") != workspace_id
            or manifest.get("generation") != generation
            or receipt.get("cleanup_state") not in {"published", "verified", "source_gc_completed", "consumed"}
        ):
            raise LegacyMigrationError("invalid legacy migration receipt")
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if receipt.get("manifest_sha256") != hashlib.sha256(encoded).hexdigest():
            raise LegacyMigrationError("legacy receipt manifest digest mismatch")
        files = manifest.get("files")
        if not isinstance(files, list):
            raise LegacyMigrationError("invalid legacy receipt manifest")
        paths: set[str] = set()
        manifest_by_path: dict[str, dict[str, Any]] = {}
        for item in files:
            if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
                raise LegacyMigrationError("invalid legacy receipt manifest")
            raw = item["path"]
            if not isinstance(raw, str) or _safe_relative(raw).as_posix() != raw or not raw.startswith("buckets/"):
                raise LegacyMigrationError("invalid legacy receipt manifest path")
            if (
                raw in paths
                or not isinstance(item["size"], int)
                or item["size"] < 0
                or not isinstance(item["sha256"], str)
                or not _SHA256.fullmatch(item["sha256"])
            ):
                raise LegacyMigrationError("invalid legacy receipt manifest")
            paths.add(raw)
            manifest_by_path[raw] = item
        source_files = receipt.get("source_files")
        if not isinstance(source_files, dict) or not set(source_files).issubset(paths):
            raise LegacyMigrationError("invalid legacy source identity")
        for raw, identity in source_files.items():
            if (
                not isinstance(identity, dict)
                or set(identity) != {"device", "inode", "size", "mtime_ns", "sha256"}
                or not _SHA256.fullmatch(str(identity.get("sha256", "")))
            ):
                raise LegacyMigrationError("invalid legacy source identity")
            if any(not isinstance(identity[key], int) or identity[key] < 0 for key in ("device", "inode", "size", "mtime_ns")):
                raise LegacyMigrationError("invalid legacy source identity")
            if identity["size"] != manifest_by_path[raw]["size"] or identity["sha256"] != manifest_by_path[raw]["sha256"]:
                raise LegacyMigrationError("source identity does not match manifest")
        if not isinstance(receipt.get("source_root"), dict) or set(receipt["source_root"]) != {"device", "inode"}:
            raise LegacyMigrationError("invalid legacy source root identity")
        verified = receipt.get("verified_files", [])
        if (
            not isinstance(verified, list)
            or len(set(verified)) != len(verified)
            or not set(verified).issubset(paths)
            or not set(verified).issubset(source_files)
        ):
            raise LegacyMigrationError("invalid cleanup evidence")

    def _load_receipts(self, workspace_id: str) -> list[dict[str, Any]]:
        self._validate_workspace_id(workspace_id)
        try:
            with self._receipt_root(workspace_id) as fd:
                names = os.listdir(fd)
                receipts: list[dict[str, Any]] = []
                for name in sorted(names):
                    if not name.startswith("receipt-") or not name.endswith(".json"):
                        continue
                    generation = name[len("receipt-") : -len(".json")]
                    if not _GENERATION.fullmatch(generation):
                        raise LegacyMigrationError("unsafe receipt filename")
                    try:
                        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
                        with os.fdopen(file_fd, "r", encoding="utf-8") as handle:
                            receipt = json.load(handle)
                    except (OSError, ValueError) as exc:
                        raise LegacyMigrationError("unreadable legacy migration receipt") from exc
                    if not isinstance(receipt, dict):
                        raise LegacyMigrationError("invalid legacy migration receipt")
                    self._validate_receipt(workspace_id, receipt)
                    receipts.append(receipt)
                return receipts
        except FileNotFoundError:
            return []

    def _reclaim_tmp_generations(self, workspace_id: str) -> None:
        """Discard only canonical unpublished directories while holding backup flock."""
        try:
            with locked_operation(), self._receipt_root(workspace_id) as receipt_fd:
                for name in os.listdir(receipt_fd):
                    generation = name.removeprefix("tmp-")
                    if not name.startswith("tmp-") or not _GENERATION.fullmatch(generation):
                        continue
                    info = os.stat(name, dir_fd=receipt_fd, follow_symlinks=False)
                    if not stat.S_ISDIR(info.st_mode):
                        continue
                    _remove_tree(receipt_fd, name)
                    _fsync(receipt_fd)
        except FileNotFoundError:
            return

    def _unreceipted_generations(self, workspace_id: str, receipts: list[dict[str, Any]]) -> list[str]:
        known = {str(receipt["generation"]) for receipt in receipts}
        try:
            with self._receipt_root(workspace_id) as receipt_fd:
                result: list[str] = []
                for name in os.listdir(receipt_fd):
                    if name in known or not _GENERATION.fullmatch(name):
                        continue
                    info = os.stat(name, dir_fd=receipt_fd, follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        result.append(name)
                return sorted(result)
        except FileNotFoundError:
            return []

    def _reclaim_published_generation(self, workspace_id: str, generation: str) -> None:
        if not _GENERATION.fullmatch(generation):
            raise LegacyMigrationError("unsafe staging generation")
        with locked_operation(), self._receipt_root(workspace_id) as receipt_fd:
            try:
                _remove_tree(receipt_fd, generation)
                _fsync(receipt_fd)
            except FileNotFoundError:
                pass

    async def _reconcile_unreceipted_generations(self, workspace_id: str, generations: list[str]) -> bool | None:
        """Return serving state when a gateway job binds an orphan generation."""
        if not generations:
            return None
        try:
            status = await control.get_legacy_import(workspace_id)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            status = {}
        bound = str(status.get("generation") or "")
        if bound in generations:
            state = status.get("state")
            _logger.warning("retaining published legacy generation without receipt", extra={"workspace_id": workspace_id, "generation": bound, "state": state})
            if state == "completed":
                return True
            return False
        # A missing job proves no generation is bound.  A different generation is
        # safe only after its GC acknowledgement, when it is a consumed identity.
        if status and not bool(status.get("gc_completed")):
            _logger.warning(
                "retaining unreceipted legacy generation while another gateway job is active", extra={"workspace_id": workspace_id, "generation": bound}
            )
            return False
        for generation in generations:
            await self._filesystem(self._reclaim_published_generation, workspace_id, generation)
        return None

    def _walk_source(self, root_fd: int) -> list[tuple[PurePosixPath, dict[str, Any]]]:
        items: list[tuple[PurePosixPath, dict[str, Any]]] = []

        def walk(fd: int, prefix: tuple[str, ...]) -> None:
            for name in sorted(os.listdir(fd)):
                if not name or "/" in name:
                    raise LegacyMigrationError("unsafe legacy source filename")
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                relative = PurePosixPath("buckets", *prefix, name)
                if stat.S_ISDIR(info.st_mode):
                    child_fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        walk(child_fd, (*prefix, name))
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(info.st_mode):
                    file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
                    try:
                        items.append((relative, _fd_identity(file_fd)))
                    finally:
                        os.close(file_fd)
                else:
                    raise LegacyMigrationError("legacy source contains non-regular entry")

        walk(root_fd, ())
        return items

    def stage(self, workspace_id: str) -> dict[str, Any]:
        """Copy source bytes through pinned no-follow descriptors and publish once."""
        self._validate_workspace_id(workspace_id)
        with locked_operation(), self._source_root(workspace_id) as source_fd, self._receipt_root(workspace_id, create=True) as receipt_fd:
            source_root = _directory_identity(source_fd)
            files = self._walk_source(source_fd)
            source_bytes = sum(item[1]["size"] for item in files)
            usage_path = self.storage_root if self.storage_root.exists() else self.storage_root.parent
            if os.statvfs(usage_path).f_bavail * os.statvfs(usage_path).f_frsize < int(source_bytes * 2.25):
                raise LegacyMigrationError("insufficient free space for legacy object-storage staging")
            generation = secrets.token_hex(16)
            tmp_name, published_name = f"tmp-{generation}", generation
            os.mkdir(tmp_name, dir_fd=receipt_fd)
            _fsync(receipt_fd)
            try:
                with _open_relative_dir(receipt_fd, (tmp_name,), create=False) as tmp_fd:
                    manifest_files: list[dict[str, Any]] = []
                    source_files: dict[str, dict[str, Any]] = {}
                    for relative, expected in files:
                        with _open_source_file(source_fd, PurePosixPath(*relative.parts[1:])) as (_, input_fd, _):
                            if _fd_identity(input_fd) != expected:
                                raise LegacyMigrationError("legacy source changed during staging")
                            with _open_relative_dir(tmp_fd, relative.parts[:-1], create=True) as output_parent:
                                output_fd = os.open(relative.parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=output_parent)
                                try:
                                    hasher = hashlib.sha256()
                                    while chunk := os.read(input_fd, _CHUNK_SIZE):
                                        hasher.update(chunk)
                                        _write_all(output_fd, chunk)
                                    _fsync(output_fd)
                                finally:
                                    os.close(output_fd)
                                _fsync(output_parent)
                            if hasher.hexdigest() != expected["sha256"] or _fd_identity(input_fd) != expected:
                                raise LegacyMigrationError("legacy source changed during staging")
                        key = relative.as_posix()
                        manifest_files.append({"path": key, "size": expected["size"], "sha256": expected["sha256"]})
                        source_files[key] = expected
                    manifest = {"version": 1, "workspace_id": workspace_id, "generation": generation, "files": manifest_files}
                    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
                    manifest_fd = os.open("manifest.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=tmp_fd)
                    try:
                        _write_all(manifest_fd, manifest_bytes)
                        _fsync(manifest_fd)
                    finally:
                        os.close(manifest_fd)
                    _fsync(tmp_fd)
                os.rename(tmp_name, published_name, src_dir_fd=receipt_fd, dst_dir_fd=receipt_fd)
                _fsync(receipt_fd)
                receipt = {
                    "version": 1,
                    "workspace_id": workspace_id,
                    "generation": generation,
                    "manifest": manifest,
                    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                    "source_files": source_files,
                    "source_root": source_root,
                    "cleanup_state": "published",
                }
                self._write_json_atomic(self._receipt_path(workspace_id, generation), receipt)
                return receipt
            except Exception:
                try:
                    _remove_tree(receipt_fd, tmp_name)
                    _fsync(receipt_fd)
                except (FileNotFoundError, OSError, LegacyMigrationError):
                    pass
                raise

    async def _filesystem(self, function: Callable[..., Any], *args: Any) -> Any:
        """Do all blocking I/O off-loop, draining a thread through repeated cancel."""
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        cancelled = False
        while True:
            try:
                result = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                cancelled = True
                continue
        if cancelled:
            raise asyncio.CancelledError
        return result

    def _record_verified(self, workspace_id: str, receipt: dict[str, Any], verified: list[str]) -> dict[str, Any]:
        self._validate_receipt(workspace_id, receipt)
        if receipt.get("cleanup_state") in {"source_gc_completed", "consumed"}:
            return receipt
        if not isinstance(verified, list) or len(set(verified)) != len(verified):
            raise LegacyMigrationError("gateway returned invalid cleanup evidence")
        paths = {item["path"] for item in receipt["manifest"]["files"]}
        if not set(verified).issubset(paths) or not set(verified).issubset(receipt["source_files"]):
            raise LegacyMigrationError("gateway returned invalid cleanup evidence")
        receipt["verified_files"] = verified
        receipt["cleanup_state"] = "verified"
        with locked_operation():
            self._write_json_atomic(self._receipt_path(workspace_id, str(receipt["generation"])), receipt)
        return receipt

    def _gc_and_record(self, workspace_id: str, receipt: dict[str, Any]) -> dict[str, Any]:
        with locked_operation():
            self._gc_verified(workspace_id, receipt)
            receipt["cleanup_state"] = "source_gc_completed"
            self._write_json_atomic(self._receipt_path(workspace_id, str(receipt["generation"])), receipt)
        return receipt

    def _consume(self, workspace_id: str, receipt: dict[str, Any]) -> None:
        self._validate_receipt(workspace_id, receipt)
        with locked_operation(), self._receipt_root(workspace_id) as receipt_fd:
            try:
                _remove_tree(receipt_fd, str(receipt["generation"]))
                _fsync(receipt_fd)
            except FileNotFoundError:
                pass
            receipt["cleanup_state"] = "consumed"
            self._write_json_atomic(self._receipt_path(workspace_id, str(receipt["generation"])), receipt)

    async def reconcile(self, workspace_id: str, *, runtime_active: bool = False) -> bool:
        await self._filesystem(self._reclaim_tmp_generations, workspace_id)
        receipts = await self._filesystem(self._load_receipts, workspace_id)
        if any(item.get("cleanup_state") == "consumed" for item in receipts):
            return True
        unreceipted = await self._filesystem(self._unreceipted_generations, workspace_id, receipts)
        orphan_state = await self._reconcile_unreceipted_generations(workspace_id, unreceipted)
        if orphan_state is not None:
            return orphan_state
        receipt = next((item for item in receipts if item.get("cleanup_state") != "consumed"), None)
        if receipt is None:
            receipt = await self._filesystem(self.stage, workspace_id)
        generation, digest = str(receipt["generation"]), str(receipt["manifest_sha256"])
        if receipt.get("cleanup_state") == "published":
            await control.submit_legacy_import(workspace_id, generation, digest)
        status = await control.get_legacy_import(workspace_id)
        if status.get("state") != "completed" or str(status.get("generation") or "") != generation or str(status.get("manifest_sha256") or "") != digest:
            return False
        if status.get("gc_completed"):
            await self._filesystem(self._consume, workspace_id, receipt)
            return True
        verified = status.get("verified_files")
        if isinstance(verified, list) and receipt.get("verified_files") != verified:
            receipt = await self._filesystem(self._record_verified, workspace_id, receipt, verified)
        if runtime_active or receipt.get("cleanup_state") == "published":
            return True
        if receipt.get("cleanup_state") == "verified":
            receipt = await self._filesystem(self._gc_and_record, workspace_id, receipt)
        if receipt.get("cleanup_state") == "source_gc_completed":
            await control.acknowledge_legacy_gc(workspace_id, generation, digest)
            await self._filesystem(self._consume, workspace_id, receipt)
        return True

    def _gc_verified(self, workspace_id: str, receipt: dict[str, Any]) -> None:
        self._validate_receipt(workspace_id, receipt)
        retained = receipt.setdefault("retained_files", {})
        try:
            with self._source_root(workspace_id) as source_fd:
                if _directory_identity(source_fd) != receipt["source_root"]:
                    for raw in receipt["verified_files"]:
                        retained[raw] = "source root replaced"
                    return
                for raw in receipt["verified_files"]:
                    relative = _safe_relative(raw)
                    expected = receipt["source_files"][raw]
                    try:
                        with _open_source_file(source_fd, PurePosixPath(*relative.parts[1:])) as (parent_fd, file_fd, name):
                            if _fd_identity(file_fd) != expected:
                                retained[raw] = "source changed"
                                continue
                            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                            opened = os.fstat(file_fd)
                            if current.st_dev != opened.st_dev or current.st_ino != opened.st_ino:
                                retained[raw] = "source entry replaced"
                                continue
                            os.unlink(name, dir_fd=parent_fd)
                    except FileNotFoundError:
                        retained[raw] = "source missing"
                    except OSError as exc:
                        retained[raw] = f"source retained: {exc.__class__.__name__}"
        except (FileNotFoundError, OSError) as exc:
            for raw in receipt["verified_files"]:
                retained.setdefault(raw, f"source root unavailable: {exc.__class__.__name__}")
        if retained:
            _logger.warning("legacy source artifacts retained after verification", extra={"workspace_id": workspace_id, "retained_files": retained})
