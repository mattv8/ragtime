"""Narrow, runtime-private Restic CLI adapter."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import functools
import hashlib
import json
import os
import re
import secrets
import selectors
import signal
import stat
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import BinaryIO, Callable, TypeVar

from .models import RESTIC_IMAGE_PATH, ResticArtifact


class ResticRepositoryError(RuntimeError):
    """A safe failure from the private Restic repository boundary."""


_HEX_ID = re.compile(r"^[0-9a-f]{64}$")
_SAFE_METADATA = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")
T = TypeVar("T")


class ResticRepository:
    """Restic lifecycle and image operations for one runtime storage root."""

    _MAX_OUTPUT_BYTES = 4 * 1024 * 1024
    _COMMAND_TIMEOUTS = {"init": 60, "cat": 30, "backup": 300, "dump": 300, "ls": 60, "snapshots": 60, "forget": 120, "check": 600, "prune": 600, "stats": 60}

    def __init__(
        self,
        *,
        repository_path: Path,
        cache_path: Path,
        password_path: Path,
        binary: Path = Path("/opt/ragtime-backup/bin/restic"),
        scratch_path: Path | None = None,
    ) -> None:
        self._repository_path = repository_path
        self._cache_path = cache_path
        self._password_path = password_path
        self._binary = binary
        self._scratch_path = scratch_path or repository_path.parent / "scratch"
        self._active_lock = threading.Lock()
        self._active: dict[int, subprocess.Popen[bytes]] = {}

    async def initialize(self, *, pass_fds: tuple[int, ...] = ()) -> str:
        return await self._blocking(self._initialize, pass_fds=pass_fds)

    async def ingest(
        self,
        image: Path,
        *,
        workspace_id: str,
        operation_id: str,
        sha256: str,
        size_bytes: int,
        pass_fds: tuple[int, ...] = (),
    ) -> ResticArtifact:
        return await self._blocking(self._ingest, image, workspace_id, operation_id, sha256, size_bytes, pass_fds=pass_fds)

    async def materialize(self, artifact: ResticArtifact, destination: Path, *, pass_fds: tuple[int, ...] = ()) -> None:
        await self._blocking(self._materialize, artifact, destination, pass_fds=pass_fds)

    async def verify(self, artifact: ResticArtifact, *, pass_fds: tuple[int, ...] = ()) -> None:
        """Verify every restored byte without writing a disposable image to disk."""
        await self._blocking(self._verify, artifact, pass_fds=pass_fds)

    async def forget(self, snapshot_ids: list[str], *, pass_fds: tuple[int, ...] = ()) -> None:
        await self._blocking(self._forget, snapshot_ids, pass_fds=pass_fds)

    async def check(self, *, read_data: bool, pass_fds: tuple[int, ...] = ()) -> None:
        await self._blocking(self._check, read_data, pass_fds=pass_fds)

    async def list_operation_snapshot_ids(self, *, workspace_id: str, operation_id: str, pass_fds: tuple[int, ...] = ()) -> list[str]:
        return await self._blocking(self._list_operation_snapshot_ids, workspace_id, operation_id, pass_fds=pass_fds)

    async def prune(self, *, max_repack_size: int, pass_fds: tuple[int, ...] = ()) -> None:
        await self._blocking(self._prune, max_repack_size, pass_fds=pass_fds)

    async def stats(self, *, pass_fds: tuple[int, ...] = ()) -> dict[str, object]:
        return await self._blocking(self._stats, pass_fds=pass_fds)

    async def _blocking(self, function: Callable[..., T], *args: object, pass_fds: tuple[int, ...] = ()) -> T:
        cancelled = threading.Event()
        work_function: Callable[[], T] = functools.partial(function, *args, cancelled, pass_fds)
        work = asyncio.create_task(asyncio.to_thread(work_function))
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            cancelled.set()
            terminate = asyncio.create_task(asyncio.to_thread(self._terminate, cancelled))
            # Killing the current child alone is insufficient: the worker may
            # still be copying/verifying an image or have not spawned yet. Drain
            # the entire worker before its caller can release storage fences.
            while not terminate.done():
                try:
                    await asyncio.shield(terminate)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if terminate.done() and not terminate.cancelled():
                terminate.exception()
            while not work.done():
                try:
                    await asyncio.shield(work)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if work.done() and not work.cancelled():
                work.exception()
            raise

    def _initialize(self, cancelled: threading.Event, pass_fds: tuple[int, ...]) -> str:
        self._ensure_safe_directory(self._repository_path.parent)
        lock_path = self._repository_path.parent / ".restic-init.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self._repository_path.is_symlink():
                raise ResticRepositoryError("Restic repository path is invalid")
            repository_exists = self._repository_path.exists()
            self._ensure_safe_directory(self._cache_path)
            self._ensure_password(repository_exists)
            if not repository_exists:
                self._run("init", "--repository-version", "2", cancelled=cancelled, pass_fds=pass_fds)
            config = self._run("cat", "config", cancelled=cancelled, pass_fds=pass_fds)
        try:
            repository_id = str(json.loads(config).get("id") or "")
        except json.JSONDecodeError as exc:
            raise ResticRepositoryError("Restic repository configuration is invalid") from exc
        if not _HEX_ID.fullmatch(repository_id):
            raise ResticRepositoryError("Restic repository identity is unavailable")
        return repository_id

    def _ingest(
        self, image: Path, workspace_id: str, operation_id: str, sha256: str, size_bytes: int, cancelled: threading.Event, pass_fds: tuple[int, ...]
    ) -> ResticArtifact:
        self._validate_metadata(workspace_id, operation_id)
        self._validate_digest(sha256, size_bytes)
        if image.is_symlink() or not image.is_file():
            raise ResticRepositoryError("Capture image is unavailable")
        repository_id = self._initialize(cancelled, pass_fds)
        self._ensure_safe_directory(self._scratch_path)
        with tempfile.TemporaryDirectory(prefix="restic-ingest-", dir=self._scratch_path) as scratch_name:
            controlled = Path(scratch_name) / "database.sqlite3"
            self._copy_verified_image(image, controlled, sha256, size_bytes)
            output = self._run(
                "backup",
                "--json",
                "--host",
                "ragtime-runtime",
                "--tag",
                f"workspace:{workspace_id}",
                "--tag",
                f"operation:{operation_id}",
                "database.sqlite3",
                cwd=Path(scratch_name),
                cancelled=cancelled,
                pass_fds=pass_fds,
            )
        snapshot_id = self._snapshot_id(output)
        self._validate_snapshot_file(snapshot_id, size_bytes, cancelled, pass_fds)
        return ResticArtifact(repository_id, snapshot_id, RESTIC_IMAGE_PATH, size_bytes, sha256)

    def _materialize(self, artifact: ResticArtifact, destination: Path, cancelled: threading.Event, pass_fds: tuple[int, ...]) -> None:
        self._validate_artifact(artifact)
        if self._initialize(cancelled, pass_fds) != artifact.repository_id:
            raise ResticRepositoryError("Stored image belongs to another repository")
        self._ensure_safe_directory(destination.parent)
        fd, temporary_name = tempfile.mkstemp(prefix="restic-materialize-", dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as output:
                self._run(
                    "dump", artifact.snapshot_id, RESTIC_IMAGE_PATH, stdout=output, max_stdout_bytes=artifact.size_bytes, cancelled=cancelled, pass_fds=pass_fds
                )
                output.flush()
                os.fsync(output.fileno())
            temporary = Path(temporary_name)
            if temporary.stat().st_size != artifact.size_bytes or self._sha256(temporary) != artifact.sha256:
                raise ResticRepositoryError("Restic materialization verification failed")
            os.replace(temporary, destination)
            self._fsync_directory(destination.parent)
        finally:
            Path(temporary_name).unlink(missing_ok=True)

    def _forget(self, snapshot_ids: list[str], cancelled: threading.Event, pass_fds: tuple[int, ...]) -> None:
        if not snapshot_ids:
            return
        if len(snapshot_ids) > 1000 or any(not _HEX_ID.fullmatch(value) for value in snapshot_ids):
            raise ResticRepositoryError("Invalid Restic snapshot identifier")
        self._run("forget", *snapshot_ids, cancelled=cancelled, pass_fds=pass_fds)

    def _verify(self, artifact: ResticArtifact, cancelled: threading.Event, pass_fds: tuple[int, ...]) -> None:
        self._validate_artifact(artifact)
        if self._initialize(cancelled, pass_fds) != artifact.repository_id:
            raise ResticRepositoryError("Stored image belongs to another repository")
        self._run(
            "dump",
            artifact.snapshot_id,
            RESTIC_IMAGE_PATH,
            verify_stdout=(artifact.sha256, artifact.size_bytes),
            max_stdout_bytes=artifact.size_bytes,
            cancelled=cancelled,
            pass_fds=pass_fds,
        )

    def _check(self, read_data: bool, cancelled: threading.Event, pass_fds: tuple[int, ...]) -> None:
        arguments = ("check", "--read-data") if read_data else ("check",)
        self._run(*arguments, cancelled=cancelled, pass_fds=pass_fds)

    def _list_operation_snapshot_ids(self, workspace_id: str, operation_id: str, cancelled: threading.Event, pass_fds: tuple[int, ...]) -> list[str]:
        self._validate_metadata(workspace_id, operation_id)
        output = self._run("snapshots", "--json", "--tag", f"workspace:{workspace_id},operation:{operation_id}", cancelled=cancelled, pass_fds=pass_fds)
        try:
            snapshots = json.loads(output)
        except json.JSONDecodeError as exc:
            raise ResticRepositoryError("Restic snapshot listing is invalid") from exc
        if not isinstance(snapshots, list):
            raise ResticRepositoryError("Restic snapshot listing is invalid")
        identifiers: list[str] = []
        for entry in snapshots:
            if not isinstance(entry, dict):
                continue
            if not {f"workspace:{workspace_id}", f"operation:{operation_id}"}.issubset(set(entry.get("tags") or [])):
                continue
            identifier = entry.get("id")
            if not isinstance(identifier, str) or not _HEX_ID.fullmatch(identifier):
                raise ResticRepositoryError("Restic snapshot listing is invalid")
            identifiers.append(identifier)
        return identifiers

    def _prune(self, max_repack_size: int, cancelled: threading.Event, pass_fds: tuple[int, ...]) -> None:
        if not 0 < max_repack_size <= 1024**4:
            raise ResticRepositoryError("Invalid Restic prune limit")
        self._run("prune", "--max-repack-size", str(max_repack_size), cancelled=cancelled, pass_fds=pass_fds)

    def _stats(self, cancelled: threading.Event, pass_fds: tuple[int, ...]) -> dict[str, object]:
        output = self._run("stats", "--json", cancelled=cancelled, pass_fds=pass_fds)
        try:
            result = json.loads(output)
        except json.JSONDecodeError as exc:
            raise ResticRepositoryError("Restic statistics are invalid") from exc
        if not isinstance(result, dict):
            raise ResticRepositoryError("Restic statistics are invalid")
        return result

    def _ensure_password(self, repository_exists: bool) -> None:
        self._ensure_safe_directory(self._password_path.parent)
        if self._password_path.exists():
            details = self._password_path.lstat()
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISREG(details.st_mode) or stat.S_IMODE(details.st_mode) != 0o600:
                raise ResticRepositoryError("Restic repository key is invalid")
            return
        if repository_exists:
            raise ResticRepositoryError("Restic repository key is unavailable")
        fd, temporary_name = tempfile.mkstemp(prefix=".repository-password-", dir=self._password_path.parent)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(secrets.token_urlsafe(48) + "\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary_name, self._password_path)
            self._fsync_directory(self._password_path.parent)
        finally:
            Path(temporary_name).unlink(missing_ok=True)

    def _run(
        self,
        *arguments: str,
        cancelled: threading.Event,
        pass_fds: tuple[int, ...],
        cwd: Path | None = None,
        stdout: BinaryIO | None = None,
        max_stdout_bytes: int | None = None,
        verify_stdout: tuple[str, int] | None = None,
    ) -> str:
        if not arguments or arguments[0] not in self._COMMAND_TIMEOUTS:
            raise ResticRepositoryError("Unsupported Restic command")
        command = [
            str(self._binary),
            "--repo",
            str(self._repository_path),
            "--cache-dir",
            str(self._cache_path),
            "--password-file",
            str(self._password_path),
            *arguments,
        ]
        environment = {"HOME": "/nonexistent", "PATH": "/usr/bin:/bin", "RESTIC_PROGRESS_FPS": "0"}
        stdout_bytes = 0
        digest = hashlib.sha256() if verify_stdout is not None else None
        selector: selectors.BaseSelector | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=pass_fds,
                start_new_session=True,
            )
        except OSError as exc:
            raise ResticRepositoryError("Restic command failed") from exc
        key = id(cancelled)
        with self._active_lock:
            self._active[key] = process
        captured = bytearray()
        deadline = time.monotonic() + self._COMMAND_TIMEOUTS[arguments[0]]
        try:
            selector = selectors.DefaultSelector()
            assert process.stdout is not None and process.stderr is not None
            for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            while selector.get_map():
                if cancelled.is_set() or time.monotonic() >= deadline:
                    self._terminate_process(process)
                    raise ResticRepositoryError("Restic command cancelled" if cancelled.is_set() else "Restic command timed out")
                for event, _ in selector.select(min(0.1, max(0.0, deadline - time.monotonic()))):
                    descriptor = event.fileobj if isinstance(event.fileobj, int) else event.fileobj.fileno()
                    block = os.read(descriptor, 65536)
                    if not block:
                        selector.unregister(event.fileobj)
                    elif event.data == "stdout" and (stdout is not None or digest is not None):
                        stdout_bytes += len(block)
                        if max_stdout_bytes is not None and stdout_bytes > max_stdout_bytes:
                            self._terminate_process(process)
                            raise ResticRepositoryError("Restic command output exceeded limit")
                        if stdout is not None:
                            stdout.write(block)
                        if digest is not None:
                            digest.update(block)
                    else:
                        if len(captured) + len(block) > self._MAX_OUTPUT_BYTES:
                            self._terminate_process(process)
                            raise ResticRepositoryError("Restic command output exceeded limit")
                        captured.extend(block)
            returncode = process.wait(timeout=5)
            if returncode == 3 and arguments[0] == "backup":
                raise ResticRepositoryError("Restic backup completed partially")
            if returncode != 0:
                raise ResticRepositoryError("Restic command failed")
            if verify_stdout is not None and digest is not None:
                if (digest.hexdigest(), stdout_bytes) != verify_stdout:
                    raise ResticRepositoryError("Restic verification failed")
                return ""
            if stdout is not None:
                return ""
            return captured.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise ResticRepositoryError("Restic command returned invalid output") from exc
        finally:
            with contextlib.suppress(Exception):
                if selector is not None:
                    selector.close()
            if process.poll() is None:
                self._terminate_process(process)
            with self._active_lock:
                self._active.pop(key, None)

    def _validate_snapshot_file(self, snapshot_id: str, expected_size: int, cancelled: threading.Event, pass_fds: tuple[int, ...]) -> None:
        if not _HEX_ID.fullmatch(snapshot_id):
            raise ResticRepositoryError("Restic backup did not produce a complete snapshot")
        output = self._run("ls", "--json", snapshot_id, cancelled=cancelled, pass_fds=pass_fds)
        for line in output.splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResticRepositoryError("Restic snapshot listing is invalid") from exc
            if (
                payload.get("struct_type") == "node"
                and payload.get("type") == "file"
                and payload.get("path") == RESTIC_IMAGE_PATH
                and payload.get("size") == expected_size
            ):
                return
        raise ResticRepositoryError("Restic snapshot is missing the capture image")

    @staticmethod
    def _snapshot_id(output: str) -> str:
        for line in output.splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResticRepositoryError("Restic backup returned invalid JSON") from exc
            if payload.get("message_type") == "summary":
                snapshot_id = payload.get("snapshot_id")
                if isinstance(snapshot_id, str) and _HEX_ID.fullmatch(snapshot_id):
                    return snapshot_id
        raise ResticRepositoryError("Restic backup did not produce a complete snapshot")

    @staticmethod
    def _validate_metadata(workspace_id: str, operation_id: str) -> None:
        if not _SAFE_METADATA.fullmatch(workspace_id) or not _SAFE_METADATA.fullmatch(operation_id):
            raise ResticRepositoryError("Invalid runtime operation metadata")

    @staticmethod
    def _validate_digest(sha256: str, size_bytes: int) -> None:
        if not _HEX_ID.fullmatch(sha256) or size_bytes < 0:
            raise ResticRepositoryError("Capture image verification failed")

    @classmethod
    def _validate_artifact(cls, artifact: ResticArtifact) -> None:
        if artifact.path != RESTIC_IMAGE_PATH or not _HEX_ID.fullmatch(artifact.repository_id) or not _HEX_ID.fullmatch(artifact.snapshot_id):
            raise ResticRepositoryError("Invalid stored image reference")
        cls._validate_digest(artifact.sha256, artifact.size_bytes)

    @staticmethod
    def _copy_verified_image(source: Path, destination: Path, sha256: str, size_bytes: int) -> None:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(source, flags)
        except OSError as exc:
            raise ResticRepositoryError("Capture image is unavailable") from exc
        digest = hashlib.sha256()
        count = 0
        try:
            with os.fdopen(fd, "rb", closefd=True) as input_file:
                if not stat.S_ISREG(os.fstat(input_file.fileno()).st_mode):
                    raise ResticRepositoryError("Capture image is unavailable")
                with destination.open("xb") as output:
                    while block := input_file.read(1024 * 1024):
                        digest.update(block)
                        count += len(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
        except OSError as exc:
            destination.unlink(missing_ok=True)
            raise ResticRepositoryError("Capture image is unavailable") from exc
        if count != size_bytes or digest.hexdigest() != sha256:
            destination.unlink(missing_ok=True)
            raise ResticRepositoryError("Capture image verification failed")

    @staticmethod
    def _ensure_safe_directory(directory: Path) -> None:
        directory = directory.absolute()
        current = Path(directory.anchor)
        for component in directory.parts[1:]:
            current /= component
            try:
                details = current.lstat()
            except FileNotFoundError:
                current.mkdir(mode=0o700)
                details = current.lstat()
            if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
                raise ResticRepositoryError("Runtime storage path is invalid")

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _terminate(self, cancelled: threading.Event) -> None:
        with self._active_lock:
            process = self._active.get(id(cancelled))
        if process is not None:
            self._terminate_process(process)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=2)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
