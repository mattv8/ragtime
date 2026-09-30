"""Bounded, SFTP-only file transfers for already-authorized SSH endpoints.

Standard SFTP rename semantics provide create-only publication. Concurrent path
mutation and an interrupted in-flight rename remain server-defined outcomes.
"""

from __future__ import annotations

import posixpath
import secrets
import stat
import threading
import time
from dataclasses import replace
from errno import ECONNABORTED, ECONNRESET, EHOSTDOWN, EHOSTUNREACH, ENETDOWN, ENETRESET, ENETUNREACH, ENOENT, EPIPE, ETIMEDOUT
from pathlib import PurePosixPath
from typing import Any, Iterator, cast

import paramiko

from ragtime.core.ssh import SSHConfig, _create_ssh_client

CHUNK_SIZE = 256 * 1024
_CLEANUP_TIMEOUT = 1.0


class _TransferError(Exception):
    pass


class _PolicyViolation(_TransferError):
    pass


class _TransferCancelled(_TransferError):
    pass


class _PublishOutcomeUnknown(_TransferError):
    pass


def _result(status: str = "ok") -> dict[str, Any]:
    return {"status": status, "bytes_transferred": 0, "files_transferred": 0, "errors": [], "skipped": []}


def _message(exc: Exception) -> str:
    if isinstance(exc, _TransferError):
        return str(exc)
    if isinstance(exc, TimeoutError):
        return "Transfer timed out."
    return "SSH transfer failed."


def _check_active(deadline: float, cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise _TransferCancelled("Transfer cancelled.")
    if time.monotonic() >= deadline:
        raise TimeoutError()


def _set_sftp_timeout(sftp: Any, deadline: float) -> None:
    channel = getattr(sftp, "get_channel", lambda: None)()
    if channel is not None and hasattr(channel, "settimeout"):
        channel.settimeout(max(0.1, deadline - time.monotonic()))


def _check_io(deadline: float, cancel_event: threading.Event | None, *sftps: Any | None) -> None:
    _check_active(deadline, cancel_event)
    for sftp in sftps:
        if sftp is not None:
            _set_sftp_timeout(sftp, deadline)


def _parts(path: str, *, label: str) -> PurePosixPath:
    candidate = PurePosixPath(path)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise _PolicyViolation(f"Invalid {label} path.")
    return candidate


def _validate_path(path: str, root: str, *, label: str) -> None:
    target = _parts(path, label=label)
    if root:
        root_path = _parts(root, label=f"{label} root")
        if target != root_path and root_path not in target.parents:
            raise _PolicyViolation(f"{label.capitalize()} path is outside the configured root.")


def _lstat(sftp: Any, path: str, *, required: bool = True) -> Any | None:
    try:
        return sftp.lstat(path)
    except (FileNotFoundError, IOError, OSError) as exc:
        if isinstance(exc, FileNotFoundError) or getattr(exc, "errno", None) == ENOENT:
            if required:
                raise _TransferError("Remote path does not exist.") from None
            return None
        raise


def _safe_path(sftp: Any, path: str, root: str, *, leaf_required: bool) -> tuple[str, Any | None]:
    # The lexical/root check happens before opening a network connection too;
    # retain it here for internal callers and checked remote ancestry.
    _validate_path(path, root, label="remote")
    target = _parts(path, label="remote")
    current = "/"
    for index, part in enumerate(target.parts[1:], start=1):
        current = posixpath.join(current, part)
        is_leaf = index == len(target.parts) - 1
        attrs = _lstat(sftp, current, required=not (is_leaf and not leaf_required))
        if attrs is None:
            continue
        if stat.S_ISLNK(attrs.st_mode):
            # This deliberately includes the configured root.
            raise _PolicyViolation("Remote path contains a symlink.")
        if not is_leaf and not stat.S_ISDIR(attrs.st_mode):
            raise _PolicyViolation("Remote path ancestor is not a directory.")
    return str(target), _lstat(sftp, str(target), required=leaf_required)


def _ordinary(attrs: Any) -> bool:
    return stat.S_ISREG(attrs.st_mode)


def _same_ssh_endpoint(source: SSHConfig | None, destination: SSHConfig | None) -> bool:
    """Conservatively identify a shared filesystem by configured host/port.

    This does not resolve DNS aliases, but deliberately ignores usernames: two
    accounts can access the same remote filesystem.
    """
    return bool(source and destination and (source.host.casefold(), source.port) == (destination.host.casefold(), destination.port))


def _same_or_descendant(path: str, parent: str) -> bool:
    candidate, ancestor = PurePosixPath(path), PurePosixPath(parent)
    return candidate == ancestor or ancestor in candidate.parents


def _preserve_metadata(sftp: Any, path: str, attrs: Any) -> None:
    mode = getattr(attrs, "st_mode", None)
    if mode is not None:
        try:
            sftp.chmod(path, mode & 0o777)
        except (AttributeError, IOError, OSError):
            pass
    mtime = getattr(attrs, "st_mtime", None)
    if mtime is not None:
        try:
            sftp.utime(path, (mtime, mtime))
        except (AttributeError, IOError, OSError):
            pass


def _publish(sftp: Any, temporary: str, target: str, *, overwrite: bool) -> None:
    if overwrite:
        rename = getattr(sftp, "posix_rename", None)
        if not callable(rename):
            raise _TransferError("Overwrite is not supported by this SSH server.")
        try:
            rename(temporary, target)
        except Exception as exc:
            if _is_transport_error(exc):
                raise
            if not isinstance(exc, (IOError, OSError)):
                raise
            raise _TransferError("Overwrite failed: this SSH server may not support POSIX rename.") from None
        return
    try:
        sftp.rename(temporary, target)
    except (IOError, OSError):
        if _lstat(sftp, target, required=False) is not None:
            raise _TransferError("Destination already exists.") from None
        raise


def _is_transport_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, EOFError, ConnectionError, paramiko.SSHException)):
        return True
    return isinstance(exc, OSError) and exc.errno in {
        ECONNABORTED,
        ECONNRESET,
        EHOSTDOWN,
        EHOSTUNREACH,
        ENETDOWN,
        ENETRESET,
        ENETUNREACH,
        EPIPE,
        ETIMEDOUT,
    }


def _cleanup_staging(sftp: Any, temporary: str | None, staging: str | None) -> None:
    _set_sftp_timeout(sftp, time.monotonic() + _CLEANUP_TIMEOUT)
    if temporary is not None:
        try:
            sftp.remove(temporary)
        except Exception:
            pass
    if staging is not None:
        try:
            sftp.rmdir(staging)
        except Exception:
            pass


def _copy_file(
    source: Any | None,
    destination: Any | None,
    source_path: str | None,
    destination_path: str | None,
    content: bytes | None,
    source_attrs: Any | None,
    *,
    overwrite: bool,
    max_file_bytes: int,
    deadline: float,
    cancel_event: threading.Event | None,
) -> bytes | int:
    _check_io(deadline, cancel_event, source, destination)
    if destination is not None:
        assert destination_path is not None
        destination_attrs = _lstat(destination, destination_path, required=False)
        if destination_attrs is not None:
            if not _ordinary(destination_attrs):
                raise _PolicyViolation("Destination must be a regular file or a new path.")
            if not overwrite:
                raise _TransferError("Destination already exists.")
    if source is None:
        if content is None:
            raise _TransferError("Inline source content is required.")
        if len(content) > max_file_bytes:
            raise _TransferError("File exceeds the maximum allowed size.")
        reader: Any = _BytesReader(content)
    else:
        assert source_path is not None
        reader = source.open(source_path, "rb")
    temporary = staging = None
    writer: Any = None
    received = bytearray() if destination is None else None
    total = 0
    try:
        if destination is not None:
            assert destination_path is not None
            parent = posixpath.dirname(destination_path)
            staging = posixpath.join(parent, f".ragtime-transfer-{secrets.token_hex(8)}")
            destination.mkdir(staging, mode=0o700)
            temporary = posixpath.join(staging, "file")
            writer = destination.open(temporary, "wx")
            # Paramiko applies the server default mode on create, so set the
            # private mode before the first source byte is written.
            destination.chmod(temporary, 0o600)
        while True:
            _check_io(deadline, cancel_event, source, destination)
            chunk = reader.read(CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > max_file_bytes:
                raise _TransferError("File exceeds the maximum allowed size.")
            if writer is None:
                assert received is not None
                received.extend(chunk)
            else:
                writer.write(chunk)
        if writer is not None:
            assert temporary is not None and destination_path is not None
            assert destination is not None
            writer.close()
            writer = None
            if source_attrs is not None:
                _check_io(deadline, cancel_event, destination)
                _preserve_metadata(destination, temporary, source_attrs)
            elif destination_attrs is not None:
                # Inline/workspace content has no source metadata. Preserve an
                # existing ordinary destination's mode only; its mtime belongs
                # to the newly uploaded content. New files remain private.
                _check_io(deadline, cancel_event, destination)
                destination.chmod(temporary, destination_attrs.st_mode & 0o777)
            _check_io(deadline, cancel_event, destination)
            try:
                _publish(destination, temporary, destination_path, overwrite=overwrite)
            except Exception as exc:
                if _is_transport_error(exc) or (
                    isinstance(exc, (OSError, IOError)) and (time.monotonic() >= deadline or (cancel_event is not None and cancel_event.is_set()))
                ):
                    raise _PublishOutcomeUnknown("Transfer outcome is unknown after publish request.") from None
                raise
            temporary = None
        return bytes(received) if received is not None else total
    finally:
        try:
            reader.close()
        except Exception:
            pass
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if destination is not None:
            _cleanup_staging(destination, temporary, staging)


class _BytesReader:
    def __init__(self, value: bytes) -> None:
        self.value, self.offset = value, 0

    def read(self, size: int) -> bytes:
        chunk = self.value[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk

    def close(self) -> None:
        pass


def _walk(sftp: Any, directory: str, *, deadline: float, cancel_event: threading.Event | None) -> Iterator[tuple[str, Any]]:
    _check_io(deadline, cancel_event, sftp)
    iterator = getattr(sftp, "listdir_iter", None)
    entries = cast(Iterator[Any], iterator(directory, read_aheads=1) if callable(iterator) else iter(sftp.listdir_attr(directory)))
    for attrs in entries:
        _check_io(deadline, cancel_event, sftp)
        name = getattr(attrs, "filename", "")
        if not name or name in {".", ".."} or "/" in name:
            continue
        path = posixpath.join(directory, name)
        checked = _lstat(sftp, path)
        assert checked is not None
        yield path, checked
        if stat.S_ISDIR(checked.st_mode):
            yield from _walk(sftp, path, deadline=deadline, cancel_event=cancel_event)


class _TransferWatchdog:
    def __init__(self, deadline: float, cancel_event: threading.Event | None) -> None:
        self.deadline, self.cancel_event = deadline, cancel_event
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.clients: list[Any] = []
        self.thread = threading.Thread(target=self._run, name="ssh-transfer-watchdog", daemon=True)

    def register(self, client: Any) -> None:
        with self.lock:
            self.clients.append(client)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop.is_set():
            cancelled = self.cancel_event is not None and self.cancel_event.is_set()
            if cancelled or time.monotonic() >= self.deadline:
                with self.lock:
                    clients = list(self.clients)
                for client in clients:
                    try:
                        client.close()
                    except Exception:
                        pass
                # Keep closing through the client/transport assignment race.
                self.stop.wait(0.05)
            else:
                self.stop.wait(min(0.05, max(0.001, self.deadline - time.monotonic())))

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=1)


def transfer_ssh_files(
    source_config: SSHConfig | None,
    destination_config: SSHConfig | None,
    source_path: str | None,
    destination_path: str | None,
    *,
    source_root: str = "",
    destination_root: str = "",
    content: bytes | None = None,
    overwrite: bool = False,
    recursive: bool = False,
    timeout: int = 300,
    max_file_bytes: int = 50 * 1024 * 1024,
    max_files: int = 500,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Transfer a single file or recursively copy a remote directory via SFTP."""
    result = _result()
    if source_config is None and destination_config is None:
        return {**result, "status": "rejected", "errors": [{"message": "An SSH endpoint is required."}]}
    if max_file_bytes < 0 or max_files < 1 or timeout <= 0:
        return {**result, "status": "rejected", "errors": [{"message": "Invalid transfer limits."}]}
    if source_config is None and (content is None or destination_path is None):
        return {**result, "status": "rejected", "errors": [{"message": "Inline source and destination path are required."}]}
    if destination_config is None and (source_path is None or content is not None or recursive):
        return {**result, "status": "rejected", "errors": [{"message": "A remote single-file source is required."}]}
    try:
        if source_config is not None:
            source_config.validate()
            assert source_path is not None
            _validate_path(source_path, source_root, label="remote")
        if destination_config is not None:
            destination_config.validate()
            assert destination_path is not None
            _validate_path(destination_path, destination_root, label="remote")
    except (ValueError, _PolicyViolation) as exc:
        return {**result, "status": "rejected", "errors": [{"message": _message(exc)}]}

    deadline = time.monotonic() + timeout
    watchdog = _TransferWatchdog(deadline, cancel_event)
    clients: list[Any] = []
    watchdog.start()
    try:
        source = destination = None
        if source_config is not None:
            _check_active(deadline, cancel_event)
            client = _create_ssh_client(
                _connection_config(source_config, deadline),
                on_client_created=watchdog.register,
                tcp_connect_timeout=min(5.0, max(0.1, deadline - time.monotonic())),
            )
            clients.append(client)
            _check_active(deadline, cancel_event)
            source = client.open_sftp()
            _check_io(deadline, cancel_event, source)
        if destination_config is not None:
            _check_active(deadline, cancel_event)
            client = _create_ssh_client(
                _connection_config(destination_config, deadline),
                on_client_created=watchdog.register,
                tcp_connect_timeout=min(5.0, max(0.1, deadline - time.monotonic())),
            )
            clients.append(client)
            _check_active(deadline, cancel_event)
            destination = client.open_sftp()
            _check_io(deadline, cancel_event, destination)
        _check_io(deadline, cancel_event, source, destination)
        src_attrs = None
        if source is not None:
            assert source_path is not None
            source_path, src_attrs = _safe_path(source, source_path, source_root, leaf_required=True)
        source_is_directory = src_attrs is not None and stat.S_ISDIR(src_attrs.st_mode)
        if source_is_directory and recursive and destination_config is not None:
            assert source_path is not None and destination_path is not None
            if _same_ssh_endpoint(source_config, destination_config) and (
                _same_or_descendant(destination_path, source_path) or _same_or_descendant(source_path, destination_path)
            ):
                result["status"] = "rejected"
                result["errors"].append({"message": "Recursive source and destination directories must not overlap on the same SSH endpoint."})
                return result
        if destination is not None:
            assert destination_path is not None
            destination_path, _ = _safe_path(destination, destination_path, destination_root, leaf_required=False)
        if source_is_directory:
            if not recursive:
                raise _PolicyViolation("Source is a directory; set recursive=true.")
            if destination is None:
                raise _TransferError("Recursive transfer requires an SSH destination.")
            assert source_path is not None and destination_path is not None
            try:
                destination.mkdir(destination_path)
            except (IOError, OSError):
                pass
            _, destination_attrs = _safe_path(destination, destination_path, destination_root, leaf_required=True)
            if destination_attrs is None or not stat.S_ISDIR(destination_attrs.st_mode):
                raise _TransferError("Recursive destination must be a directory.")
            visited_entries = 0
            for path, attrs in _walk(source, source_path, deadline=deadline, cancel_event=cancel_event):
                _check_io(deadline, cancel_event, source, destination)
                visited_entries += 1
                if visited_entries > max_files:
                    raise _TransferError("Transfer exceeds the maximum recursive entry count.")
                rel, target = posixpath.relpath(path, source_path), posixpath.join(destination_path, posixpath.relpath(path, source_path))
                if stat.S_ISDIR(attrs.st_mode):
                    try:
                        destination.mkdir(target)
                    except (IOError, OSError):
                        pass
                    _safe_path(destination, target, destination_root, leaf_required=True)
                elif stat.S_ISLNK(attrs.st_mode) or not _ordinary(attrs):
                    result["skipped"].append({"path": rel, "message": "Skipped symlink or special file."})
                else:
                    try:
                        _safe_path(destination, target, destination_root, leaf_required=False)
                        copied = _copy_file(
                            source,
                            destination,
                            path,
                            target,
                            None,
                            attrs,
                            overwrite=overwrite,
                            max_file_bytes=max_file_bytes,
                            deadline=deadline,
                            cancel_event=cancel_event,
                        )
                    except (_TransferCancelled, TimeoutError):
                        raise
                    except Exception as exc:
                        result["errors"].append({"path": rel, "message": _message(exc)})
                    else:
                        result["bytes_transferred"] += int(copied)
                        result["files_transferred"] += 1
            if result["errors"]:
                result["status"] = "transfer_failed"
        else:
            if src_attrs is not None and not _ordinary(src_attrs):
                raise _PolicyViolation("Source must be a regular file.")
            copied = _copy_file(
                source,
                destination,
                source_path,
                destination_path,
                content,
                src_attrs,
                overwrite=overwrite,
                max_file_bytes=max_file_bytes,
                deadline=deadline,
                cancel_event=cancel_event,
            )
            if destination is None:
                assert isinstance(copied, bytes)
                result["content"] = copied
                result["bytes_transferred"] = len(copied)
            else:
                result["bytes_transferred"] = int(copied)
            result["files_transferred"] = 1
    except _PolicyViolation as exc:
        result["status"] = "rejected"
        result["errors"].append({"message": _message(exc)})
    except Exception as exc:
        result["status"] = "transfer_failed"
        result["errors"].append({"message": _message(exc)})
    finally:
        for client in clients:
            try:
                client.close()
            except Exception:
                pass
        watchdog.close()
    return result


def _connection_config(config: SSHConfig, deadline: float) -> SSHConfig:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError()
    configured = config.timeout if config.timeout > 0 else 30
    return replace(config, timeout=min(configured, max(0.1, remaining)))
