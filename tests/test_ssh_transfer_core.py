import stat
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

from ragtime.core.ssh import SSHConfig
from ragtime.core.ssh_transfer import CHUNK_SIZE, transfer_ssh_files


class _File:
    def __init__(self, sftp, path, mode):
        self.sftp, self.path, self.mode, self.offset = sftp, path, mode, 0

    def read(self, size):
        value = self.sftp.files[self.path][self.offset : self.offset + size]
        self.offset += len(value)
        if self.sftp.on_read:
            self.sftp.on_read()
        return value

    def write(self, value):
        self.sftp.files[self.path] = self.sftp.files.get(self.path, b"") + value
        if self.sftp.on_write:
            self.sftp.on_write()
        return len(value)

    def close(self):
        pass


class _SFTP:
    def __init__(self, files=None, symlinks=(), specials=(), file_modes=None):
        self.files = dict(files or {})
        self.symlinks = set(symlinks)
        self.specials = set(specials)
        self.file_modes = dict(file_modes or {})
        self.dirs = {"/", "/root"}
        self.removed = []
        self.max_read = 0
        self.on_read: Callable[[], None] | None = None
        self.on_write: Callable[[], None] | None = None
        self.operations = []

    def lstat(self, path):
        if path in self.symlinks:
            return SimpleNamespace(st_mode=stat.S_IFLNK, st_mtime=1)
        if path in self.specials:
            return SimpleNamespace(st_mode=stat.S_IFIFO | 0o600, st_mtime=1)
        if path in self.files:
            return SimpleNamespace(st_mode=stat.S_IFREG | self.file_modes.get(path, 0o644), st_mtime=1, filename=path.rsplit("/", 1)[-1])
        prefix = path.rstrip("/") + "/"
        if path in self.dirs or any(key.startswith(prefix) for key in self.files) or path == "/":
            return SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_mtime=1)
        raise FileNotFoundError(path)

    def open(self, path, mode):
        if "x" in mode and path in self.files:
            raise IOError("exists")
        if "w" in mode or "x" in mode:
            self.files[path] = b""
        return _File(self, path, mode)

    def rename(self, old, new):
        if new in self.files:
            raise IOError("exists")
        self.operations.append(("rename", old, new))
        self.files[new] = self.files.pop(old)
        self.file_modes[new] = self.file_modes.pop(old, 0o644)

    def posix_rename(self, old, new):
        self.files[new] = self.files.pop(old)
        self.file_modes[new] = self.file_modes.pop(old, 0o644)

    def remove(self, path):
        self.removed.append(path)
        self.files.pop(path, None)

    def listdir_attr(self, directory):
        prefix = directory.rstrip("/") + "/"
        names = set()
        for path in self.files:
            if path.startswith(prefix):
                names.add(path[len(prefix) :].split("/", 1)[0])
        return [SimpleNamespace(filename=name, st_mode=self.lstat(prefix + name).st_mode) for name in names]

    def listdir_iter(self, directory, read_aheads=1):
        yield from sorted(self.listdir_attr(directory), key=lambda entry: entry.filename)

    def mkdir(self, path, mode=0o777):
        self.dirs.add(path)
        self.operations.append(("mkdir", path, mode))

    def rmdir(self, path):
        self.dirs.discard(path)
        self.operations.append(("rmdir", path))

    def chmod(self, *args):
        self.operations.append(("chmod", *args))

    def utime(self, *args):
        self.operations.append(("utime", *args))


class _Client:
    def __init__(self, sftp):
        self.sftp = sftp
        self.closed = False

    def open_sftp(self):
        return self.sftp

    def close(self):
        self.closed = True


def _config():
    return SSHConfig(host="host", user="user", password="pw")


def _run(source, destination, **kwargs):
    clients = iter([_Client(source), _Client(destination)])

    def create(*args, **kwargs):
        client = next(clients)
        kwargs.get("on_client_created", lambda _: None)(client)
        return client

    with patch("ragtime.core.ssh_transfer._create_ssh_client", side_effect=create):
        return transfer_ssh_files(
            _config(),
            _config(),
            kwargs.pop("source_path", "/root/a"),
            kwargs.pop("destination_path", "/root/b"),
            source_root="/root",
            destination_root="/root",
            **kwargs,
        )


def test_binary_streaming_and_exact_cap():
    data = b"x" * (CHUNK_SIZE + 9)
    destination = _SFTP()
    result = _run(_SFTP({"/root/a": data}), destination, max_file_bytes=len(data))
    assert result["status"] == "ok" and destination.files["/root/b"] == data and result["bytes_transferred"] == len(data)
    result = _run(_SFTP({"/root/a": data}), _SFTP(), max_file_bytes=len(data) - 1)
    assert result["status"] == "transfer_failed" and "maximum" in result["errors"][0]["message"]


def test_nested_symlink_and_root_escape_rejected_before_write():
    result = _run(_SFTP({"/root/link/a": b"x"}, {"/root/link"}), _SFTP(), source_path="/root/link/a")
    assert result["status"] == "rejected" and "symlink" in result["errors"][0]["message"]
    result = _run(_SFTP({"/root/a": b"x"}), _SFTP(), destination_path="/outside/b")
    assert result["status"] == "rejected" and "outside" in result["errors"][0]["message"]


def test_no_clobber_and_failed_publish_cleanup():
    destination = _SFTP({"/root/b": b"old"})
    result = _run(_SFTP({"/root/a": b"new"}), destination)
    assert result["status"] == "transfer_failed" and destination.files["/root/b"] == b"old"
    assert result["errors"] == [{"message": "Destination already exists."}]
    assert not any(".ragtime-transfer-" in directory for directory in destination.dirs)


def test_recursive_cap_is_partial_and_skips_symlink():
    source = _SFTP({"/root/tree/a": b"a", "/root/tree/b": b"b", "/root/tree/link": b"x"}, {"/root/tree/link"})
    result = _run(source, _SFTP(), source_path="/root/tree", destination_path="/root/out", recursive=True, max_files=1)
    assert result["status"] == "transfer_failed" and result["files_transferred"] == 1


def test_recursive_same_endpoint_same_or_nested_destination_is_rejected_before_write():
    for destination_path in ("/root/tree", "/root/tree/copy"):
        endpoint = _SFTP({"/root/tree/a": b"a"})

        result = _run(
            endpoint,
            endpoint,
            source_path="/root/tree",
            destination_path=destination_path,
            recursive=True,
            max_files=2,
        )

        assert result["status"] == "rejected"
        assert result["errors"] == [{"message": "Recursive source and destination directories must not overlap on the same SSH endpoint."}]
        assert "/root/tree/copy" not in endpoint.dirs
        assert set(endpoint.files) == {"/root/tree/a"}
        assert endpoint.operations == []


def test_recursive_same_endpoint_parent_destination_is_rejected_before_write():
    endpoint = _SFTP({"/root/tree/inner/a": b"a"})

    result = _run(endpoint, endpoint, source_path="/root/tree/inner", destination_path="/root/tree", recursive=True)

    assert result["status"] == "rejected"
    assert "overlap" in result["errors"][0]["message"]
    assert endpoint.operations == []


def test_recursive_skipped_entries_are_bounded_by_entry_limit():
    source = _SFTP(
        {
            "/root/tree/link-a": b"a",
            "/root/tree/link-b": b"b",
            "/root/tree/link-c": b"c",
        },
        {"/root/tree/link-a", "/root/tree/link-b", "/root/tree/link-c"},
    )

    result = _run(
        source,
        _SFTP(),
        source_path="/root/tree",
        destination_path="/root/out",
        recursive=True,
        max_files=2,
    )

    assert result["status"] == "transfer_failed"
    assert len(result["skipped"]) == 2
    assert result["errors"] == [{"message": "Transfer exceeds the maximum recursive entry count."}]


def test_cancel_and_timeout_cleanup_temp_files():
    event = threading.Event()
    event.set()
    destination = _SFTP()
    result = _run(_SFTP({"/root/a": b"x"}), destination, cancel_event=event)
    assert result["status"] == "transfer_failed" and not destination.files


def test_cancellation_during_chunks_prevents_publish_and_cleans_temp():
    event = threading.Event()
    source = _SFTP({"/root/a": b"x" * (CHUNK_SIZE + 1)})
    source.on_read = event.set
    destination = _SFTP()
    result = _run(source, destination, cancel_event=event)
    assert result["status"] == "transfer_failed"
    assert "/root/b" not in destination.files
    assert destination.removed


def test_deadline_before_publish_prevents_publish_and_cleans_temp():
    now = [0.0]
    source = _SFTP({"/root/a": b"x"})
    destination = _SFTP()
    destination.on_write = lambda: now.__setitem__(0, 2.0)
    with patch("ragtime.core.ssh_transfer.time.monotonic", side_effect=lambda: now[0]):
        result = _run(source, destination, timeout=1)
    assert result["status"] == "transfer_failed"
    assert "/root/b" not in destination.files
    assert destination.removed


def test_recursive_order_cap_and_per_file_failure_are_deterministic():
    source = _SFTP({"/root/tree/b": b"b", "/root/tree/a": b"a"})
    destination = _SFTP()
    result = _run(source, destination, source_path="/root/tree", destination_path="/root/out", recursive=True, max_files=1)
    assert result["status"] == "transfer_failed"
    assert destination.files["/root/out/a"] == b"a"
    assert "maximum recursive entry count" in result["errors"][0]["message"]


def test_no_clobber_race_unsupported_overwrite_and_metadata_are_safe():
    destination = _SFTP()
    destination.on_write = lambda: destination.files.setdefault("/root/b", b"racer")
    result = _run(_SFTP({"/root/a": b"new"}), destination)
    assert result["status"] == "transfer_failed" and destination.files["/root/b"] == b"racer"

    class NoPosixRename(_SFTP):
        posix_rename: Any = None

    result = _run(_SFTP({"/root/a": b"new"}), NoPosixRename(), overwrite=True)
    assert result["status"] == "transfer_failed" and "not supported" in result["errors"][0]["message"]

    class BrokenPosixRename(_SFTP):
        def posix_rename(self, old, new):
            raise IOError("unsupported")

    result = _run(_SFTP({"/root/a": b"new"}), BrokenPosixRename(), overwrite=True)
    assert result["status"] == "transfer_failed" and "POSIX rename" in result["errors"][0]["message"]

    class LostPosixRename(_SFTP):
        def __init__(self, exc):
            super().__init__()
            self.exc = exc

        def posix_rename(self, old, new):
            raise self.exc

    for exc in (TimeoutError(), EOFError()):
        result = _run(_SFTP({"/root/a": b"new"}), LostPosixRename(exc), overwrite=True)
        assert result["status"] == "transfer_failed"
        assert result["errors"] == [{"message": "Transfer outcome is unknown after publish request."}]

    destination = _SFTP()
    result = _run(_SFTP({"/root/a": b"new"}), destination)
    assert result["status"] == "ok"
    assert destination.operations.index(next(op for op in destination.operations if op[0] == "chmod")) < destination.operations.index(
        next(op for op in destination.operations if op[0] == "rename")
    )
    mkdir = next(op for op in destination.operations if op[0] == "mkdir" and ".ragtime-transfer-" in op[1])
    assert mkdir[2] == 0o700
    assert any(op[0] == "chmod" and op[-1] == 0o600 for op in destination.operations)


def test_inline_upload_keeps_new_files_private_and_preserves_existing_mode_without_mtime():
    new_destination = _SFTP()
    with patch("ragtime.core.ssh_transfer._create_ssh_client", return_value=_Client(new_destination)):
        result = transfer_ssh_files(None, _config(), None, "/root/new", content=b"new", destination_root="/root")
    assert result["status"] == "ok"
    assert any(op[0] == "chmod" and op[-1] == 0o600 for op in new_destination.operations)

    existing_destination = _SFTP({"/root/existing": b"old"}, file_modes={"/root/existing": 0o755})
    with patch("ragtime.core.ssh_transfer._create_ssh_client", return_value=_Client(existing_destination)):
        result = transfer_ssh_files(None, _config(), None, "/root/existing", content=b"new", destination_root="/root", overwrite=True)
    assert result["status"] == "ok"
    assert existing_destination.files["/root/existing"] == b"new"
    assert any(op[0] == "chmod" and op[-1] == 0o755 for op in existing_destination.operations)
    assert not any(op[0] == "utime" for op in existing_destination.operations)


def test_invalid_config_permission_errors_and_download_cap():
    with patch("ragtime.core.ssh_transfer._create_ssh_client") as create:
        result = transfer_ssh_files(SSHConfig(host="", user="u", password="pw"), None, "/a", None)
    assert result["status"] == "rejected" and not create.called

    source = _SFTP({"/root/a": b"x"})
    source.lstat = lambda path: (_ for _ in ()).throw(PermissionError()) if path == "/root/a" else _SFTP.lstat(source, path)
    result = _run(source, _SFTP())
    assert result["status"] == "transfer_failed" and result["errors"][0]["message"] == "SSH transfer failed."

    source = _SFTP({"/root/a": b"xx"})
    with patch("ragtime.core.ssh_transfer._create_ssh_client", return_value=_Client(source)):
        result = transfer_ssh_files(_config(), None, "/root/a", None, source_root="/root", max_file_bytes=1)
    assert result["status"] == "transfer_failed" and "maximum" in result["errors"][0]["message"]


def test_policy_paths_reject_before_network_and_directory_message_is_accurate():
    with patch("ragtime.core.ssh_transfer._create_ssh_client") as create:
        result = transfer_ssh_files(_config(), None, "relative", None)
    assert result["status"] == "rejected" and not create.called

    result = _run(_SFTP({"/root/tree/a": b"x"}), _SFTP(), source_path="/root/tree")
    assert result["status"] == "rejected" and "recursive=true" in result["errors"][0]["message"]


def test_recursive_existing_special_target_is_preserved_and_later_file_copies_without_reading_it():
    source = _SFTP({"/root/tree/a": b"blocked", "/root/tree/b": b"copied"})
    opened = []
    source_open = source.open
    source.open = lambda path, mode: (opened.append(path) if mode == "rb" else None) or source_open(path, mode)
    destination = _SFTP(specials={"/root/out/a"})

    result = _run(source, destination, source_path="/root/tree", destination_path="/root/out", recursive=True, overwrite=True)

    assert result["status"] == "transfer_failed"
    assert result["errors"] == [{"path": "a", "message": "Destination must be a regular file or a new path."}]
    assert "/root/out/a" in destination.specials
    assert destination.files["/root/out/b"] == b"copied"
    assert opened == ["/root/tree/b"]


def test_recursive_preexisting_file_fails_before_reading_that_source_and_continues():
    source = _SFTP({"/root/tree/a": b"blocked", "/root/tree/b": b"copied"})
    opened = []
    source_open = source.open
    source.open = lambda path, mode: (opened.append(path) if mode == "rb" else None) or source_open(path, mode)
    destination = _SFTP({"/root/out/a": b"existing"})

    result = _run(source, destination, source_path="/root/tree", destination_path="/root/out", recursive=True)

    assert result["status"] == "transfer_failed"
    assert result["errors"] == [{"path": "a", "message": "Destination already exists."}]
    assert destination.files["/root/out/a"] == b"existing"
    assert destination.files["/root/out/b"] == b"copied"
    assert opened == ["/root/tree/b"]


def test_recursive_same_host_different_users_rejects_nested_destination_before_writes():
    endpoint = _SFTP({"/root/tree/a": b"a"})
    clients = iter([_Client(endpoint), _Client(endpoint)])

    def create(*args, **kwargs):
        client = next(clients)
        kwargs["on_client_created"](client)
        return client

    with patch("ragtime.core.ssh_transfer._create_ssh_client", side_effect=create):
        result = transfer_ssh_files(
            SSHConfig(host="host", user="source", password="pw"),
            SSHConfig(host="host", user="destination", password="pw"),
            "/root/tree",
            "/root/tree/copy",
            source_root="/root",
            destination_root="/root",
            recursive=True,
        )
    assert result["status"] == "rejected"
    assert endpoint.operations == []


def test_watchdog_interrupts_blocked_sftp_handshake_promptly():
    started, released = threading.Event(), threading.Event()

    class BlockingClient(_Client):
        def open_sftp(self):
            started.set()
            released.wait(1)
            raise TimeoutError()

        def close(self):
            super().close()
            released.set()

    client = BlockingClient(_SFTP())

    def create(*args, **kwargs):
        kwargs["on_client_created"](client)
        return client

    with patch("ragtime.core.ssh_transfer._create_ssh_client", side_effect=create):
        start = time.monotonic()
        result = transfer_ssh_files(_config(), None, "/root/a", None, source_root="/root", timeout=1)
    assert started.is_set() and client.closed and time.monotonic() - start < 1.5
    assert result["status"] == "transfer_failed"
