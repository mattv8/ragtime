"""Conservative, descriptor-pinned source state tokens for SQLite capture."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager

_CHUNK_SIZE = 1024 * 1024
_SIDECAR_SUFFIXES = ("", "-wal", "-journal")


def _metadata(details: os.stat_result) -> tuple[int, int, int, int, int]:
    return (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns, details.st_ctime_ns)


def _frame(digest: "hashlib._Hash", payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


class _Entry:
    def __init__(self, name: str, fd: int | None, details: os.stat_result | None) -> None:
        self.name = name
        self.fd = fd
        self.details = details


class PinnedSourceState:
    """Hold source descriptors while proving their pinned entries stayed stable."""

    def __init__(self, directory_fd: int, database_name: str) -> None:
        self._directory_fd = directory_fd
        self._entries: list[_Entry] = []
        self._invalid = False
        self._token_observed = False
        self.main_fd: int | None = None
        if not database_name or "/" in database_name or database_name in {".", ".."}:
            self._invalid = True
            return
        try:
            for suffix in _SIDECAR_SUFFIXES:
                name = database_name + suffix
                try:
                    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
                except FileNotFoundError:
                    if suffix == "":
                        self._invalid = True
                    self._entries.append(_Entry(name, None, None))
                    continue
                details = os.fstat(fd)
                if not stat.S_ISREG(details.st_mode):
                    os.close(fd)
                    self._invalid = True
                    break
                entry = _Entry(name, fd, details)
                self._entries.append(entry)
                if suffix == "":
                    self.main_fd = fd
        except OSError:
            self._invalid = True
        if self._invalid:
            self.close()

    def close(self) -> None:
        for entry in self._entries:
            if entry.fd is not None:
                os.close(entry.fd)
                entry.fd = None
        self.main_fd = None

    def _entry_matches(self, entry: _Entry) -> bool:
        try:
            current = os.stat(entry.name, dir_fd=self._directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return entry.details is None
        if entry.details is None or not stat.S_ISREG(current.st_mode):
            return False
        return _metadata(current) == _metadata(entry.details)

    def refresh_after_warmup(self) -> bool:
        """Accept SQLite's one-time readonly-open metadata side effect once.

        This deliberately refreshes metadata only before any state observation;
        identities and directory presence must still match the pinned FDs.
        """
        if self._invalid or self._token_observed or self.main_fd is None:
            return False
        refreshed: list[os.stat_result | None] = []
        for entry in self._entries:
            try:
                current = os.stat(entry.name, dir_fd=self._directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                if entry.details is not None:
                    self._invalid = True
                    return False
                refreshed.append(None)
                continue
            if entry.details is None or not stat.S_ISREG(current.st_mode) or entry.fd is None:
                self._invalid = True
                return False
            try:
                pinned = os.fstat(entry.fd)
            except OSError:
                self._invalid = True
                return False
            if not stat.S_ISREG(pinned.st_mode) or (current.st_dev, current.st_ino) != (pinned.st_dev, pinned.st_ino):
                self._invalid = True
                return False
            refreshed.append(current)
        for entry, details in zip(self._entries, refreshed):
            entry.details = details
        return True

    def token(self) -> str | None:
        """Return a content token only if every pinned entry stayed unchanged."""
        self._token_observed = True
        if self._invalid or self.main_fd is None or any(not self._entry_matches(entry) for entry in self._entries):
            self._invalid = True
            return None
        journal = self._entries[2]
        if journal.details is not None and journal.details.st_size:
            return None
        digest = hashlib.sha256(b"ragtime-sqlite-source-state-v1\0")
        for entry in self._entries:
            _frame(digest, entry.name.encode("utf-8", "surrogateescape"))
            if entry.fd is None:
                _frame(digest, b"absent")
                continue
            details = entry.details
            if details is None:
                self._invalid = True
                return None
            try:
                before = os.fstat(entry.fd)
                if _metadata(before) != _metadata(details):
                    self._invalid = True
                    return None
                _frame(digest, b"present")
                _frame(digest, repr(_metadata(before)).encode("ascii"))
                os.lseek(entry.fd, 0, os.SEEK_SET)
                while chunk := os.read(entry.fd, _CHUNK_SIZE):
                    digest.update(chunk)
                if _metadata(os.fstat(entry.fd)) != _metadata(before):
                    self._invalid = True
                    return None
            except OSError:
                self._invalid = True
                return None
        if any(not self._entry_matches(entry) for entry in self._entries):
            self._invalid = True
            return None
        return digest.hexdigest()


@contextmanager
def pinned_source_state(directory_fd: int, database_name: str) -> Iterator[PinnedSourceState]:
    """Pin main/WAL/journal descriptors for a probe or full capture lifetime."""
    state = PinnedSourceState(directory_fd, database_name)
    try:
        yield state
    finally:
        state.close()


def source_state_token(directory_fd: int, database_name: str) -> str | None:
    """Probe a SQLite source through a pinned directory descriptor."""
    with pinned_source_state(directory_fd, database_name) as state:
        return state.token()
