"""The deliberately small, Linux-only exec boundary for copied mount rsync.

This module is executed in a fresh process, never via ``preexec_fn`` from a
worker thread.  Exit 77 is reserved for Landlock setup being unavailable
*before* rsync has run; exit 78 is a launcher validation/internal error.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import stat
import sys

_RSYNC = "/usr/bin/rsync"
_EXIT_UNAVAILABLE = 77
_EXIT_INVALID = 78
_PR_SET_NO_NEW_PRIVS = 38
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1

# linux/landlock.h -- ABI 3 introduced TRUNCATE, required by this boundary.
_EXECUTE = 1 << 0
_WRITE_FILE = 1 << 1
_READ_FILE = 1 << 2
_READ_DIR = 1 << 3
_REMOVE_DIR = 1 << 4
_REMOVE_FILE = 1 << 5
_MAKE_CHAR = 1 << 6
_MAKE_DIR = 1 << 7
_MAKE_REG = 1 << 8
_MAKE_SOCK = 1 << 9
_MAKE_FIFO = 1 << 10
_MAKE_BLOCK = 1 << 11
_MAKE_SYM = 1 << 12
_REFER = 1 << 13
_TRUNCATE = 1 << 14
_SOURCE_RIGHTS = _READ_FILE | _READ_DIR
_DESTINATION_RIGHTS = _WRITE_FILE | _READ_FILE | _READ_DIR | _REMOVE_DIR | _REMOVE_FILE | _MAKE_DIR | _MAKE_REG | _MAKE_SYM | _REFER | _TRUNCATE
_HANDLED_RIGHTS = _SOURCE_RIGHTS | _DESTINATION_RIGHTS | _EXECUTE | _MAKE_CHAR | _MAKE_SOCK | _MAKE_FIFO | _MAKE_BLOCK
_READ_EXEC_RIGHTS = _READ_FILE | _READ_DIR | _EXECUTE


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32), ("reserved", ctypes.c_uint32)]


def _syscall_numbers() -> tuple[int, int, int]:
    # Landlock syscall numbers are shared by x86_64 and aarch64.
    if os.uname().machine not in {"x86_64", "aarch64"}:
        raise OSError(errno.ENOSYS, "unsupported architecture for Landlock")
    return 444, 445, 446


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def landlock_abi() -> int:
    create, _add, _restrict = _syscall_numbers()
    result = _libc().syscall(create, 0, 0, _LANDLOCK_CREATE_RULESET_VERSION)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def trial_ruleset() -> None:
    """Verify ruleset creation without changing the caller's restrictions."""
    if landlock_abi() < 3:
        raise OSError(errno.EOPNOTSUPP, "Landlock ABI 3 is required")
    create, _add, _restrict = _syscall_numbers()
    attr = _RulesetAttr(_HANDLED_RIGHTS)
    fd = _libc().syscall(create, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    os.close(fd)


def _add_path_rule(ruleset_fd: int, parent_fd: int, rights: int) -> None:
    _create, add, _restrict = _syscall_numbers()
    attr = _PathBeneathAttr(rights, parent_fd, 0)
    result = _libc().syscall(add, ruleset_fd, _LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(attr), 0)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _validate_directory_fd(fd: int) -> None:
    if fd < 0:
        raise ValueError(f"descriptor {fd} is not a directory")
    directory_stat = os.fstat(fd)
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise ValueError(f"descriptor {fd} is not a directory")
    if directory_stat.st_nlink == 0:
        raise ValueError(f"descriptor {fd} refers to a retired directory")


def _validate_distinct_roots(source_fd: int, destination_fd: int) -> None:
    source = os.fstat(source_fd)
    destination = os.fstat(destination_fd)
    if (source.st_dev, source.st_ino) == (destination.st_dev, destination.st_ino):
        raise ValueError("source and destination are identical")
    # /proc fd links retain the resolved directory identity.  This catches the
    # nested-root case before delete semantics can be reached.
    source_path = os.path.realpath(f"/proc/self/fd/{source_fd}")
    destination_path = os.path.realpath(f"/proc/self/fd/{destination_fd}")
    try:
        common = os.path.commonpath((source_path, destination_path))
    except ValueError:
        # Different mount/path styles cannot be compared lexically; distinct
        # descriptors are still protected by their separately pinned roots.
        common = ""
    if common in {source_path, destination_path}:
        raise ValueError("source and destination roots overlap")


def _validate_protected_paths(protected_paths: tuple[str, ...]) -> None:
    for path in protected_paths:
        if not path or path.startswith("/") or "\x00" in path or any(part in {"", ".", ".."} for part in path.split("/")):
            raise ValueError("protected paths must be normalized relative paths")


def _install_landlock(source_fd: int, destination_fd: int) -> None:
    trial_ruleset()
    create, _add, restrict = _syscall_numbers()
    handled = _HANDLED_RIGHTS
    attr = _RulesetAttr(handled)
    ruleset_fd = _libc().syscall(create, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if ruleset_fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    try:
        _add_path_rule(ruleset_fd, source_fd, _SOURCE_RIGHTS)
        _add_path_rule(ruleset_fd, destination_fd, _DESTINATION_RIGHTS)
        for path in ("/usr", "/lib", "/lib64"):
            if os.path.isdir(path):
                fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    _add_path_rule(ruleset_fd, fd, _READ_EXEC_RIGHTS)
                finally:
                    os.close(fd)
        if os.path.isfile("/etc/ld.so.cache"):
            fd = os.open("/etc/ld.so.cache", os.O_RDONLY)
            try:
                _add_path_rule(ruleset_fd, fd, _READ_FILE)
            finally:
                os.close(fd)
        if _libc().prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        if _libc().syscall(restrict, ruleset_fd, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    finally:
        os.close(ruleset_fd)


def _rsync_argv(source_fd: int, destination_fd: int, protected_paths: tuple[str, ...]) -> list[str]:
    argv = [_RSYNC, "--recursive", "--links", "--perms", "--times", "--xattrs", "--checksum", "--modify-window=-1", "--delete-delay"]
    for path in protected_paths:
        # rsync filter syntax: a leading slash anchors it; escaping avoids
        # glob/filter metacharacters broadening a protected descendant.
        escaped = "".join("\\" + char if char in "*?[]\\" else char for char in path)
        argv.extend(("--exclude", f"/{escaped}", "--exclude", f"/{escaped}/***"))
    return [*argv, "--", f"/proc/self/fd/{source_fd}/", f"/proc/self/fd/{destination_fd}/"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fd", type=int, required=True)
    parser.add_argument("--destination-fd", type=int, required=True)
    parser.add_argument("--protected-path", action="append", default=[])
    args = parser.parse_args()
    try:
        _validate_directory_fd(args.source_fd)
        _validate_directory_fd(args.destination_fd)
        _validate_distinct_roots(args.source_fd, args.destination_fd)
        _validate_protected_paths(tuple(args.protected_path))
    except Exception as exc:
        print(f"mount-sync launcher validation: {exc}", file=sys.stderr)
        raise SystemExit(_EXIT_INVALID)
    try:
        _install_landlock(args.source_fd, args.destination_fd)
    except OSError as exc:
        if exc.errno in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EPERM, errno.EACCES}:
            print(f"mount-sync confinement unavailable: {exc}", file=sys.stderr)
            raise SystemExit(_EXIT_UNAVAILABLE)
        print(f"mount-sync launcher failure: {exc}", file=sys.stderr)
        raise SystemExit(_EXIT_INVALID)
    except Exception as exc:
        print(f"mount-sync launcher failure: {exc}", file=sys.stderr)
        raise SystemExit(_EXIT_INVALID)
    try:
        os.execve(_RSYNC, _rsync_argv(args.source_fd, args.destination_fd, tuple(args.protected_path)), {"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
    except FileNotFoundError as exc:
        print(f"mount-sync rsync unavailable before execution: {exc}", file=sys.stderr)
        raise SystemExit(_EXIT_UNAVAILABLE)
    except OSError as exc:
        print(f"mount-sync exec failure: {exc}", file=sys.stderr)
        raise SystemExit(_EXIT_INVALID)


if __name__ == "__main__":
    main()
