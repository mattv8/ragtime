"""Per-workspace sandbox using Linux namespaces + pivot_root/chroot.

This module is the single entry point for all process spawning inside
workspace sessions.  Every user-facing command (PTY shell, devserver,
bootstrap, one-shot exec) **must** go through :func:`spawn_sandboxed` or
:func:`prepare_sandbox_pty`.

Architecture
------------
* When ``CAP_SYS_ADMIN`` (or equivalent mount authority) is available the
  sandbox uses ``pivot_root`` for strongest confinement.
* Otherwise it falls back to ``chroot`` with additional path-escape
  prevention (private mount namespace, read-only bind mounts of host
  system directories, and ``/proc`` remount inside the sandbox).
* A per-workspace ``rootfs`` tree is lazily provisioned under
  ``<workspace_dir>/../rootfs/`` with read-only bind mounts of
  ``/bin``, ``/usr``, ``/lib``, ``/lib64``, ``/etc`` from the host
  container, plus a writable ``/tmp`` and ``/workspace`` (the project
  files).

All of this runs inside the existing runtime Docker container — no
Firecracker, Docker-in-Docker, LXC, or k8s primitives are used.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import ctypes.util
import errno
import json
import logging
import os
import posixpath
import shutil
import signal
import stat
import struct
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Callable, Sequence

from ragtime.core.file_constants import DEFAULT_EXCLUDE_DIR_NAMES, GENERATED_BYTECODE_EXTENSIONS

from ..core.shared import has_cap_sys_admin
from .mount_sync import MountSyncError, MountSyncUnavailable, mount_sync_available, sync_copied_mount

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Clone flag constants (from <linux/sched.h>)
CLONE_NEWNS = 0x00020000  # New mount namespace
CLONE_NEWUTS = 0x04000000  # New UTS namespace
CLONE_NEWIPC = 0x08000000  # New IPC namespace
CLONE_NEWPID = 0x20000000  # New PID namespace
CLONE_NEWNET = 0x40000000  # New network namespace
CLONE_NEWUSER = 0x10000000  # New user namespace

# Mount flags
MS_BIND = 4096
MS_REC = 16384
MS_RDONLY = 1
MS_REMOUNT = 32
MS_PRIVATE = 1 << 18  # 262144
MS_NOSUID = 2
MS_NODEV = 4
MS_NOEXEC = 8
MNT_DETACH = 2

# Syscall numbers (x86_64)
SYS_MOUNT = 165
SYS_UMOUNT2 = 166
SYS_UNSHARE = 272

# prctl(2) constants
PR_CAPBSET_DROP = 24
PR_SET_PDEATHSIG = 1
PR_SET_NO_NEW_PRIVS = 38
LINUX_CAPABILITY_VERSION_3 = 0x20080522
_CAPABILITY_WORDS = 2
_MAX_CAPABILITY_INDEX = 63
_SANDBOX_LAUNCH_PROTOCOL_VERSION = 1
_MAX_LAUNCH_RECORD_BYTES = 64 * 1024
_SANDBOX_LAUNCH_STARTUP_TIMEOUT_SECONDS = 10.0
_SANDBOX_CGROUP_PIDS_FLOOR = 64
_SANDBOX_CGROUP_PIDS_MIN = 128
_SANDBOX_CGROUP_PIDS_DEFAULT = 512
_SANDBOX_CGROUP_PIDS_MAX = 1024
_SANDBOX_CGROUP_PIDS_RESERVE = 128
_SANDBOX_CGROUP_PARENT = "/sys/fs/cgroup/ragtime-sandboxes"
_STARTUP_CONCURRENCY_MAX = 4
_STARTUP_CONCURRENCY_CPU_DIVISOR = 2
_STARTUP_CONCURRENCY_BYTES_PER_SLOT = 2 * 1024 * 1024 * 1024

# Directories from the host container to bind-mount read-only into each
# workspace rootfs (lightweight — no copy, shared pages).
_HOST_RO_BIND_DIRS = ["/bin", "/usr", "/lib", "/sbin"]
# /lib64 may not exist on all images
_HOST_RO_BIND_DIRS_OPTIONAL = ["/lib64", "/lib32", "/libx32"]
# Directories created writable inside the sandbox rootfs
_SANDBOX_WRITABLE_DIRS = [
    "/tmp",
    "/var",
    "/var/tmp",
    "/run",
    "/dev",
    "/dev/pts",
    "/dev/shm",
]
# Path inside sandbox where project files are mounted
SANDBOX_WORKSPACE_MOUNT = "/workspace"
# Path inside sandbox where /proc is mounted
SANDBOX_PROC_MOUNT = "/proc"
_WORKSPACE_LEGACY_RECOVERY_DIR = "_legacy_workspace_recoveries"
# Marker file recording the sandbox layout the workspace was last provisioned
# with.  Lives at the workspace root (sibling of ``files/`` and ``rootfs/``)
# so it survives both kinds of layout migration.  This gives bootstrap-time
# code a single, queryable source of truth for the previous-vs-current
# sandbox mode, which is what we use to detect host-level capability flips
# (for example, the operator toggling ``cap_add: [SYS_ADMIN]`` or
# ``privileged: true`` in compose) and react to them rather than relying on
# mtime heuristics across two co-existing copies of the workspace tree.
_SANDBOX_LAYOUT_MARKER_FILENAME = "_ragtime_sandbox_layout.json"
_SANDBOX_LAYOUT_MARKER_VERSION = 1
# Indexing/search exclusions used when mirroring or scanning a workspace for
# discovery; intentionally broad to keep search indexes lean.
_WORKSPACE_SYNC_SKIP_DIRS = DEFAULT_EXCLUDE_DIR_NAMES
# Recovery migrates files that were previously written only into the legacy
# rootfs workspace copy back into canonical workspace files.  Default to
# preserving everything (runtime dependencies, build outputs, framework caches,
# config dirs) and only skip directories that are unsafe or wasteful to
# migrate: VCS metadata and pure tool caches that re-generate on demand.
# This policy is intentionally independent of indexing excludes.
_WORKSPACE_RECOVERY_SKIP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        ".cache",
        "coverage",
    }
)
_WORKSPACE_SYNC_SKIP_SUFFIXES = GENERATED_BYTECODE_EXTENSIONS

# Minimal /usr payload required for chroot fallback operation when mounts are
# unavailable (non-CAP_SYS_ADMIN mode without mount namespace).
_CHROOT_USR_INCLUDE_PATHS = (
    "bin",
    "sbin",
    "lib",
    "libexec",
    "local/bin",
    "local/sbin",
    "local/lib",
    "share/nodejs",
    "share/zoneinfo",
    "share/terminfo",
)
_CHROOT_USR_SYNC_VERSION = "8"
_CHROOT_USR_SYNC_STAMP = ".ragtime_usr_sync_version"
_SANDBOX_SYSTEM_SYNC_MARKER_FILENAME = "_ragtime_sandbox_system_sync.json"
_SANDBOX_SYSTEM_SYNC_MARKER_VERSION = 1

# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

_libc_name = ctypes.util.find_library("c")
_libc = ctypes.CDLL(_libc_name or "libc.so.6", use_errno=True)


def _can_unshare_flags(flags: int) -> bool:
    """Probe whether unshare(flags) is permitted without perturbing this process."""
    try:
        # CPython 3.12+ warns that fork() in a multi-threaded process can lead
        # to deadlocks. This probe is intentionally short-lived (immediate
        # _exit) and touches no locks or shared state, so the warning is benign here.
        # So we suppress it locally (but not globally).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
    except Exception:
        return False
    if pid == 0:
        try:
            ret = _libc.unshare(flags)
        except Exception:
            os._exit(1)
        os._exit(0 if ret == 0 else 1)
    _, status = os.waitpid(pid, 0)
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


@dataclass
class SandboxCapabilities:
    """Detected sandbox capabilities of the runtime container."""

    has_cap_sys_admin: bool = False
    can_pivot_root: bool = False
    can_user_ns: bool = False
    can_mount: bool = False
    unshare_flags: int = 0
    dropped_unshare_flags: int = 0
    mount_namespace: bool = False
    pid_namespace: bool = False
    uts_namespace: bool = False
    ipc_namespace: bool = False
    cgroup_pids_available: bool = False
    cgroup_pids_parent: str | None = None
    cgroup_pids_max: int | None = None
    drop_capabilities: bool = True
    no_new_privs: bool = True
    mode: str = "unavailable"  # "pivot_root" | "chroot" | "unavailable"

    @property
    def available(self) -> bool:
        return self.mode in ("pivot_root", "chroot")


def _validate_launch_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"sandbox launch field {field_name} must be a string")
    if "\x00" in value:
        raise ValueError(f"sandbox launch field {field_name} must not contain NUL bytes")
    return value


@dataclass(frozen=True)
class SandboxLaunchSpec:
    workspace_id: str
    workspace_files_path: str
    rootfs_path: str
    mode: str
    cwd: str
    argv: tuple[str, ...]
    unshare_flags: int
    pty: bool = False
    has_cap_sys_admin: bool = False
    can_mount: bool = False
    can_pivot_root: bool = False
    cgroup_pids_available: bool = False
    cgroup_pids_parent: str | None = None
    cgroup_pids_max: int | None = None
    drop_capabilities: bool = True
    no_new_privs: bool = True

    def __post_init__(self) -> None:
        for field_name in ("workspace_id", "workspace_files_path", "rootfs_path", "cwd"):
            _validate_launch_string(getattr(self, field_name), field_name)
        if self.cgroup_pids_parent is not None:
            _validate_launch_string(self.cgroup_pids_parent, "cgroup_pids_parent")
        if self.mode not in {"pivot_root", "chroot"}:
            raise ValueError(f"unsupported sandbox launch mode: {self.mode}")
        if not self.argv:
            raise ValueError("sandbox launch argv must not be empty")
        for index, value in enumerate(self.argv):
            _validate_launch_string(value, f"argv[{index}]")
        if not isinstance(self.unshare_flags, int):
            raise ValueError("sandbox launch field unshare_flags must be an integer")
        for field_name in (
            "pty",
            "has_cap_sys_admin",
            "can_mount",
            "can_pivot_root",
            "cgroup_pids_available",
            "drop_capabilities",
            "no_new_privs",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"sandbox launch field {field_name} must be a boolean")
        if self.cgroup_pids_max is not None and not isinstance(self.cgroup_pids_max, int):
            raise ValueError("sandbox launch field cgroup_pids_max must be an integer or None")

    def to_record(self) -> dict[str, Any]:
        return {
            "version": _SANDBOX_LAUNCH_PROTOCOL_VERSION,
            "workspace_id": self.workspace_id,
            "workspace_files_path": self.workspace_files_path,
            "rootfs_path": self.rootfs_path,
            "mode": self.mode,
            "cwd": self.cwd,
            "argv": list(self.argv),
            "unshare_flags": self.unshare_flags,
            "pty": self.pty,
            "has_cap_sys_admin": self.has_cap_sys_admin,
            "can_mount": self.can_mount,
            "can_pivot_root": self.can_pivot_root,
            "cgroup_pids_available": self.cgroup_pids_available,
            "cgroup_pids_parent": self.cgroup_pids_parent,
            "cgroup_pids_max": self.cgroup_pids_max,
            "drop_capabilities": self.drop_capabilities,
            "no_new_privs": self.no_new_privs,
        }

    def to_sandbox_spec(self) -> SandboxSpec:
        return SandboxSpec(
            workspace_id=self.workspace_id,
            workspace_files_path=Path(self.workspace_files_path),
            rootfs_path=Path(self.rootfs_path),
            mode=self.mode,
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> SandboxLaunchSpec:
        if not isinstance(record, dict):
            raise ValueError("sandbox launch record must be a dictionary")
        version = record.get("version")
        if version != _SANDBOX_LAUNCH_PROTOCOL_VERSION:
            raise ValueError(f"unsupported sandbox launch protocol version: {version}")
        argv = record.get("argv")
        if not isinstance(argv, list):
            raise ValueError("sandbox launch field argv must be a list of strings")
        unshare_flags = record.get("unshare_flags")
        if not isinstance(unshare_flags, int):
            raise ValueError("sandbox launch field unshare_flags must be an integer")
        return cls(
            workspace_id=_validate_launch_string(record.get("workspace_id"), "workspace_id"),
            workspace_files_path=_validate_launch_string(record.get("workspace_files_path"), "workspace_files_path"),
            rootfs_path=_validate_launch_string(record.get("rootfs_path"), "rootfs_path"),
            mode=_validate_launch_string(record.get("mode"), "mode"),
            cwd=_validate_launch_string(record.get("cwd"), "cwd"),
            argv=tuple(_validate_launch_string(value, f"argv[{index}]") for index, value in enumerate(argv)),
            unshare_flags=unshare_flags,
            pty=record.get("pty", False),
            has_cap_sys_admin=record.get("has_cap_sys_admin", False),
            can_mount=record.get("can_mount", False),
            can_pivot_root=record.get("can_pivot_root", False),
            cgroup_pids_available=record.get("cgroup_pids_available", False),
            cgroup_pids_parent=record.get("cgroup_pids_parent"),
            cgroup_pids_max=record.get("cgroup_pids_max"),
            drop_capabilities=record.get("drop_capabilities", True),
            no_new_privs=record.get("no_new_privs", True),
        )


@dataclass(frozen=True)
class SandboxLaunchStatus:
    stage: str
    errno: int | None
    message: str

    def __post_init__(self) -> None:
        _validate_launch_string(self.stage, "stage")
        _validate_launch_string(self.message, "message")
        if self.errno is not None and not isinstance(self.errno, int):
            raise ValueError("sandbox launch field errno must be an integer or None")

    def to_record(self) -> dict[str, Any]:
        return {
            "version": _SANDBOX_LAUNCH_PROTOCOL_VERSION,
            "stage": self.stage,
            "errno": self.errno,
            "message": self.message,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> SandboxLaunchStatus:
        if not isinstance(record, dict):
            raise ValueError("sandbox launch status record must be a dictionary")
        version = record.get("version")
        if version != _SANDBOX_LAUNCH_PROTOCOL_VERSION:
            raise ValueError(f"unsupported sandbox launch protocol version: {version}")
        return cls(
            stage=_validate_launch_string(record.get("stage"), "stage"),
            errno=record.get("errno"),
            message=_validate_launch_string(record.get("message"), "message"),
        )


class SandboxLaunchError(RuntimeError):
    def __init__(self, status: SandboxLaunchStatus):
        self.status = status
        super().__init__(f"sandbox launch failed during {status.stage}: {status.message}")


def _encode_launch_record(record: dict[str, Any]) -> bytes:
    payload = json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(payload) > _MAX_LAUNCH_RECORD_BYTES:
        raise ValueError(f"launch record exceeds {_MAX_LAUNCH_RECORD_BYTES} bytes")
    return struct.pack(">I", len(payload)) + payload


def _decode_launch_record(payload: bytes) -> dict[str, Any]:
    if len(payload) < 4:
        raise ValueError("sandbox launch record is truncated")
    encoded_size = struct.unpack(">I", payload[:4])[0]
    body = payload[4:]
    if encoded_size != len(body):
        raise ValueError("sandbox launch record length prefix does not match payload size")
    if encoded_size > _MAX_LAUNCH_RECORD_BYTES:
        raise ValueError(f"launch record exceeds {_MAX_LAUNCH_RECORD_BYTES} bytes")
    record = json.loads(body.decode("utf-8"))
    if not isinstance(record, dict):
        raise ValueError("sandbox launch record must decode to a dictionary")
    return record


def _launch_spec_from_spawn_request(
    spec: SandboxSpec,
    command: Sequence[str],
    *,
    cwd: str | None,
    pty: bool,
    caps: SandboxCapabilities,
) -> SandboxLaunchSpec:
    return SandboxLaunchSpec(
        workspace_id=spec.workspace_id,
        workspace_files_path=str(spec.workspace_files_path),
        rootfs_path=str(spec.rootfs_path),
        mode=spec.mode,
        cwd=cwd or spec.sandbox_workspace,
        argv=tuple(command),
        unshare_flags=caps.unshare_flags,
        pty=pty,
        has_cap_sys_admin=caps.has_cap_sys_admin,
        can_mount=caps.can_mount,
        can_pivot_root=caps.can_pivot_root and spec.mode == "pivot_root",
        cgroup_pids_available=caps.cgroup_pids_available,
        cgroup_pids_parent=caps.cgroup_pids_parent,
        cgroup_pids_max=caps.cgroup_pids_max,
        drop_capabilities=caps.drop_capabilities,
        no_new_privs=caps.no_new_privs,
    )


def _capabilities_from_launch_spec(spec: SandboxLaunchSpec) -> SandboxCapabilities:
    launch_flags = spec.unshare_flags
    return SandboxCapabilities(
        has_cap_sys_admin=spec.has_cap_sys_admin,
        can_pivot_root=spec.can_pivot_root and spec.mode == "pivot_root",
        can_user_ns=bool(launch_flags & CLONE_NEWUSER),
        can_mount=spec.can_mount,
        unshare_flags=launch_flags,
        dropped_unshare_flags=0,
        mount_namespace=bool(launch_flags & CLONE_NEWNS),
        pid_namespace=bool(launch_flags & CLONE_NEWPID),
        uts_namespace=bool(launch_flags & CLONE_NEWUTS),
        ipc_namespace=bool(launch_flags & CLONE_NEWIPC),
        cgroup_pids_available=spec.cgroup_pids_available,
        cgroup_pids_parent=spec.cgroup_pids_parent,
        cgroup_pids_max=spec.cgroup_pids_max,
        drop_capabilities=spec.drop_capabilities,
        no_new_privs=spec.no_new_privs,
        mode=spec.mode,
    )


def _pipe_cloexec() -> tuple[int, int]:
    if hasattr(os, "pipe2"):
        return os.pipe2(os.O_CLOEXEC)
    return os.pipe()


def _write_all_fd(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError(errno.EIO, "short write to sandbox launch pipe")
        offset += written


def _read_exact_fd(fd: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            raise ValueError("sandbox launch record is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_launch_record_from_fd(fd: int) -> dict[str, Any] | None:
    header = os.read(fd, 4)
    if not header:
        return None
    if len(header) != 4:
        raise ValueError("sandbox launch record is truncated")
    payload_size = struct.unpack(">I", header)[0]
    if payload_size > _MAX_LAUNCH_RECORD_BYTES:
        raise ValueError(f"launch record exceeds {_MAX_LAUNCH_RECORD_BYTES} bytes")
    payload = _read_exact_fd(fd, payload_size)
    return _decode_launch_record(header + payload)


def _read_launch_spec_from_fd(fd: int) -> SandboxLaunchSpec:
    record = _read_launch_record_from_fd(fd)
    if record is None:
        raise ValueError("sandbox launch spec pipe closed before sending data")
    return SandboxLaunchSpec.from_record(record)


def _read_launch_status_from_fd(fd: int) -> SandboxLaunchStatus | None:
    record = _read_launch_record_from_fd(fd)
    if record is None:
        return None
    return SandboxLaunchStatus.from_record(record)


def _write_launch_status_to_fd(fd: int, status: SandboxLaunchStatus) -> None:
    _write_all_fd(fd, _encode_launch_record(status.to_record()))


async def terminate_process_group(
    process: asyncio.subprocess.Process,
    *,
    timeout: float = 2.0,
) -> None:
    if process.returncode is not None:
        return

    def _signal_process_group(signum: signal.Signals) -> bool:
        try:
            os.killpg(os.getpgid(process.pid), signum)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            signal_process = process.terminate if signum == signal.SIGTERM else process.kill
            try:
                signal_process()
                return True
            except ProcessLookupError:
                return False

    if not _signal_process_group(signal.SIGTERM):
        with contextlib.suppress(Exception):
            await process.wait()
        return

    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
        return
    except asyncio.TimeoutError:
        pass

    if not _signal_process_group(signal.SIGKILL):
        with contextlib.suppress(Exception):
            await process.wait()
        return

    with contextlib.suppress(Exception):
        await process.wait()


async def _cleanup_failed_launcher_process(process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(Exception):
        await terminate_process_group(process, timeout=1.0)


_capabilities_cache: dict[str, SandboxCapabilities] = {}
_rootfs_provision_locks: dict[str, threading.Lock] = {}
_rootfs_provision_locks_guard = threading.Lock()


def _unshare_flag_names(flags: int) -> list[str]:
    return [name for bit, name in _UNSHARE_FLAG_NAMES if flags & bit]


def _read_cgroup_limit(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _detect_memory_limit_bytes() -> int | None:
    cgroup_value = _read_cgroup_limit(Path("/sys/fs/cgroup/memory.max"))
    if cgroup_value is not None:
        return cgroup_value
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if not line.startswith("MemTotal:"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                return int(parts[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _runtime_max_sessions() -> int:
    raw = os.getenv("RUNTIME_MAX_SESSIONS", "12").strip()
    try:
        value = int(raw)
    except ValueError:
        return 12
    return value if value > 0 else 12


def _calculate_sandbox_pids_max(
    *,
    total_pids_limit: int | None,
    memory_limit_bytes: int | None,
    cpu_count: int | None,
    max_sessions: int,
) -> int:
    cpu = max(1, int(cpu_count or 1))
    memory_gib = 0
    if memory_limit_bytes is not None and memory_limit_bytes > 0:
        memory_gib = max(1, memory_limit_bytes // (1024 * 1024 * 1024))

    resource_target = _SANDBOX_CGROUP_PIDS_DEFAULT
    if cpu_count is not None or memory_limit_bytes is not None:
        resource_target = (cpu * 64) + (memory_gib * 64)
    resource_target = max(_SANDBOX_CGROUP_PIDS_MIN, min(_SANDBOX_CGROUP_PIDS_MAX, resource_target))

    if total_pids_limit is None:
        return resource_target

    session_count = max(1, int(max_sessions or 1))
    reserve = min(max(_SANDBOX_CGROUP_PIDS_RESERVE, total_pids_limit // 10), _SANDBOX_CGROUP_PIDS_MAX)
    per_session_budget = max(_SANDBOX_CGROUP_PIDS_FLOOR, (max(0, total_pids_limit - reserve) // session_count))
    return max(_SANDBOX_CGROUP_PIDS_FLOOR, min(resource_target, per_session_budget, _SANDBOX_CGROUP_PIDS_MAX))


def _default_sandbox_pids_max(root: Path) -> int:
    return _calculate_sandbox_pids_max(
        total_pids_limit=_read_cgroup_limit(root / "pids.max"),
        memory_limit_bytes=_detect_memory_limit_bytes(),
        cpu_count=os.cpu_count(),
        max_sessions=_runtime_max_sessions(),
    )


def recommended_startup_concurrency() -> int:
    caps = detect_capabilities()
    if not caps.pid_namespace:
        return 1

    cpu_slots = max(1, (os.cpu_count() or 1) // _STARTUP_CONCURRENCY_CPU_DIVISOR)
    memory_limit = _detect_memory_limit_bytes()
    if memory_limit is None:
        memory_slots = cpu_slots
    else:
        memory_slots = max(1, memory_limit // _STARTUP_CONCURRENCY_BYTES_PER_SLOT)
    return max(1, min(_STARTUP_CONCURRENCY_MAX, cpu_slots, memory_slots, _runtime_max_sessions()))


def _detect_cgroup_pids_limit() -> tuple[bool, str | None, int | None]:
    parent = Path(_SANDBOX_CGROUP_PARENT)
    root = parent.parent
    if not (root / "cgroup.controllers").exists():
        return False, None, None
    try:
        controllers = (root / "cgroup.controllers").read_text(encoding="utf-8").split()
    except OSError:
        return False, None, None
    if "pids" not in controllers:
        return False, None, None
    try:
        pids_max = _default_sandbox_pids_max(root)
        parent.mkdir(parents=True, exist_ok=True)
        try:
            (root / "cgroup.subtree_control").write_text("+pids", encoding="utf-8")
        except OSError as exc:
            if exc.errno not in {errno.EBUSY, errno.EPERM, errno.EACCES, errno.EROFS}:
                raise
        probe = parent / ".probe"
        probe.mkdir(exist_ok=True)
        (probe / "pids.max").write_text(str(pids_max), encoding="utf-8")
        try:
            probe.rmdir()
        except OSError:
            pass
        return True, str(parent), pids_max
    except OSError:
        return False, None, None


def detect_capabilities() -> SandboxCapabilities:
    """Detect what sandbox primitives are available (cached after first call)."""
    cached = _capabilities_cache.get("caps")
    if cached is not None:
        return cached

    caps = SandboxCapabilities()
    caps.has_cap_sys_admin = has_cap_sys_admin()
    caps.can_user_ns = _can_unshare_flags(CLONE_NEWUSER)

    requested_flags = CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWIPC | CLONE_NEWPID
    supported_flags = 0
    for flag in (CLONE_NEWNS, CLONE_NEWUTS, CLONE_NEWIPC, CLONE_NEWPID):
        if caps.has_cap_sys_admin and _can_unshare_flags(flag):
            supported_flags |= flag
    if caps.can_user_ns and not caps.has_cap_sys_admin:
        user_mount_flags = CLONE_NEWUSER | CLONE_NEWNS
        if _can_unshare_flags(user_mount_flags):
            supported_flags |= user_mount_flags

    # Probe the final usable combination once. Some kernels/security profiles
    # accept individual flags but reject the combined call used by real spawns.
    candidate_flags = supported_flags & (requested_flags | CLONE_NEWUSER)
    if candidate_flags and not _can_unshare_flags(candidate_flags):
        reduced_flags = 0
        for flag in (CLONE_NEWUSER, CLONE_NEWNS, CLONE_NEWUTS, CLONE_NEWIPC, CLONE_NEWPID):
            trial = reduced_flags | (candidate_flags & flag)
            if trial and _can_unshare_flags(trial):
                reduced_flags = trial
        candidate_flags = reduced_flags

    caps.unshare_flags = candidate_flags
    caps.dropped_unshare_flags = requested_flags & ~candidate_flags
    caps.mount_namespace = bool(candidate_flags & CLONE_NEWNS)
    # CLONE_NEWPID support means the launcher can fork a namespace-local PID 1
    # before exec. The sandbox itself can still pivot_root without it.
    caps.pid_namespace = bool(candidate_flags & CLONE_NEWPID)
    caps.uts_namespace = bool(candidate_flags & CLONE_NEWUTS)
    caps.ipc_namespace = bool(candidate_flags & CLONE_NEWIPC)
    caps.can_mount = caps.mount_namespace and (caps.has_cap_sys_admin or bool(candidate_flags & CLONE_NEWUSER))
    caps.can_pivot_root = caps.has_cap_sys_admin and caps.can_mount

    cgroup_available, cgroup_parent, cgroup_pids_max = _detect_cgroup_pids_limit()
    caps.cgroup_pids_available = cgroup_available
    caps.cgroup_pids_parent = cgroup_parent
    caps.cgroup_pids_max = cgroup_pids_max

    if caps.can_pivot_root:
        caps.mode = "pivot_root"
    elif os.geteuid() == 0:
        caps.mode = "chroot"
    else:
        caps.mode = "unavailable"

    _capabilities_cache["caps"] = caps
    logger.info(
        "Sandbox capabilities detected: mode=%s, cap_sys_admin=%s, user_ns=%s, mount=%s, "
        "pivot_root=%s, unshare_flags=%s, dropped_unshare_flags=%s, pid_namespace=%s, "
        "cgroup_pids=%s, cgroup_pids_max=%s, drop_caps=%s, no_new_privs=%s",
        caps.mode,
        caps.has_cap_sys_admin,
        caps.can_user_ns,
        caps.can_mount,
        caps.can_pivot_root,
        _unshare_flag_names(caps.unshare_flags),
        _unshare_flag_names(caps.dropped_unshare_flags),
        caps.pid_namespace,
        caps.cgroup_pids_available,
        caps.cgroup_pids_max,
        caps.drop_capabilities,
        caps.no_new_privs,
    )
    if caps.dropped_unshare_flags:
        logger.warning(
            "Sandbox namespace support degraded; launcher will run without: %s",
            _unshare_flag_names(caps.dropped_unshare_flags),
        )
    return caps


def _pivot_root_unshare_flags(caps: SandboxCapabilities) -> int:
    if caps.unshare_flags:
        return caps.unshare_flags
    flags = CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWIPC | CLONE_NEWPID
    if caps.can_user_ns and not caps.has_cap_sys_admin:
        flags |= CLONE_NEWUSER
    return flags


def _rootfs_provision_lock(rootfs_path: Path) -> threading.Lock:
    key = str(rootfs_path)
    with _rootfs_provision_locks_guard:
        lock = _rootfs_provision_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _rootfs_provision_locks[key] = lock
        return lock


# ---------------------------------------------------------------------------
# Rootfs provisioning
# ---------------------------------------------------------------------------


@dataclass
class SandboxSpec:
    """Describes the sandbox layout for a workspace session."""

    workspace_id: str
    workspace_files_path: Path  # Host path: .../workspaces/<id>/files
    rootfs_path: Path  # Host path: .../workspaces/<id>/rootfs
    sandbox_workspace: str = SANDBOX_WORKSPACE_MOUNT  # Path inside sandbox
    mode: str = "chroot"  # "pivot_root" | "chroot"


def workspace_mirror_required(spec: SandboxSpec, caps: SandboxCapabilities) -> bool:
    if caps.can_mount:
        return False
    return True


def _chroot_system_sync_required(spec: SandboxSpec, caps: SandboxCapabilities) -> bool:
    return spec.mode == "chroot" and caps.mode == "chroot" and not caps.can_mount


def _sandbox_layout_marker_path(spec: SandboxSpec) -> Path:
    return spec.rootfs_path.parent / _SANDBOX_LAYOUT_MARKER_FILENAME


def _read_sandbox_layout_marker(spec: SandboxSpec) -> dict[str, Any] | None:
    """Return the recorded layout marker for ``spec`` or None if absent/invalid.

    The marker is the single source of truth for the sandbox mode this
    workspace was last provisioned with.  It allows future bootstrap
    changes to detect cross-mode transitions (for example, the operator
    flipping CAP_SYS_ADMIN / ``privileged: true`` in compose) without
    relying on heuristics over the on-disk workspace tree.
    """
    path = _sandbox_layout_marker_path(spec)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning(
            "Ignoring unreadable sandbox layout marker for %s at %s: %s",
            spec.workspace_id,
            path,
            exc,
        )
        return None
    if not isinstance(data, dict):
        return None
    return data


def _write_sandbox_layout_marker(spec: SandboxSpec, caps: SandboxCapabilities) -> None:
    """Persist the current sandbox mode for ``spec``.

    Called after successful provisioning and after a successful in-mode
    cleanup-time reconcile, so a subsequent provision can tell whether
    the runtime's capability profile has changed since the last session.
    """
    path = _sandbox_layout_marker_path(spec)
    payload = {
        "version": _SANDBOX_LAYOUT_MARKER_VERSION,
        "mode": caps.mode,
        "can_mount": bool(caps.can_mount),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning(
            "Failed to persist sandbox layout marker for %s at %s: %s",
            spec.workspace_id,
            path,
            exc,
        )


def _safe_legacy_archive_path(workspace_root: Path, label: str) -> Path:
    archive_root = workspace_root / _WORKSPACE_LEGACY_RECOVERY_DIR
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base = archive_root / f"{label}-{timestamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = Path(f"{base}-{suffix}")
    return candidate


def _copy_workspace_symlink(src: Path, dst: Path) -> str:
    target = os.readlink(src)
    if dst.is_symlink() and os.readlink(dst) == target:
        return "same"
    if dst.exists() and not dst.is_symlink():
        return "preserved_canonical"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    os.symlink(target, dst)
    shutil.copystat(src, dst, follow_symlinks=False)
    return "copied"


def _copy_workspace_file_if_needed(src: Path, dst: Path, *, prefer_source: bool = False) -> str:
    """Copy ``src`` to ``dst`` when content differs.

    When ``prefer_source`` is True, the source side is treated as the
    authoritative copy and always wins on conflict.  This is the correct
    semantics for legacy/mode-transition reconciliation, where ``src``
    is the previous authoritative location (e.g. ``rootfs/workspace`` from
    a prior chroot-mode session) and ``dst`` may have been touched by a
    fresh, half-initialized canonical bootstrap.  Comparing mtimes across
    those two snapshots is unsafe and silently discards real data when
    the canonical copy happens to be newer.
    """
    src_stat = src.stat()
    if dst.exists():
        if dst.is_file():
            dst_stat = dst.stat()
            if src_stat.st_size == dst_stat.st_size and _files_have_same_content(src, dst):
                return "same"
            if not prefer_source and src_stat.st_mtime <= dst_stat.st_mtime:
                return "preserved_canonical"
        else:
            return "preserved_canonical"

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.syncing-{os.getpid()}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return "copied"


def _workspace_tree_has_meaningful_content(
    workspace_root: Path,
    canonical_root: Path,
    *,
    skip_dirs: frozenset[str] = _WORKSPACE_RECOVERY_SKIP_DIRS,
) -> bool:
    """Return whether a mirrored workspace contains restorable project content.

    A chroot-only sandbox keeps a copied workspace tree under ``rootfs/workspace``
    because bind mounts are unavailable. That copy is useful for legacy recovery
    when it contains edits that never made it back to the canonical ``files/``
    tree, but it is dangerous to treat every non-empty mirrored tree as
    authoritative because routine bootstrap copies also make it non-empty.

    This helper ignores known cache / VCS directories and sidecar metadata and
    compares the mirrored tree against canonical ``files/`` content. We only
    reconcile when the mirrored tree contains project files that are missing
    from canonical storage or whose bytes differ, which indicates real legacy
    edits rather than a routine bootstrap mirror.
    """
    if not workspace_root.is_dir():
        return False

    for root, dirs, files in os.walk(workspace_root, topdown=True, followlinks=False):
        kept_dirs = []
        for dirname in dirs:
            if dirname in skip_dirs:
                continue
            kept_dirs.append(dirname)
        dirs[:] = kept_dirs

        for filename in files:
            path = Path(root) / filename
            if path.suffix in _WORKSPACE_SYNC_SKIP_SUFFIXES:
                continue
            if filename.endswith(".artifact.json"):
                continue
            relative_path = path.relative_to(workspace_root)
            canonical_path = canonical_root / relative_path
            if not canonical_path.is_file():
                return True
            try:
                if path.stat().st_size != canonical_path.stat().st_size:
                    return True
                if not _files_have_same_content(path, canonical_path):
                    return True
            except OSError:
                return True

    return False


def _files_have_same_content(left: Path, right: Path, *, chunk_size: int = 1024 * 1024) -> bool:
    """Byte-compare files without filecmp's process-global cache."""
    with left.open("rb") as left_file, right.open("rb") as right_file:
        while True:
            left_chunk = left_file.read(chunk_size)
            right_chunk = right_file.read(chunk_size)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _sync_workspace_copy_to_canonical(
    spec: SandboxSpec,
    source_workspace: Path,
    *,
    skip_dirs: frozenset[str] = _WORKSPACE_SYNC_SKIP_DIRS,
    prefer_source: bool = False,
) -> dict[str, int]:
    canonical = spec.workspace_files_path
    stats = {
        "copied": 0,
        "same": 0,
        "preserved_canonical": 0,
        "skipped": 0,
        "errors": 0,
    }
    if not source_workspace.is_dir() or not canonical.is_dir():
        return stats

    for root, dirs, files in os.walk(source_workspace, topdown=True, followlinks=False):
        kept_dirs = []
        for dirname in dirs:
            if dirname in skip_dirs:
                stats["skipped"] += 1
                continue
            kept_dirs.append(dirname)
        dirs[:] = kept_dirs

        root_path = Path(root)
        relative_root = root_path.relative_to(source_workspace)
        for filename in files:
            src = root_path / filename
            if src.suffix in _WORKSPACE_SYNC_SKIP_SUFFIXES:
                stats["skipped"] += 1
                continue
            dst = canonical / relative_root / filename
            try:
                if src.is_symlink():
                    result = _copy_workspace_symlink(src, dst)
                elif src.is_file():
                    result = _copy_workspace_file_if_needed(src, dst, prefer_source=prefer_source)
                else:
                    stats["skipped"] += 1
                    continue
                stats[result] = stats.get(result, 0) + 1
            except Exception as exc:
                stats["errors"] += 1
                logger.warning(
                    "workspace sync failed for %s -> %s in %s: %s",
                    src,
                    dst,
                    spec.workspace_id,
                    exc,
                )
    return stats


def _reconcile_workspace_copy(spec: SandboxSpec, *, label: str, prefer_source: bool = True) -> None:
    source_workspace = spec.rootfs_path / spec.sandbox_workspace.lstrip("/")
    if not source_workspace.is_dir() or not spec.workspace_files_path.is_dir():
        return
    try:
        has_recoverable_content = _workspace_tree_has_meaningful_content(
            source_workspace,
            spec.workspace_files_path,
        )
    except OSError:
        return
    if not has_recoverable_content:
        return

    # ``source_workspace`` is the previous authoritative location for the
    # workspace tree: either a same-mode chroot session that wrote into
    # ``rootfs/workspace`` directly, or a prior chroot-mode session whose
    # data is now stranded under a pivot_root-capable runtime.  In both
    # cases the canonical ``files/`` copy may have been newly initialized
    # by the current mode and contain only a half-baked skeleton, so a
    # source-wins copy is the only safe policy.  See the matching
    # explanation in ``_copy_workspace_file_if_needed``.
    stats = _sync_workspace_copy_to_canonical(
        spec,
        source_workspace,
        skip_dirs=_WORKSPACE_RECOVERY_SKIP_DIRS,
        prefer_source=prefer_source,
    )
    archive = _safe_legacy_archive_path(spec.rootfs_path.parent, label)
    try:
        source_workspace.rename(archive)
    except OSError as exc:
        logger.warning(
            "Failed to archive legacy workspace copy for %s at %s: %s",
            spec.workspace_id,
            source_workspace,
            exc,
        )
        return
    _ensure_real_directory(source_workspace)
    logger.info(
        "Reconciled legacy sandbox workspace for %s: copied=%s same=%s preserved_canonical=%s skipped=%s errors=%s archive=%s",
        spec.workspace_id,
        stats.get("copied", 0),
        stats.get("same", 0),
        stats.get("preserved_canonical", 0),
        stats.get("skipped", 0),
        stats.get("errors", 0),
        archive,
    )


def provision_rootfs(spec: SandboxSpec) -> None:
    """Create the rootfs directory tree for a workspace sandbox.

    This is idempotent — safe to call on every session start.  It creates
    the directory skeleton and records what bind mounts are needed, but
    does NOT perform bind mounts (those happen in the forked child via
    :func:`_setup_sandbox_mounts`).
    """
    rootfs = spec.rootfs_path
    rootfs.mkdir(parents=True, exist_ok=True)

    # Create mount-point directories for host read-only binds
    for d in _HOST_RO_BIND_DIRS + _HOST_RO_BIND_DIRS_OPTIONAL:
        target = rootfs / d.lstrip("/")
        target.mkdir(parents=True, exist_ok=True)

    # Create writable dirs
    for d in _SANDBOX_WRITABLE_DIRS:
        target = rootfs / d.lstrip("/")
        target.mkdir(parents=True, exist_ok=True)

    # /proc mount point
    proc_dir = rootfs / "proc"
    proc_dir.mkdir(parents=True, exist_ok=True)

    # /workspace mount point (project files). Older chroot-only runtimes
    # wrote directly into this rootfs copy; reconcile it before the mount
    # capable path shadows it with the canonical files/ bind mount.  When
    # the workspace was last provisioned in a different sandbox mode
    # (typically because the host's capability profile changed — for
    # example, the operator enabled CAP_SYS_ADMIN or ``privileged: true``
    # in compose), this reconcile is what migrates the previous
    # authoritative copy into the new one.  The marker below records the
    # mode we're provisioning under so the next provision can detect the
    # transition without inspecting the workspace tree itself.
    ws_dir = rootfs / spec.sandbox_workspace.lstrip("/")
    caps = detect_capabilities()
    previous_marker = _read_sandbox_layout_marker(spec)
    previous_mode = (previous_marker or {}).get("mode") if previous_marker else None
    if previous_mode and previous_mode != caps.mode:
        logger.info(
            "Sandbox layout transition for %s: %s -> %s; running workspace reconcile",
            spec.workspace_id,
            previous_mode,
            caps.mode,
        )
    _reconcile_workspace_copy(spec, label="chroot-workspace")
    _ensure_real_directory(ws_dir)

    # In mount-capable sandbox modes the child bind-mounts the real
    # workspace over this directory, so mirroring would only add startup I/O.
    # Chroot fallback without mounts still needs a copied workspace tree.
    workspace_src = spec.workspace_files_path
    if workspace_mirror_required(spec, caps) and workspace_src.is_dir():
        try:
            shutil.copytree(
                str(workspace_src),
                str(ws_dir),
                dirs_exist_ok=True,
                symlinks=True,
                ignore_dangling_symlinks=True,
                copy_function=_copy_file_if_changed,
            )
        except Exception as exc:
            logger.warning("provision_rootfs: workspace mirror failed: %s", exc)

    # Minimal /etc files needed for basic operation
    _provision_etc(rootfs)

    # Create /dev/null, /dev/zero, /dev/urandom, /dev/random stubs
    _provision_dev(rootfs)

    # /home directory for user shells
    home_dir = rootfs / "home"
    home_dir.mkdir(parents=True, exist_ok=True)

    # /root for root user home
    root_home = rootfs / "root"
    root_home.mkdir(parents=True, exist_ok=True)

    # Record the sandbox mode used for this provision so the next session
    # can detect a capability flip (and trigger the layout-transition
    # reconcile above) without inspecting the workspace tree itself.
    _write_sandbox_layout_marker(spec, caps)


def materialize_mounts(
    spec: SandboxSpec,
    mounts: list[dict[str, Any]],
    *,
    clear_targets: Sequence[str] | None = None,
    cancel_event: threading.Event | None = None,
    timeout_seconds: float = 180.0,
) -> None:
    """Copy mount sources into the sandbox rootfs under their target paths.

    Each mount dict must have ``source_local_path`` and ``target_path``.
    Content is copied read-only by default.  When a mount spec requests
    ``runtime_mount_mode=live_bind`` and the runtime has mount authority,
    the source is bind-mounted read-only onto the workspace target instead
    so host-side changes remain visible without rewriting source perms.
    """
    rootfs = spec.rootfs_path
    caps = detect_capabilities()

    def target_parts(target: str) -> tuple[str, ...] | None:
        raw = (target or "").strip().replace("\\", "/")
        if not raw or "\x00" in raw:
            return None
        parts = tuple(part for part in raw.lstrip("/").split("/") if part)
        return parts if parts and all(part not in (".", "..") for part in parts) else None

    def workspace_parts(target: str) -> tuple[str, ...] | None:
        raw = posixpath.normpath((target or "").strip().replace("\\", "/"))
        prefix = SANDBOX_WORKSPACE_MOUNT.rstrip("/") + "/"
        if not raw.startswith(prefix):
            return None
        parts = tuple(part for part in raw[len(prefix) :].split("/") if part)
        return parts if parts and all(part not in (".", "..") for part in parts) else None

    def bind_mount(source: Path, root: Path, parts: tuple[str, ...], read_only: bool) -> None:
        _unmount_pinned_directory(root, parts)
        with _pinned_directory(root, parts, create=True) as dest:
            _syscall_mount(str(source), dest, None, MS_BIND | MS_REC)
        if read_only:
            # Re-open after the bind: the original fd names the covered mount.
            with _pinned_directory(root, parts) as mounted_dest:
                _syscall_mount(str(source), mounted_dest, None, MS_BIND | MS_REMOUNT | MS_RDONLY | MS_REC)

    requested_clear_targets = clear_targets or [str(mount.get("target_path") or "") for mount in mounts]
    copied_mounts: list[tuple[dict[str, Any], tuple[str, ...]]] = []
    for mount in mounts:
        parts = target_parts(str(mount.get("target_path") or ""))
        live_parts = workspace_parts(str(mount.get("target_path") or ""))
        is_explicit_live_bind = str(mount.get("runtime_mount_mode") or "") == "live_bind"
        if mount.get("source_local_path") and parts is not None and not is_explicit_live_bind and not (caps.can_mount and live_parts is not None):
            copied_mounts.append((mount, parts))

    # Reject source/destination aliases before the legacy clear path can touch
    # a source tree.  The sync launcher repeats this check against pinned FDs.
    for mount, parts in copied_mounts:
        source_path = Path(str(mount["source_local_path"]))
        if not source_path.is_dir():
            continue
        source_resolved = source_path.resolve()
        destination_resolved = rootfs.joinpath(*parts).resolve(strict=False)
        if (
            source_resolved == destination_resolved
            or source_resolved.is_relative_to(destination_resolved)
            or destination_resolved.is_relative_to(source_resolved)
        ):
            raise ValueError(f"Workspace mount source and destination overlap: {source_path} -> {rootfs.joinpath(*parts)}")

    # Select this once per refresh: valid copied destinations stay intact only
    # when the confined helper is available for the complete refresh.
    sync_available = mount_sync_available() if copied_mounts else False
    copied_target_parts = {parts for mount, parts in copied_mounts if Path(str(mount["source_local_path"])).is_dir()}

    for target in requested_clear_targets:
        parts = target_parts(str(target))
        if parts is None or parts in copied_target_parts:
            continue
        try:
            live_parts = workspace_parts(str(target))
            if live_parts is not None:
                _unmount_pinned_directory(spec.workspace_files_path, live_parts)
            _clear_pinned_directory(rootfs, parts)
        except OSError as exc:
            logger.warning("materialize_mounts: rejected unsafe target %s: %s", target, exc)

    ordered_mounts = sorted(
        mounts,
        key=lambda mount: len(target_parts(str(mount.get("target_path") or "")) or ()),
    )
    for mount in ordered_mounts:
        source, target = mount.get("source_local_path", ""), mount.get("target_path", "")
        parts = target_parts(str(target))
        if not source or parts is None:
            continue
        source_path = Path(source)
        live_parts = workspace_parts(str(target))
        if not source_path.is_dir():
            if str(mount.get("runtime_mount_mode") or "") == "live_bind":
                raise FileNotFoundError(f"Workspace mount source is not available in the runtime container: {source}")
            logger.warning("materialize_mounts: Workspace mount source is not available: %s", source)
            continue
        if str(mount.get("runtime_mount_mode") or "") == "live_bind" and live_parts is None:
            raise ValueError(f"Live workspace mount target must be under {SANDBOX_WORKSPACE_MOUNT}: {target}")
        if caps.can_mount and live_parts is not None:
            bind_mount(source_path, spec.workspace_files_path, live_parts, bool(mount.get("read_only", True)))
            continue
        if str(mount.get("runtime_mount_mode") or "") == "live_bind":
            raise PermissionError("Live workspace mounts require runtime mount authority. Enable SYS_ADMIN or privileged mode for the runtime container.")
        if sync_available:
            _prepare_copied_mount_target(rootfs, parts, caps)
            protected_paths = _nested_mount_protected_paths(parts, mounts, target_parts)
            try:
                with (
                    _pinned_directory_fd(source_path, ()) as source_fd,
                    _pinned_directory_fd(rootfs, parts) as destination_fd,
                ):
                    _validate_pinned_source_generation(source_path, source_fd)
                    try:
                        sync_copied_mount(
                            source_fd,
                            destination_fd,
                            protected_paths=protected_paths,
                            cancel_event=cancel_event,
                            timeout_seconds=timeout_seconds,
                        )
                    except MountSyncUnavailable:
                        logger.warning(
                            "materialize_mounts: confined rsync unavailable for %s; using full copy fallback",
                            target,
                        )
                        _validate_pinned_source_generation(source_path, source_fd)
                        _copy_mount_fallback(Path(f"/proc/self/fd/{source_fd}"), rootfs, parts)
                        _validate_pinned_source_generation(source_path, source_fd)
                    else:
                        _validate_pinned_source_generation(source_path, source_fd)
            except OSError:
                # Pinned-target and rsync-path failures are materialization
                # failures, never a warning that allows startup to continue.
                raise
        else:
            logger.warning(
                "materialize_mounts: confined rsync unavailable; using full copy fallback for %s",
                target,
            )
            try:
                with _pinned_directory_fd(source_path, ()) as source_fd:
                    _validate_pinned_source_generation(source_path, source_fd)
                    _copy_mount_fallback(Path(f"/proc/self/fd/{source_fd}"), rootfs, parts)
                    _validate_pinned_source_generation(source_path, source_fd)
            except OSError as exc:
                logger.warning("materialize_mounts: fallback copy failed %s -> %s: %s", source, target, exc)


def _nested_mount_protected_paths(
    parent_parts: tuple[str, ...],
    mounts: Sequence[dict[str, Any]],
    parse_target: Callable[[str], tuple[str, ...] | None],
) -> tuple[str, ...]:
    """Find minimal literal descendant targets that a parent sync must skip."""
    candidates = sorted(
        {
            "/".join(parts[len(parent_parts) :])
            for mount in mounts
            if (parts := parse_target(str(mount.get("target_path") or ""))) and len(parts) > len(parent_parts) and parts[: len(parent_parts)] == parent_parts
        },
        key=lambda value: (value.count("/"), value),
    )
    protected: list[str] = []
    for candidate in candidates:
        if not any(candidate == prefix or candidate.startswith(prefix + "/") for prefix in protected):
            protected.append(candidate)
    return tuple(protected)


def _prepare_copied_mount_target(
    rootfs: Path,
    parts: tuple[str, ...],
    caps: SandboxCapabilities,
) -> None:
    """Ensure a retained copied target is a safe real directory for rsync."""
    target = rootfs.joinpath(*parts)
    if os.path.ismount(target):
        if not caps.can_mount:
            raise PermissionError(f"Refusing to sync into mounted target without mount authority: {target}")
        _unmount_pinned_directory(rootfs, parts)
    with _pinned_directory_fd(rootfs, parts[:-1], create=True) as parent_fd:
        try:
            entry = os.lstat(parts[-1], dir_fd=parent_fd)
        except FileNotFoundError:
            entry = None
    if entry is not None and not stat.S_ISDIR(entry.st_mode):
        _clear_pinned_directory(rootfs, parts)
    with _pinned_directory_fd(rootfs, parts, create=True):
        pass


def _copy_mount_fallback(source: Path, rootfs: Path, parts: tuple[str, ...]) -> None:
    """Perform the existing full-copy behavior, clearing immediately before it."""
    _clear_pinned_directory(rootfs, parts)
    with _pinned_directory(rootfs, parts, create=True) as destination:
        shutil.copytree(
            str(source),
            destination,
            dirs_exist_ok=True,
            symlinks=True,
            ignore_dangling_symlinks=True,
            copy_function=_copy_file,
        )


def _validate_pinned_source_generation(source_path: Path, source_fd: int) -> None:
    """Reject cache-root replacement while a copied mount is materialized."""
    pinned = os.fstat(source_fd)
    try:
        current = os.lstat(source_path)
    except OSError as exc:
        raise MountSyncError(f"Workspace mount source generation disappeared: {source_path}") from exc
    if (
        not stat.S_ISDIR(pinned.st_mode)
        or pinned.st_nlink == 0
        or not stat.S_ISDIR(current.st_mode)
        or (pinned.st_dev, pinned.st_ino) != (current.st_dev, current.st_ino)
    ):
        raise MountSyncError(f"Workspace mount source generation changed during materialization: {source_path}")


def _provision_etc(rootfs: Path) -> None:
    """Ensure minimal /etc content exists in the sandbox rootfs."""
    etc = rootfs / "etc"
    etc.mkdir(parents=True, exist_ok=True)

    # /etc/passwd — minimal entries
    passwd = etc / "passwd"
    if not passwd.exists():
        passwd.write_text(
            "root:x:0:0:root:/root:/bin/bash\nnobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n",
            encoding="utf-8",
        )

    # /etc/group
    group = etc / "group"
    if not group.exists():
        group.write_text(
            "root:x:0:\nnogroup:x:65534:\n",
            encoding="utf-8",
        )

    # /etc/hostname
    hostname = etc / "hostname"
    if not hostname.exists():
        hostname.write_text("sandbox\n", encoding="utf-8")

    # /etc/hosts — ensure localhost always resolves inside sandboxed processes.
    # Some user projects (including Vite middleware/HMR internals and DB clients)
    # perform explicit DNS lookups for "localhost". If /etc/hosts is missing,
    # those lookups can fail with ENOTFOUND even though loopback networking works.
    hosts = etc / "hosts"
    if not hosts.exists():
        hosts.write_text(
            "127.0.0.1 localhost\n"
            "::1 localhost ip6-localhost ip6-loopback\n"
            "fe00::0 ip6-localnet\n"
            "ff00::0 ip6-mcastprefix\n"
            "ff02::1 ip6-allnodes\n"
            "ff02::2 ip6-allrouters\n",
            encoding="utf-8",
        )

    # /etc/resolv.conf — copy from host container
    host_resolv = Path("/etc/resolv.conf")
    sandbox_resolv = etc / "resolv.conf"
    if not sandbox_resolv.exists() and host_resolv.exists():
        try:
            sandbox_resolv.write_text(
                host_resolv.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        except Exception:
            sandbox_resolv.write_text("nameserver 127.0.0.1\n", encoding="utf-8")

    # /etc/nsswitch.conf
    nsswitch = etc / "nsswitch.conf"
    if not nsswitch.exists():
        nsswitch.write_text(
            "passwd: files\ngroup: files\nhosts: files dns\n",
            encoding="utf-8",
        )

    # /etc/ld.so.conf — tell the dynamic linker where to find shared libs
    ld_so_conf = etc / "ld.so.conf"
    if not ld_so_conf.exists():
        ld_so_conf.write_text(
            "/usr/local/lib\n/usr/local/lib/x86_64-linux-gnu\n/usr/lib/x86_64-linux-gnu\n/lib/x86_64-linux-gnu\n",
            encoding="utf-8",
        )

    # /etc/ssl — CA certificates so npm/curl/wget can verify TLS
    host_ssl = Path("/etc/ssl")
    sandbox_ssl = etc / "ssl"
    if host_ssl.is_dir() and not sandbox_ssl.exists():
        try:
            shutil.copytree(
                str(host_ssl),
                str(sandbox_ssl),
                symlinks=True,
                ignore_dangling_symlinks=True,
            )
        except Exception:
            pass  # Non-fatal — npm will still work with --strict-ssl=false


def _provision_dev(rootfs: Path) -> None:
    """Create basic /dev device nodes in the sandbox rootfs.

    If mknod fails (common in unprivileged containers) we just create
    regular placeholder files — they will be bind-mounted over from the
    host in the mount phase when CAP_SYS_ADMIN is available.
    """
    dev = rootfs / "dev"
    dev.mkdir(parents=True, exist_ok=True)

    # (name, major, minor, mode)
    devices = [
        ("null", 1, 3, 0o666),
        ("zero", 1, 5, 0o666),
        ("random", 1, 8, 0o666),
        ("urandom", 1, 9, 0o666),
        ("tty", 5, 0, 0o666),
    ]
    for name, major, minor, mode in devices:
        path = dev / name
        if path.exists():
            continue
        try:
            os.mknod(
                str(path),
                stat.S_IFCHR | mode,
                os.makedev(major, minor),
            )
        except (PermissionError, OSError):
            # Fallback: create a regular file placeholder
            path.touch(exist_ok=True)


def _ensure_real_directory(path: Path) -> None:
    """Ensure a path is a real directory (not a symlink or file)."""
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        path.unlink(missing_ok=True)
    path.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def _pinned_directory(root: Path, parts: Sequence[str], *, create: bool = False):
    """Pin a directory below root without following symlink components."""
    with _pinned_directory_fd(root, parts, create=create) as fd:
        yield f"/proc/self/fd/{fd}"


@contextlib.contextmanager
def _pinned_directory_fd(root: Path, parts: Sequence[str], *, create: bool = False):
    """Yield an open, O_NOFOLLOW-pinned directory descriptor below ``root``."""
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(root, flags)
    try:
        for part in parts:
            if part in ("", ".", ".."):
                raise ValueError("unsafe sandbox destination component")
            try:
                next_fd = os.open(part, flags, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, dir_fd=fd)
                next_fd = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


def _unmount_pinned_directory(root: Path, parts: Sequence[str]) -> None:
    """Unmount only: canonical live workspace files must not be cleared."""
    try:
        with _pinned_directory(root, parts) as path:
            _syscall_umount2(path, MNT_DETACH)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, errno.ENOENT}:
            raise


def _clear_pinned_directory(root: Path, parts: Sequence[str]) -> None:
    """Clear a rootfs destination through a pinned parent descriptor."""
    if not parts:
        raise ValueError("refusing to clear sandbox root")
    with _pinned_directory(root, parts[:-1], create=True) as parent_path:
        # ``parent_path`` names the descriptor opened with O_NOFOLLOW above;
        # /proc/self/fd itself is necessarily a symlink and must not be checked
        # as though it were an untrusted destination component.
        fd = os.open(parent_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            try:
                entry = os.lstat(parts[-1], dir_fd=fd)
            except FileNotFoundError:
                return
            if stat.S_ISDIR(entry.st_mode):
                shutil.rmtree(parts[-1], dir_fd=fd)
            else:
                os.unlink(parts[-1], dir_fd=fd)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# Mount helpers (called inside forked child, AFTER unshare)
# ---------------------------------------------------------------------------


def _syscall_mount(
    source: str,
    target: str,
    fstype: str | None,
    flags: int,
    data: str | None = None,
) -> int:
    """Thin wrapper around mount(2) via ctypes."""
    src = source.encode() if source else None
    tgt = target.encode()
    fs = fstype.encode() if fstype else None
    d = data.encode() if data else None
    ret = _libc.mount(src, tgt, fs, flags, d)
    if ret != 0:
        err = ctypes.get_errno()
        errno_name = errno.errorcode.get(err, str(err))
        raise OSError(err, f"mount({source}, {target}, {fstype}, {flags}): {errno_name} {os.strerror(err)}")
    return ret


def _syscall_umount2(target: str, flags: int = 0) -> int:
    tgt = target.encode()
    ret = _libc.umount2(tgt, flags)
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"umount2({target}, {flags}): {os.strerror(err)}")
    return ret


def _syscall_pivot_root(new_root: str, put_old: str) -> int:
    """Thin wrapper around pivot_root(2) via ctypes."""
    ret = _libc.pivot_root(new_root.encode(), put_old.encode())
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"pivot_root({new_root}, {put_old}): {os.strerror(err)}")
    return ret


def _setup_sandbox_mounts(spec: SandboxSpec, *, mount_proc: bool) -> None:
    """Perform bind mounts and /proc mount inside the sandbox.

    MUST be called in a child process that has already done
    ``unshare(CLONE_NEWNS)`` (i.e. private mount namespace).
    """
    rootfs = str(spec.rootfs_path)

    # Make our mount namespace private so changes don't leak out
    _syscall_mount("none", "/", None, MS_REC | MS_PRIVATE)

    # Bind mount rootfs onto itself (required for pivot_root)
    _syscall_mount(rootfs, rootfs, None, MS_BIND | MS_REC)

    # Bind mount host system directories read-only
    for d in _HOST_RO_BIND_DIRS:
        src = d
        dst = os.path.join(rootfs, d.lstrip("/"))
        if not os.path.isdir(src):
            continue
        _syscall_mount(src, dst, None, MS_BIND | MS_REC)
        # Remount read-only
        _syscall_mount(src, dst, None, MS_BIND | MS_REMOUNT | MS_RDONLY | MS_REC)

    for d in _HOST_RO_BIND_DIRS_OPTIONAL:
        src = d
        dst = os.path.join(rootfs, d.lstrip("/"))
        if not os.path.isdir(src):
            continue
        try:
            _syscall_mount(src, dst, None, MS_BIND | MS_REC)
            _syscall_mount(src, dst, None, MS_BIND | MS_REMOUNT | MS_RDONLY | MS_REC)
        except OSError:
            pass  # Optional — not fatal

    # Bind mount project files into /workspace
    ws_src = str(spec.workspace_files_path)
    ws_dst = os.path.join(rootfs, spec.sandbox_workspace.lstrip("/"))

    # Ensure target is a real directory in case older sessions left a symlink.
    ws_dst_path = Path(ws_dst)
    _ensure_real_directory(ws_dst_path)

    _syscall_mount(ws_src, ws_dst, None, MS_BIND | MS_REC)

    # Bind mount host /dev devices we need
    for dev_name in ("null", "zero", "random", "urandom", "tty"):
        src = f"/dev/{dev_name}"
        dst = os.path.join(rootfs, "dev", dev_name)
        if os.path.exists(src):
            try:
                _syscall_mount(src, dst, None, MS_BIND)
            except OSError:
                pass

    # Bind mount /dev/pts
    dev_pts_src = "/dev/pts"
    dev_pts_dst = os.path.join(rootfs, "dev/pts")
    if os.path.isdir(dev_pts_src):
        try:
            _syscall_mount(dev_pts_src, dev_pts_dst, None, MS_BIND)
        except OSError:
            pass

    # Mount a fresh /proc only when this process also has a private PID
    # namespace. Without CLONE_NEWPID, a new procfs exposes the runtime
    # container's process tree to every workspace and makes fork/job-control
    # failures much harder to reason about.
    if mount_proc:
        proc_dst = os.path.join(rootfs, "proc")
        try:
            _syscall_mount("proc", proc_dst, "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC)
        except OSError:
            pass  # Non-fatal; some commands will degrade

    # Mount /dev/shm as tmpfs
    shm_dst = os.path.join(rootfs, "dev/shm")
    try:
        _syscall_mount("tmpfs", shm_dst, "tmpfs", MS_NOSUID | MS_NODEV, "size=64m")
    except OSError:
        pass


def _do_pivot_root(spec: SandboxSpec) -> None:
    """Execute pivot_root(2) to switch the process root."""
    rootfs = str(spec.rootfs_path)
    old_root = os.path.join(rootfs, ".pivot_old")
    os.makedirs(old_root, exist_ok=True)

    # After the self-bind in ``_setup_sandbox_mounts`` the safest kernel-facing
    # form is to chdir into the new root and pivot using paths relative to that
    # mount point. Real privileged launches from disposable tmpfs-backed test
    # workspaces can fail with ESRCH when using the original absolute paths even
    # though the self-bind succeeded and the same topology is otherwise valid.
    # Relative paths keep both arguments anchored to the freshly bound mount.
    os.chdir(rootfs)
    _syscall_pivot_root(".", ".pivot_old")
    os.chdir("/")

    # Unmount old root and remove mount point
    _syscall_umount2("/.pivot_old", 2)  # MNT_DETACH = 2
    try:
        os.rmdir("/.pivot_old")
    except OSError:
        pass


def _do_chroot(spec: SandboxSpec) -> None:
    """Fall back to chroot(2) when pivot_root is not available."""
    rootfs = str(spec.rootfs_path)
    os.chroot(rootfs)
    os.chdir("/")


# ---------------------------------------------------------------------------
# User namespace helpers
# ---------------------------------------------------------------------------


def _setup_user_namespace_mappings() -> None:
    """Write uid/gid mappings when in a user namespace.

    Maps container root (uid 0) -> sandbox uid 0 (1:1 identity mapping).
    """
    uid = os.getuid()
    gid = os.getgid()
    Path("/proc/self/setgroups").write_text("deny", encoding="utf-8")
    Path("/proc/self/uid_map").write_text(f"0 {uid} 1\n", encoding="utf-8")
    Path("/proc/self/gid_map").write_text(f"0 {gid} 1\n", encoding="utf-8")


def _set_parent_death_signal(expected_parent_pid: int) -> None:
    ret = _libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def _set_no_new_privs() -> None:
    ret = _libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    # PR_GET_NO_NEW_PRIVS is available without procfs, unlike status parsing.
    if _libc.prctl(39, 0, 0, 0, 0) != 1:
        err = ctypes.get_errno()
        raise OSError(err or errno.EPERM, "no_new_privs postcondition was not established")


def _sanitize_cgroup_component(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    return sanitized[:120] or "workspace"


def _sandbox_cgroup_path(spec: SandboxSpec, caps: SandboxCapabilities) -> Path | None:
    if not caps.cgroup_pids_available or not caps.cgroup_pids_parent:
        return None
    return Path(caps.cgroup_pids_parent) / _sanitize_cgroup_component(spec.workspace_id)


def _prepare_sandbox_cgroup(spec: SandboxSpec, caps: SandboxCapabilities) -> None:
    cgroup_path = _sandbox_cgroup_path(spec, caps)
    if cgroup_path is None or caps.cgroup_pids_max is None:
        return
    try:
        cgroup_path.mkdir(parents=True, exist_ok=True)
        (cgroup_path / "pids.max").write_text(str(caps.cgroup_pids_max), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to prepare sandbox pids cgroup for %s: %s", spec.workspace_id, exc)


def _assign_current_process_to_sandbox_cgroup(spec: SandboxSpec, caps: SandboxCapabilities) -> None:
    cgroup_path = _sandbox_cgroup_path(spec, caps)
    if cgroup_path is None:
        return
    (cgroup_path / "cgroup.procs").write_text(str(os.getpid()), encoding="utf-8")


def _read_cgroup_process_ids(cgroup_path: Path) -> list[int]:
    try:
        raw = (cgroup_path / "cgroup.procs").read_text(encoding="utf-8")
    except OSError:
        return []
    current_pid = os.getpid()
    process_ids: list[int] = []
    for line in raw.splitlines():
        try:
            process_id = int(line.strip())
        except ValueError:
            continue
        if process_id > 0 and process_id != current_pid:
            process_ids.append(process_id)
    return sorted(set(process_ids))


def _terminate_sandbox_cgroup_processes(
    spec: SandboxSpec,
    caps: SandboxCapabilities,
    *,
    timeout: float = 1.0,
) -> None:
    cgroup_path = _sandbox_cgroup_path(spec, caps)
    if cgroup_path is None or not cgroup_path.exists():
        return

    process_ids = _read_cgroup_process_ids(cgroup_path)
    if not process_ids:
        return

    logger.info(
        "Terminating %s lingering sandbox process(es) for workspace %s",
        len(process_ids),
        spec.workspace_id,
    )
    for process_id in process_ids:
        try:
            os.kill(process_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            logger.debug("Failed to terminate sandbox process %s: %s", process_id, exc)

    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        remaining = _read_cgroup_process_ids(cgroup_path)
        if not remaining:
            return
        time.sleep(0.05)

    remaining = _read_cgroup_process_ids(cgroup_path)
    for process_id in remaining:
        try:
            os.kill(process_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            logger.debug("Failed to kill sandbox process %s: %s", process_id, exc)


class _CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


def _drop_process_capabilities(*, no_new_privs: bool) -> None:
    for cap_index in range(_MAX_CAPABILITY_INDEX + 1):
        ret = _libc.prctl(PR_CAPBSET_DROP, cap_index, 0, 0, 0)
        if ret == 0:
            continue
        err = ctypes.get_errno()
        if err == errno.EINVAL:
            break
        if err not in {errno.EPERM, errno.EACCES}:
            logger.debug("Failed to drop capability %s from bounding set: %s", cap_index, os.strerror(err))

    header = _CapHeader(version=LINUX_CAPABILITY_VERSION_3, pid=0)
    data = (_CapData * _CAPABILITY_WORDS)()
    ret = _libc.capset(ctypes.byref(header), ctypes.byref(data))
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"failed to clear sandbox process capabilities: {os.strerror(err)}")

    readback = (_CapData * _CAPABILITY_WORDS)()
    ret = _libc.capget(ctypes.byref(header), ctypes.byref(readback))
    if ret != 0 or any(word.effective or word.permitted or word.inheritable for word in readback):
        err = ctypes.get_errno()
        raise OSError(err or errno.EPERM, "sandbox capability drop postcondition was not established")

    if no_new_privs:
        _set_no_new_privs()


# ---------------------------------------------------------------------------
# Namespace capability helpers
# ---------------------------------------------------------------------------


_UNSHARE_FLAG_NAMES: tuple[tuple[int, str], ...] = (
    (CLONE_NEWNS, "CLONE_NEWNS"),
    (CLONE_NEWUTS, "CLONE_NEWUTS"),
    (CLONE_NEWIPC, "CLONE_NEWIPC"),
    (CLONE_NEWPID, "CLONE_NEWPID"),
    (CLONE_NEWUSER, "CLONE_NEWUSER"),
)


def _sync_system_dirs_for_chroot(spec: SandboxSpec) -> None:
    """Sync system files only for the no-mount chroot fallback.

    This intentionally pays an independent-inode copy cost on first launch;
    normal starts reuse the persisted rootfs and generation marker. See
    ``docs/userspace-runtime-performance.md`` before optimizing this path:
    product priority is warm starts and public app loading, never writable
    system hardlinks or weaker sandbox isolation.
    """
    rootfs = spec.rootfs_path
    usr_dst = rootfs / "usr"
    usr_stamp = rootfs / _CHROOT_USR_SYNC_STAMP
    usr_stamp_value = ""
    if usr_stamp.exists() and usr_stamp.is_file():
        try:
            usr_stamp_value = usr_stamp.read_text(encoding="utf-8").strip()
        except Exception:
            usr_stamp_value = ""
    usr_needs_sync = usr_stamp_value != _CHROOT_USR_SYNC_VERSION or not usr_dst.exists() or not any(usr_dst.iterdir())
    migration_marker_valid = _system_sync_marker_matches(spec)

    system_dirs = _HOST_RO_BIND_DIRS + _HOST_RO_BIND_DIRS_OPTIONAL
    sync_succeeded = True
    for d in system_dirs:
        src = Path(d)
        if not src.is_dir():
            continue
        dst = rootfs / d.lstrip("/")
        try:
            if src == Path("/usr"):
                if usr_needs_sync:
                    _sync_usr_for_chroot(src, dst, force=True)
                continue
            if not usr_needs_sync and dst.exists() and any(dst.iterdir()):
                # Already populated (from a previous session)
                continue
            else:
                _remove_conflicting_system_symlinks(src, dst)
                shutil.copytree(str(src), str(dst), dirs_exist_ok=True, symlinks=True, ignore_dangling_symlinks=True, copy_function=_copy_system_file_detached)
        except Exception as exc:
            sync_succeeded = False
            logger.warning("Failed to sync %s into rootfs: %s", d, exc)

    # Earlier versions used hard links for regular system files. Detach every
    # remaining multiply-linked regular file, including stale destination-only
    # files, once for each rootfs generation. The marker is host-owned beside
    # rootfs, never the workload-writable in-rootfs version stamp.
    if not migration_marker_valid and not _detach_legacy_system_hardlinks(rootfs, system_dirs):
        sync_succeeded = False

    if usr_needs_sync and sync_succeeded:
        usr_stamp.parent.mkdir(parents=True, exist_ok=True)
        usr_stamp.write_text(_CHROOT_USR_SYNC_VERSION, encoding="utf-8")

    if not sync_succeeded:
        raise RuntimeError("failed to safely synchronize chroot system directories")

    if not migration_marker_valid:
        _write_system_sync_marker(spec)

    # Workspace files are mirrored by provision_rootfs() before the launcher
    # process enters the sandbox. Repeating that copy later in the launcher
    # path after user-namespace setup can lose write access to the root-owned
    # rootfs tree on restored/bind-mounted data directories.


def _sync_usr_for_chroot(src_usr: Path, dst_usr: Path, *, force: bool = False) -> None:
    """Sync a minimal, runtime-focused subset of /usr for chroot fallback.

    Copying all of /usr in no-mount chroot mode can create very large rootfs
    trees.  This function keeps the payload bounded to binaries/libs and
    small runtime metadata needed by common tools (including Node/npm).

    Parameters
    ----------
    force:
        When *True* (typically triggered by a ``_CHROOT_USR_SYNC_VERSION``
        bump), already-populated destination directories are re-synced
        via ``copytree(dirs_exist_ok=True)`` so that newly installed
        binaries (e.g. esbuild) appear in existing rootfs trees.

    Symlink safety
    --------------
    * ``shutil.copytree(symlinks=True)`` preserves symlinks as-is (no
      recursion into symlinked directories).
    * ``ignore_dangling_symlinks=True`` silently skips source symlinks
      whose targets do not exist on the host.
    * ``dirs_exist_ok=True`` allows incremental updates without rmtree.
    * Self-referential symlinks inside ``/usr/share/nodejs`` (e.g.
      ``libnpmteam/node_modules -> ../npm/node_modules``) are preserved
      and resolve correctly after chroot because the full subtree is
      present.
    * External relative symlinks (e.g. ``../../javascript/...``) will
      dangle inside the sandbox — acceptable since those are optional
      assets (prettify, man pages) and Node/npm do not depend on them.
    * Regular files are atomically copied into independent destination inodes;
      symlink conflicts from a forced re-sync are cleared before copytree so
      existing valid symlinks and destination-only package files are retained.
    """
    dst_usr.mkdir(parents=True, exist_ok=True)
    for rel in _CHROOT_USR_INCLUDE_PATHS:
        src = src_usr / rel
        if not src.exists():
            continue
        # Skip source paths that are themselves symlinks pointing outside
        # the expected /usr subtree (prevents following unexpected mounts).
        if src.is_symlink():
            try:
                resolved = src.resolve(strict=True)
                if not str(resolved).startswith("/usr/"):
                    logger.debug(
                        "Skipping external symlink in /usr sync: %s -> %s",
                        src,
                        resolved,
                    )
                    continue
            except OSError:
                continue  # dangling symlink
        dst = dst_usr / rel
        if src.is_dir():
            if not force and dst.exists() and any(dst.iterdir()):
                continue
            if force:
                _remove_conflicting_system_symlinks(src, dst)
            shutil.copytree(
                str(src),
                str(dst),
                dirs_exist_ok=True,
                symlinks=True,
                ignore_dangling_symlinks=True,
                copy_function=_copy_system_file_detached,
            )
        else:
            if not force and dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            _copy_system_file_detached(str(src), str(dst))


def _remove_conflicting_system_symlinks(src_root: Path, dst_root: Path) -> None:
    """Clear only destination entries occupied by source symlinks for copytree."""
    for current_root, dirs, files in os.walk(src_root, followlinks=False):
        source_current = Path(current_root)
        relative = source_current.relative_to(src_root)
        for name in [*dirs, *files]:
            source = source_current / name
            if not source.is_symlink():
                continue
            destination = dst_root / relative / name
            try:
                if destination.is_symlink() or destination.is_file():
                    destination.unlink()
                elif destination.exists():
                    shutil.rmtree(destination)
            except FileNotFoundError:
                pass


def _system_sync_marker_path(spec: SandboxSpec) -> Path:
    return spec.rootfs_path.parent / _SANDBOX_SYSTEM_SYNC_MARKER_FILENAME


def _system_sync_marker_matches(spec: SandboxSpec) -> bool:
    try:
        rootfs_stat = spec.rootfs_path.stat()
        marker = json.loads(_system_sync_marker_path(spec).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        isinstance(marker, dict)
        and marker.get("version") == _SANDBOX_SYSTEM_SYNC_MARKER_VERSION
        and marker.get("system_sync_version") == _CHROOT_USR_SYNC_VERSION
        and marker.get("rootfs_device") == rootfs_stat.st_dev
        and marker.get("rootfs_inode") == rootfs_stat.st_ino
    )


def _write_system_sync_marker(spec: SandboxSpec) -> None:
    rootfs_stat = spec.rootfs_path.stat()
    marker_path = _system_sync_marker_path(spec)
    payload = {
        "version": _SANDBOX_SYSTEM_SYNC_MARKER_VERSION,
        "system_sync_version": _CHROOT_USR_SYNC_VERSION,
        "rootfs_device": rootfs_stat.st_dev,
        "rootfs_inode": rootfs_stat.st_ino,
    }
    temporary = marker_path.with_name(f".{marker_path.name}.tmp-{os.getpid()}")
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, marker_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _copy_system_file_detached(src: str, dst: str) -> None:
    """Atomically replace one system file, never writing through a hardlink."""
    destination = Path(dst)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(destination.parent, flags)
    temporary_name: str | None = None
    try:
        temporary_fd, temporary_path = tempfile.mkstemp(
            prefix=f".{destination.name}.syncing-",
            dir=f"/proc/self/fd/{parent_fd}",
        )
        temporary_name = Path(temporary_path).name
        os.close(temporary_fd)
        shutil.copy2(src, f"/proc/self/fd/{parent_fd}/{temporary_name}")
        os.replace(temporary_name, destination.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        temporary_name = None
    finally:
        if temporary_name is not None:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=parent_fd)
        os.close(parent_fd)


def _detach_legacy_system_hardlinks(rootfs: Path, system_dirs: Sequence[str]) -> bool:
    """Detach all multiply-linked regular files under chroot system trees."""
    succeeded = True
    for directory in system_dirs:
        destination_root = rootfs / directory.lstrip("/")
        if not destination_root.is_dir():
            continue
        for current_root, _dirs, files in os.walk(destination_root, followlinks=False):
            for filename in files:
                path = Path(current_root) / filename
                try:
                    if stat.S_ISREG(path.stat(follow_symlinks=False).st_mode) and path.stat(follow_symlinks=False).st_nlink > 1:
                        _copy_system_file_detached(str(path), str(path))
                except OSError as exc:
                    succeeded = False
                    logger.warning("Failed to detach legacy system hardlink %s: %s", path, exc)
    return succeeded


def _copy_file(src: str, dst: str) -> None:
    """Copy a file, replacing destination when present."""
    shutil.copy2(src, dst)


def _copy_file_if_changed(src: str, dst: str) -> None:
    """Avoid rewriting mirrored workspace files only after exact comparison."""
    destination = Path(dst)
    source = Path(src)
    source_stat = source.stat()
    try:
        destination_stat = destination.stat()
    except FileNotFoundError:
        shutil.copy2(src, dst)
        return

    if (
        stat.S_ISREG(destination_stat.st_mode)
        and source_stat.st_size == destination_stat.st_size
        and stat.S_IMODE(source_stat.st_mode) == stat.S_IMODE(destination_stat.st_mode)
        and source_stat.st_mtime_ns == destination_stat.st_mtime_ns
        and _files_have_same_content(source, destination)
    ):
        return
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Public API: spawn sandboxed processes
# ---------------------------------------------------------------------------


def get_sandbox_spec(
    workspace_id: str,
    workspace_root: Path,
    workspace_files_path: Path,
) -> SandboxSpec:
    """Build a SandboxSpec for the given workspace.

    ``workspace_root`` is the parent directory (e.g. ``.../workspaces/<id>``).
    ``workspace_files_path`` is the actual project files directory.
    """
    caps = detect_capabilities()
    rootfs_path = workspace_root / "rootfs"
    return SandboxSpec(
        workspace_id=workspace_id,
        workspace_files_path=workspace_files_path,
        rootfs_path=rootfs_path,
        mode=caps.mode,
    )


def ensure_sandbox_ready(spec: SandboxSpec) -> None:
    """Provision the rootfs directory tree (idempotent)."""
    start = time.monotonic()
    with _rootfs_provision_lock(spec.rootfs_path):
        provision_rootfs(spec)
        caps = detect_capabilities()
        _prepare_sandbox_cgroup(spec, caps)
        if _chroot_system_sync_required(spec, caps):
            _sync_system_dirs_for_chroot(spec)

    elapsed = time.monotonic() - start
    if elapsed >= 1:
        logger.info(
            "Sandbox rootfs ready for workspace %s in %.2fs (mode=%s)",
            spec.workspace_id,
            elapsed,
            spec.mode,
        )


def sandbox_env(
    spec: SandboxSpec,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the environment dict for a sandboxed process."""
    env: dict[str, str] = {
        "HOME": "/root",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:./node_modules/.bin",
        "TERM": "xterm-256color",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "SHELL": "/bin/bash",
        "USER": "root",
        "LOGNAME": "root",
        "HOSTNAME": "sandbox",
        "PWD": spec.sandbox_workspace,
        "NODE_PATH": "/usr/local/lib/node_modules:/usr/lib/nodejs:/usr/lib/x86_64-linux-gnu/nodejs:/usr/share/nodejs",
        "TMPDIR": "/tmp",
        "LD_LIBRARY_PATH": "/usr/local/lib:/usr/lib:/lib",
    }
    if extra_env:
        env.update(extra_env)
    return env


async def spawn_sandboxed(
    spec: SandboxSpec,
    command: Sequence[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    stdout: int | IO[Any] | None = None,
    stderr: int | IO[Any] | None = None,
    stdin: int | IO[Any] | None = None,
    pty: bool = False,
    ensure_ready: bool = True,
) -> asyncio.subprocess.Process:
    """Spawn a command inside the workspace sandbox.

    This is the ONLY way user commands should be executed.
    """
    if ensure_ready:
        await asyncio.to_thread(ensure_sandbox_ready, spec)
    caps = detect_capabilities()
    if not caps.available:
        raise RuntimeError(
            f"Sandbox is not available in this container (mode={caps.mode}). "
            "The runtime container must run as root and ideally with "
            "CAP_SYS_ADMIN for full namespace isolation."
        )

    launch_spec = _launch_spec_from_spawn_request(spec, command, cwd=cwd, pty=pty, caps=caps)
    effective_env = sandbox_env(spec, env)

    # Set PWD to the requested cwd inside the sandbox
    if cwd:
        effective_env["PWD"] = cwd

    spec_read_fd, spec_write_fd = _pipe_cloexec()
    status_read_fd, status_write_fd = _pipe_cloexec()
    process: asyncio.subprocess.Process | None = None

    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "runtime.worker.sandbox_launcher",
            "--spec-fd",
            str(spec_read_fd),
            "--status-fd",
            str(status_write_fd),
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            env=effective_env,
            pass_fds=(spec_read_fd, status_write_fd),
            start_new_session=not pty,
        )
        try:
            os.close(spec_read_fd)
        except OSError:
            pass
        spec_read_fd = -1
        try:
            os.close(status_write_fd)
        except OSError:
            pass
        status_write_fd = -1

        try:
            _write_all_fd(spec_write_fd, _encode_launch_record(launch_spec.to_record()))
            os.close(spec_write_fd)
            spec_write_fd = -1
        except OSError as exc:
            await _cleanup_failed_launcher_process(process)
            raise SandboxLaunchError(SandboxLaunchStatus(stage="write_spec", errno=exc.errno, message=str(exc))) from exc

        try:
            status = await asyncio.wait_for(
                asyncio.to_thread(_read_launch_status_from_fd, status_read_fd),
                timeout=_SANDBOX_LAUNCH_STARTUP_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            await _cleanup_failed_launcher_process(process)
            raise SandboxLaunchError(
                SandboxLaunchStatus(stage="startup_timeout", errno=None, message=str(exc) or "sandbox launcher startup timed out")
            ) from exc
        except ValueError as exc:
            await _cleanup_failed_launcher_process(process)
            raise SandboxLaunchError(SandboxLaunchStatus(stage="read_status", errno=None, message=str(exc))) from exc

        if status is None:
            if process.returncode is not None:
                await process.wait()
                raise SandboxLaunchError(
                    SandboxLaunchStatus(
                        stage="launcher_exit",
                        errno=None,
                        message=f"launcher exited unexpectedly with code {process.returncode}",
                    )
                )
            return process

        await process.wait()
        if status.errno == errno.ENOENT:
            raise FileNotFoundError(status.errno, status.message, command[0] if command else None)
        raise SandboxLaunchError(status)
    finally:
        for fd in (spec_read_fd, spec_write_fd, status_read_fd, status_write_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass


def cleanup_sandbox(spec: SandboxSpec) -> None:
    """Best-effort cleanup of sandbox rootfs mounts.

    Called on session stop.  In pivot_root mode mounts are per-PID so
    they disappear when the process exits.  For chroot mode with copied
    system dirs, we optionally clean up to reclaim disk.
    """
    rootfs = spec.rootfs_path
    if not rootfs.exists():
        return

    caps = detect_capabilities()
    if workspace_mirror_required(spec, caps):
        # Routine session stop: prefer canonical files/ with mtime-based
        # comparison so stale rootfs content does not overwrite newer edits
        # that arrived through the ragtime API. Mode transitions (provision
        # path) continue to use source-wins to recover stranded chroot-era data.
        _reconcile_workspace_copy(spec, label="chroot-workspace-cleanup", prefer_source=False)
        # Refresh the marker so a subsequent provision sees the most
        # recently used mode even when no transition occurred.
        _write_sandbox_layout_marker(spec, caps)

    _terminate_sandbox_cgroup_processes(spec, caps)

    # Unmount any lingering bind mounts (best effort)
    try:
        mounts_data = Path("/proc/mounts").read_text(encoding="utf-8")
        rootfs_str = str(rootfs)
        for line in mounts_data.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].startswith(rootfs_str):
                mount_point = parts[1]
                try:
                    _syscall_umount2(mount_point, 2)  # MNT_DETACH
                except OSError:
                    pass
    except Exception:
        pass

    cgroup_path = _sandbox_cgroup_path(spec, caps)
    if cgroup_path is not None:
        try:
            cgroup_path.rmdir()
        except OSError:
            pass

    logger.info("Sandbox cleanup completed for rootfs: %s", rootfs)


def sandbox_diagnostics() -> dict[str, Any]:
    """Return a dict of sandbox capability information for health/debug endpoints."""
    caps = detect_capabilities()
    return {
        "sandbox_mode": caps.mode,
        "sandbox_available": caps.available,
        "has_cap_sys_admin": caps.has_cap_sys_admin,
        "can_pivot_root": caps.can_pivot_root,
        "can_user_ns": caps.can_user_ns,
        "can_mount": caps.can_mount,
        "unshare_flags": caps.unshare_flags,
        "unshare_flag_names": _unshare_flag_names(caps.unshare_flags),
        "dropped_unshare_flags": caps.dropped_unshare_flags,
        "dropped_unshare_flag_names": _unshare_flag_names(caps.dropped_unshare_flags),
        "mount_namespace": caps.mount_namespace,
        "pid_namespace": caps.pid_namespace,
        "uts_namespace": caps.uts_namespace,
        "ipc_namespace": caps.ipc_namespace,
        "cgroup_pids_available": caps.cgroup_pids_available,
        "cgroup_pids_parent": caps.cgroup_pids_parent,
        "cgroup_pids_max": caps.cgroup_pids_max,
        "drop_capabilities": caps.drop_capabilities,
        "no_new_privs": caps.no_new_privs,
        "euid": os.geteuid(),
        "egid": os.getegid(),
    }
