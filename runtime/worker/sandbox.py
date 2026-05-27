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
import ctypes
import ctypes.util
import errno
import filecmp
import logging
import os
import posixpath
import shutil
import stat
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Sequence

from ragtime.core.file_constants import DEFAULT_EXCLUDE_DIR_NAMES, GENERATED_BYTECODE_EXTENSIONS

from ..core.shared import has_cap_sys_admin

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
SYS_CAPSET = 126
SYS_PIVOT_ROOT = 155
SYS_MOUNT = 165
SYS_UMOUNT2 = 166
SYS_UNSHARE = 272

# prctl(2) constants
PR_CAPBSET_DROP = 24
PR_SET_NO_NEW_PRIVS = 38
LINUX_CAPABILITY_VERSION_3 = 0x20080522
_CAPABILITY_WORDS = 2
_MAX_CAPABILITY_INDEX = 63
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
_WORKSPACE_SYNC_SKIP_DIRS = DEFAULT_EXCLUDE_DIR_NAMES
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
_CHROOT_USR_SYNC_VERSION = "5"
_CHROOT_USR_SYNC_STAMP = ".ragtime_usr_sync_version"

# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

_libc_name = ctypes.util.find_library("c")
_libc = ctypes.CDLL(_libc_name or "libc.so.6", use_errno=True)


def _can_unshare_flags(flags: int) -> bool:
    """Probe whether unshare(flags) is permitted without perturbing this process."""
    try:
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
    caps.pid_namespace = bool(candidate_flags & CLONE_NEWPID)
    caps.uts_namespace = bool(candidate_flags & CLONE_NEWUTS)
    caps.ipc_namespace = bool(candidate_flags & CLONE_NEWIPC)
    caps.can_mount = caps.mount_namespace and (caps.has_cap_sys_admin or bool(candidate_flags & CLONE_NEWUSER))
    caps.can_pivot_root = caps.has_cap_sys_admin and caps.can_mount and caps.pid_namespace

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
            "Sandbox namespace support degraded; unsupported flags dropped: %s",
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


def _workspace_mirror_required(spec: SandboxSpec, caps: SandboxCapabilities) -> bool:
    if caps.can_mount:
        return False
    return True


def _chroot_system_sync_required(spec: SandboxSpec, caps: SandboxCapabilities) -> bool:
    return spec.mode == "chroot" and caps.mode == "chroot" and not caps.can_mount


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


def _copy_workspace_file_if_needed(src: Path, dst: Path) -> str:
    src_stat = src.stat()
    if dst.exists():
        if dst.is_file():
            dst_stat = dst.stat()
            if src_stat.st_size == dst_stat.st_size and filecmp.cmp(src, dst, shallow=False):
                return "same"
            if src_stat.st_mtime <= dst_stat.st_mtime:
                return "preserved_canonical"
        else:
            return "preserved_canonical"

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.syncing-{os.getpid()}")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return "copied"


def _sync_workspace_copy_to_canonical(spec: SandboxSpec, source_workspace: Path) -> dict[str, int]:
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
            if dirname in _WORKSPACE_SYNC_SKIP_DIRS:
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
                    result = _copy_workspace_file_if_needed(src, dst)
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


def _reconcile_workspace_copy(spec: SandboxSpec, *, label: str) -> None:
    source_workspace = spec.rootfs_path / spec.sandbox_workspace.lstrip("/")
    if not source_workspace.is_dir() or not spec.workspace_files_path.is_dir():
        return
    try:
        has_content = any(source_workspace.iterdir())
    except OSError:
        return
    if not has_content:
        return

    stats = _sync_workspace_copy_to_canonical(spec, source_workspace)
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
    # capable path shadows it with the canonical files/ bind mount.
    ws_dir = rootfs / spec.sandbox_workspace.lstrip("/")
    _reconcile_workspace_copy(spec, label="chroot-workspace")
    _ensure_real_directory(ws_dir)

    # In mount-capable sandbox modes the child bind-mounts the real
    # workspace over this directory, so mirroring would only add startup I/O.
    # Chroot fallback without mounts still needs a copied workspace tree.
    caps = detect_capabilities()
    workspace_src = spec.workspace_files_path
    if _workspace_mirror_required(spec, caps) and workspace_src.is_dir():
        try:
            shutil.copytree(
                str(workspace_src),
                str(ws_dir),
                dirs_exist_ok=True,
                symlinks=True,
                ignore_dangling_symlinks=True,
                copy_function=_copy_file,
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


def materialize_mounts(
    spec: SandboxSpec,
    mounts: list[dict[str, Any]],
    *,
    clear_targets: Sequence[str] | None = None,
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

    def resolve_target_path(target: str) -> Path | None:
        normalized = (target or "").strip().replace("\\", "/").lstrip("/")
        if not normalized:
            return None
        candidate = rootfs / normalized
        try:
            candidate.relative_to(rootfs)
        except ValueError:
            logger.warning(
                "materialize_mounts: target %s escapes rootfs, skipping",
                target,
            )
            return None
        return candidate

    def resolve_workspace_bind_target(target: str) -> Path | None:
        raw = (target or "").strip().replace("\\", "/")
        if not raw or "\x00" in raw:
            return None
        normalized = posixpath.normpath(raw)
        workspace_prefix = SANDBOX_WORKSPACE_MOUNT.rstrip("/") + "/"
        if not normalized.startswith(workspace_prefix):
            return None
        relative = normalized[len(workspace_prefix) :].strip("/")
        if not relative or relative == ".":
            return None
        parts = relative.split("/")
        if any(part in ("", ".", "..") for part in parts):
            return None
        return spec.workspace_files_path.joinpath(*parts)

    def unmount_if_mounted(path: Path) -> None:
        try:
            if os.path.ismount(path):
                _syscall_umount2(str(path), MNT_DETACH)
        except OSError as exc:
            logger.warning("materialize_mounts: failed to unmount %s: %s", path, exc)

    def copy_mount_source(source_path: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            str(source_path),
            str(dest),
            symlinks=True,
            ignore_dangling_symlinks=True,
            copy_function=_copy_file,
        )

    def bind_mount_source(source_path: Path, dest: Path, *, read_only: bool) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        unmount_if_mounted(dest)
        _ensure_real_directory(dest)
        _syscall_mount(str(source_path), str(dest), None, MS_BIND | MS_REC)
        if read_only:
            _syscall_mount(str(source_path), str(dest), None, MS_BIND | MS_REMOUNT | MS_RDONLY | MS_REC)

    for target in clear_targets or [str(mount.get("target_path") or "") for mount in mounts]:
        live_dest = resolve_workspace_bind_target(str(target or ""))
        if live_dest is not None:
            unmount_if_mounted(live_dest)
        dest = resolve_target_path(str(target or ""))
        if dest is None:
            continue
        try:
            if dest.is_symlink() or dest.is_file():
                dest.unlink(missing_ok=True)
            elif dest.exists():
                shutil.rmtree(dest)
        except Exception as exc:
            logger.warning(
                "materialize_mounts: failed to clear %s before sync: %s",
                dest,
                exc,
            )

    for mount in mounts:
        source = mount.get("source_local_path", "")
        target = mount.get("target_path", "")
        if not source or not target:
            continue
        source_path = Path(source)
        if not source_path.is_dir():
            message = f"Workspace mount source is not available in the runtime container: {source}"
            if str(mount.get("runtime_mount_mode") or "") == "live_bind":
                raise FileNotFoundError(message)
            logger.warning("materialize_mounts: %s", message)
            continue
        dest = resolve_target_path(str(target))
        if dest is None:
            continue

        if str(mount.get("runtime_mount_mode") or "") == "live_bind":
            live_dest = resolve_workspace_bind_target(str(target))
            if live_dest is None:
                raise ValueError(f"Live workspace mount target must be under {SANDBOX_WORKSPACE_MOUNT}: {target}")
            if caps.can_mount:
                bind_mount_source(
                    source_path,
                    live_dest,
                    read_only=bool(mount.get("read_only", True)),
                )
                continue
            raise PermissionError("Live workspace mounts require runtime mount authority. Enable SYS_ADMIN or privileged mode for the runtime container.")

        try:
            copy_mount_source(source_path, dest)
        except Exception as exc:
            logger.warning(
                "materialize_mounts: failed to copy %s -> %s: %s",
                source,
                dest,
                exc,
            )


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
        raise OSError(err, f"mount({source}, {target}, {fstype}, {flags:#x}): {os.strerror(err)}")
    return ret


def _syscall_umount2(target: str, flags: int = 0) -> int:
    tgt = target.encode()
    ret = _libc.umount2(tgt, flags)
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"umount2({target}, {flags}): {os.strerror(err)}")
    return ret


def _syscall_pivot_root(new_root: str, put_old: str) -> int:
    """pivot_root(2) via syscall."""
    ret = _libc.syscall(SYS_PIVOT_ROOT, new_root.encode(), put_old.encode())
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

    _syscall_pivot_root(rootfs, old_root)
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
    try:
        Path("/proc/self/setgroups").write_text("deny", encoding="utf-8")
    except (PermissionError, OSError):
        pass
    try:
        Path("/proc/self/uid_map").write_text(f"0 {uid} 1\n", encoding="utf-8")
    except (PermissionError, OSError):
        pass
    try:
        Path("/proc/self/gid_map").write_text(f"0 {gid} 1\n", encoding="utf-8")
    except (PermissionError, OSError):
        pass


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
    try:
        (cgroup_path / "cgroup.procs").write_text(str(os.getpid()), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to assign sandbox process to pids cgroup for %s: %s", spec.workspace_id, exc)


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
    ret = _libc.syscall(SYS_CAPSET, ctypes.byref(header), ctypes.byref(data))
    if ret != 0:
        err = ctypes.get_errno()
        logger.warning("Failed to clear sandbox process capabilities: %s", os.strerror(err))

    if no_new_privs:
        ret = _libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
        if ret != 0:
            err = ctypes.get_errno()
            logger.warning("Failed to set no_new_privs for sandbox process: %s", os.strerror(err))


# ---------------------------------------------------------------------------
# Pre-exec sandbox entry (called from forked child before exec)
# ---------------------------------------------------------------------------


_UNSHARE_FLAG_NAMES: tuple[tuple[int, str], ...] = (
    (CLONE_NEWNS, "CLONE_NEWNS"),
    (CLONE_NEWUTS, "CLONE_NEWUTS"),
    (CLONE_NEWIPC, "CLONE_NEWIPC"),
    (CLONE_NEWPID, "CLONE_NEWPID"),
    (CLONE_NEWUSER, "CLONE_NEWUSER"),
)


def _probe_unshare(flags: int, label: str) -> str:
    """Fork, attempt unshare(flags) in the child, return a result string."""
    import errno as _errno

    try:
        r, w = os.pipe()
        pid = os.fork()
    except OSError as exc:
        return f"{label} fork_failed={exc}"
    if pid == 0:
        os.close(r)
        ctypes.set_errno(0)
        ret = _libc.unshare(flags)
        err = ctypes.get_errno()
        if ret == 0:
            payload = f"{label} 0x{flags:x} OK"
        else:
            pn = _errno.errorcode.get(err, str(err))
            payload = f"{label} 0x{flags:x} FAIL errno={pn} ({err}) {os.strerror(err)}"
        try:
            os.write(w, payload.encode())
        except OSError:
            pass
        os._exit(0)
    os.close(w)
    try:
        data = os.read(r, 4096).decode(errors="replace")
    except OSError:
        data = f"{label} read_failed"
    os.close(r)
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass
    return data


def _persist_diagnostic(lines: list[str]) -> None:
    try:
        debug_path = os.environ.get("RUNTIME_SANDBOX_DEBUG_LOG", "/tmp/sandbox-unshare-debug.log")
        with open(debug_path, "a", encoding="utf-8") as fh:
            fh.write(f"--- {time.time():.3f} pid={os.getpid()} ---\n")
            for line in lines:
                fh.write(line + "\n")
            fh.write("\n")
    except OSError:
        pass


def _probe_unshare_combinations(unshare_flags: int, caps: SandboxCapabilities) -> None:
    """Startup-time probe: log result of unshare with and without
    CLONE_NEWUSER (and each flag individually) so we can identify which
    flag the kernel/security profile is rejecting. Runs in forked
    children so it does not perturb the real preexec process.
    """
    lines: list[str] = []

    def _emit(msg: str) -> None:
        lines.append(msg)
        logger.error("%s", msg)

    _emit(
        "unshare probe header: requested=0x%x pid=%d ppid=%d euid=%d egid=%d "
        "has_cap_sys_admin=%s can_user_ns=%s can_mount=%s"
        % (
            unshare_flags,
            os.getpid(),
            os.getppid(),
            os.geteuid(),
            os.getegid(),
            caps.has_cap_sys_admin,
            caps.can_user_ns,
            caps.can_mount,
        )
    )

    # Original combo (the suspected-buggy set: includes CLONE_NEWUSER)
    original_combo = unshare_flags | CLONE_NEWUSER
    _emit("probe: " + _probe_unshare(original_combo, "WITH_NEWUSER"))

    # Fixed combo (current production candidate, no CLONE_NEWUSER)
    fixed_combo = unshare_flags & ~CLONE_NEWUSER
    if fixed_combo != original_combo:
        _emit("probe: " + _probe_unshare(fixed_combo, "WITHOUT_NEWUSER"))

    # Individual flag isolation
    for bit, name in _UNSHARE_FLAG_NAMES:
        _emit("probe: " + _probe_unshare(bit, f"SOLO_{name}"))

    _persist_diagnostic(lines)


def _diagnose_unshare_failure(
    unshare_flags: int,
    err: int,
    caps: SandboxCapabilities,
) -> None:
    """Called only when the real unshare in the preexec child fails."""
    import errno as _errno

    requested = [name for bit, name in _UNSHARE_FLAG_NAMES if unshare_flags & bit]
    errno_name = _errno.errorcode.get(err, str(err))
    msg = "unshare diagnostic: real_call_failed flags=0x%x [%s] errno=%s (%d) %s" % (unshare_flags, ",".join(requested), errno_name, err, os.strerror(err))
    logger.error("%s", msg)
    _persist_diagnostic([msg])


def _sandbox_preexec(spec: SandboxSpec, caps: SandboxCapabilities, target_cwd: str | None = None) -> None:
    """Configure the sandbox inside the forked child, before exec.

    This function is called as the ``preexec_fn`` (or equivalent) in the
    forked child process.  It MUST NOT return on failure — it must
    ``os._exit(126)`` so the parent sees a clean failure.
    """
    try:
        if caps.mode in {"pivot_root", "chroot"}:
            _assign_current_process_to_sandbox_cgroup(spec, caps)

            unshare_flags = caps.unshare_flags
            if unshare_flags:
                ret = _libc.unshare(unshare_flags)
                if ret != 0:
                    err = ctypes.get_errno()
                    logger.error("unshare() failed: %s", os.strerror(err))
                    _diagnose_unshare_failure(unshare_flags, err, caps)
                    os._exit(126)

            if caps.can_user_ns and (unshare_flags & CLONE_NEWUSER):
                _setup_user_namespace_mappings()

        if caps.mode == "pivot_root":
            _setup_sandbox_mounts(spec, mount_proc=caps.pid_namespace)
            _do_pivot_root(spec)

        elif caps.mode == "chroot":
            if caps.can_mount:
                try:
                    _setup_sandbox_mounts(spec, mount_proc=caps.pid_namespace)
                except OSError as exc:
                    logger.warning(
                        "Sandbox mount setup failed in chroot mode, continuing with basic chroot: %s",
                        exc,
                    )
                    _sync_system_dirs_for_chroot(spec)
            else:
                _sync_system_dirs_for_chroot(spec)

            _do_chroot(spec)

        else:
            logger.error("Sandbox mode '%s' is not supported", caps.mode)
            os._exit(126)

        # Set hostname only after entering a private UTS namespace.
        if caps.uts_namespace:
            try:
                import socket

                socket.sethostname("sandbox")
            except Exception:
                pass

        # chdir to the correct directory inside sandbox
        if target_cwd:
            try:
                os.chdir(target_cwd)
            except OSError:
                os.chdir(spec.sandbox_workspace)
        else:
            os.chdir(spec.sandbox_workspace)

        if caps.drop_capabilities:
            _drop_process_capabilities(no_new_privs=caps.no_new_privs)
        elif caps.no_new_privs:
            ret = _libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0)
            if ret != 0:
                err = ctypes.get_errno()
                logger.warning("Failed to set no_new_privs for sandbox process: %s", os.strerror(err))

    except Exception as exc:
        logger.error("Sandbox setup failed: %s", exc, exc_info=True)
        os._exit(126)


def _sync_system_dirs_for_chroot(spec: SandboxSpec) -> None:
    """When no mount namespace is available, sync essential system files.

    This copies a minimal set of binaries/libraries into the rootfs so
    chroot is functional. Symlinks are preserved to avoid recursive loops
    from entries such as ``/bin/X11``.
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

    for d in _HOST_RO_BIND_DIRS:
        src = Path(d)
        if not src.is_dir():
            continue
        dst = rootfs / d.lstrip("/")
        try:
            if src == Path("/usr"):
                if usr_needs_sync:
                    _sync_usr_for_chroot(src, dst, force=True)
                    usr_stamp.parent.mkdir(parents=True, exist_ok=True)
                    usr_stamp.write_text(
                        _CHROOT_USR_SYNC_VERSION,
                        encoding="utf-8",
                    )
                continue
            if dst.exists() and any(dst.iterdir()):
                # Already populated (from a previous session)
                continue
            else:
                shutil.copytree(
                    str(src),
                    str(dst),
                    dirs_exist_ok=True,
                    symlinks=True,
                    ignore_dangling_symlinks=True,
                    copy_function=_link_or_copy,
                )
        except Exception as exc:
            logger.warning("Failed to sync %s into rootfs: %s", d, exc)

    for d in _HOST_RO_BIND_DIRS_OPTIONAL:
        src = Path(d)
        if not src.is_dir():
            continue
        dst = rootfs / d.lstrip("/")
        if dst.exists() and any(dst.iterdir()):
            continue
        try:
            shutil.copytree(
                str(src),
                str(dst),
                dirs_exist_ok=True,
                symlinks=True,
                ignore_dangling_symlinks=True,
                copy_function=_link_or_copy,
            )
        except Exception:
            pass

    # Workspace files are mirrored by provision_rootfs() before the child
    # process enters the sandbox.  Repeating that copy from preexec after
    # user-namespace setup can lose write access to the root-owned rootfs
    # tree on restored/bind-mounted data directories.


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
    * ``_link_or_copy`` is the copy function for regular files only
      (not called for symlinks); it tries a hard-link first and falls
      back to ``shutil.copy2``.
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
            shutil.copytree(
                str(src),
                str(dst),
                dirs_exist_ok=True,
                symlinks=True,
                ignore_dangling_symlinks=True,
                copy_function=_link_or_copy,
            )
        else:
            if not force and dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(str(src), str(dst))


def _link_or_copy(src: str, dst: str) -> None:
    """Try hard link for a regular file, fall back to copy.

    Only called by ``shutil.copytree`` for regular files (not symlinks).
    Hard-links share inode/pages with the host and save disk; when they
    fail (cross-device, read-only source, etc.) we fall back to copy2.
    """
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_file(src: str, dst: str) -> None:
    """Copy a file, replacing destination when present."""
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


def make_sandbox_preexec(spec: SandboxSpec, target_cwd: str | None = None) -> Any:
    """Return a preexec_fn callable that enters the sandbox.

    Use this with ``asyncio.create_subprocess_exec(..., preexec_fn=fn)``.
    """
    caps = detect_capabilities()
    if not caps.available:
        raise RuntimeError(
            f"Sandbox is not available in this container (mode={caps.mode}). "
            "The runtime container must run as root and ideally with "
            "CAP_SYS_ADMIN for full namespace isolation."
        )

    def _preexec() -> None:
        _sandbox_preexec(spec, caps, target_cwd=target_cwd)

    return _preexec


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
    start_new_session: bool = False,
    ensure_ready: bool = True,
) -> asyncio.subprocess.Process:
    """Spawn a command inside the workspace sandbox.

    This is the ONLY way user commands should be executed.
    """
    if ensure_ready:
        await asyncio.to_thread(ensure_sandbox_ready, spec)
    preexec_fn = make_sandbox_preexec(spec, target_cwd=cwd)
    effective_env = sandbox_env(spec, env)

    # Set PWD to the requested cwd inside the sandbox
    if cwd:
        effective_env["PWD"] = cwd

    return await asyncio.create_subprocess_exec(
        *command,
        cwd=None,  # preexec_fn will chdir to the correct directory
        env=effective_env,
        stdout=stdout,
        stderr=stderr,
        stdin=stdin,
        preexec_fn=preexec_fn,
        start_new_session=start_new_session,
    )


def prepare_sandbox_pty_preexec(spec: SandboxSpec, target_cwd: str | None = None) -> Any:
    """Return a preexec_fn for PTY processes (same sandbox, but needs
    to set up the PTY slave fd before entering the sandbox).

    The returned callable enters the sandbox.  The caller is responsible
    for setting up the PTY master/slave pair and passing the slave fd to
    the subprocess.
    """
    return make_sandbox_preexec(spec, target_cwd=target_cwd)


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
    if _workspace_mirror_required(spec, caps):
        _reconcile_workspace_copy(spec, label="chroot-workspace-cleanup")

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
