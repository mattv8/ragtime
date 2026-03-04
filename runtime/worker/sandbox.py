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
import logging
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Sequence

from runtime.shared import has_cap_sys_admin

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

# Syscall numbers (x86_64)
SYS_PIVOT_ROOT = 155
SYS_MOUNT = 165
SYS_UMOUNT2 = 166
SYS_UNSHARE = 272

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
    "share/zoneinfo",
    "share/terminfo",
)

# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------

_libc_name = ctypes.util.find_library("c")
_libc = ctypes.CDLL(_libc_name or "libc.so.6", use_errno=True)


def _can_unshare_userns() -> bool:
    """Probe whether user namespaces are permitted in this container."""
    try:
        pid = os.fork()
        if pid == 0:
            # Child: try unshare(CLONE_NEWUSER)
            ret = _libc.unshare(CLONE_NEWUSER)
            os._exit(0 if ret == 0 else 1)
        else:
            _, status = os.waitpid(pid, 0)
            return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    except Exception:
        return False


@dataclass
class SandboxCapabilities:
    """Detected sandbox capabilities of the runtime container."""

    has_cap_sys_admin: bool = False
    can_pivot_root: bool = False
    can_user_ns: bool = False
    can_mount: bool = False
    mode: str = "unavailable"  # "pivot_root" | "chroot" | "unavailable"

    @property
    def available(self) -> bool:
        return self.mode in ("pivot_root", "chroot")


_capabilities_cache: dict[str, SandboxCapabilities] = {}


def detect_capabilities() -> SandboxCapabilities:
    """Detect what sandbox primitives are available (cached after first call)."""
    cached = _capabilities_cache.get("caps")
    if cached is not None:
        return cached

    caps = SandboxCapabilities()
    caps.has_cap_sys_admin = has_cap_sys_admin()
    caps.can_user_ns = _can_unshare_userns()

    # Test mount capability by trying a private mount namespace
    if caps.has_cap_sys_admin:
        caps.can_mount = True
        caps.can_pivot_root = True
        caps.mode = "pivot_root"
    else:
        # Even without CAP_SYS_ADMIN, chroot may work if we are uid 0
        if os.geteuid() == 0:
            caps.mode = "chroot"
            caps.can_mount = False  # mount() requires CAP_SYS_ADMIN
        else:
            caps.mode = "unavailable"

    _capabilities_cache["caps"] = caps
    logger.info(
        "Sandbox capabilities detected: mode=%s, cap_sys_admin=%s, "
        "user_ns=%s, mount=%s, pivot_root=%s",
        caps.mode,
        caps.has_cap_sys_admin,
        caps.can_user_ns,
        caps.can_mount,
        caps.can_pivot_root,
    )
    return caps


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

    # /workspace mount point (project files)
    ws_dir = rootfs / spec.sandbox_workspace.lstrip("/")
    _ensure_real_directory(ws_dir)

    # Pre-populate workspace mirror so the first forked child doesn't
    # need a full copy.  Uses dirs_exist_ok=True (idempotent).
    workspace_src = spec.workspace_files_path
    if workspace_src.is_dir():
        import shutil

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


def _provision_etc(rootfs: Path) -> None:
    """Ensure minimal /etc content exists in the sandbox rootfs."""
    etc = rootfs / "etc"
    etc.mkdir(parents=True, exist_ok=True)

    # /etc/passwd — minimal entries
    passwd = etc / "passwd"
    if not passwd.exists():
        passwd.write_text(
            "root:x:0:0:root:/root:/bin/bash\n"
            "nobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n",
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
        raise OSError(
            err, f"mount({source}, {target}, {fstype}, {flags:#x}): {os.strerror(err)}"
        )
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


def _setup_sandbox_mounts(spec: SandboxSpec) -> None:
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

    # Mount /proc
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


# ---------------------------------------------------------------------------
# Pre-exec sandbox entry (called from forked child before exec)
# ---------------------------------------------------------------------------


def _sandbox_preexec(
    spec: SandboxSpec, caps: SandboxCapabilities, target_cwd: str | None = None
) -> None:
    """Configure the sandbox inside the forked child, before exec.

    This function is called as the ``preexec_fn`` (or equivalent) in the
    forked child process.  It MUST NOT return on failure — it must
    ``os._exit(126)`` so the parent sees a clean failure.
    """
    try:
        if caps.mode == "pivot_root":
            # Full namespace isolation + pivot_root
            unshare_flags = CLONE_NEWNS | CLONE_NEWUTS | CLONE_NEWIPC | CLONE_NEWPID
            if caps.can_user_ns:
                unshare_flags |= CLONE_NEWUSER
            # NET namespace - only if we have full caps (network isolation)
            if caps.has_cap_sys_admin:
                # Don't isolate network - devserver needs localhost access
                pass

            ret = _libc.unshare(unshare_flags)
            if ret != 0:
                err = ctypes.get_errno()
                logger.error("unshare() failed: %s", os.strerror(err))
                os._exit(126)

            if caps.can_user_ns and (unshare_flags & CLONE_NEWUSER):
                _setup_user_namespace_mappings()

            _setup_sandbox_mounts(spec)
            _do_pivot_root(spec)

        elif caps.mode == "chroot":
            # chroot-only fallback — still provides filesystem confinement
            # Try to get a private mount namespace even without CAP_SYS_ADMIN
            # (may work if user namespaces are available)
            got_mount_ns = False
            if caps.can_user_ns:
                ret = _libc.unshare(CLONE_NEWUSER | CLONE_NEWNS)
                if ret == 0:
                    _setup_user_namespace_mappings()
                    got_mount_ns = True

            if got_mount_ns:
                # We have a mount namespace, do bind mounts
                try:
                    _setup_sandbox_mounts(spec)
                except OSError as exc:
                    # Mount failed but we can still chroot
                    logger.warning(
                        "Sandbox mount setup failed in chroot mode, "
                        "continuing with basic chroot: %s",
                        exc,
                    )
                    _sync_system_dirs_for_chroot(spec)
            else:
                # No mount namespace available — perform minimal setup
                # Sync essential system dirs for chroot operation
                _sync_system_dirs_for_chroot(spec)

            _do_chroot(spec)

        else:
            logger.error("Sandbox mode '%s' is not supported", caps.mode)
            os._exit(126)

        # Set hostname inside the sandbox (UTS namespace)
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

    for d in _HOST_RO_BIND_DIRS:
        src = Path(d)
        if not src.is_dir():
            continue
        dst = rootfs / d.lstrip("/")
        if dst.exists() and any(dst.iterdir()):
            # Already populated (from a previous session)
            continue
        try:
            if src == Path("/usr"):
                _sync_usr_for_chroot(src, dst)
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

    # Mirror workspace files into /workspace for no-mount chroot mode.
    # We keep /workspace as a real directory because symlinking to an
    # absolute host path becomes invalid after chroot.
    #
    # IMPORTANT: We do NOT rmtree /workspace here.  This function runs in
    # every forked child (preexec_fn).  Deleting the directory destroys
    # the inode and invalidates the cwd of any already-running sandboxed
    # process (e.g. the devserver), causing it to serve from "/" instead
    # of "/workspace".  Instead we use an incremental copytree that adds
    # new / updated files while preserving the directory inode.
    workspace_src = spec.workspace_files_path
    ws_dst = rootfs / spec.sandbox_workspace.lstrip("/")

    _ensure_real_directory(ws_dst)
    if workspace_src.is_dir():
        try:
            shutil.copytree(
                str(workspace_src),
                str(ws_dst),
                dirs_exist_ok=True,
                symlinks=True,
                ignore_dangling_symlinks=True,
                copy_function=_copy_file,
            )
        except Exception as exc:
            logger.warning("Failed to mirror workspace into rootfs: %s", exc)


def _sync_usr_for_chroot(src_usr: Path, dst_usr: Path) -> None:
    """Sync a minimal, runtime-focused subset of /usr for chroot fallback.

    Copying all of /usr in no-mount chroot mode can create very large rootfs
    trees. This function keeps the payload bounded to binaries/libs and small
    runtime metadata needed by common tools.
    """

    dst_usr.mkdir(parents=True, exist_ok=True)
    for rel in _CHROOT_USR_INCLUDE_PATHS:
        src = src_usr / rel
        if not src.exists():
            continue
        dst = dst_usr / rel
        if src.is_dir():
            shutil.copytree(
                str(src),
                str(dst),
                dirs_exist_ok=True,
                symlinks=True,
                ignore_dangling_symlinks=True,
                copy_function=_link_or_copy,
            )
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(str(src), str(dst))


def _link_or_copy(src: str, dst: str) -> None:
    """Try hard link, fall back to copy."""
    try:
        os.link(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def _copy_file(src: str, dst: str) -> None:
    """Copy a file, replacing destination when present."""
    import shutil

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
    provision_rootfs(spec)


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
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:"
        "./node_modules/.bin",
        "TERM": "xterm-256color",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "SHELL": "/bin/bash",
        "USER": "root",
        "LOGNAME": "root",
        "HOSTNAME": "sandbox",
        "PWD": spec.sandbox_workspace,
        "NODE_PATH": "/usr/local/lib/node_modules",
        "TMPDIR": "/tmp",
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
) -> asyncio.subprocess.Process:
    """Spawn a command inside the workspace sandbox.

    This is the ONLY way user commands should be executed.
    """
    ensure_sandbox_ready(spec)
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


def prepare_sandbox_pty_preexec(
    spec: SandboxSpec, target_cwd: str | None = None
) -> Any:
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
        "euid": os.geteuid(),
        "egid": os.getegid(),
    }
