from __future__ import annotations

import argparse
import ctypes
import errno
import fcntl
import os
import shutil
import signal
import socket
import termios
from typing import NoReturn, cast

from .sandbox import (
    CLONE_NEWPID,
    CLONE_NEWUSER,
    SandboxLaunchSpec,
    SandboxLaunchStatus,
    _assign_current_process_to_sandbox_cgroup,
    _capabilities_from_launch_spec,
    _do_chroot,
    _do_pivot_root,
    _drop_process_capabilities,
    _libc,
    _read_launch_spec_from_fd,
    _set_no_new_privs,
    _set_parent_death_signal,
    _setup_sandbox_mounts,
    _setup_user_namespace_mappings,
    _sync_system_dirs_for_chroot,
    _write_launch_status_to_fd,
)

_TERMINATION_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
_TINI_PATH = "/usr/bin/tini"
_FORWARDED_SIGNALS: set[int] = set()


def _block_termination_signals() -> set[signal.Signals] | None:
    if not hasattr(signal, "pthread_sigmask"):
        return None
    return cast(set[signal.Signals], signal.pthread_sigmask(signal.SIG_BLOCK, _TERMINATION_SIGNALS))


def _restore_parent_signal_mask(previous: set[signal.Signals] | None) -> None:
    if previous is None or not hasattr(signal, "pthread_sigmask"):
        return
    signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _reset_termination_signal_dispositions() -> None:
    for signum in _TERMINATION_SIGNALS:
        signal.signal(signum, signal.SIG_DFL)


def _setup_controlling_terminal() -> None:
    os.setsid()
    try:
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    except OSError:
        pass
    try:
        os.tcsetpgrp(0, os.getpid())
    except OSError:
        pass


def _unshare_or_raise(flags: int) -> None:
    if not flags:
        return
    ret = _libc.unshare(flags)
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def _resolve_workload_executable(argv: tuple[str, ...], env: dict[str, str]) -> str:
    executable = argv[0]
    if "/" in executable:
        if os.path.isfile(executable) and os.access(executable, os.X_OK):
            return executable
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), executable)
    resolved = shutil.which(executable, path=env.get("PATH"))
    if not resolved:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), executable)
    return resolved


def _write_startup_failure(status_fd: int, stage: str, exc: BaseException) -> None:
    status = SandboxLaunchStatus(
        stage=stage,
        errno=getattr(exc, "errno", None),
        message=str(exc) or exc.__class__.__name__,
    )
    try:
        _write_launch_status_to_fd(status_fd, status)
    except Exception:
        pass


def _exit_with_startup_failure(status_fd: int, stage: str, exc: BaseException) -> NoReturn:
    _write_startup_failure(status_fd, stage, exc)
    os._exit(126)


def _arm_status_fd_cloexec(status_fd: int) -> None:
    flags = fcntl.fcntl(status_fd, fcntl.F_GETFD)
    fcntl.fcntl(status_fd, fcntl.F_SETFD, flags | fcntl.FD_CLOEXEC)


def _enter_rootfs_and_exec(
    spec: SandboxLaunchSpec,
    status_fd: int,
    *,
    mount_proc: bool,
    use_tini: bool,
    blocked_signals: set[signal.Signals] | None = None,
) -> NoReturn:
    sandbox_spec = spec.to_sandbox_spec()
    caps = _capabilities_from_launch_spec(spec)

    try:
        if caps.can_mount:
            try:
                _setup_sandbox_mounts(sandbox_spec, mount_proc=mount_proc)
            except OSError:
                if spec.mode != "chroot":
                    raise
                _sync_system_dirs_for_chroot(sandbox_spec)
    except Exception as exc:
        _exit_with_startup_failure(status_fd, "mounts", exc)

    try:
        if spec.mode == "pivot_root":
            _do_pivot_root(sandbox_spec)
        elif spec.mode == "chroot":
            _do_chroot(sandbox_spec)
        else:
            raise RuntimeError(f"unsupported sandbox mode: {spec.mode}")
    except Exception as exc:
        _exit_with_startup_failure(status_fd, "rootfs", exc)

    if caps.uts_namespace:
        try:
            socket.sethostname("sandbox")
        except Exception:
            pass

    try:
        os.chdir(spec.cwd)
    except Exception as exc:
        _exit_with_startup_failure(status_fd, "cwd", exc)

    env = dict(os.environ)

    try:
        resolved_executable = _resolve_workload_executable(spec.argv, env)
    except Exception as exc:
        _exit_with_startup_failure(status_fd, "resolve_executable", exc)

    try:
        if caps.drop_capabilities:
            _drop_process_capabilities(no_new_privs=caps.no_new_privs)
        elif caps.no_new_privs:
            _set_no_new_privs()
    except Exception as exc:
        _exit_with_startup_failure(status_fd, "capabilities", exc)

    _reset_termination_signal_dispositions()
    if blocked_signals is None and hasattr(signal, "pthread_sigmask"):
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _TERMINATION_SIGNALS)
    else:
        _restore_parent_signal_mask(blocked_signals)

    executable = resolved_executable
    argv = list(spec.argv)
    stage = "exec_workload"
    if use_tini:
        if not (os.path.isfile(_TINI_PATH) and os.access(_TINI_PATH, os.X_OK)):
            _exit_with_startup_failure(
                status_fd,
                "exec_tini",
                FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), _TINI_PATH),
            )
        executable = _TINI_PATH
        argv = [_TINI_PATH, "-g", "--", *spec.argv]
        stage = "exec_tini"

    try:
        _arm_status_fd_cloexec(status_fd)
        os.execvpe(executable, argv, env)
    except Exception as exc:
        _exit_with_startup_failure(status_fd, stage, exc)

    raise AssertionError("os.execvpe returned unexpectedly")


def _run_pid_init(spec: SandboxLaunchSpec, status_fd: int, launcher_pid: int) -> NoReturn:
    try:
        _set_parent_death_signal(launcher_pid)
    except Exception as exc:
        _exit_with_startup_failure(status_fd, "parent_death", exc)

    _enter_rootfs_and_exec(spec, status_fd, mount_proc=True, use_tini=True)


def _forward_signal_to_child(signum: int, child_pid: int) -> None:
    _FORWARDED_SIGNALS.add(signum)
    try:
        os.kill(child_pid, signum)
    except ProcessLookupError:
        pass


def _wait_and_mirror_child(child_pid: int) -> NoReturn:
    while True:
        try:
            _, status = os.waitpid(child_pid, 0)
            break
        except InterruptedError:
            continue

    if os.WIFSIGNALED(status):
        signum = os.WTERMSIG(status)
        if signum in _FORWARDED_SIGNALS:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        os._exit(128 + signum)

    os._exit(os.WEXITSTATUS(status))


def _run_launcher(spec: SandboxLaunchSpec, status_fd: int) -> NoReturn:
    worker_pid = os.getppid()
    try:
        _set_parent_death_signal(worker_pid)
    except Exception as exc:
        _exit_with_startup_failure(status_fd, "parent_death", exc)

    sandbox_spec = spec.to_sandbox_spec()
    caps = _capabilities_from_launch_spec(spec)

    try:
        _assign_current_process_to_sandbox_cgroup(sandbox_spec, caps)
    except Exception as exc:
        _exit_with_startup_failure(status_fd, "cgroup", exc)

    try:
        if spec.pty:
            _setup_controlling_terminal()
    except Exception as exc:
        _exit_with_startup_failure(status_fd, "pty", exc)

    blocked = _block_termination_signals()

    try:
        _unshare_or_raise(spec.unshare_flags)
    except Exception as exc:
        _exit_with_startup_failure(status_fd, "unshare", exc)

    if spec.unshare_flags & CLONE_NEWUSER:
        try:
            _setup_user_namespace_mappings()
        except Exception as exc:
            _exit_with_startup_failure(status_fd, "user_mapping", exc)

    if not (spec.unshare_flags & CLONE_NEWPID):
        _enter_rootfs_and_exec(spec, status_fd, mount_proc=False, use_tini=False, blocked_signals=blocked)

    try:
        child_pid = os.fork()
    except Exception as exc:
        _exit_with_startup_failure(status_fd, "fork_pid_init", exc)

    if child_pid == 0:
        _run_pid_init(spec, status_fd, os.getppid())

    def _handle_forwarded_signal(received: int, _frame: object) -> None:
        _forward_signal_to_child(received, child_pid)

    for signum in _TERMINATION_SIGNALS:
        signal.signal(signum, _handle_forwarded_signal)

    try:
        os.close(status_fd)
    except OSError:
        pass
    _restore_parent_signal_mask(blocked)
    _wait_and_mirror_child(child_pid)


def main() -> NoReturn:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-fd", type=int, required=True)
    parser.add_argument("--status-fd", type=int, required=True)
    args = parser.parse_args()

    try:
        spec = _read_launch_spec_from_fd(args.spec_fd)
    except Exception as exc:
        _exit_with_startup_failure(args.status_fd, "read_spec", exc)
    finally:
        try:
            os.close(args.spec_fd)
        except OSError:
            pass

    _run_launcher(spec, args.status_fd)


if __name__ == "__main__":
    main()
