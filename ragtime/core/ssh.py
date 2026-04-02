"""
SSH Utilities - Robust SSH execution with multiple auth methods.

Supports:
- Password-only authentication
- Key-only authentication (from file path or content)
- Key + passphrase authentication
"""

import io
import os as _os
import shlex
import shutil
import socket
import stat
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path as _Path
from pathlib import PurePosixPath
from typing import Optional

import paramiko

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Default scheduler values for userspace SSH mount auto-sync watch mode.
USERSPACE_MOUNT_WATCH_INTERVAL_SECONDS = 5.0
USERSPACE_MOUNT_WATCH_JITTER_SECONDS = 3.0

# Common stderr noise patterns to filter out (shell initialization artifacts)
# These occur when shell RC files run terminal commands without a TTY
STDERR_NOISE_PATTERNS = [
    "tput: No value for $TERM and no -T specified",
]


def _filter_stderr_noise(stderr: str) -> str:
    """Filter out known harmless stderr noise from SSH output.

    Some SSH servers have shell initialization scripts that produce
    stderr output when run without a TTY (e.g., tput commands).
    This filters those known-harmless warnings.
    """
    if not stderr:
        return stderr

    lines = stderr.splitlines()
    filtered_lines = [
        line
        for line in lines
        if not any(pattern in line for pattern in STDERR_NOISE_PATTERNS)
    ]
    return "\n".join(filtered_lines)


class SSHAuthMethod(str, Enum):
    """SSH authentication methods."""

    PASSWORD = "password"
    KEY_FILE = "key_file"
    KEY_CONTENT = "key_content"


@dataclass
class SSHResult:
    """Result from SSH command execution."""

    stdout: str
    stderr: str
    exit_code: int
    success: bool

    @property
    def output(self) -> str:
        """Combined output, preferring stdout but including stderr if present."""
        if self.stderr and not self.stdout:
            return self.stderr
        if self.stderr:
            return f"{self.stdout}\n\nSTDERR:\n{self.stderr}"
        return self.stdout


@dataclass
class SSHConfig:
    """SSH connection configuration.

    Supports multiple authentication methods:
    - Password only: set password, leave key fields empty
    - Key file: set key_path (and optionally key_passphrase)
    - Key content: set key_content (and optionally key_passphrase)
    """

    host: str
    user: str
    port: int = 22
    password: Optional[str] = None
    key_path: Optional[str] = None
    key_content: Optional[str] = None
    key_passphrase: Optional[str] = None
    timeout: int = 30

    @property
    def auth_method(self) -> SSHAuthMethod:
        """Determine which auth method to use based on config."""
        if self.key_content:
            return SSHAuthMethod.KEY_CONTENT
        if self.key_path:
            return SSHAuthMethod.KEY_FILE
        return SSHAuthMethod.PASSWORD

    def validate(self) -> None:
        """Validate that required fields are present."""
        if not self.host:
            raise ValueError("SSH host is required")
        if not self.user:
            raise ValueError("SSH user is required")

        # Must have at least one auth method
        if not any([self.password, self.key_path, self.key_content]):
            raise ValueError(
                "SSH authentication required: provide password, key_path, or key_content"
            )


def _load_private_key(
    key_content: Optional[str] = None,
    key_path: Optional[str] = None,
    passphrase: Optional[str] = None,
) -> paramiko.PKey:
    """Load a private key from content or file path."""
    if key_content:
        key_data = key_content
    elif key_path:
        with open(key_path, "r") as f:
            key_data = f.read()
    else:
        raise ValueError("Either key_content or key_path must be provided")

    key_file = io.StringIO(key_data)
    passphrase_bytes = passphrase.encode() if passphrase else None
    key_classes = [
        paramiko.RSAKey,
        paramiko.Ed25519Key,
        paramiko.ECDSAKey,
        paramiko.DSSKey,
    ]

    last_error = None
    for key_class in key_classes:
        try:
            key_file.seek(0)
            return key_class.from_private_key(key_file, password=passphrase_bytes)
        except Exception as e:
            last_error = e
    raise ValueError(f"Failed to load private key: {last_error}")


def _build_connect_kwargs(config: SSHConfig) -> dict:
    """Build paramiko connect kwargs from SSHConfig."""
    kwargs = {
        "hostname": config.host,
        "port": config.port,
        "username": config.user,
        "timeout": config.timeout,
        "allow_agent": False,
        "look_for_keys": False,
    }

    if config.auth_method == SSHAuthMethod.PASSWORD:
        kwargs["password"] = config.password
    else:
        kwargs["pkey"] = _load_private_key(
            key_content=config.key_content,
            key_path=config.key_path,
            passphrase=config.key_passphrase,
        )
        if config.password:
            kwargs["password"] = config.password

    return kwargs


def _create_ssh_client(config: SSHConfig) -> paramiko.SSHClient:
    """Create and connect an SSH client."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    logger.debug(
        f"SSH connecting to {config.user}@{config.host}:{config.port} using {config.auth_method.value}"
    )
    client.connect(**_build_connect_kwargs(config))
    return client


def _drain_channel(
    channel, stdout_chunks: list[bytes], stderr_chunks: list[bytes]
) -> None:
    """Drain any remaining data from the channel buffers."""
    while channel.recv_ready():
        stdout_chunks.append(channel.recv(4096))
    while channel.recv_stderr_ready():
        stderr_chunks.append(channel.recv_stderr(4096))


def _error_result(message: str) -> SSHResult:
    """Create a failed SSHResult with the given error message."""
    return SSHResult(stdout="", stderr=message, exit_code=-1, success=False)


def execute_ssh_command(
    config: SSHConfig,
    command: str,
    input_data: Optional[str] = None,
) -> SSHResult:
    """
    Execute a command over SSH.

    Args:
        config: SSH connection configuration
        command: Command to execute on remote host
        input_data: Optional data to send to command's stdin

    Returns:
        SSHResult with stdout, stderr, exit_code, and success flag
    """
    config.validate()
    client = None

    try:
        client = _create_ssh_client(config)

        # Execute command
        stdin, stdout, stderr = client.exec_command(command, timeout=config.timeout)

        # Send input data if provided
        if input_data:
            stdin.write(input_data)
            stdin.channel.shutdown_write()

        # Read output with total timeout
        channel = stdout.channel
        start_time = time.time()
        stdout_chunks = []
        stderr_chunks = []
        timed_out = False

        while True:
            # check timeout
            if config.timeout and (time.time() - start_time > config.timeout):
                logger.warning(
                    f"SSH command execution timed out after {config.timeout}s"
                )
                timed_out = True
                channel.close()
                break

            if channel.exit_status_ready():
                break

            # Check for data
            if channel.recv_ready():
                stdout_chunks.append(channel.recv(4096))

            if channel.recv_stderr_ready():
                stderr_chunks.append(channel.recv_stderr(4096))

            # Small sleep to yield CPU and prevent tight loop
            time.sleep(0.1)

        # Drain any remaining buffered data
        _drain_channel(channel, stdout_chunks, stderr_chunks)

        stdout_data = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        stderr_data = b"".join(stderr_chunks).decode("utf-8", errors="replace")

        # Filter out harmless stderr noise (e.g., tput warnings) at the source
        stderr_data = _filter_stderr_noise(stderr_data)

        if timed_out:
            exit_code = -1
            # Append timeout note to stderr but mark as success so partial stdout is returned
            timeout_msg = (
                f"[Command execution timed out after {config.timeout} seconds]"
            )
            stderr_data = (
                f"{stderr_data}\n{timeout_msg}" if stderr_data else timeout_msg
            )
        else:
            exit_code = channel.recv_exit_status()

        return SSHResult(
            stdout=stdout_data,
            stderr=stderr_data,
            exit_code=exit_code,
            success=(exit_code == 0) or timed_out,
        )

    except paramiko.AuthenticationException as e:
        logger.error(f"SSH authentication failed: {e}")
        return _error_result(f"SSH authentication failed: {e}")
    except socket.timeout:
        logger.error(f"SSH connection timed out after {config.timeout}s")
        return _error_result(f"SSH connection timed out after {config.timeout}s")
    except socket.error as e:
        logger.error(f"SSH connection error: {e}")
        return _error_result(f"SSH connection error: {e}")
    except Exception as e:
        logger.error(f"SSH error: {e}")
        return _error_result(f"SSH error: {e}")
    finally:
        if client:
            client.close()


def test_ssh_connection(config: SSHConfig) -> SSHResult:
    """
    Test SSH connection with a simple command.

    Returns SSHResult with connection test output.
    """
    return execute_ssh_command(config, "echo 'SSH connection successful'")


def expand_env_vars_via_ssh(config: SSHConfig, command: str) -> tuple[str, str | None]:
    """
    Expand environment variables in a command via SSH.

    Uses printf with the command to expand variables on the remote host,
    then returns the expanded command for validation.

    Args:
        config: SSH connection configuration
        command: Command containing env vars like $HOME, ${PATH}, etc.

    Returns:
        Tuple of (expanded_command, error_message). Error is None on success.
    """
    # Use printf %s to expand variables without executing the command
    # We wrap in single quotes and escape existing single quotes
    escaped_command = command.replace("'", "'\"'\"'")
    expand_cmd = f"printf '%s' '{escaped_command}'"

    result = execute_ssh_command(config, expand_cmd)
    if not result.success:
        return command, f"Failed to expand environment variables: {result.stderr}"

    return result.stdout, None


def ssh_config_from_dict(config_dict: dict) -> SSHConfig:
    """
    Create SSHConfig from a connection config dictionary.

    Maps common field names from tool configs to SSHConfig fields.
    Handles both 'ssh_' prefixed and non-prefixed field names.
    """

    def get_field(name: str, default=None):
        """Get field with or without ssh_ prefix."""
        return config_dict.get(f"ssh_{name}") or config_dict.get(name, default)

    return SSHConfig(
        host=get_field("host", ""),
        user=get_field("user", ""),
        port=int(get_field("port", 22)),
        password=get_field("password"),
        key_path=get_field("key_path"),
        key_content=get_field("key_content"),
        key_passphrase=get_field("key_passphrase"),
        timeout=int(get_field("timeout", 30)),
    )


# =============================================================================
# SSH Tunnel Support
# =============================================================================


@dataclass
class SSHTunnelConfig:
    """Configuration for SSH tunnel to a database.

    The tunnel forwards a local port to a remote host:port as seen from the SSH server.
    This allows connecting to databases that are only accessible from the SSH server
    (e.g., localhost:3306 on a remote server).
    """

    # SSH connection settings
    ssh_host: str
    ssh_user: str
    ssh_port: int = 22
    ssh_password: Optional[str] = None
    ssh_key_path: Optional[str] = None
    ssh_key_content: Optional[str] = None
    ssh_key_passphrase: Optional[str] = None

    # Remote endpoint (as seen from SSH server)
    remote_host: str = "127.0.0.1"  # Usually localhost on the remote server
    remote_port: int = 3306  # Default MySQL port

    # Local binding
    local_port: int = 0  # 0 = auto-assign

    timeout: int = 30

    def to_ssh_config(self) -> SSHConfig:
        """Convert to SSHConfig for connection."""
        return SSHConfig(
            host=self.ssh_host,
            user=self.ssh_user,
            port=self.ssh_port,
            password=self.ssh_password,
            key_path=self.ssh_key_path,
            key_content=self.ssh_key_content,
            key_passphrase=self.ssh_key_passphrase,
            timeout=self.timeout,
        )

    def validate(self) -> None:
        """Validate tunnel configuration."""
        if not self.ssh_host:
            raise ValueError("SSH host is required for tunnel")
        if not self.ssh_user:
            raise ValueError("SSH user is required for tunnel")
        if not any([self.ssh_password, self.ssh_key_path, self.ssh_key_content]):
            raise ValueError(
                "SSH authentication required: provide password, key_path, or key_content"
            )


class SSHTunnel:
    """
    SSH tunnel manager for database connections.

    Usage:
        tunnel = SSHTunnel(config)
        with tunnel:
            # Connect to database at ('127.0.0.1', tunnel.local_port)
            conn = pymysql.connect(host='127.0.0.1', port=tunnel.local_port, ...)

    Or for async:
        tunnel = SSHTunnel(config)
        tunnel.start()
        try:
            # Use tunnel.local_port
        finally:
            tunnel.stop()
    """

    def __init__(self, config: SSHTunnelConfig):
        self.config = config
        self._client: Optional[paramiko.SSHClient] = None
        self._transport: Optional[paramiko.Transport] = None
        self._local_port: int = 0
        self._server_socket: Optional[socket.socket] = None
        self._forward_thread: Optional[socket.socket] = None
        self._running = False
        self._connections: list = []

    @property
    def local_port(self) -> int:
        """Get the local port the tunnel is bound to."""
        return self._local_port

    @property
    def local_bind_address(self) -> tuple[str, int]:
        """Get the local bind address tuple."""
        return ("127.0.0.1", self._local_port)

    def start(self) -> int:
        """
        Start the SSH tunnel.

        Returns:
            The local port number to connect to.
        """
        self.config.validate()
        ssh_config = self.config.to_ssh_config()
        self._client = _create_ssh_client(ssh_config)
        self._transport = self._client.get_transport()

        # Create local server socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind(("127.0.0.1", self.config.local_port))
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)  # For clean shutdown

        self._local_port = self._server_socket.getsockname()[1]
        self._running = True

        # Start forwarding thread
        self._forward_thread = threading.Thread(target=self._forward_loop, daemon=True)
        self._forward_thread.start()

        logger.info(
            f"SSH tunnel started: localhost:{self._local_port} -> "
            f"{self.config.remote_host}:{self.config.remote_port} "
            f"via {self.config.ssh_host}"
        )

        return self._local_port

    def _forward_loop(self) -> None:
        """Accept connections and forward them through SSH."""
        while self._running:
            try:
                client_socket, addr = self._server_socket.accept()
                logger.debug(f"SSH tunnel connection from {addr}")

                # Open channel to remote host
                try:
                    channel = self._transport.open_channel(
                        "direct-tcpip",
                        (self.config.remote_host, self.config.remote_port),
                        addr,
                    )
                except Exception as e:
                    logger.error(f"Failed to open SSH channel: {e}")
                    client_socket.close()
                    continue

                if channel is None:
                    logger.error("SSH channel open returned None")
                    client_socket.close()
                    continue

                # Start bidirectional forwarding threads
                self._connections.append((client_socket, channel))

                t1 = threading.Thread(
                    target=self._forward_data,
                    args=(client_socket, channel, "local->remote"),
                    daemon=True,
                )
                t2 = threading.Thread(
                    target=self._forward_data,
                    args=(channel, client_socket, "remote->local"),
                    daemon=True,
                )
                t1.start()
                t2.start()

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"SSH tunnel accept error: {e}")
                break

    def _forward_data(self, src, dst, direction: str) -> None:
        """Forward data between sockets."""
        try:
            while self._running:
                try:
                    data = src.recv(4096)
                    if not data:
                        break
                    dst.sendall(data)
                except Exception:
                    break
        except Exception:
            pass
        finally:
            try:
                src.close()
            except Exception:
                pass
            try:
                dst.close()
            except Exception:
                pass

    def stop(self) -> None:
        """Stop the SSH tunnel."""
        self._running = False

        # Close all forwarded connections
        for client_socket, channel in self._connections:
            try:
                client_socket.close()
            except Exception:
                pass
            try:
                channel.close()
            except Exception:
                pass
        self._connections.clear()

        # Close server socket
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None

        # Close SSH connection
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._transport = None

        logger.debug("SSH tunnel stopped")

    def __enter__(self) -> "SSHTunnel":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


# =============================================================================
# SSH File Sync (best-effort live sync for userspace mounts)
# =============================================================================


@dataclass
class SSHSyncResult:
    """Result of an SSH directory sync operation."""

    files_synced: int
    errors: list[str]
    success: bool
    backend_used: str = "rsync"
    notice: str | None = None


def _remote_join(root: str, relative_path: str) -> str:
    normalized_root = root.rstrip("/") or "/"
    if not relative_path:
        return normalized_root
    relative = relative_path.strip("/")
    return f"{normalized_root}/{relative}" if normalized_root != "/" else f"/{relative}"


def _scan_remote_tree(
    sftp: paramiko.SFTPClient,
    remote_root: str,
    *,
    max_files: int,
    max_file_size_bytes: int,
) -> tuple[dict[str, tuple[int, int]], set[str], list[str]]:
    """Collect remote file metadata relative to *remote_root*."""
    errors: list[str] = []
    files: dict[str, tuple[int, int]] = {}
    directories: set[str] = {""}
    queue: deque[tuple[str, str]] = deque([(_remote_join(remote_root, ""), "")])

    while queue and len(files) < max_files:
        current_remote_dir, current_relative_dir = queue.popleft()
        try:
            entries = sftp.listdir_attr(current_remote_dir)
        except Exception as exc:
            errors.append(f"listdir {current_remote_dir}: {exc}")
            continue

        for entry in entries:
            if entry.filename in (".", ".."):
                continue
            child_relative = (
                entry.filename
                if not current_relative_dir
                else f"{current_relative_dir}/{entry.filename}"
            )
            child_remote = _remote_join(remote_root, child_relative)
            mode = entry.st_mode or 0
            if stat.S_ISDIR(mode):
                directories.add(child_relative)
                queue.append((child_remote, child_relative))
                continue
            if not stat.S_ISREG(mode):
                continue
            if (entry.st_size or 0) > max_file_size_bytes:
                continue
            files[child_relative] = (int(entry.st_size or 0), int(entry.st_mtime or 0))
            if len(files) >= max_files:
                break

    return files, directories, errors


def _scan_local_tree(
    local_root: _Path,
    *,
    max_files: int,
    max_file_size_bytes: int,
) -> tuple[dict[str, tuple[int, int]], set[str], list[str]]:
    """Collect local file metadata relative to *local_root*."""
    errors: list[str] = []
    files: dict[str, tuple[int, int]] = {}
    directories: set[str] = {""}

    if not local_root.exists():
        return files, directories, errors

    for root, _dirnames, filenames in _os.walk(str(local_root)):
        root_path = _Path(root)
        relative_dir = root_path.relative_to(local_root)
        relative_dir_str = "" if str(relative_dir) == "." else relative_dir.as_posix()
        directories.add(relative_dir_str)
        for filename in filenames:
            if len(files) >= max_files:
                return files, directories, errors
            local_file = root_path / filename
            relative_file = (
                filename if not relative_dir_str else f"{relative_dir_str}/{filename}"
            )
            try:
                stat_result = local_file.stat()
            except Exception as exc:
                errors.append(f"stat {local_file}: {exc}")
                continue
            if stat_result.st_size > max_file_size_bytes:
                continue
            files[relative_file] = (
                int(stat_result.st_size),
                int(stat_result.st_mtime),
            )

    return files, directories, errors


def _ensure_remote_directory(
    sftp: paramiko.SFTPClient,
    remote_root: str,
    relative_dir: str,
    created_dirs: set[str],
) -> None:
    """Create *relative_dir* and its parents on the remote server if missing."""
    normalized_relative = relative_dir.strip("/")
    if normalized_relative in created_dirs:
        return
    current = ""
    for part in PurePosixPath(normalized_relative).parts:
        current = part if not current else f"{current}/{part}"
        if current in created_dirs:
            continue
        remote_dir = _remote_join(remote_root, current)
        try:
            sftp.stat(remote_dir)
        except IOError:
            sftp.mkdir(remote_dir)
        created_dirs.add(current)


def _set_local_mtime(path: _Path, mtime_seconds: int) -> None:
    timestamp = float(max(0, mtime_seconds))
    _os.utime(path, (timestamp, timestamp))


def _set_remote_mtime(
    sftp: paramiko.SFTPClient,
    remote_path: str,
    mtime_seconds: int,
) -> None:
    timestamp = int(max(0, mtime_seconds))
    sftp.utime(remote_path, (timestamp, timestamp))


def _download_sftp_file(
    sftp: paramiko.SFTPClient,
    remote_root: str,
    local_root: _Path,
    relative_path: str,
    remote_mtime: int,
) -> None:
    remote_file = _remote_join(remote_root, relative_path)
    local_file = local_root / PurePosixPath(relative_path)
    local_file.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_file, str(local_file))
    _set_local_mtime(local_file, remote_mtime)


def _upload_sftp_file(
    sftp: paramiko.SFTPClient,
    remote_root: str,
    local_root: _Path,
    relative_path: str,
    created_remote_dirs: set[str],
) -> None:
    local_file = local_root / PurePosixPath(relative_path)
    remote_file = _remote_join(remote_root, relative_path)
    remote_parent = PurePosixPath(relative_path).parent
    remote_parent_str = "" if str(remote_parent) == "." else remote_parent.as_posix()
    if remote_parent_str:
        _ensure_remote_directory(sftp, remote_root, remote_parent_str, created_remote_dirs)
    sftp.put(str(local_file), remote_file)
    _set_remote_mtime(sftp, remote_file, int(local_file.stat().st_mtime))


def _sync_ssh_directory_merge(
    sftp: paramiko.SFTPClient,
    remote_root: str,
    local_root: _Path,
    *,
    max_files: int,
    max_file_size_bytes: int,
) -> SSHSyncResult:
    """Bidirectional merge where newest mtime wins and nothing is deleted."""
    errors: list[str] = []
    files_synced = 0
    remote_files, remote_dirs, remote_errors = _scan_remote_tree(
        sftp,
        remote_root,
        max_files=max_files,
        max_file_size_bytes=max_file_size_bytes,
    )
    local_files, local_dirs, local_errors = _scan_local_tree(
        local_root,
        max_files=max_files,
        max_file_size_bytes=max_file_size_bytes,
    )
    errors.extend(remote_errors)
    errors.extend(local_errors)

    local_root.mkdir(parents=True, exist_ok=True)
    created_remote_dirs = set(remote_dirs)
    for relative_dir in sorted(remote_dirs - local_dirs):
        if relative_dir:
            (local_root / PurePosixPath(relative_dir)).mkdir(parents=True, exist_ok=True)
    for relative_dir in sorted(local_dirs - remote_dirs):
        if relative_dir:
            try:
                _ensure_remote_directory(sftp, remote_root, relative_dir, created_remote_dirs)
            except Exception as exc:
                errors.append(f"mkdir {relative_dir}: {exc}")

    for relative_path in sorted(set(remote_files) | set(local_files)):
        remote_meta = remote_files.get(relative_path)
        local_meta = local_files.get(relative_path)
        try:
            if remote_meta is None and local_meta is not None:
                _upload_sftp_file(
                    sftp,
                    remote_root,
                    local_root,
                    relative_path,
                    created_remote_dirs,
                )
                files_synced += 1
                continue

            if local_meta is None and remote_meta is not None:
                _download_sftp_file(
                    sftp,
                    remote_root,
                    local_root,
                    relative_path,
                    remote_meta[1],
                )
                files_synced += 1
                continue

            if remote_meta is None or local_meta is None:
                continue

            remote_size, remote_mtime = remote_meta
            local_size, local_mtime = local_meta
            if remote_mtime > local_mtime:
                _download_sftp_file(
                    sftp,
                    remote_root,
                    local_root,
                    relative_path,
                    remote_mtime,
                )
                files_synced += 1
            elif local_mtime > remote_mtime:
                _upload_sftp_file(
                    sftp,
                    remote_root,
                    local_root,
                    relative_path,
                    created_remote_dirs,
                )
                files_synced += 1
            elif remote_size != local_size:
                _download_sftp_file(
                    sftp,
                    remote_root,
                    local_root,
                    relative_path,
                    remote_mtime,
                )
                files_synced += 1
        except Exception as exc:
            errors.append(f"sync {relative_path}: {exc}")

    return SSHSyncResult(
        files_synced=files_synced,
        errors=errors,
        success=len(errors) == 0,
        backend_used="paramiko",
    )


def _sync_ssh_directory_delete(
    sftp: paramiko.SFTPClient,
    remote_root: str,
    local_root: _Path,
    *,
    max_files: int,
    max_file_size_bytes: int,
) -> SSHSyncResult:
    """Remote-wins sync that deletes local files absent on the remote."""
    local_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = local_root.parent / (
        f".{local_root.name}.sync-{_os.getpid()}-{threading.get_ident()}-{int(time.time() * 1000)}"
    )
    if temp_root.exists():
        shutil.rmtree(temp_root, ignore_errors=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    files_synced = 0
    try:
        remote_files, remote_dirs, remote_errors = _scan_remote_tree(
            sftp,
            remote_root,
            max_files=max_files,
            max_file_size_bytes=max_file_size_bytes,
        )
        errors.extend(remote_errors)

        for relative_dir in sorted(remote_dirs):
            if relative_dir:
                (temp_root / PurePosixPath(relative_dir)).mkdir(parents=True, exist_ok=True)

        for relative_path, (_size, remote_mtime) in sorted(remote_files.items()):
            try:
                _download_sftp_file(
                    sftp,
                    remote_root,
                    temp_root,
                    relative_path,
                    remote_mtime,
                )
                files_synced += 1
            except Exception as exc:
                errors.append(f"get {_remote_join(remote_root, relative_path)}: {exc}")

        if local_root.is_symlink() or local_root.is_file():
            local_root.unlink(missing_ok=True)
        elif local_root.exists():
            shutil.rmtree(local_root)
        temp_root.replace(local_root)
    except Exception as exc:
        errors.append(f"SSH sync error: {exc}")
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)

    return SSHSyncResult(
        files_synced=files_synced,
        errors=errors,
        success=len(errors) == 0,
        backend_used="paramiko",
    )


def is_rsync_missing_error(output: str | None) -> bool:
    """Return True when stderr/stdout clearly shows remote rsync is unavailable."""
    normalized = (output or "").lower()
    if not normalized:
        return False
    patterns = (
        "rsync: command not found",
        "rsync: not found",
        "bash: rsync: command not found",
        "sh: 1: rsync: not found",
        "remote command not found",
    )
    return any(pattern in normalized for pattern in patterns)


def check_remote_rsync_available(config: SSHConfig) -> tuple[bool | None, str | None]:
    """Check whether the remote SSH endpoint has an rsync binary available.

    Returns ``(True, None)`` when rsync is available, ``(False, message)`` when
    the remote explicitly lacks rsync, and ``(None, message)`` when the probe
    could not be completed because the SSH command itself failed.
    """
    config.validate()
    temp_key_file: str | None = None
    temp_dir: str | None = None
    askpass_path: str | None = None
    ssh_wrapper_path: str | None = None

    try:
        temp_dir = tempfile.mkdtemp(prefix="ragtime_rsync_probe_")

        if config.key_content:
            temp_key_file = _os.path.join(temp_dir, "id_rsa")
            with open(temp_key_file, "w", encoding="utf-8") as fh:
                fh.write(config.key_content)
                if not config.key_content.endswith("\n"):
                    fh.write("\n")
            _os.chmod(temp_key_file, 0o600)

        ssh_args = _build_rsync_ssh_cmd(config, temp_key_path=temp_key_file)
        if config.password or config.key_passphrase:
            askpass_path = _write_rsync_askpass_script(temp_dir)
        ssh_wrapper_path = _write_rsync_ssh_wrapper(
            ssh_args,
            askpass_path=askpass_path,
            temp_dir=temp_dir,
        )

        env = _os.environ.copy()
        if config.password:
            env["RAGTIME_SSH_PASSWORD"] = config.password
        if config.key_passphrase:
            env["RAGTIME_SSH_KEY_PASSPHRASE"] = config.key_passphrase

        remote_command = (
            'env PATH="$PATH:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin" '
            "sh -lc 'if command -v rsync >/dev/null 2>&1 "
            "|| [ -x /usr/local/bin/rsync ] "
            "|| [ -x /usr/bin/rsync ] "
            "|| [ -x /bin/rsync ]; "
            "then printf __RAGTIME_RSYNC_AVAILABLE__; "
            "else printf __RAGTIME_RSYNC_MISSING__; fi'"
        )
        proc = subprocess.run(
            [ssh_wrapper_path, f"{config.user}@{config.host}", remote_command],
            capture_output=True,
            text=True,
            timeout=config.timeout,
            env=env,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        stdout_data = (proc.stdout or "").strip()
        stderr_data = _filter_stderr_noise(proc.stderr or "").strip()

        if proc.returncode != 0:
            error_msg = stderr_data or stdout_data or "Failed to check remote rsync availability"
            return None, error_msg

        if "__RAGTIME_RSYNC_AVAILABLE__" in stdout_data:
            return True, None
        if "__RAGTIME_RSYNC_MISSING__" in stdout_data:
            return False, "Remote server does not have rsync installed"
        return None, stdout_data or stderr_data or "Unexpected response while checking remote rsync availability"
    except subprocess.TimeoutExpired:
        return None, f"Timed out checking remote rsync availability after {config.timeout}s"
    except Exception as exc:
        return None, f"Failed to check remote rsync availability: {exc}"
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def sync_ssh_directory(
    config: SSHConfig,
    remote_path: str,
    local_path: str,
    *,
    max_files: int = 5000,
    max_file_size_bytes: int = 50 * 1024 * 1024,
    sync_deletes: bool = False,
) -> SSHSyncResult:
    """Sync a remote directory to a local path via Paramiko/SFTP.

    This is the built-in fallback for SSH mounts when the remote server does
    not have ``rsync`` installed. It mirrors the userspace rsync semantics:
    - ``sync_deletes=True``: remote wins and local-only files are deleted
    - ``sync_deletes=False``: two-way merge where newest mtime wins
    """
    config.validate()
    local_root = _Path(local_path)
    client: Optional[paramiko.SSHClient] = None
    sftp: Optional[paramiko.SFTPClient] = None

    try:
        client = _create_ssh_client(config)
        sftp = client.open_sftp()
        result = (
            _sync_ssh_directory_delete(
                sftp,
                remote_path,
                local_root,
                max_files=max_files,
                max_file_size_bytes=max_file_size_bytes,
            )
            if sync_deletes
            else _sync_ssh_directory_merge(
                sftp,
                remote_path,
                local_root,
                max_files=max_files,
                max_file_size_bytes=max_file_size_bytes,
            )
        )
        return result
    except paramiko.AuthenticationException as exc:
        return SSHSyncResult(
            files_synced=0,
            errors=[f"SSH auth failed: {exc}"],
            success=False,
            backend_used="paramiko",
        )
    except Exception as exc:
        return SSHSyncResult(
            files_synced=0,
            errors=[f"SSH sync error: {exc}"],
            success=False,
            backend_used="paramiko",
        )
    finally:
        if sftp:
            sftp.close()
        if client:
            client.close()


# =============================================================================
# Rsync-based SSH file sync
# =============================================================================


def _build_rsync_ssh_cmd(
    config: SSHConfig,
    *,
    temp_key_path: str | None = None,
) -> list[str]:
    """Build SSH command arguments for rsync transport.

    This mirrors Ragtime's Paramiko behavior as closely as OpenSSH allows:
    - password-only auth
    - key file or key content auth
    - encrypted keys via key passphrase
    - key plus password fallback / second factor
    """
    parts = [
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
    ]
    if config.port and config.port != 22:
        parts.extend(["-p", str(config.port)])
    key = temp_key_path or config.key_path
    if key:
        parts.extend(["-i", key])
        parts.extend(["-o", "IdentitiesOnly=yes"])
    if config.timeout:
        parts.extend(["-o", f"ConnectTimeout={config.timeout}"])

    if key and config.password:
        parts.extend(
            [
                "-o",
                "PreferredAuthentications=publickey,keyboard-interactive,password",
                "-o",
                "PasswordAuthentication=yes",
                "-o",
                "KbdInteractiveAuthentication=yes",
                "-o",
                "NumberOfPasswordPrompts=3",
            ]
        )
    elif key:
        parts.extend(
            [
                "-o",
                "PreferredAuthentications=publickey",
                "-o",
                "PasswordAuthentication=no",
                "-o",
                "KbdInteractiveAuthentication=no",
            ]
        )
    else:
        parts.extend(
            [
                "-o",
                "PreferredAuthentications=keyboard-interactive,password",
                "-o",
                "PubkeyAuthentication=no",
                "-o",
                "PasswordAuthentication=yes",
                "-o",
                "KbdInteractiveAuthentication=yes",
                "-o",
                "NumberOfPasswordPrompts=3",
            ]
        )

    return parts


def _write_rsync_askpass_script(temp_dir: str) -> str:
    """Create an askpass helper that serves key passphrases and passwords."""
    script_path = _os.path.join(temp_dir, "ssh-askpass.py")
    script = """#!/usr/bin/env python3
import os
import sys

prompt = " ".join(sys.argv[1:]).lower()
password = os.environ.get("RAGTIME_SSH_PASSWORD", "")
passphrase = os.environ.get("RAGTIME_SSH_KEY_PASSPHRASE", "")

if "passphrase" in prompt:
    value = passphrase
elif "password" in prompt or "verification code" in prompt:
    value = password or passphrase
else:
    value = passphrase or password

sys.stdout.write(value)
sys.stdout.write("\\n")
"""
    with open(script_path, "w", encoding="utf-8") as handle:
        handle.write(script)
    _os.chmod(script_path, 0o700)
    return script_path


def _write_rsync_ssh_wrapper(
    ssh_args: list[str],
    *,
    askpass_path: str | None,
    temp_dir: str,
) -> str:
    """Create a wrapper script for rsync's ``-e`` transport command."""
    wrapper_path = _os.path.join(temp_dir, "ssh-wrapper.sh")
    quoted_ssh = " ".join(shlex.quote(part) for part in ssh_args)
    script_lines = ["#!/bin/sh", "set -eu"]
    if askpass_path:
        script_lines.extend(
            [
                'export DISPLAY="${DISPLAY:-:0}"',
                f"export SSH_ASKPASS={shlex.quote(askpass_path)}",
                'export SSH_ASKPASS_REQUIRE="force"',
            ]
        )
    # Keep stdin attached so rsync can stream its protocol over the SSH transport.
    script_lines.append(f'exec {quoted_ssh} "$@"')
    with open(wrapper_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(script_lines) + "\n")
    _os.chmod(wrapper_path, 0o700)
    return wrapper_path


def rsync_ssh_directory(
    config: SSHConfig,
    remote_path: str,
    local_path: str,
    *,
    sync_deletes: bool = False,
    timeout_seconds: int = 300,
) -> SSHSyncResult:
    """Sync a remote directory to a local path using the ``rsync`` binary over SSH.

    When *sync_deletes* is True a single rsync pass with ``--delete`` is
    executed (remote wins, local-only files are removed).

    When *sync_deletes* is False two rsync passes run:
      1. remote -> local  (merge, no delete — picks up remote changes)
      2. local  -> remote (merge, no delete, ``--update`` — pushes local-only
         or newer-mtime local files back to the remote)
    This implements "newest mtime wins" bidirectional merge.

    Returns:
        SSHSyncResult with a best-effort ``files_synced`` count.
    """
    config.validate()
    local_root = _Path(local_path)
    local_root.mkdir(parents=True, exist_ok=True)

    normalized_remote = remote_path.rstrip("/") + "/"
    remote_spec = f"{config.user}@{config.host}:{normalized_remote}"

    errors: list[str] = []
    files_synced = 0
    temp_key_file: str | None = None
    temp_dir: str | None = None
    askpass_path: str | None = None
    ssh_wrapper_path: str | None = None

    try:
        temp_dir = tempfile.mkdtemp(prefix="ragtime_rsync_")

        # If auth is key_content, write it to a secure temp file.
        if config.key_content:
            temp_key_file = _os.path.join(temp_dir, "id_rsa")
            with open(temp_key_file, "w", encoding="utf-8") as fh:
                fh.write(config.key_content)
                if not config.key_content.endswith("\n"):
                    fh.write("\n")
            _os.chmod(temp_key_file, 0o600)

        ssh_args = _build_rsync_ssh_cmd(config, temp_key_path=temp_key_file)
        if config.password or config.key_passphrase:
            askpass_path = _write_rsync_askpass_script(temp_dir)
        ssh_wrapper_path = _write_rsync_ssh_wrapper(
            ssh_args,
            askpass_path=askpass_path,
            temp_dir=temp_dir,
        )

        # Base rsync flags: archive mode, compress, partial (resume),
        # itemize changes (for counting transferred files).
        base_flags = [
            "-az",
            "--partial",
            "--itemize-changes",
            "--timeout=60",
        ]

        def _run_rsync(args: list[str]) -> tuple[int, str, str]:
            """Execute rsync with the generated SSH wrapper."""
            cmd: list[str] = []
            env = _os.environ.copy()
            if config.password:
                env["RAGTIME_SSH_PASSWORD"] = config.password
            if config.key_passphrase:
                env["RAGTIME_SSH_KEY_PASSPHRASE"] = config.key_passphrase
            cmd.append("rsync")
            cmd.extend(args)
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
                check=False,
            )
            return proc.returncode, proc.stdout, proc.stderr

        def _count_transferred(stdout: str) -> int:
            """Count file transfers from rsync itemize-changes output."""
            count = 0
            for line in stdout.splitlines():
                # Itemized lines start with a change indicator like ">f" (receiving file)
                # or "<f" (sending file). Directories are "cd" or ".d".
                stripped = line.strip()
                if stripped and len(stripped) > 2 and stripped[0] in "><c." and "f" in stripped[:3]:
                    count += 1
            return count

        # --- Pass 1: remote -> local ---
        pull_args = list(base_flags) + ["-e", ssh_wrapper_path or "ssh"]
        if sync_deletes:
            pull_args.append("--delete")
        pull_args.extend([remote_spec, str(local_root) + "/"])

        rc, stdout, stderr = _run_rsync(pull_args)
        files_synced += _count_transferred(stdout)
        if rc != 0:
            # rsync exit code 23 = partial transfer (some files couldn't be read)
            # rsync exit code 24 = partial transfer (source files vanished)
            if rc in (23, 24):
                errors.append(f"rsync pull partial error (exit {rc}): {stderr.strip()[:300]}")
            else:
                errors.append(f"rsync pull failed (exit {rc}): {stderr.strip()[:300]}")
                return SSHSyncResult(files_synced=files_synced, errors=errors, success=False)

        # --- Pass 2: local -> remote (bidirectional merge, only when not deleting) ---
        if not sync_deletes:
            push_args = list(base_flags) + ["-e", ssh_wrapper_path or "ssh", "--update"]
            push_args.extend([str(local_root) + "/", remote_spec])
            rc2, stdout2, stderr2 = _run_rsync(push_args)
            files_synced += _count_transferred(stdout2)
            if rc2 != 0 and rc2 not in (23, 24):
                errors.append(f"rsync push failed (exit {rc2}): {stderr2.strip()[:300]}")

    except subprocess.TimeoutExpired:
        errors.append(f"rsync timed out after {timeout_seconds}s")
    except FileNotFoundError:
        errors.append("rsync binary not found — ensure rsync is installed in the container")
    except Exception as exc:
        errors.append(f"rsync error: {exc}")
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return SSHSyncResult(
        files_synced=files_synced,
        errors=errors,
        success=len(errors) == 0,
    )


def ssh_tunnel_config_from_dict(
    config_dict: dict, default_remote_port: int = 3306
) -> SSHTunnelConfig:
    """
    Create SSHTunnelConfig from a connection config dictionary.

    Expects fields prefixed with 'ssh_tunnel_' for tunnel settings.
    The remote host/port for the tunnel are taken from the main 'host'/'port'
    fields in the config (these represent the remote endpoint from the SSH
    server's perspective when tunneling).
    """

    def _coerce_int(value, default: int) -> int:
        """Best-effort int conversion that tolerates blanks and wrong types."""
        try:
            if value is None:
                return default
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    return default
                return int(stripped)
            return int(value)
        except (TypeError, ValueError):
            return default

    # Remote endpoint is the main host/port - this is where the database is
    # from the SSH server's perspective (usually 127.0.0.1 / localhost)
    remote_host = config_dict.get("host", "127.0.0.1") or "127.0.0.1"
    remote_port = _coerce_int(
        config_dict.get("port", default_remote_port), default_remote_port
    )

    return SSHTunnelConfig(
        ssh_host=config_dict.get("ssh_tunnel_host", ""),
        ssh_user=config_dict.get("ssh_tunnel_user", ""),
        ssh_port=_coerce_int(config_dict.get("ssh_tunnel_port", 22), 22),
        ssh_password=config_dict.get("ssh_tunnel_password"),
        ssh_key_path=config_dict.get("ssh_tunnel_key_path"),
        ssh_key_content=config_dict.get("ssh_tunnel_key_content"),
        ssh_key_passphrase=config_dict.get("ssh_tunnel_key_passphrase"),
        remote_host=remote_host,
        remote_port=remote_port,
        timeout=_coerce_int(config_dict.get("ssh_tunnel_timeout", 30), 30),
    )


def test_ssh_tunnel(config: SSHTunnelConfig) -> tuple[bool, str]:
    """
    Test SSH tunnel connectivity.

    Verifies that the SSH connection can be established and a channel
    to the remote endpoint can be opened.

    Returns:
        Tuple of (success, message)
    """
    try:
        config.validate()
    except ValueError as e:
        return False, str(e)

    client = None
    try:
        client = _create_ssh_client(config.to_ssh_config())
        transport = client.get_transport()

        # Try to open a channel to verify the remote endpoint is reachable
        try:
            channel = transport.open_channel(
                "direct-tcpip",
                (config.remote_host, config.remote_port),
                ("127.0.0.1", 0),
                timeout=10,
            )
            if channel:
                channel.close()
                return (
                    True,
                    f"SSH tunnel test successful: {config.ssh_host} -> "
                    f"{config.remote_host}:{config.remote_port}",
                )
            else:
                return (
                    False,
                    f"Could not open channel to {config.remote_host}:{config.remote_port}",
                )
        except Exception as e:
            return (
                False,
                f"SSH connected but cannot reach {config.remote_host}:{config.remote_port}: {e}",
            )

    except paramiko.AuthenticationException as e:
        return False, f"SSH authentication failed: {e}"
    except socket.timeout:
        return False, f"SSH connection timed out"
    except socket.error as e:
        return False, f"SSH connection error: {e}"
    except Exception as e:
        return False, f"SSH tunnel test failed: {e}"
    finally:
        if client:
            client.close()


def build_ssh_tunnel_config(config: dict, host: str, port: int) -> dict | None:
    """
    Build a normalized SSH tunnel config dict from a tool connection config.

    This extracts ssh_tunnel_* prefixed fields from the config and builds
    a dict suitable for passing to ssh_tunnel_config_from_dict().

    Args:
        config: Tool connection config dict with ssh_tunnel_* fields
        host: Remote host (database host from SSH server's perspective)
        port: Remote port (database port from SSH server's perspective)

    Returns:
        Dict for ssh_tunnel_config_from_dict(), or None if SSH tunnel not enabled
    """
    if not config.get("ssh_tunnel_enabled"):
        return None

    return {
        "host": host or "127.0.0.1",
        "port": port,
        "ssh_tunnel_host": config.get("ssh_tunnel_host"),
        "ssh_tunnel_port": config.get("ssh_tunnel_port"),
        "ssh_tunnel_user": config.get("ssh_tunnel_user"),
        "ssh_tunnel_password": config.get("ssh_tunnel_password"),
        "ssh_tunnel_key_path": config.get("ssh_tunnel_key_path"),
        "ssh_tunnel_key_content": config.get("ssh_tunnel_key_content"),
        "ssh_tunnel_key_passphrase": config.get("ssh_tunnel_key_passphrase"),
    }
