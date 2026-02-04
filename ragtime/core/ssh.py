"""
SSH Utilities - Robust SSH execution with multiple auth methods.

Supports:
- Password-only authentication
- Key-only authentication (from file path or content)
- Key + passphrase authentication
"""

import io
import socket
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import paramiko

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

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
