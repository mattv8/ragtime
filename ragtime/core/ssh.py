"""
SSH Utilities - Robust SSH execution with multiple auth methods.

Supports:
- Password-only authentication
- Key-only authentication (from file path or content)
- Key + passphrase authentication
"""

import io
import socket
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import paramiko

from ragtime.core.logging import get_logger

logger = get_logger(__name__)


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
    """
    Load a private key from content or file path.

    Supports RSA, Ed25519, ECDSA, and DSA key types.
    Handles keys with or without passphrases.
    """
    if key_content:
        key_data = key_content
    elif key_path:
        with open(key_path, "r") as f:
            key_data = f.read()
    else:
        raise ValueError("Either key_content or key_path must be provided")

    # Try different key types
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
        except paramiko.ssh_exception.SSHException as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"Failed to load private key: {last_error}")


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

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Build connection kwargs
        connect_kwargs = {
            "hostname": config.host,
            "port": config.port,
            "username": config.user,
            "timeout": config.timeout,
            "allow_agent": False,
            "look_for_keys": False,
        }

        # Add authentication
        auth_method = config.auth_method
        logger.debug(f"SSH connecting to {config.user}@{config.host}:{config.port} using {auth_method.value}")

        if auth_method == SSHAuthMethod.PASSWORD:
            connect_kwargs["password"] = config.password
        else:
            # Key-based auth (either from content or file)
            pkey = _load_private_key(
                key_content=config.key_content,
                key_path=config.key_path,
                passphrase=config.key_passphrase,
            )
            connect_kwargs["pkey"] = pkey
            # If password is also provided with key, it may be used for passphrase
            # but paramiko handles this in the key loading

        client.connect(**connect_kwargs)

        # Execute command
        stdin, stdout, stderr = client.exec_command(command, timeout=config.timeout)

        # Send input data if provided
        if input_data:
            stdin.write(input_data)
            stdin.channel.shutdown_write()

        # Read output
        stdout_data = stdout.read().decode("utf-8", errors="replace")
        stderr_data = stderr.read().decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()

        return SSHResult(
            stdout=stdout_data,
            stderr=stderr_data,
            exit_code=exit_code,
            success=exit_code == 0,
        )

    except paramiko.AuthenticationException as e:
        logger.error(f"SSH authentication failed: {e}")
        return SSHResult(
            stdout="",
            stderr=f"SSH authentication failed: {e}",
            exit_code=-1,
            success=False,
        )
    except socket.timeout as e:
        logger.error(f"SSH connection timed out: {e}")
        return SSHResult(
            stdout="",
            stderr=f"SSH connection timed out after {config.timeout}s",
            exit_code=-1,
            success=False,
        )
    except socket.error as e:
        logger.error(f"SSH connection error: {e}")
        return SSHResult(
            stdout="",
            stderr=f"SSH connection error: {e}",
            exit_code=-1,
            success=False,
        )
    except Exception as e:
        logger.error(f"SSH error: {e}")
        return SSHResult(
            stdout="",
            stderr=f"SSH error: {e}",
            exit_code=-1,
            success=False,
        )
    finally:
        client.close()


def test_ssh_connection(config: SSHConfig) -> SSHResult:
    """
    Test SSH connection with a simple command.

    Returns SSHResult with connection test output.
    """
    return execute_ssh_command(config, "echo 'SSH connection successful'")


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
