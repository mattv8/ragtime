"""Helpers for running Docker CLI commands on a remote host over SSH."""

from __future__ import annotations

import shlex
from typing import Any

from ragtime.core.ssh import SSHConfig, SSHResult, execute_ssh_command


def _build_heredoc_command(command: list[str], input_data: str) -> str:
    """Build a remote shell command that feeds input_data to command via heredoc."""
    delimiter = "RAGTIME_DOCKER_STDIN"
    while delimiter in input_data:
        delimiter = f"{delimiter}_X"
    return f"{shlex.join(command)} <<'{delimiter}'\n{input_data}\n{delimiter}"


def docker_ssh_config_from_dict(config: dict[str, Any], *, timeout: int = 30) -> SSHConfig | None:
    """Build an SSHConfig from docker_ssh_* fields in a tool config."""
    if config.get("docker_ssh_enabled") is False:
        return None

    host = str(config.get("docker_ssh_host") or "")
    user = str(config.get("docker_ssh_user") or "")
    if not host or not user:
        return None

    return SSHConfig(
        host=host,
        port=int(config.get("docker_ssh_port") or 22),
        user=user,
        password=str(password) if (password := config.get("docker_ssh_password")) else None,
        key_path=str(key_path) if (key_path := config.get("docker_ssh_key_path")) else None,
        key_content=str(key_content) if (key_content := config.get("docker_ssh_key_content")) else None,
        key_passphrase=str(key_passphrase) if (key_passphrase := config.get("docker_ssh_key_passphrase")) else None,
        timeout=timeout,
    )


def execute_docker_command_on_remote_host(
    ssh_config: SSHConfig,
    command: list[str],
    input_data: str | None = None,
) -> SSHResult:
    """Execute a Docker CLI command on a remote host over SSH."""
    remote_command = _build_heredoc_command(command, input_data) if input_data is not None else shlex.join(command)
    return execute_ssh_command(ssh_config, remote_command)
