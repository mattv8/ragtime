"""Helpers for running Docker CLI commands on a remote host over SSH."""

from __future__ import annotations

import shlex
from typing import Any

from ragtime.core.ssh import SSHConfig, SSHResult, execute_ssh_command

DOCKER_OUTPUT_BEGIN_MARKER = "__RAGTIME_DOCKER_OUTPUT_BEGIN__"
DOCKER_OUTPUT_END_MARKER = "__RAGTIME_DOCKER_OUTPUT_END__"


def _build_heredoc_command(command: list[str], input_data: str) -> str:
    """Build a remote shell command that feeds input_data to command via heredoc."""
    delimiter = "RAGTIME_DOCKER_STDIN"
    while delimiter in input_data:
        delimiter = f"{delimiter}_X"
    return f"{shlex.join(command)} <<'{delimiter}'\n{input_data}\n{delimiter}"


def _wrap_with_output_markers(command: str) -> str:
    """Wrap a remote command so its stdout is delimited by sentinel markers.

    Remote hosts can print login banners/MOTD to stdout for SSH exec sessions
    (e.g. Proxmox LXC helper scripts), which corrupts Docker JSON/template
    output. Markers let the client isolate the command's real stdout while
    preserving the command's exit status.
    """
    return (
        f"printf '%s\\n' {shlex.quote(DOCKER_OUTPUT_BEGIN_MARKER)}\n"
        f"{command}\n"
        "status=$?\n"
        f"printf '\\n%s\\n' {shlex.quote(DOCKER_OUTPUT_END_MARKER)}\n"
        "exit $status"
    )


def _extract_marked_stdout(stdout: str) -> str:
    """Return only the stdout between the sentinel markers.

    Preserves the command's own newlines exactly (callers parse line-oriented
    JSON and pipe-delimited fields). Falls back to raw stdout when the begin
    marker is absent, and to everything after the begin marker when the end
    marker is absent (e.g. truncated output).
    """
    begin_index = stdout.find(DOCKER_OUTPUT_BEGIN_MARKER)
    if begin_index < 0:
        return stdout

    content_start = begin_index + len(DOCKER_OUTPUT_BEGIN_MARKER)
    if stdout.startswith("\r\n", content_start):
        content_start += 2
    elif stdout.startswith("\n", content_start):
        content_start += 1

    end_index = stdout.find(DOCKER_OUTPUT_END_MARKER, content_start)
    if end_index < 0:
        return stdout[content_start:]

    content = stdout[content_start:end_index]
    # The wrapper prints a newline before the end marker; remove that separator
    # without stripping newlines that came from the command itself.
    if content.endswith("\r\n"):
        return content[:-2]
    if content.endswith("\n"):
        return content[:-1]
    return content


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
    docker_command = _build_heredoc_command(command, input_data) if input_data is not None else shlex.join(command)
    result = execute_ssh_command(ssh_config, _wrap_with_output_markers(docker_command))
    result.stdout = _extract_marked_stdout(result.stdout)
    return result
