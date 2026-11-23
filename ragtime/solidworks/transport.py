"""SSH transport for the SolidWorks host."""

import asyncio
import threading
import time
from pathlib import Path
from typing import Literal, Protocol

import paramiko

from ragtime.core.ssh import RAGTIME_OUTPUT_LIMIT_MARKER, SSHConfig, SSHResult, _create_ssh_client, execute_ssh_command
from ragtime.solidworks.host_activity import run_blocking
from ragtime.solidworks.protocol import SolidworksProtocolError, encode_request, parse_response

TransportErrorKind = Literal["ssh_unreachable", "ssh_auth_failed", "timeout", "no_response"]
SOLIDWORKS_MAX_OUTPUT_BYTES = 1024 * 1024
OUTPUT_LIMIT_MESSAGE = "The SolidWorks agent returned too much output."


class SolidworksTransportError(Exception):
    """Transport failure with a safe, stable classification."""

    def __init__(self, kind: TransportErrorKind, message: str, *, output_limit: bool = False):
        super().__init__(message)
        self.kind = kind
        self.output_limit = output_limit


class SolidworksTransport(Protocol):
    async def call(self, request: dict, *, timeout_seconds: float) -> dict: ...

    async def run_powershell(self, script: str, *, timeout_seconds: float) -> SSHResult: ...

    async def upload_file(self, local_path: Path, remote_path: str, *, timeout_seconds: float) -> None: ...


def build_ssh_config(connection_config: dict, *, timeout: float) -> SSHConfig:
    """Map the Windows shell connection fields to a PowerShell SSH config with a total deadline."""
    if not timeout > 0:
        raise ValueError("SolidWorks SSH timeout must be a positive number of seconds")
    return SSHConfig(
        host=connection_config.get("host", ""),
        port=connection_config.get("port", 22),
        user=connection_config.get("user", ""),
        password=connection_config.get("password") or None,
        key_path=connection_config.get("key_path") or None,
        key_content=connection_config.get("key_content") or None,
        key_passphrase=connection_config.get("key_passphrase") or None,
        timeout=timeout,
        total_timeout=timeout,
        shell="powershell",
        host_keys=connection_config.get("host_keys"),
        pairing_id=connection_config.get("pairing_id"),
        transport=connection_config.get("transport"),
        max_output_bytes=SOLIDWORKS_MAX_OUTPUT_BYTES,
    )


def _classify_failure(result: SSHResult) -> SolidworksTransportError:
    if RAGTIME_OUTPUT_LIMIT_MARKER in result.stderr:
        return SolidworksTransportError("no_response", OUTPUT_LIMIT_MESSAGE, output_limit=True)
    stderr = result.stderr.lower()
    if "timed out" in stderr or "timeout" in stderr:
        return SolidworksTransportError("timeout", "The SolidWorks agent request timed out.")
    if "ssh authentication failed" in stderr or "authenticationexception" in stderr:
        return SolidworksTransportError("ssh_auth_failed", "SSH authentication failed; check the configured credentials.")
    if result.exit_code == -1 and (stderr.startswith("ssh ") or not stderr):
        return SolidworksTransportError("ssh_unreachable", "SSH connection failed; check the SSH host and connection.")
    return SolidworksTransportError("no_response", "The SolidWorks agent did not return a valid response.")


class SshSolidworksTransport:
    """Transport constructed from a full tool configuration dictionary."""

    def __init__(self, config: dict):
        self.connection_config = config.get("connection_config") or {}

    async def run_powershell(self, script: str, *, timeout_seconds: float) -> SSHResult:
        config = build_ssh_config(self.connection_config, timeout=timeout_seconds)
        return await run_blocking(execute_ssh_command, config, script)

    async def call(self, request: dict, *, timeout_seconds: float) -> dict:
        encoded = encode_request(request)
        command = f"& (Join-Path $env:LOCALAPPDATA 'RagtimeSolidWorks\\host\\ragtime-solidworks-host.exe') call --request-b64 '{encoded}'"
        try:
            result = await self.run_powershell(command, timeout_seconds=timeout_seconds)
        except paramiko.AuthenticationException:
            raise SolidworksTransportError("ssh_auth_failed", "SSH authentication failed; check the configured credentials.") from None
        except TimeoutError:
            raise SolidworksTransportError("timeout", "The SolidWorks agent request timed out.") from None
        except (OSError, ValueError, paramiko.SSHException):
            raise SolidworksTransportError("ssh_unreachable", "SSH connection failed; check the SSH host and connection.") from None
        try:
            return parse_response(result.stdout)
        except SolidworksProtocolError:
            raise _classify_failure(result) from None

    async def upload_file(self, local_path: Path, remote_path: str, *, timeout_seconds: float) -> None:
        if "\\" in remote_path:
            raise ValueError("SFTP destination must use forward slashes")
        config = build_ssh_config(self.connection_config, timeout=timeout_seconds)

        def upload() -> None:
            started = time.monotonic()
            client = _create_ssh_client(config)
            timer = None
            try:
                if timeout_seconds > 0:
                    remaining = timeout_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("SFTP upload timed out")
                    timer = threading.Timer(remaining, client.close)
                    timer.daemon = True
                    timer.start()
                sftp = client.open_sftp()
                try:
                    if timeout_seconds > 0:
                        remaining = timeout_seconds - (time.monotonic() - started)
                        if remaining <= 0:
                            raise TimeoutError("SFTP upload timed out")
                        channel = sftp.get_channel()
                        if channel is None:
                            raise OSError("SFTP channel is not open")
                        channel.settimeout(remaining)
                    sftp.put(str(local_path), remote_path)
                finally:
                    sftp.close()
            finally:
                if timer is not None:
                    timer.cancel()
                client.close()

        work = run_blocking(upload)
        if timeout_seconds > 0:
            await asyncio.wait_for(work, timeout_seconds)
        else:
            await work
