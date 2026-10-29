"""Bounded, authenticated removal of paired Windows remote clients."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from dataclasses import dataclass, replace
from importlib.resources import files
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from ragtime.core.ssh import SSHConfig, execute_ssh_command, ssh_config_from_dict

CleanupMode = Literal["disable_ssh", "pairing_only"]
StepStatus = Literal["confirmed", "failed", "unconfirmed", "not_requested"]
PAIRING_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
REMOVAL_RESULT_PREFIX = "RAGTIME_REMOVAL_RESULT:"
TOTAL_CLEANUP_TIMEOUT_SECONDS = 30
_VALID_STATUSES = {"confirmed", "failed", "unconfirmed", "not_requested"}


@dataclass(frozen=True)
class CleanupResult:
    mode: CleanupMode
    steps: dict[str, StepStatus]
    warnings: list[str]
    local_removal_command: str
    local_removal_instructions: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "mode": self.mode,
            "steps": self.steps,
            "warnings": self.warnings,
            "local_removal_command": self.local_removal_command,
        }
        if self.local_removal_instructions:
            result["local_removal_instructions"] = self.local_removal_instructions
        return result


def validate_cleanup_request(mode: str | None, confirm_machine_wide: bool | None) -> CleanupMode | None:
    if mode is None:
        if confirm_machine_wide is not None:
            raise ValueError("confirm_machine_wide is only valid with ssh_cleanup=disable_ssh")
        return None
    if mode not in {"disable_ssh", "pairing_only"}:
        raise ValueError("ssh_cleanup must be disable_ssh or pairing_only")
    if mode == "disable_ssh" and confirm_machine_wide is not True:
        raise ValueError("confirm_machine_wide=true is required with ssh_cleanup=disable_ssh")
    if mode == "pairing_only" and confirm_machine_wide is not None:
        raise ValueError("confirm_machine_wide is only valid with ssh_cleanup=disable_ssh")
    return cast(CleanupMode, mode)


def is_windows_pairing_tool(tool_type: Any, connection_config: dict[str, Any] | None) -> bool:
    type_value = str(getattr(tool_type, "value", tool_type))
    return type_value == "windows_shell" and is_paired_tool(connection_config)


def is_paired_tool(connection_config: dict[str, Any] | None) -> bool:
    pairing_id = (connection_config or {}).get("pairing_id")
    return isinstance(pairing_id, str) and PAIRING_ID_PATTERN.fullmatch(pairing_id) is not None


def local_removal_command(server_url: str, pairing_id: str, mode: CleanupMode, target_user: str) -> str:
    if PAIRING_ID_PATTERN.fullmatch(pairing_id) is None:
        raise ValueError("Invalid pairing id")
    origin = urlsplit(server_url)
    if origin.scheme != "https" or not origin.netloc:
        return ""
    script_url = f"{server_url.rstrip('/')}/indexes/remote/install.ps1"
    quoted_script_url = script_url.replace("'", "''")
    invocation = _local_invocation(pairing_id, mode, target_user)
    return f"$p=Join-Path $env:TEMP ragtime-install.ps1; iwr -UseBasicParsing '{quoted_script_url}' -OutFile $p; {invocation}"


def _local_invocation(pairing_id: str, mode: CleanupMode, target_user: str, *, transferred: bool = False) -> str:
    arguments = base64.b64encode(json.dumps({"pairing_id": pairing_id, "target_user": target_user}, separators=(",", ":")).encode("utf-8")).decode("ascii")
    script_path = "$p=Join-Path (Get-Location) install.ps1; " if transferred else "$p=Join-Path $env:TEMP ragtime-install.ps1; "
    command = script_path + (
        f"$a=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{arguments}'))|ConvertFrom-Json; "
        "& $p -Uninstall $a.pairing_id -TargetUser $a.target_user"
    )
    if mode == "disable_ssh":
        command += " -CleanupMode DisableSsh -ConfirmMachineWide"
    else:
        command += " -CleanupMode PairingOnly"
    encoded_command = base64.b64encode(command.encode("utf-16le")).decode("ascii")
    return f"powershell.exe -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded_command}"


def _unconfirmed_result(mode: CleanupMode, command: str, warning: str, instructions: str | None = None) -> CleanupResult:
    return CleanupResult(
        mode=mode,
        steps={"pairing_cleanup": "unconfirmed", "ssh_service": "unconfirmed" if mode == "disable_ssh" else "not_requested"},
        warnings=[warning],
        local_removal_command=command,
        local_removal_instructions=instructions,
    )


def _parse_removal_result(stdout: str, mode: CleanupMode, command: str) -> CleanupResult | None:
    lines = [line for line in stdout.splitlines() if line.startswith(REMOVAL_RESULT_PREFIX)]
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1][len(REMOVAL_RESULT_PREFIX) :])
        steps = payload["steps"]
        pairing_cleanup = steps["pairing_cleanup"]
        ssh_service = steps["ssh_service"]
        warnings = payload.get("warnings", [])
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    if not isinstance(pairing_cleanup, str) or not isinstance(ssh_service, str):
        return None
    if (
        pairing_cleanup not in _VALID_STATUSES
        or ssh_service not in _VALID_STATUSES
        or not isinstance(warnings, list)
        or not all(isinstance(warning, str) for warning in warnings)
    ):
        return None
    return CleanupResult(
        mode=mode,
        steps={"pairing_cleanup": cast(StepStatus, pairing_cleanup), "ssh_service": cast(StepStatus, ssh_service)},
        warnings=warnings,
        local_removal_command=command,
    )


def _remote_command(arguments: dict[str, str | bool]) -> str:
    encoded_arguments = base64.b64encode(json.dumps(arguments, separators=(",", ":")).encode("utf-8")).decode("ascii")
    return (
        "$raw=[Console]::In.ReadToEnd();"
        "$path=Join-Path $env:TEMP ('ragtime-removal-'+[guid]::NewGuid().ToString()+'.ps1');"
        "[IO.File]::WriteAllText($path,$raw,(New-Object Text.UTF8Encoding($false)));"
        f"$RAGTIME_REMOVAL_ARGS='{encoded_arguments}';"
        "try {$argsJson=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($RAGTIME_REMOVAL_ARGS))|ConvertFrom-Json;"
        "& $path -Uninstall $argsJson.pairing_id -TargetUser $argsJson.target_user -CleanupMode $argsJson.cleanup_mode "
        "-ConfirmMachineWide:([bool]$argsJson.confirm_machine_wide)} finally {Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue}"
    )


async def execute_windows_cleanup(
    connection_config: dict[str, Any], pairing_id: str, mode: CleanupMode, confirm_machine_wide: bool, *, server_url: str = ""
) -> CleanupResult:
    """Stage the packaged script over the existing pinned SSH connection once."""
    if PAIRING_ID_PATTERN.fullmatch(pairing_id) is None:
        raise ValueError("Invalid pairing id")
    started = time.monotonic()
    target_user = ssh_config_from_dict({"user": connection_config.get("user"), "ssh_user": connection_config.get("ssh_user")}).user
    command = local_removal_command(server_url, pairing_id, mode, target_user)
    instructions = (
        None
        if command
        else (
            "Transfer the maintained install.ps1 through a trusted channel to the Windows host. "
            "In an Administrator PowerShell in its directory, run:\n" + _local_invocation(pairing_id, mode, target_user, transferred=True)
        )
    )
    try:
        ssh_config: SSHConfig = ssh_config_from_dict(connection_config)
        ssh_config.shell = "powershell"
        ssh_config.timeout = min(float(ssh_config.timeout), TOTAL_CLEANUP_TIMEOUT_SECONDS)
        arguments = {
            "pairing_id": pairing_id,
            "cleanup_mode": "DisableSsh" if mode == "disable_ssh" else "PairingOnly",
            "confirm_machine_wide": confirm_machine_wide,
            "target_user": ssh_config.user,
        }
        script = files("ragtime.remote_client.scripts").joinpath("install.ps1").read_text(encoding="utf-8")
        remaining = TOTAL_CLEANUP_TIMEOUT_SECONDS - (time.monotonic() - started)
        if remaining <= 0:
            return _unconfirmed_result(mode, command, "Remote cleanup timed out before connection", instructions)
        ssh_config.timeout = min(ssh_config.timeout, remaining)
        ssh_config.total_timeout = remaining
        result = await asyncio.wait_for(asyncio.to_thread(execute_ssh_command, ssh_config, _remote_command(arguments), script), timeout=remaining)
    except asyncio.TimeoutError:
        return _unconfirmed_result(mode, command, "Remote cleanup timed out before confirmation", instructions)
    except Exception as exc:
        return _unconfirmed_result(mode, command, f"Remote cleanup could not start: {exc}", instructions)
    parsed = _parse_removal_result(result.stdout, mode, command)
    if parsed is not None:
        return replace(parsed, local_removal_instructions=instructions)
    warning = result.stderr.strip() or "Remote cleanup ended without a confirmed removal result"
    return _unconfirmed_result(mode, command, warning, instructions)
