"""Probe the Windows account and report SolidWorks host availability."""

import json
import logging
import re
import time
from collections.abc import Callable

import paramiko

from ragtime.core.tool_timeouts import resolve_effective_command_timeout
from ragtime.solidworks.host_artifact import get_host_artifact
from ragtime.solidworks.pdm_link import PdmLinkError, build_pdm_context, resolve_actor_label, resolve_pdm_link
from ragtime.solidworks.protocol import build_request
from ragtime.solidworks.transport import SolidworksTransport, SolidworksTransportError, SshSolidworksTransport, _classify_failure

logger = logging.getLogger(__name__)

PROBE_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$exe = Join-Path $env:LOCALAPPDATA 'RagtimeSolidWorks\host\ragtime-solidworks-host.exe'
$present = Test-Path -LiteralPath $exe -PathType Leaf
$version = $null
$features = @()
if ($present) {
    try {
        $output = & $exe version 2>$null
        foreach ($line in $output) {
            try { $v = $line | ConvertFrom-Json; if ($v.host_version) { $version = [string]$v.host_version }; $features = @($v.features | ForEach-Object { [string]$_ }) } catch {}
        }
    } catch {}
}
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
$admin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$interactive = $null
try {
    $username = ($identity.Name -split '\\')[-1]
    $sessions = @(quser.exe $username 2>&1)
    if ($LASTEXITCODE -eq 0 -and $sessions.Count -ge 2) {
        $interactive = $true
    } elseif (($sessions -join ' ') -match 'No User exists') {
        $interactive = $false
    }
} catch {}
$versions = $null
try {
    $versions = @(
        Get-ChildItem 'HKLM:\SOFTWARE\SolidWorks' -ErrorAction Stop |
        ForEach-Object {
            if ($_.PSChildName -match '^SOLIDWORKS (\d{4})$' -and [int]$Matches[1] -ge 2022) {
                [int]$Matches[1]
            }
        } | Sort-Object -Unique
    )
} catch {}
@{exe_present=$present; host_version=$version; host_features=$features; account=@{name=$identity.Name; is_administrator=$admin};
  interactive=$interactive; installed_versions=$versions} | ConvertTo-Json -Compress -Depth 4
exit 0
"""

DEFAULT_TOOL_TIMEOUT_SECONDS = 100
STATUS_TIMEOUT_CEILING_SECONDS = 60

_MESSAGES = {
    "ssh_unreachable": "SSH connection failed; check the host and connection.",
    "ssh_auth_failed": "SSH authentication failed; check the configured credentials.",
    "host_not_running": "The SolidWorks host is not running or did not respond.",
    "no_interactive_session": "The account has no interactive desktop session.",
    "internal_error": "The SolidWorks host status could not be checked.",
    "version_not_installed": "The configured SolidWorks version is not installed.",
}
PDM_VIEWS_WARNING = "The PDM vault view could not be checked."
_HOST_ERROR_CODES = {
    "invalid_request",
    "protocol_mismatch",
    "unknown_operation",
    "invalid_arguments",
    "write_not_allowed",
    "host_not_running",
    "host_busy",
    "host_stopping",
    "no_interactive_session",
    "version_not_installed",
    "version_mismatch",
    "solidworks_start_failed",
    "solidworks_not_running",
    "instance_limit",
    "instance_timeout",
    "operation_timeout",
    "document_not_found",
    "document_not_open",
    "not_supported_for_document_type",
    "entity_not_found",
    "stale_snapshot",
    "result_too_large",
    "solidworks_error",
    "internal_error",
}


def empty_host_status() -> dict:
    """Return the status shape with every field at its unknown default."""
    return {
        "reachable": False,
        "installed": False,
        "running": False,
        "instances_reported": False,
        "host_version": None,
        "host_features": [],
        "activity": None,
        "auto_update": None,
        "expected_host_version": None,
        "update_available": False,
        "account": None,
        "session": None,
        "solidworks": None,
        "pdm": None,
        "error": None,
        "warning": None,
    }


def _status_timeout_seconds(config: dict) -> int:
    """Use the tool's timeout, capped for the status probe and host call; never zero or negative."""
    configured = config.get("timeout_max_seconds")
    tool_maximum = configured if type(configured) is int and configured > 0 else DEFAULT_TOOL_TIMEOUT_SECONDS
    timeout = resolve_effective_command_timeout(None, tool_maximum)
    return min(timeout, STATUS_TIMEOUT_CEILING_SECONDS) if timeout > 0 else STATUS_TIMEOUT_CEILING_SECONDS


def _error(code: str) -> dict:
    if code in _HOST_ERROR_CODES and code not in _MESSAGES:
        return {"code": code, "message": "The SolidWorks host could not report its status."}
    if code not in _MESSAGES:
        code = "host_not_running"
    return {"code": code, "message": _MESSAGES[code]}


def _probe_object(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict) and type(value.get("exe_present")) is bool:
            return value
    raise ValueError("Invalid probe response")


def _versions(value: object) -> list[int] | None:
    if not isinstance(value, list) or any(type(year) is not int or year < 2022 for year in value):
        return None
    return value


def _features(value: object) -> list[str]:
    if not isinstance(value, list) or len(value) > 20 or any(not isinstance(item, str) or not re.fullmatch(r"[a-z0-9-]{1,40}", item) for item in value):
        return []
    return value


def _activity(value: object) -> dict | None:
    if not isinstance(value, dict):
        return None
    if any(type(value.get(key)) is not int or value[key] < 0 for key in ("in_flight", "idle_seconds")) or type(value.get("pdm_busy")) is not bool:
        return None
    return {key: value[key] for key in ("in_flight", "idle_seconds", "pdm_busy")}


def _pdm_views(value: object) -> list[dict] | None:
    if not isinstance(value, list):
        return None
    views = []
    for view in value:
        if (
            not isinstance(view, dict)
            or not isinstance(view.get("vault"), str)
            or not isinstance(view.get("root"), str)
            or type(view.get("logged_in")) is not bool
            or (view.get("user") is not None and not isinstance(view.get("user"), str))
        ):
            return None
        views.append({"vault": view["vault"], "root": view["root"], "logged_in": view["logged_in"], "user": view.get("user")})
    return views


def _pdm_block(value: object) -> dict | None:
    """Copy the host's client facts; anything malformed means the host did not report them."""
    if not isinstance(value, dict):
        return None
    client_version = value.get("client_version")
    views = _pdm_views(value.get("views"))
    if type(value.get("client_installed")) is not bool or (client_version is not None and not isinstance(client_version, str)) or views is None:
        return None
    return {"client_installed": value["client_installed"], "client_version": client_version, "views": views, "linked": None}


def _ok_result(response: object, operation: str) -> dict | None:
    if (
        not isinstance(response, dict)
        or response.get("operation") != operation
        or response.get("ok") is not True
        or not isinstance(response.get("result"), dict)
    ):
        return None
    return response["result"]


def _host_context(config: dict, timeout: int) -> dict:
    connection = config.get("connection_config") or {}
    return {
        "solidworks_version": connection.get("solidworks_version"),
        "instance_mode": connection.get("instance_mode", "background"),
        "max_instances": connection.get("max_instances", 1),
        "idle_shutdown_seconds": connection.get("idle_shutdown_seconds", 900),
        "allow_write": bool(config.get("allow_write", False)),
        "deadline_seconds": max(5, timeout - 10),
        "max_result_bytes": 32768,
    }


class SolidworksHostStatusService:
    """Report status without starting a SolidWorks instance."""

    def __init__(self, transport_factory: Callable[[dict], SolidworksTransport] = SshSolidworksTransport):
        self.transport_factory = transport_factory

    async def host_status(self, config: dict, *, probe_only: bool = False, include_pdm: bool = True) -> dict:
        """Report host status; probe_only answers from the probe alone and never calls the host."""
        started = time.monotonic()
        status = empty_host_status()
        try:
            artifact = get_host_artifact()
            status["expected_host_version"] = artifact.host_version if artifact else None
            timeout = _status_timeout_seconds(config)
            transport = self.transport_factory(config)
            result = await transport.run_powershell(PROBE_SCRIPT, timeout_seconds=timeout)
            try:
                probe = _probe_object(result.stdout)
            except ValueError:
                raise _classify_failure(result) from None
            status["reachable"] = True
            status["installed"] = probe["exe_present"]
            version = probe.get("host_version")
            status["host_version"] = version if isinstance(version, str) and version else None
            status["host_features"] = _features(probe.get("host_features"))
            status["update_available"] = bool(status["installed"] and status["host_version"] and artifact and version != artifact.host_version)
            account = probe.get("account")
            if isinstance(account, dict) and isinstance(account.get("name"), str) and type(account.get("is_administrator")) is bool:
                status["account"] = {"name": account["name"], "is_administrator": account["is_administrator"]}
            interactive = probe.get("interactive")
            status["session"] = {"interactive": interactive if type(interactive) is bool else None}
            versions = _versions(probe.get("installed_versions"))
            if versions is not None:
                status["solidworks"] = {"installed_versions": versions, "instances": []}
            if probe_only or not status["installed"]:
                return status
            context = _host_context(config, timeout)
            response = await transport.call(build_request("status", {}, context), timeout_seconds=timeout)
            host = _ok_result(response, "status")
            if host is None:
                host_error = response.get("error") if isinstance(response, dict) else None
                code = host_error.get("code") if isinstance(host_error, dict) else None
                if not isinstance(code, str) or code not in _HOST_ERROR_CODES:
                    code = "host_not_running"
                status["error"] = _error("no_interactive_session" if interactive is False else code)
                return status
            status["running"] = True
            status["activity"] = _activity(host.get("activity"))
            if isinstance(host.get("host_version"), str) and host["host_version"]:
                status["host_version"] = host["host_version"]
                status["update_available"] = bool(artifact and host["host_version"] != artifact.host_version)
            session = host.get("session")
            if isinstance(session, dict) and type(session.get("interactive")) is bool:
                status["session"] = {"interactive": session["interactive"]}
            solidworks = host.get("solidworks")
            if isinstance(solidworks, dict):
                host_versions = _versions(solidworks.get("installed_versions"))
                instances = solidworks.get("instances")
                if host_versions is not None and isinstance(instances, list) and all(isinstance(item, dict) for item in instances):
                    status["solidworks"] = {"installed_versions": host_versions, "instances": instances}
                    status["instances_reported"] = True
            status["pdm"] = _pdm_block(host.get("pdm"))
            if include_pdm and status["pdm"] is not None:
                await self._add_pdm_views(config, transport, context, status, timeout - (time.monotonic() - started))
        except paramiko.AuthenticationException:
            status["error"] = _error("ssh_auth_failed")
        except SolidworksTransportError as exc:
            code = exc.kind if exc.kind in ("ssh_auth_failed", "ssh_unreachable") else "host_not_running"
            if status["installed"] and status["session"] and status["session"]["interactive"] is False:
                code = "no_interactive_session"
            status["error"] = _error(code)
        except (OSError, paramiko.SSHException, TimeoutError):
            status["error"] = _error("host_not_running" if status["reachable"] else "ssh_unreachable")
        except Exception:
            status["error"] = _error("internal_error")
        finally:
            try:
                from ragtime.solidworks.host_updater import host_updater  # inline-import: keep

                if status["installed"] and status["host_version"]:
                    host_updater.observe(config, status["host_version"])
                status["auto_update"] = host_updater.state_for(config.get("id") or "")
            except Exception as exc:
                logger.warning("Host updater status hook failed: %s", type(exc).__name__)
        return status

    async def _add_pdm_views(self, config: dict, transport: SolidworksTransport, context: dict, status: dict, remaining: float) -> None:
        """Replace the host's view facts with the linked vault's live state; a failure only adds a warning."""
        pdm = status["pdm"]
        try:
            link = await resolve_pdm_link(config)
        except PdmLinkError:
            pdm["linked"] = None
            return
        except Exception as exc:
            # Logged by type only: the message or config may carry connection details.
            logger.warning("PDM link could not be resolved for the SolidWorks host status: %s", type(exc).__name__)
            status["warning"] = status["warning"] or PDM_VIEWS_WARNING
            return
        views = None
        if remaining >= 1:
            try:
                actor = await resolve_actor_label()
                pdm_context = build_pdm_context(config, link, actor)
                response = await transport.call(
                    build_request("pdm.status", {}, {**context, "pdm": pdm_context}),
                    timeout_seconds=int(remaining),
                )
                result = _ok_result(response, "pdm.status")
                views = _pdm_views(result.get("views")) if result else None
            except Exception:
                views = None
        if views is None:
            status["warning"] = status["warning"] or PDM_VIEWS_WARNING
            return
        pdm["views"] = views
        linked = result.get("linked") if result else None
        pdm["linked"] = linked if isinstance(linked, dict) else None

    async def test_connection(self, config: dict) -> tuple[bool, str, dict | None]:
        status = await self.host_status(config)
        account = status["account"]
        session = status["session"]
        solidworks = status["solidworks"]
        details = {
            "host_version": status["host_version"],
            "account": account["name"] if account else None,
            "solidworks_versions": solidworks["installed_versions"] if solidworks else None,
            "session_interactive": session["interactive"] if session else None,
        }
        pdm = status["pdm"]
        linked = pdm["linked"] if pdm else None
        if linked and linked.get("logged_in") is True:
            details.update(
                pdm_client_version=pdm["client_version"],
                pdm_vault=linked.get("vault"),
                pdm_vault_root=linked.get("root"),
                pdm_user=linked.get("user"),
            )
        error = status["error"]
        if not status["reachable"] or not status["running"]:
            error = error or _error("host_not_running")
        else:
            connection = config.get("connection_config") or {}
            pinned = connection.get("solidworks_version")
            if (
                pinned is not None
                and connection.get("instance_mode", "background") == "background"
                and solidworks is not None
                and pinned not in solidworks["installed_versions"]
            ):
                error = _error("version_not_installed")
        if error:
            return False, error["message"], {"error_code": error["code"], **details}
        return True, f"SolidWorks host {details['host_version']} is running as {details['account']}.", details
