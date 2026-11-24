"""Install and remove the SolidWorks host over SSH."""

import json
import re
import unicodedata
from collections.abc import Callable

import paramiko

from ragtime.solidworks.host_artifact import get_host_artifact, verify_host_artifact
from ragtime.solidworks.host_status import empty_host_status
from ragtime.solidworks.transport import SolidworksTransport, SolidworksTransportError, SshSolidworksTransport, _classify_failure

RESOLVE_TIMEOUT_SECONDS = 30
STAGING_TIMEOUT_SECONDS = 30
UPLOAD_TIMEOUT_SECONDS = 60
INSTALL_TIMEOUT_SECONDS = 90
GATED_INSTALL_TIMEOUT_SECONDS = 240
MAX_IDLE_GATE_SECONDS = 86400
UNINSTALL_TIMEOUT_SECONDS = 90

RESOLVE_SCRIPT = "$env:LOCALAPPDATA | ConvertTo-Json -Compress; exit 0"
STAGING_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
try {
  New-Item -ItemType Directory -Force -Path (Join-Path $env:LOCALAPPDATA 'RagtimeSolidWorks\staging') | Out-Null
  Write-Output 'RAGTIME_STAGING:ok'
  exit 0
} catch {
  Write-Output 'RAGTIME_STAGING:failed'
  exit 1
}
"""

INSTALL_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$exitWaitSeconds = 20
$pollMs = 250
$retryTries = 20
$retryDelayMs = 500
$idleGate = '__IDLE_GATE__'
$root = Join-Path $env:LOCALAPPDATA 'RagtimeSolidWorks'
$hostDir = Join-Path $root 'host'
$new = Join-Path $root 'host.new'
$old = Join-Path $root 'host.old'
$failed = Join-Path $root 'host.failed'
$zip = Join-Path (Join-Path $root 'staging') 'host-__PREFIX__.zip'
$exe = Join-Path $hostDir 'ragtime-solidworks-host.exe'
$oldExe = Join-Path $old 'ragtime-solidworks-host.exe'
function Restore-Current {
  if (Test-Path -LiteralPath $exe -PathType Leaf) {
    try { & $exe install | Out-Null } catch {}
  }
}
try {
  if (-not (Test-Path -LiteralPath $exe -PathType Leaf) -and (Test-Path -LiteralPath $oldExe -PathType Leaf)) {
    Remove-Item -LiteralPath $failed -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $hostDir) { Rename-Item -LiteralPath $hostDir -NewName 'host.failed' }
    Rename-Item -LiteralPath $old -NewName 'host'
  }
  if ((Test-Path -LiteralPath $exe -PathType Leaf) -and (Test-Path -LiteralPath $old)) {
    Remove-Item -LiteralPath $old -Recurse -Force -ErrorAction SilentlyContinue
  }
} catch {
  Write-Output 'RAGTIME_INSTALL:locked'
  exit 1
}
$hadOld = Test-Path -LiteralPath $exe -PathType Leaf
try {
  if ((Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant() -ne '__SHA__') {
    Write-Output 'RAGTIME_INSTALL:hash_mismatch'
    exit 1
  }
  if (Test-Path -LiteralPath $new) { Remove-Item -LiteralPath $new -Recurse -Force }
  Expand-Archive -LiteralPath $zip -DestinationPath $new -Force
  if (-not (Test-Path -LiteralPath (Join-Path $new 'ragtime-solidworks-host.exe') -PathType Leaf)) { throw 'staged host has no exe' }
} catch {
  Remove-Item -LiteralPath $new -Recurse -Force -ErrorAction SilentlyContinue
  Write-Output 'RAGTIME_INSTALL:install_failed'
  exit 1
}
try {
  if ($hadOld) {
    if ($idleGate -ne '') {
      & $exe stop --if-idle ([int]$idleGate) | Out-Null
      if ($LASTEXITCODE -eq 3) {
        Remove-Item -LiteralPath $new -Recurse -Force -ErrorAction SilentlyContinue
        Write-Output 'RAGTIME_INSTALL:busy'
        exit 1
      }
      if ($LASTEXITCODE -ne 0) { throw 'idle stop failed' }
    }
    & $exe uninstall | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'uninstall failed' }
    $end = (Get-Date).AddSeconds($exitWaitSeconds)
    do { if (-not (Get-Process -Name 'ragtime-solidworks-host' -ErrorAction SilentlyContinue | Where-Object { try { $_.Path -and $_.Path.StartsWith("$hostDir\", [StringComparison]::OrdinalIgnoreCase) } catch { $false } })) { break }; Start-Sleep -Milliseconds $pollMs } while ((Get-Date) -lt $end)
  }
} catch {
  Remove-Item -LiteralPath $new -Recurse -Force -ErrorAction SilentlyContinue
  Write-Output 'RAGTIME_INSTALL:install_failed'
  exit 1
}
$hadDir = Test-Path -LiteralPath $hostDir
try {
  if ($hadDir) { for ($i = 1; ; $i++) { try { Rename-Item -LiteralPath $hostDir -NewName 'host.old'; break } catch { if ($i -ge $retryTries) { throw }; Start-Sleep -Milliseconds $retryDelayMs } } }
} catch {
  Remove-Item -LiteralPath $new -Recurse -Force -ErrorAction SilentlyContinue
  Restore-Current
  Write-Output 'RAGTIME_INSTALL:locked'
  exit 1
}
try {
  Rename-Item -LiteralPath $new -NewName 'host'
} catch {
  if ($hadDir) { Rename-Item -LiteralPath $old -NewName 'host' -ErrorAction SilentlyContinue }
  Remove-Item -LiteralPath $new -Recurse -Force -ErrorAction SilentlyContinue
  Restore-Current
  Write-Output 'RAGTIME_INSTALL:install_failed'
  exit 1
}
try {
  $output = & $exe install
  $text = $output | Out-String
  if ($LASTEXITCODE -ne 0 -or $text -notmatch '"ok"\s*:\s*true') { throw 'new host did not install' }
  $status = @($output)[-1]
} catch {
  try { & $exe uninstall | Out-Null } catch {}
  Remove-Item -LiteralPath $failed -Recurse -Force -ErrorAction SilentlyContinue
  try {
    Rename-Item -LiteralPath $hostDir -NewName 'host.failed'
  } catch {
    Write-Output 'RAGTIME_INSTALL:install_failed'
    exit 1
  }
  $restored = $false
  if ($hadDir) {
    try {
      Rename-Item -LiteralPath $old -NewName 'host'
      if (Test-Path -LiteralPath $exe -PathType Leaf) { & $exe install | Out-Null; $restored = ($LASTEXITCODE -eq 0) }
    } catch {}
  }
  Remove-Item -LiteralPath $failed -Recurse -Force -ErrorAction SilentlyContinue
  if ($restored) { Write-Output 'RAGTIME_INSTALL:rolled_back' } else { Write-Output 'RAGTIME_INSTALL:install_failed' }
  exit 1
}
Write-Output $status
Remove-Item -LiteralPath $old -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
exit 0
"""

UNINSTALL_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$exitWaitSeconds = 20
$pollMs = 250
$retryTries = 20
$retryDelayMs = 500
$hostDir = Join-Path (Join-Path $env:LOCALAPPDATA 'RagtimeSolidWorks') 'host'
$exe = Join-Path $hostDir 'ragtime-solidworks-host.exe'
try {
  if (Test-Path -LiteralPath $exe) {
    & $exe uninstall | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Output 'RAGTIME_UNINSTALL:uninstall_failed'; exit 1 }
    $end = (Get-Date).AddSeconds($exitWaitSeconds)
    do { if (-not (Get-Process -Name 'ragtime-solidworks-host' -ErrorAction SilentlyContinue | Where-Object { try { $_.Path -and $_.Path.StartsWith("$hostDir\", [StringComparison]::OrdinalIgnoreCase) } catch { $false } })) { break }; Start-Sleep -Milliseconds $pollMs } while ((Get-Date) -lt $end)
  }
  if (Test-Path -LiteralPath $hostDir) {
    for ($i = 1; ; $i++) {
      try { Remove-Item -LiteralPath $hostDir -Recurse -Force -ErrorAction Stop; break } catch { if ($i -ge $retryTries) { break }; Start-Sleep -Milliseconds $retryDelayMs }
    }
    if (Test-Path -LiteralPath $hostDir) {
      Write-Output 'RAGTIME_UNINSTALL:locked'
      exit 0
    }
  }
  Write-Output 'RAGTIME_UNINSTALL:ok'
  exit 0
} catch {
  Write-Output 'RAGTIME_UNINSTALL:uninstall_failed'
  exit 1
}
"""

LOCKED_UNINSTALL_WARNING = "The SolidWorks agent was stopped and its logon task removed, but some files could not be deleted."

MESSAGES = {
    "ssh_unreachable": "SSH connection failed; check the SSH host and connection.",
    "ssh_auth_failed": "SSH authentication failed; check the configured credentials.",
    "upload_failed": "The SolidWorks agent upload failed.",
    "artifact_mismatch": "The packaged SolidWorks agent zip does not match its manifest; rebuild the agent package.",
    "hash_mismatch": "The uploaded SolidWorks agent failed its integrity check.",
    "install_failed": "The SolidWorks agent installation failed.",
    "install_rolled_back": "The SolidWorks agent installation failed; the previous version was restored.",
    "locked": "The SolidWorks agent files are locked; close the SolidWorks agent and try again.",
    "uninstall_failed": "The SolidWorks agent could not be stopped or uninstalled.",
    "host_busy": "The SolidWorks agent is in use. The update will run when it is idle.",
    "host_busy_retry": "The SolidWorks agent is busy with another request or update. Try again in a minute.",
}
_ERROR_CODES = {"artifact_mismatch": "install_failed", "install_rolled_back": "install_failed", "locked": "install_failed", "host_busy_retry": "host_busy"}
_INSTALL_SENTINELS = {
    "busy": "host_busy",
    "hash_mismatch": "hash_mismatch",
    "locked": "locked",
    "install_failed": "install_failed",
    "rolled_back": "install_rolled_back",
}


class HostArtifactUnavailableError(Exception):
    """The packaged host artifact is unavailable."""


def _remote_path(value: object, prefix: str) -> str:
    if not isinstance(value, str) or not re.match(r"^[A-Za-z]:[\\/]", value):
        raise ValueError("Invalid LOCALAPPDATA")
    if any(unicodedata.category(char) == "Cc" for char in value):
        raise ValueError("Invalid LOCALAPPDATA")
    parts = value.replace("\\", "/").split("/")
    if any(part in (".", "..") for part in parts) or ":" in value[2:]:
        raise ValueError("Invalid LOCALAPPDATA")
    return value.replace("\\", "/").rstrip("/") + f"/RagtimeSolidWorks/staging/host-{prefix}.zip"


def _transport_code(exc: Exception, fallback: str) -> str:
    if isinstance(exc, SolidworksTransportError):
        return exc.kind if exc.kind in ("ssh_auth_failed", "ssh_unreachable") else fallback
    if isinstance(exc, paramiko.AuthenticationException):
        return "ssh_auth_failed"
    if isinstance(exc, (paramiko.SSHException, OSError)):
        return "ssh_unreachable"
    return fallback


def _sentinel(stdout: str, prefix: str) -> str | None:
    """Return the text after the last line starting with prefix, or None when no such line exists."""
    values = [line.strip()[len(prefix) :] for line in stdout.splitlines() if line.strip().startswith(prefix)]
    return values[-1] if values else None


def _status_envelope(stdout: str) -> dict | None:
    """Return the last stdout line that is a successful status response envelope."""
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and value.get("operation") == "status" and value.get("ok") is True:
            return value
    return None


def _resolved_local_app_data(result) -> object:
    try:
        return json.loads(result.stdout.strip())
    except ValueError:
        raise _classify_failure(result) from None


class SolidworksHostInstaller:
    def __init__(self, transport_factory: Callable[[dict], SolidworksTransport] = SshSolidworksTransport, status_service=None):
        if status_service is None:
            from ragtime.solidworks.host_status import SolidworksHostStatusService

            status_service = SolidworksHostStatusService(transport_factory=transport_factory)
        self.transport_factory = transport_factory
        self.status_service = status_service

    async def _status(self, config: dict, code: str | None = None, *, warning: str | None = None, probe_only: bool = False) -> dict:
        try:
            status = await self.status_service.host_status(config, probe_only=probe_only)
        except Exception:
            status = empty_host_status()
        if code:
            status["error"] = {"code": _ERROR_CODES.get(code, code), "message": MESSAGES[code]}
        status["warning"] = warning
        return status

    async def busy_status(self, config: dict) -> dict:
        return await self._status(config, "host_busy_retry")

    async def install_host(self, config: dict, *, idle_gate_seconds: int | None = None) -> dict:
        if idle_gate_seconds is not None and (type(idle_gate_seconds) is not int or not 0 <= idle_gate_seconds <= MAX_IDLE_GATE_SECONDS):
            raise ValueError("idle_gate_seconds must be an integer between 0 and 86400")
        artifact = get_host_artifact()
        if artifact is None:
            raise HostArtifactUnavailableError("SolidWorks agent package is unavailable.")
        if not isinstance(artifact.sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact.sha256):
            return await self._status(config, "install_failed")
        if not verify_host_artifact(artifact):
            return await self._status(config, "artifact_mismatch")
        transport = self.transport_factory(config)
        try:
            result = await transport.run_powershell(RESOLVE_SCRIPT, timeout_seconds=RESOLVE_TIMEOUT_SECONDS)
            remote = _remote_path(_resolved_local_app_data(result), artifact.sha256[:16])
            result = await transport.run_powershell(STAGING_SCRIPT, timeout_seconds=STAGING_TIMEOUT_SECONDS)
            if _sentinel(result.stdout, "RAGTIME_STAGING:") != "ok":
                raise _classify_failure(result)
        except Exception as exc:
            return await self._status(config, _transport_code(exc, "install_failed"))
        try:
            await transport.upload_file(artifact.path, remote, timeout_seconds=UPLOAD_TIMEOUT_SECONDS)
        except Exception:
            return await self._status(config, "upload_failed")
        script = (
            INSTALL_SCRIPT.replace("__SHA__", artifact.sha256)
            .replace("__PREFIX__", artifact.sha256[:16])
            .replace("__IDLE_GATE__", str(idle_gate_seconds) if idle_gate_seconds is not None else "")
        )
        timeout = GATED_INSTALL_TIMEOUT_SECONDS if idle_gate_seconds is not None else INSTALL_TIMEOUT_SECONDS
        try:
            result = await transport.run_powershell(script, timeout_seconds=timeout)
        except Exception as exc:
            return await self._status(config, _transport_code(exc, "install_failed"))
        sentinel = _sentinel(result.stdout, "RAGTIME_INSTALL:")
        if sentinel is not None:
            return await self._status(config, _INSTALL_SENTINELS.get(sentinel, "install_failed"))
        if _status_envelope(result.stdout) is not None:
            return await self._status(config)
        if not result.stdout.strip():
            return await self._status(config, _transport_code(_classify_failure(result), "install_failed"))
        return await self._status(config, "install_failed")

    async def uninstall_host(self, config: dict) -> dict:
        try:
            result = await self.transport_factory(config).run_powershell(UNINSTALL_SCRIPT, timeout_seconds=UNINSTALL_TIMEOUT_SECONDS)
        except Exception as exc:
            return await self._status(config, _transport_code(exc, "uninstall_failed"))
        sentinel = _sentinel(result.stdout, "RAGTIME_UNINSTALL:")
        if sentinel == "ok":
            return await self._status(config)
        if sentinel == "locked":
            return await self._status(config, warning=LOCKED_UNINSTALL_WARNING, probe_only=True)
        if not result.stdout.strip():
            return await self._status(config, _transport_code(_classify_failure(result), "uninstall_failed"))
        return await self._status(config, "uninstall_failed")
