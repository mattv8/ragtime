import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "ragtime" / "remote_client" / "scripts"


PAIRING_ID = "123e4567-e89b-12d3-a456-426614174000"
OTHER_PAIRING_ID = "123e4567-e89b-12d3-a456-426614174001"


def _installer_functions() -> str:
    installer = (SCRIPTS / "install.ps1").read_text()
    functions, marker, _ = installer.partition("# RAGTIME_INSTALLER_ENTRYPOINT")
    assert marker, "install.ps1 must delimit its function library from its entrypoint"
    return functions


def _pwsh() -> Path:
    configured_pwsh = os.environ.get("RAGTIME_TEST_PWSH") or shutil.which("pwsh")
    if not configured_pwsh:
        pytest.skip("PowerShell runtime integration test requires pwsh or RAGTIME_TEST_PWSH")
    pwsh = Path(configured_pwsh)
    if not pwsh.is_file():
        pytest.skip("RAGTIME_TEST_PWSH does not point to an executable")
    return pwsh


def test_windows_removal_rejects_non_uuid_before_resolving_an_account():
    pwsh = _pwsh()
    result = subprocess.run(
        [
            str(pwsh),
            "-NoProfile",
            "-File",
            str(SCRIPTS / "install.ps1"),
            "-Uninstall",
            "not-a-pairing-id",
            "-CleanupMode",
            "PairingOnly",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "valid UUID" in result.stderr
    assert "Cannot resolve target account" not in result.stderr


def test_windows_removal_only_removes_an_exact_pairing_comment_and_preserves_other_keys(tmp_path):
    pwsh = _pwsh()
    target = tmp_path / "authorized_keys"
    target.write_text(
        "ssh-ed25519 OTHER unrelated\n"
        f"ssh-ed25519 REMOVE # ragtime-remote:{PAIRING_ID}\n"
        f"ssh-ed25519 REMOVE-WHITESPACE # ragtime-remote:{PAIRING_ID}   \n"
        f"ssh-ed25519 KEEP # ragtime-remote:{OTHER_PAIRING_ID}\n"
        f"ssh-ed25519 KEEP suffix # ragtime-remote:{PAIRING_ID} extra\n",
        encoding="utf-8",
    )
    harness = tmp_path / "remove-owned-key.ps1"
    harness.write_text(
        _installer_functions() + f"\nRemove-PairingKeyLines '{target}' '{PAIRING_ID}'\n",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)
    assert target.read_text(encoding="utf-8") == (
        f"ssh-ed25519 OTHER unrelated\nssh-ed25519 KEEP # ragtime-remote:{OTHER_PAIRING_ID}\nssh-ed25519 KEEP suffix # ragtime-remote:{PAIRING_ID} extra\n"
    )


def test_windows_removal_result_is_compact_json_with_only_contract_statuses(tmp_path):
    pwsh = _pwsh()
    harness = tmp_path / "result.ps1"
    harness.write_text(
        _installer_functions() + "\nWrite-RemovalResult ([ordered]@{ pairing_cleanup = 'confirmed'; ssh_service = 'not_requested' }) @('kept')\n",
        encoding="utf-8",
    )
    result = subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)
    assert result.stdout.splitlines()[-1] == (
        'RAGTIME_REMOVAL_RESULT:{"steps":{"pairing_cleanup":"confirmed","ssh_service":"not_requested"},"warnings":["kept"]}'
    )


def test_windows_removal_disables_sshd_before_stopping_it(tmp_path):
    pwsh = _pwsh()
    harness = tmp_path / "service-order.ps1"
    harness.write_text(
        _installer_functions()
        + """
$trace = @()
function Test-Elevated { $true }
function Get-Service { [pscustomobject]@{ Status = 'Running' } }
function Set-Service { $script:trace += 'disable' }
function Stop-Service { $script:trace += 'stop' }
$steps = New-RemovalSteps 'DisableSsh'
$warnings = New-Object System.Collections.ArrayList
Invoke-SshServiceCleanup $steps $warnings
if (($trace -join ',') -ne 'disable,stop') { throw "unexpected service order: $($trace -join ',')" }
if ($steps.ssh_service -ne 'confirmed') { throw "unexpected service status: $($steps.ssh_service)" }
""",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)


def test_windows_removal_reports_service_privilege_failure_without_calling_service_cmdlets(tmp_path):
    pwsh = _pwsh()
    harness = tmp_path / "service-privilege.ps1"
    harness.write_text(
        _installer_functions()
        + """
function Test-Elevated { $false }
function Set-Service { throw 'Set-Service must not be called' }
function Stop-Service { throw 'Stop-Service must not be called' }
$steps = New-RemovalSteps 'DisableSsh'
$warnings = New-Object System.Collections.ArrayList
Invoke-SshServiceCleanup $steps $warnings
if ($steps.ssh_service -ne 'failed') { throw "unexpected service status: $($steps.ssh_service)" }
if ($warnings.Count -ne 1 -or $warnings[0] -notmatch 'Administrator privileges') { throw 'missing privilege warning' }
""",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)


def test_windows_removal_missing_resources_are_idempotent(tmp_path):
    pwsh = _pwsh()
    profile = tmp_path / "profile"
    program_data = tmp_path / "program-data"
    harness = tmp_path / "missing-resources.ps1"
    harness.write_text(
        _installer_functions()
        + f"""
$env:ProgramData = '{program_data}'
function Test-Elevated {{ $true }}
function Get-ScheduledTask {{ Write-Error -Category ObjectNotFound -ErrorAction Stop 'missing task' }}
$identity = [pscustomobject]@{{ Name = 'user'; Sid = 'S-1-5-21-1000'; Profile = '{profile}' }}
$steps = New-RemovalSteps 'PairingOnly'
$warnings = New-Object System.Collections.ArrayList
Invoke-PairingCleanup '{PAIRING_ID}' $identity $steps $warnings
if ($steps.pairing_cleanup -ne 'confirmed') {{ throw "missing resources were not idempotent: $($steps.pairing_cleanup)" }}
if ($warnings.Count -ne 0) {{ throw 'missing resources emitted warnings' }}
""",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)


def test_windows_removal_finalizer_uses_distinct_task_and_reports_unconfirmed(tmp_path):
    pwsh = _pwsh()
    profile = tmp_path / "profile"
    program_data = tmp_path / "program-data"
    harness = tmp_path / "finalizer.ps1"
    harness.write_text(
        _installer_functions()
        + f"""
$env:ProgramData = '{program_data}'
$trace = @()
function Test-Elevated {{ $true }}
function Get-SidProfile {{ param($Sid) '{profile}' }}
function Initialize-SystemDirectory {{ param($Path) New-Item -ItemType Directory -Force -Path $Path | Out-Null }}
function Set-SystemOnlyAcl {{ param($Path) }}
function New-ScheduledTaskAction {{ [pscustomobject]@{{}} }}
function New-ScheduledTaskTrigger {{ [pscustomobject]@{{}} }}
function New-ScheduledTaskSettingsSet {{
    param($ExecutionTimeLimit, [switch]$AllowStartIfOnBatteries, [switch]$DontStopIfGoingOnBatteries, [switch]$StartWhenAvailable)
    if (-not $AllowStartIfOnBatteries -or -not $DontStopIfGoingOnBatteries) {{ throw 'finalizer battery settings were omitted' }}
    [pscustomobject]@{{}}
}}
function Register-ScheduledTask {{ param($TaskName) $script:trace += "register:$TaskName" }}
function Start-ScheduledTask {{ throw 'finalizer must wait for its one-shot trigger' }}
$identity = [pscustomobject]@{{ Name = 'DOMAIN\\user'; Sid = 'S-1-5-21-1000'; Profile = '{profile}' }}
$steps = New-RemovalSteps 'DisableSsh'
$warnings = New-Object System.Collections.ArrayList
if (-not (Start-RemovalFinalizer '{PAIRING_ID}' 'DisableSsh' $identity $steps $warnings)) {{ throw 'finalizer did not launch' }}
$expected = 'RagtimeRemoteRemoval-{PAIRING_ID}'
if (($trace -join ',') -ne "register:$expected") {{ throw "wrong finalizer trace: $($trace -join ',')" }}
if ($steps.pairing_cleanup -ne 'unconfirmed' -or $steps.ssh_service -ne 'unconfirmed') {{ throw 'finalizer claimed completion' }}
""",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)


def test_windows_removal_refuses_reparse_pairing_directory(tmp_path):
    pwsh = _pwsh()
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "pairing-link"
    link.symlink_to(target, target_is_directory=True)
    harness = tmp_path / "reparse.ps1"
    harness.write_text(
        _installer_functions()
        + f"""
function Test-Elevated {{ $false }}
try {{ Remove-PairingDirectory '{link}'; throw 'reparse point was removed' }}
catch {{ if ($_.Exception.Message -notmatch 'Refusing reparse point') {{ throw }} }}
if (-not (Test-Path -LiteralPath '{target}')) {{ throw 'reparse target was changed' }}
""",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)


def test_windows_removal_rejects_uuid_with_a_trailing_newline(tmp_path):
    pwsh = _pwsh()
    harness = tmp_path / "strict-uuid.ps1"
    harness.write_text(
        _installer_functions()
        + f"""
try {{ Assert-PairingId "{PAIRING_ID}`n"; throw 'newline UUID was accepted' }}
catch {{ if ($_.Exception.Message -notmatch 'valid UUID') {{ throw }} }}
""",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)


def test_windows_removal_failed_finalizer_does_not_fall_back_inline(tmp_path):
    pwsh = _pwsh()
    profile = tmp_path / "profile"
    harness = tmp_path / "finalizer-failure.ps1"
    harness.write_text(
        _installer_functions()
        + f"""
function Test-Elevated {{ $true }}
function Test-TunnelTaskPresent {{ $true }}
function Start-RemovalFinalizer {{ param($PairingId, $Mode, $Identity, $Steps, $Warnings) Set-RemovalStepStatus $Steps 'pairing_cleanup' 'failed'; [void]$Warnings.Add('launch failed'); return $false }}
function Invoke-PairingCleanup {{ throw 'inline fallback is unsafe' }}
$identity = [pscustomobject]@{{ Name = 'user'; Sid = 'S-1-5-21-1000'; Profile = '{profile}' }}
Invoke-Removal '{PAIRING_ID}' 'PairingOnly' $identity
""",
        encoding="utf-8",
    )
    result = subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)
    assert '"pairing_cleanup":"failed"' in result.stdout
    assert "inline fallback is unsafe" not in result.stderr


def test_windows_removal_nonadmin_reports_protected_resources_unconfirmed(tmp_path):
    pwsh = _pwsh()
    profile = tmp_path / "profile"
    harness = tmp_path / "nonadmin-resources.ps1"
    harness.write_text(
        _installer_functions()
        + f"""
function Test-Elevated {{ $false }}
function Get-ScheduledTask {{ Write-Error -Category ObjectNotFound -ErrorAction Stop 'missing task' }}
$identity = [pscustomobject]@{{ Name = 'user'; Sid = 'S-1-5-21-1000'; Profile = '{profile}' }}
$steps = New-RemovalSteps 'PairingOnly'
$warnings = New-Object System.Collections.ArrayList
Invoke-PairingCleanup '{PAIRING_ID}' $identity $steps $warnings
if ($steps.pairing_cleanup -ne 'unconfirmed') {{ throw "protected state was claimed complete: $($steps.pairing_cleanup)" }}
if ($warnings.Count -lt 1) {{ throw 'missing protected-state warning' }}
""",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)


def test_windows_removal_refuses_elevated_recursive_profile_cleanup(tmp_path):
    pwsh = _pwsh()
    profile = tmp_path / "profile"
    state = profile / "AppData" / "Local" / "RagtimeRemote" / PAIRING_ID
    state.mkdir(parents=True)
    harness = tmp_path / "elevated-profile.ps1"
    harness.write_text(
        _installer_functions()
        + f"""
function Test-Elevated {{ $true }}
function Get-ScheduledTask {{ Write-Error -Category ObjectNotFound -ErrorAction Stop 'missing task' }}
$env:ProgramData = '{tmp_path / "program-data"}'
$identity = [pscustomobject]@{{ Name = 'user'; Sid = 'S-1-5-21-1000'; Profile = '{profile}' }}
$steps = New-RemovalSteps 'PairingOnly'
$warnings = New-Object System.Collections.ArrayList
Invoke-PairingCleanup '{PAIRING_ID}' $identity $steps $warnings
if ($steps.pairing_cleanup -ne 'failed') {{ throw "unsafe elevated profile traversal was allowed: $($steps.pairing_cleanup)" }}
if (-not (Test-Path -LiteralPath '{state}')) {{ throw 'profile state was removed' }}
""",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)


def test_windows_removal_never_calls_recursive_helper_for_absent_elevated_profile(tmp_path):
    pwsh = _pwsh()
    harness = tmp_path / "absent-profile.ps1"
    harness.write_text(
        _installer_functions()
        + f"""
function Test-Elevated {{ $true }}
function Get-ScheduledTask {{ Write-Error -Category ObjectNotFound -ErrorAction Stop 'missing' }}
function Remove-PairingDirectory {{ param($Path) if ($Path.Contains('profile')) {{ throw 'unsafe profile helper call' }} }}
$env:ProgramData = '{tmp_path / "system"}'
$identity = [pscustomobject]@{{ Name = 'user'; Sid = 'S-1-5-21-1000'; Profile = '{tmp_path / "profile"}' }}
$steps = New-RemovalSteps 'PairingOnly'
$warnings = New-Object System.Collections.ArrayList
Invoke-PairingCleanup '{PAIRING_ID}' $identity $steps $warnings
if (($warnings -join ',') -match 'unsafe profile helper call') {{ throw 'elevated profile helper was called' }}
""",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)


def test_windows_finalizer_rejects_context_with_wrong_registered_profile(tmp_path):
    pwsh = _pwsh()
    harness = tmp_path / "context.ps1"
    harness.write_text(
        _installer_functions()
        + f"""
function Get-SidProfile {{ param($Sid) '{tmp_path / "registered"}' }}
$context = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes('{{"Name":"DOMAIN\\\\user","Sid":"S-1-5-21-1000","Profile":"{tmp_path / "wrong"}"}}'))
try {{ Get-FinalizerIdentity $context; throw 'wrong profile accepted' }}
catch {{ if ($_.Exception.Message -notmatch 'does not match') {{ throw }} }}
""",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)


def test_windows_protected_system_path_rejects_reparse_ancestor(tmp_path):
    pwsh = _pwsh()
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "program-data"
    link.symlink_to(real, target_is_directory=True)
    harness = tmp_path / "system-ancestor.ps1"
    harness.write_text(
        _installer_functions()
        + f"""
$env:ProgramData = '{link}'
try {{ Assert-ProtectedSystemPath (Join-Path $env:ProgramData 'RagtimeRemote\\missing'); throw 'ancestor accepted' }}
catch {{ if ($_.Exception.Message -notmatch 'Refusing reparse point') {{ throw }} }}
""",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)


def test_windows_invalid_finalizer_context_still_cleans_owned_artifacts(tmp_path):
    pwsh = _pwsh()
    harness = tmp_path / "invalid-finalizer.ps1"
    harness.write_text(
        _installer_functions()
        + f"""
$cleaned = $false
function Complete-RemovalFinalizer {{ $script:cleaned = $true }}
function Invoke-Removal {{ throw 'destructive cleanup must not execute' }}
try {{ Invoke-RemovalFinalizer '{PAIRING_ID}' 'PairingOnly' 'invalid'; throw 'invalid context accepted' }}
catch {{ if ($_.Exception.Message -notmatch 'context is invalid') {{ throw }} }}
if (-not $cleaned) {{ throw 'finalizer artifacts were not cleaned' }}
""",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)


def test_windows_failed_task_registration_removes_staged_payload(tmp_path):
    pwsh = _pwsh()
    harness = tmp_path / "failed-staging.ps1"
    harness.write_text(
        _installer_functions()
        + f"""
$env:ProgramData = '{tmp_path / "system"}'
function Test-Elevated {{ $true }}
function Get-SidProfile {{ '{tmp_path / "profile"}' }}
function Initialize-SystemDirectory {{ param($Path) New-Item -ItemType Directory -Force -Path $Path | Out-Null }}
function Set-SystemOnlyAcl {{ }}
function New-ScheduledTaskAction {{ [pscustomobject]@{{}} }}
function New-ScheduledTaskTrigger {{ [pscustomobject]@{{}} }}
function New-ScheduledTaskSettingsSet {{ [pscustomobject]@{{}} }}
function Register-ScheduledTask {{ throw 'registration denied' }}
$identity = [pscustomobject]@{{ Name = 'user'; Sid = 'S-1-5-21-1000'; Profile = '{tmp_path / "profile"}' }}
$steps = New-RemovalSteps 'DisableSsh'
$warnings = New-Object System.Collections.ArrayList
if (Start-RemovalFinalizer '{PAIRING_ID}' 'DisableSsh' $identity $steps $warnings) {{ throw 'registration failure claimed launch' }}
if (Test-Path -LiteralPath (Join-Path $env:ProgramData 'RagtimeRemote/removal-payloads/{PAIRING_ID}.ps1')) {{ throw 'staged payload leaked' }}
if ($steps.pairing_cleanup -ne 'failed') {{ throw 'registration failure was not reported' }}
""",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)


def test_windows_finalizer_waits_for_parent_readiness_before_cleanup(tmp_path):
    pwsh = _pwsh()
    staging = tmp_path / "AppData/Local/RagtimeRemote/removal-payloads"
    staging.mkdir(parents=True)
    harness = staging / f"{PAIRING_ID}.ps1"
    harness.write_text(
        _installer_functions()
        + f"""
function Test-Elevated {{ $false }}
function Get-FinalizerIdentity {{ [pscustomobject]@{{ Profile = '{tmp_path}'; Name = 'user'; Sid = 'S-1-5-21-1000' }} }}
function Start-Sleep {{ throw 'readiness pending' }}
function Complete-RemovalFinalizer {{ $script:cleaned = $true }}
function Invoke-Removal {{ throw 'destructive cleanup preceded readiness' }}
$cleaned = $false
try {{ Invoke-RemovalFinalizer '{PAIRING_ID}' 'PairingOnly' 'context'; throw 'missing gate accepted' }}
catch {{ if ($_.Exception.Message -ne 'readiness pending') {{ throw }} }}
if (-not $cleaned) {{ throw 'pending finalizer was not cleaned' }}
""",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("staged", [False, True])
def test_windows_invalid_context_only_removes_independently_verified_staged_script(tmp_path, staged):
    pwsh = _pwsh()
    staging = tmp_path / "AppData/Local/RagtimeRemote/removal-payloads"
    staging.mkdir(parents=True)
    harness = staging / f"{PAIRING_ID}.ps1" if staged else tmp_path / "downloaded-installer.ps1"
    ready = Path(f"{harness}.ready")
    ready.write_text("ready", encoding="utf-8")
    trace = tmp_path / "task-cleanup"
    harness.write_text(
        _installer_functions()
        + f"""
function Test-Elevated {{ $false }}
function Get-OriginalIdentity {{ [pscustomobject]@{{ Profile = '{tmp_path}'; Name = 'user'; Sid = 'S-1-5-21-1000' }} }}
function Unregister-ScheduledTask {{ Set-Content -LiteralPath '{trace}' -Value 'unregistered' }}
try {{ Invoke-RemovalFinalizer '{PAIRING_ID}' 'PairingOnly' 'invalid'; throw 'invalid context accepted' }}
catch {{ if ($_.Exception.Message -notmatch 'context is invalid') {{ throw }} }}
""",
        encoding="utf-8",
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)
    assert harness.exists() is not staged
    assert ready.exists() is not staged
    assert trace.exists() is staged
