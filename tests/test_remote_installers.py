import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "ragtime" / "remote_client" / "scripts"


def test_posix_installer_uses_material_before_authorizing_key_and_retries_nonce():
    script = (SCRIPTS / "install.sh").read_text()
    material = script.index("enroll/material")
    authorization = script.index('touch "$KEY_FILE"')
    assert material < authorization
    assert "nonce_file" in script
    assert "attempt" in script
    assert "Authorization: Bearer %s" in script
    assert '--config "$curl_config"' in script


def test_posix_installer_preserves_ssh_policy_and_owned_uninstall_boundary():
    script = (SCRIPTS / "install.sh").read_text()
    assert "TAG_PREFIX='# ragtime-remote:'" in script
    assert 'grep -Fv "${TAG_PREFIX}$PAIRING_ID"' in script
    assert "--proto =https" in script
    assert "--insecure" not in script
    assert "apt-get" in script and "rc-service" in script
    assert "sha256_file" in script
    assert "tunnel.json" in script


def test_posix_openrc_tunnel_service_is_supervised_and_owned_uninstall_removes_it():
    script = (SCRIPTS / "install.sh").read_text()
    assert "#!/sbin/openrc-run" in script
    assert "supervisor=supervise-daemon" in script
    assert 'command_user="$TARGET_USER"' in script
    assert 'rc-update add "$(basename "$unit")" default' in script
    assert 'rc-service "ragtime-remote-$PAIRING_ID" stop' in script
    assert 'rc-update del "ragtime-remote-$PAIRING_ID" default' in script


def test_posix_user_systemd_unit_restarts_at_login_without_claiming_boot_persistence():
    script = (SCRIPTS / "install.sh").read_text()
    user_unit = script.split('unit="$TARGET_HOME/.config/systemd/user/', 1)[1].split("  elif command -v rc-service", 1)[0]
    assert "[Install]" in user_unit
    assert "WantedBy=default.target" in user_unit
    assert "systemctl --user enable --now" in user_unit


def test_posix_openrc_tunnel_install_and_uninstall_generate_owned_unit(tmp_path):
    init_dir = tmp_path / "init.d"
    log_dir = tmp_path / "log"
    init_dir.mkdir()
    log_dir.mkdir()
    installer = tmp_path / "install.sh"
    installer.write_text(
        (SCRIPTS / "install.sh").read_text().replace("command -v systemctl", "false").replace("/etc/init.d", str(init_dir)).replace("/var/log", str(log_dir))
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    host_key_dir = tmp_path / "ssh"
    host_key_dir.mkdir()
    (host_key_dir / "ssh_host_ed25519_key.pub").write_text("ssh-ed25519 AAAAHOST host\n")
    trace = tmp_path / "openrc-trace"
    commands = {
        "curl": '#!/bin/sh\nfor arg in "$@"; do case $arg in *enroll/material) printf \'%s\' \'{"pairing_id":"pair_openrc","public_key":"ssh-ed25519 AAAA test","ws_url":"wss://example/tunnel"}\'; exit;; esac; done\nprintf \'%s\' \'{"pairing_id":"pair_openrc","status":"consumed","transport":"tunnel","credential":"credential"}\'\n',
        "sshd": "#!/bin/sh\n[ \"$1\" = -T ] && printf 'port 22\\n'\n",
        "uname": "#!/bin/sh\nprintf 'Linux\\n'\n",
        "id": "#!/bin/sh\ncase \"$1\" in -u) printf '0\\n';; -un) printf 'root\\n';; *) printf 'root\\n';; esac\n",
        "rc-service": '#!/bin/sh\nprintf \'service %s\\n\' "$*" >> "$OPENRC_TRACE"\n',
        "rc-update": '#!/bin/sh\nprintf \'update %s\\n\' "$*" >> "$OPENRC_TRACE"\n',
        "chown": "#!/bin/sh\nexit 0\n",
    }
    for name, content in commands.items():
        command = bin_dir / name
        command.write_text(content)
        command.chmod(0o755)
    client = tmp_path / "tunnel-client"
    client.write_text("#!/bin/sh\n")
    client.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "OPENRC_TRACE": str(trace),
        "RAGTIME_SSH_HOST_KEY_DIR": str(host_key_dir),
    }
    subprocess.run(
        ["sh", str(installer), "--server-url", "https://example", "--token", "token", "--mode", "tunnel", "--tunnel-client", str(client)],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    unit = init_dir / "ragtime-remote-pair_openrc"
    assert "supervisor=supervise-daemon" in unit.read_text()
    subprocess.run(["sh", str(installer), "--uninstall", "pair_openrc"], env=environment, check=True, capture_output=True, text=True)
    assert not unit.exists()
    assert "update add ragtime-remote-pair_openrc default" in trace.read_text()
    assert "update del ragtime-remote-pair_openrc default" in trace.read_text()


def test_windows_installer_uses_exact_sid_and_never_rewrites_account_names():
    script = (SCRIPTS / "install.ps1").read_text()
    assert "SecurityIdentifier($Sid)" in script
    assert "Get-OriginalIdentity" in script
    assert "administrators_authorized_keys" in script
    assert "ProfileList\\$($sid.Value)" in script
    assert "sshd AuthorizedKeysFile" in script
    assert "cannot be enrolled from this elevated session" not in script
    assert "DOMAIN/" not in script
    assert "-SkipCertificateCheck" not in script


def test_windows_non_elevated_fallback_uses_readable_config_for_current_standard_user_only():
    script = (SCRIPTS / "install.ps1").read_text()
    assert "return Get-NonElevatedAuthorizedKeyPath $Identity" in script
    assert "Cannot validate a different target account without elevation" in script
    assert "function Test-AdministratorMembership" in script
    assert "ClaimTypes]::DenyOnlySid" in script
    assert "may use sshd administrators_authorized_keys. Re-run elevated" in script
    assert "Get-Content -LiteralPath $configPath" in script


def test_offline_payloads_include_supplied_public_key_and_bounded_facts_fields():
    sh = (SCRIPTS / "install.sh").read_text()
    ps = (SCRIPTS / "install.ps1").read_text()
    for script in (sh, ps):
        assert "pairing_id" in script
        assert "public_key" in script
        assert "host_keys" in script
        assert "python_available" in script


def test_generated_offline_commands_supply_pairing_material_without_an_online_token():
    routes = (SCRIPTS.parent / "routes.py").read_text()
    assert '"offline_sh": f"sh ./install.sh' in routes
    assert '"offline_powershell": f"powershell.exe' in routes
    assert "iwr -UseBasicParsing" not in routes.split('"offline_powershell"', 1)[1].split("\n", 1)[0]
    assert "--mode offline --pairing-id" in routes
    assert "--public-key" in routes


def test_windows_acl_helpers_use_set_acl_and_harden_system_owned_directories():
    script = (SCRIPTS / "install.ps1").read_text()
    assert "[IO.File]::SetAccessControl" not in script
    assert "Set-Acl -LiteralPath $Path -AclObject $Acl" in script
    assert "$acl.SetOwner($administrators)" in script
    assert "FileAttributes]::ReparsePoint" in script
    assert "Get-ItemProperty -LiteralPath" in script
    assert "GetFullPath((Join-Path $Identity.Profile '.ssh\\authorized_keys'))" in script
    assert "-T -C" in script


def test_powershell_runtime_exposes_the_portable_acl_extension_api():
    pwsh = _pwsh()
    result = subprocess.run(
        [
            str(pwsh),
            "-NoProfile",
            "-Command",
            "[IO.FileSystemAclExtensions].GetMethods() | Where-Object Name -eq SetAccessControl | Measure-Object | Select-Object -ExpandProperty Count",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert int(result.stdout.strip()) >= 2


def test_posix_online_rerun_reuses_nonce_and_adds_one_tagged_key(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    host_key_dir = tmp_path / "ssh"
    host_key_dir.mkdir()
    (host_key_dir / "ssh_host_ed25519_key.pub").write_text("ssh-ed25519 AAAAHOST host\n")
    calls = tmp_path / "calls"
    (bin_dir / "curl").write_text(
        "#!/bin/sh\n"
        'for arg in "$@"; do case $arg in *enroll/material) '
        'printf \'%s\' \'{"pairing_id":"pair_1","public_key":"ssh-ed25519 AAAA test","ws_url":"wss://example/tunnel","tunnel_client_sha256":"00"}\'; exit;; esac; done\n'
        'printf \'%s\\n\' "$@" >> "$CALLS"\nprintf \'%s\' \'{"pairing_id":"pair_1","status":"paired","transport":"direct"}\'\n'
    )
    (bin_dir / "sshd").write_text("#!/bin/sh\n[ \"$1\" = -T ] && printf 'port 2222\\n'\n")
    (bin_dir / "uname").write_text("#!/bin/sh\nprintf 'Linux\\n'\n")
    for command in bin_dir.iterdir():
        command.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "CALLS": str(calls),
        "RAGTIME_SSH_HOST_KEY_DIR": str(host_key_dir),
    }
    command = ["sh", str(SCRIPTS / "install.sh"), "--server-url", "https://example", "--token", "token", "--mode", "direct"]
    subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
    subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
    assert (tmp_path / ".ssh" / "authorized_keys").read_text().count("# ragtime-remote:pair_1") == 1
    payloads = [line for line in calls.read_text().splitlines() if line.startswith('{"nonce"')]
    assert len(payloads) == 2
    assert payloads[0] == payloads[1]
    assert '"mode":"direct"' in payloads[0]


def test_posix_auto_without_tunnel_prerequisites_enrolls_direct_instead_of_failing(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    host_key_dir = tmp_path / "ssh"
    host_key_dir.mkdir()
    (host_key_dir / "ssh_host_ed25519_key.pub").write_text("ssh-ed25519 AAAAHOST host\n")
    calls = tmp_path / "calls"
    (bin_dir / "curl").write_text(
        "#!/bin/sh\n"
        'for arg in "$@"; do case $arg in *enroll/material) '
        'printf \'%s\' \'{"pairing_id":"pair_auto","public_key":"ssh-ed25519 AAAA test"}\'; exit;; esac; done\n'
        'printf \'%s\\n\' "$@" >> "$CALLS"\nprintf \'%s\' \'{"pairing_id":"pair_auto","status":"paired","transport":"direct"}\'\n'
    )
    (bin_dir / "sshd").write_text("#!/bin/sh\n[ \"$1\" = -T ] && printf 'port 22\\n'\n")
    (bin_dir / "uname").write_text("#!/bin/sh\nprintf 'Linux\\n'\n")
    (bin_dir / "python3").write_text("#!/bin/sh\nexit 1\n")
    for command in bin_dir.iterdir():
        command.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "CALLS": str(calls),
        "RAGTIME_SSH_HOST_KEY_DIR": str(host_key_dir),
    }
    result = subprocess.run(
        ["sh", str(SCRIPTS / "install.sh"), "--server-url", "https://example", "--token", "token", "--mode", "auto"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Auto mode will enroll direct SSH only" in result.stderr
    payload = calls.read_text()
    assert '"mode":"direct"' in payload
    assert "Bundled tunnel mode requires Python 3.8" in payload


def test_posix_replaces_rotated_key_for_its_pairing_tag(tmp_path):
    script = (SCRIPTS / "install.sh").read_text()
    assert "awk -v tag=" in script
    assert "if (!done++) print key" in script


def _pwsh() -> Path:
    configured_pwsh = os.environ.get("RAGTIME_TEST_PWSH") or shutil.which("pwsh")
    if not configured_pwsh:
        pytest.skip("PowerShell runtime integration test requires pwsh or RAGTIME_TEST_PWSH")
    pwsh = Path(configured_pwsh)
    if not pwsh.is_file():
        pytest.skip("RAGTIME_TEST_PWSH does not point to an executable")
    return pwsh


def test_powershell_key_replacement_function_preserves_unrelated_key_layouts(tmp_path):
    pwsh = _pwsh()
    installer = (SCRIPTS / "install.ps1").read_text()
    functions = installer.split("$identity = Get-OriginalIdentity", 1)[0]
    target = tmp_path / "authorized_keys"
    harness = tmp_path / "key-test.ps1"
    for initial, expected in [
        ("", "ssh-ed25519 NEW # ragtime-remote:pair\n"),
        ("ssh-ed25519 OTHER user\n", "ssh-ed25519 OTHER user\nssh-ed25519 NEW # ragtime-remote:pair\n"),
        ("ssh-ed25519 OTHER one\nssh-rsa OTHER two\n", "ssh-ed25519 OTHER one\nssh-rsa OTHER two\nssh-ed25519 NEW # ragtime-remote:pair\n"),
    ]:
        target.write_bytes(initial.encode())
        harness.write_text(
            functions
            + f"\nSet-AuthorizedKey '{target}' '# ragtime-remote:pair' 'ssh-ed25519 OLD'\nSet-AuthorizedKey '{target}' '# ragtime-remote:pair' 'ssh-ed25519 NEW'\n"
        )
        subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)
        assert target.read_text(encoding="utf-8") == expected
        assert not target.read_bytes().startswith(b"\xef\xbb\xbf")


def test_powershell_admin_membership_handles_deny_only_sid_and_blocks_profile_fallback(tmp_path):
    pwsh = _pwsh()
    installer = (SCRIPTS / "install.ps1").read_text()
    functions = installer.split("$identity = Get-OriginalIdentity", 1)[0]
    profile = tmp_path / "profile"
    program_data = tmp_path / "program-data"
    harness = tmp_path / "deny-only-test.ps1"
    harness.write_text(
        functions
        + f"""
$env:ProgramData = '{program_data}'
$profile = '{profile}'
$sid = 'S-1-5-21-1000'
$identity = [pscustomobject]@{{ Name = 'user'; Sid = $sid; Profile = $profile }}
$user = [pscustomobject]@{{ Value = $sid }}
$normal = [pscustomobject]@{{ User = $user; Groups = @(); Claims = @() }}
if (Test-AdministratorMembership $normal) {{ throw 'normal user was classified as administrator' }}
$enabledAdmin = [pscustomobject]@{{ User = $user; Groups = @([pscustomobject]@{{ Value = 'S-1-5-32-544' }}); Claims = @() }}
if (-not (Test-AdministratorMembership $enabledAdmin)) {{ throw 'enabled administrator group was missed' }}
$unrelatedDeny = [pscustomobject]@{{ User = $user; Groups = @(); Claims = @([Security.Claims.Claim]::new([Security.Claims.ClaimTypes]::DenyOnlySid, 'S-1-5-32-545')) }}
if (Test-AdministratorMembership $unrelatedDeny) {{ throw 'unrelated deny-only SID was accepted' }}
$filteredAdmin = [pscustomobject]@{{ User = $user; Groups = @(); Claims = @([Security.Claims.Claim]::new([Security.Claims.ClaimTypes]::DenyOnlySid, 'S-1-5-32-544')) }}
if (-not (Test-AdministratorMembership $filteredAdmin)) {{ throw 'deny-only administrators SID was missed' }}
$profileKey = Join-Path $profile '.ssh\\authorized_keys'
try {{ Get-NonElevatedAuthorizedKeyPath $identity $filteredAdmin; throw 'filtered admin selected a profile key' }} catch {{ if ($_.Exception.Message -notmatch 'Re-run elevated') {{ throw }} }}
if (Test-Path -LiteralPath $profileKey) {{ throw 'filtered admin wrote profile authorized_keys' }}
$configPath = Join-Path $env:ProgramData 'ssh\\sshd_config'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $configPath) | Out-Null
Set-Content -LiteralPath $configPath -Value 'AuthorizedKeysFile .ssh/authorized_keys'
if ((Get-NonElevatedAuthorizedKeyPath $identity $normal) -ne [IO.Path]::GetFullPath($profileKey)) {{ throw 'standard user did not use the configured profile key' }}
"""
    )
    subprocess.run([str(pwsh), "-NoProfile", "-File", str(harness)], check=True, capture_output=True, text=True)
