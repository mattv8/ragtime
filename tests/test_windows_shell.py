import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from prisma import Json
from prisma.enums import AuthProvider, UserRole
from prisma.models import User

from ragtime.core.security import validate_powershell_command
from ragtime.core.ssh import (
    POWERSHELL_TIMEOUT_SENTINEL,
    SSHConfig,
    SSHResult,
    _encode_powershell_payload,
    _wrap_powershell_command,
    build_powershell_working_directory_command,
    execute_ssh_command,
)
from ragtime.indexer.routes import (
    SSHBrowseRequest,
    _browse_powershell_ssh_filesystem,
    _get_ssh_cache_key,
    _powershell_path_from_browser_path,
    _ssh_browse_config,
    browse_ssh_filesystem,
)
from ragtime.rag.components import RAGComponents


class _Channel:
    def exit_status_ready(self):
        return True

    def recv_ready(self):
        return False

    def recv_stderr_ready(self):
        return False

    def recv(self, _size):
        return b""

    def recv_stderr(self, _size):
        return b""

    def recv_exit_status(self):
        return 124


class _Client:
    def exec_command(self, command, timeout):
        self.command = command
        stream = type("Stream", (), {"channel": _Channel()})()
        return type("Input", (), {"write": lambda *_: None, "channel": type("Channel", (), {"shutdown_write": lambda *_: None})()})(), stream, stream

    def close(self):
        pass


def test_powershell_wrapper_uses_encoded_command_and_preserves_native_124():
    wrapped = _wrap_powershell_command(30)
    encoded = wrapped.rsplit(" ", 1)[1]
    script = base64.b64decode(encoded).decode("utf-16le")
    decoded_payload = base64.b64decode(_encode_powershell_payload("Write-Output 'héllo'; cmd /c exit 124")).decode("utf-16le")
    assert "ReadLine" in script
    assert "UTF8Encoding" in decoded_payload
    assert "cmd /c exit 124" in decoded_payload
    assert "Start-Job" not in script
    assert "taskkill.exe" in script
    assert POWERSHELL_TIMEOUT_SENTINEL in script

    client = _Client()
    with patch("ragtime.core.ssh._create_ssh_client", return_value=client):
        result = execute_ssh_command(SSHConfig(host="host", user="user", password="pw", shell="powershell"), "cmd /c exit 124")
    assert result.exit_code == 124
    assert not result.success
    assert "timed out" not in result.stderr.lower()


def test_powershell_read_only_guardrails_are_not_a_directory_sandbox():
    assert validate_powershell_command("Get-ChildItem C:\\Users")[0]
    assert not validate_powershell_command("Remove-Item C:\\Users\\file.txt")[0]
    assert validate_powershell_command("Remove-Item C:\\Users\\file.txt", allow_write=True)[0]
    assert not validate_powershell_command("ri C:\\Users\\file.txt")[0]
    assert not validate_powershell_command("curl https://example.invalid -OutFile C:\\x")[0]
    assert not validate_powershell_command("cmd /c del C:\\x")[0]
    assert not validate_powershell_command("[IO.File]::WriteAllText('C:\\x', 'x')")[0]
    assert validate_powershell_command("Get-ChildItem x 2>$null")[0]
    assert validate_powershell_command("Write-Output 'a>b'")[0]


def test_powershell_browse_keeps_unicode_paths_and_singleton_arrays():
    result = SSHResult(
        stdout=json.dumps({"name": "Résumé", "path": "C:\\Users\\Ada Lovelace\\Résumé", "is_dir": True, "size": None}),
        stderr="",
        exit_code=0,
        success=True,
    )
    config = SSHConfig(host="host", user="user", password="pw", shell="powershell")
    with patch("ragtime.indexer.routes.execute_ssh_command", return_value=result):
        response = asyncio.run(_browse_powershell_ssh_filesystem(config, "/C:/Users/Ada Lovelace"))
    assert response.entries[0].path == "/C:/Users/Ada Lovelace/Résumé"
    assert response.entries[0].is_dir


def test_windows_browse_paths_require_drive_roots_and_escape_curly_quotes():
    assert _powershell_path_from_browser_path("/C:") == "C:\\"
    assert _powershell_path_from_browser_path("/C:/Users") == "C:\\Users"
    try:
        _powershell_path_from_browser_path("/C/Users")
    except ValueError:
        pass
    else:
        raise AssertionError("missing drive colon was accepted")

    config = SSHConfig(host="host", user="user", password="pw", shell="powershell")
    with patch("ragtime.indexer.routes.execute_ssh_command", return_value=SSHResult(stdout="[]", stderr="", exit_code=0, success=True)) as execute:
        asyncio.run(_browse_powershell_ssh_filesystem(config, "/C:/Bob’s files"))
    assert "Bob’’s files" in execute.call_args.args[1]


def test_tunnel_heartbeat_cache_keys_are_pairing_and_shell_specific():
    base = {"host": "remote-relay", "port": 22, "user": "Administrator", "transport": "tunnel"}
    assert _get_ssh_cache_key({**base, "pairing_id": "one", "shell": "powershell"}) != _get_ssh_cache_key({**base, "pairing_id": "two", "shell": "powershell"})
    assert _get_ssh_cache_key({**base, "pairing_id": "one", "shell": "powershell"}) != _get_ssh_cache_key({**base, "pairing_id": "one", "shell": "posix"})


def test_windows_tool_composes_literal_working_directory_before_prefix_and_command():
    config = {
        "tool_type": "windows_shell",
        "name": "Windows",
        "connection_config": {
            "host": "host",
            "user": "user",
            "password": "pw",
            "working_directory": "/C:/Bob’s Files",
            "command_prefix": "Write-Output prefix; ",
        },
    }
    result = SSHResult(stdout="ok", stderr="", exit_code=0, success=True)
    tool = asyncio.run(RAGComponents()._create_ssh_tool(config, "windows", "tool"))
    with patch("ragtime.rag.components.execute_ssh_command", return_value=result) as execute:
        response = asyncio.run(tool.ainvoke({"command": "Write-Output command", "reason": "test"}))
    assert json.loads(response)["status"] == "completed"
    composed = execute.call_args.args[1]
    assert composed == "$ErrorActionPreference = 'Stop'; Set-Location -LiteralPath 'C:\\Bob’’s Files'; Write-Output prefix; Write-Output command"


def test_paired_browse_hydrates_server_credentials_for_both_shells():
    for shell in ("posix", "powershell"):
        pairing = {"id": "pairing", "toolId": "tool", "status": "paired"}
        saved = SimpleNamespace(
            connection_config={
                "host": "saved-host",
                "user": "saved-user",
                "port": 2222,
                "key_content": "server-only-private-key",
                "host_keys": ["ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEdvb2Q="],
                "pairing_id": "pairing",
                "transport": "direct",
                "shell": shell,
            }
        )
        with (
            patch("ragtime.remote_client.repository.remote_pairing_repository.get", AsyncMock(return_value=pairing)),
            patch("ragtime.indexer.routes.repository.get_tool_config", AsyncMock(return_value=saved)),
            patch("ragtime.indexer.routes.execute_ssh_command", return_value=SSHResult(stdout="[]", stderr="", exit_code=0, success=True)) as execute,
        ):
            request = SSHBrowseRequest(pairing_id="pairing", host="attacker", user="attacker", shell="posix")
            config = asyncio.run(_ssh_browse_config(request))
            user = User(
                id="test-user",
                username="test-user",
                authProvider=AuthProvider.local,
                cachedGroups=cast(Json, "[]"),
                role=UserRole.user,
                roleManuallySet=False,
                createdAt=datetime.now(UTC),
                updatedAt=datetime.now(UTC),
                securityGeneration=0,
            )
            response = asyncio.run(browse_ssh_filesystem(request, user))
        assert response.error is None
        assert execute.call_args.args[0].host == "saved-host"
        assert execute.call_args.args[0].shell == shell
        assert config.key_content == "server-only-private-key"


def test_local_pwsh_parses_outer_wrapper_and_runs_inner_encoded_payload():
    configured_pwsh = os.environ.get("RAGTIME_TEST_PWSH") or shutil.which("pwsh")
    if not configured_pwsh:
        pytest.skip("PowerShell runtime integration test requires pwsh or RAGTIME_TEST_PWSH")
    pwsh = Path(configured_pwsh)
    if not pwsh.is_file():
        pytest.skip("RAGTIME_TEST_PWSH does not point to an executable")
    user_command = "Write-Output 'héllo'; $line = [Console]::In.ReadLine(); Write-Output \"stdin:$line\"; exit 124"
    wrapped = _wrap_powershell_command(5)
    outer = base64.b64decode(wrapped.rsplit(" ", 1)[1]).decode("utf-16le")
    payload = _encode_powershell_payload(user_command)
    assert "StandardInputEncoding" not in outer
    parser_env = {**os.environ, "RAGTIME_TEST_SCRIPT": base64.b64encode(outer.encode("utf-8")).decode("ascii")}
    parsed = subprocess.run(
        [
            str(pwsh),
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$s = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($env:RAGTIME_TEST_SCRIPT)); $tokens = $null; $errors = $null; [System.Management.Automation.Language.Parser]::ParseInput($s, [ref]$tokens, [ref]$errors) | Out-Null; if ($errors.Count) { exit 1 }",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=parser_env,
    )
    assert parsed.returncode == 0, parsed.stderr
    inner = subprocess.run(
        [str(pwsh), "-NoProfile", "-NonInteractive", "-EncodedCommand", payload],
        input="data\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert inner.returncode == 124
    assert "héllo" in inner.stdout
    assert "stdin:data" in inner.stdout

    local_outer = outer.replace("FileName = 'powershell.exe'", f"FileName = '{pwsh.resolve()}'")
    local_outer_encoded = base64.b64encode(local_outer.encode("utf-16le")).decode("ascii")
    supervised = subprocess.run(
        [str(pwsh), "-NoProfile", "-NonInteractive", "-EncodedCommand", local_outer_encoded],
        input=f"{payload}\ndata\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert supervised.returncode == 124
    assert "héllo" in supervised.stdout
    assert "stdin:data" in supervised.stdout

    unicode_input = "naïve\n雪\n"
    native_exit_command = "cmd /c exit 47" if os.name == "nt" else "/bin/sh -c 'exit 47'"
    unicode_payload = _encode_powershell_payload(
        f"$inputText = [Console]::In.ReadToEnd(); [Console]::Out.Write('stdin:' + $inputText); [Console]::Error.Write('child-error'); & {native_exit_command}"
    )
    unicode_supervised = subprocess.run(
        [str(pwsh), "-NoProfile", "-NonInteractive", "-EncodedCommand", local_outer_encoded],
        input=f"{unicode_payload}\n{unicode_input}",
        capture_output=True,
        text=True,
        check=False,
    )
    assert unicode_supervised.returncode == 47
    assert unicode_supervised.stdout == f"stdin:{unicode_input}"
    assert unicode_supervised.stderr == "child-error"

    eof_payload = _encode_powershell_payload("if ([Console]::In.ReadToEnd() -eq '') { Write-Output 'eof' } else { exit 1 }")
    eof_supervised = subprocess.run(
        [str(pwsh), "-NoProfile", "-NonInteractive", "-EncodedCommand", local_outer_encoded],
        input=f"{eof_payload}\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert eof_supervised.returncode == 0
    assert eof_supervised.stdout == "eof\n"

    native_shell = "cmd /c" if os.name == "nt" else "/bin/sh -c"
    missing_path = "$env:TEMP\\missing-ragtime-path" if os.name == "nt" else "/missing-ragtime-path"
    native_then_cmdlet = subprocess.run(
        [str(pwsh), "-NoProfile", "-NonInteractive", "-EncodedCommand", _encode_powershell_payload(f"& {native_shell} 'exit 7'; Write-Output done")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert native_then_cmdlet.returncode == 0
    failed_cmdlet = subprocess.run(
        [str(pwsh), "-NoProfile", "-NonInteractive", "-EncodedCommand", _encode_powershell_payload(f"& {native_shell} 'exit 0'; Get-Item {missing_path}")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed_cmdlet.returncode == 1

    missing_directory = subprocess.run(
        [
            str(pwsh),
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            _encode_powershell_payload(build_powershell_working_directory_command("/C:/missing-ragtime-path", "Write-Output should-not-run")),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_directory.returncode == 1
    assert "should-not-run" not in missing_directory.stdout

    long_payload = _encode_powershell_payload("Write-Output 'x' " + ("# padding\n" * 3000))
    assert len(_wrap_powershell_command(30)) < 8191
    assert len(long_payload) > 8191

    timeout_outer = base64.b64decode(_wrap_powershell_command(1).rsplit(" ", 1)[1]).decode("utf-16le")
    timeout_outer = timeout_outer.replace("FileName = 'powershell.exe'", f"FileName = '{pwsh.resolve()}'")
    timeout_encoded = base64.b64encode(timeout_outer.encode("utf-16le")).decode("ascii")
    timed_out = subprocess.run(
        [str(pwsh), "-NoProfile", "-NonInteractive", "-EncodedCommand", timeout_encoded],
        input=f"{_encode_powershell_payload('Start-Sleep -Seconds 5')}\n",
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert timed_out.returncode != 0
    assert POWERSHELL_TIMEOUT_SENTINEL in timed_out.stderr


@pytest.mark.parametrize(
    ("command", "expected_exit_code", "expected_stdout"),
    [
        ("Write-Error 'boom'; Write-Output ok; return", 1, "ok\n"),
        ("& {native_shell} 'exit 47'; return", 47, ""),
        ("$errorCount = 999; Write-Error 'boom'; Write-Output ok", 1, "ok\n"),
        ("param($value = 'x'); Write-Output $value", 0, "x\n"),
        ("Write-Output ok; return", 0, "ok\n"),
        ("throw 'boom'", 1, ""),
        ("exit 3", 3, ""),
    ],
)
def test_local_pwsh_payload_keeps_user_scope_and_exit_contract(command, expected_exit_code, expected_stdout):
    configured_pwsh = os.environ.get("RAGTIME_TEST_PWSH") or shutil.which("pwsh")
    if not configured_pwsh:
        pytest.skip("PowerShell runtime integration test requires pwsh or RAGTIME_TEST_PWSH")
    pwsh = Path(configured_pwsh)
    if not pwsh.is_file():
        pytest.skip("RAGTIME_TEST_PWSH does not point to an executable")

    native_shell = "cmd /c" if os.name == "nt" else "/bin/sh -c"
    payload = _encode_powershell_payload(command.format(native_shell=native_shell))
    result = subprocess.run(
        [str(pwsh), "-NoProfile", "-NonInteractive", "-EncodedCommand", payload],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == expected_exit_code
    assert result.stdout == expected_stdout
