from __future__ import annotations

import asyncio
import os
import signal
import unittest
from unittest import mock

from ragtime.core import claude_code
from ragtime.core.claude_code import ClaudeCodeStatus


class _HangingProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.pid: int | None = None
        self.communicate_started = asyncio.Event()
        self.terminated = False
        self.killed = False
        self.waited = False

    async def communicate(self):  # type: ignore[no-untyped-def]
        self.communicate_started.set()
        await asyncio.Event().wait()

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int | None:
        self.waited = True
        return self.returncode


class _PidProcess(_HangingProcess):
    def __init__(self, *, wait_timeouts: int = 0) -> None:
        super().__init__()
        self.pid = 12345
        self.wait_timeouts = wait_timeouts

    async def wait(self) -> int | None:
        self.waited = True
        if self.wait_timeouts > 0:
            self.wait_timeouts -= 1
            raise asyncio.TimeoutError()
        return self.returncode


class ClaudeCodeStatusTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        claude_code._STATUS_IN_FLIGHT = None

    async def asyncTearDown(self) -> None:
        claude_code._STATUS_IN_FLIGHT = None

    async def test_run_claude_command_reaps_timed_out_process(self) -> None:
        process = _HangingProcess()

        async def fake_create_subprocess_exec(*args, **kwargs):  # type: ignore[no-untyped-def]
            return process

        with mock.patch("asyncio.create_subprocess_exec", fake_create_subprocess_exec):
            result_task = asyncio.create_task(claude_code._run_claude_command("claude", "--version", timeout=0.01))
            await asyncio.wait_for(process.communicate_started.wait(), timeout=1.0)
            result = await result_task

        self.assertTrue(result.timed_out)
        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)

    async def test_terminate_process_uses_group_signal_only_when_requested(self) -> None:
        process = _PidProcess()

        with mock.patch("os.killpg") as killpg:
            await claude_code._terminate_process(process, process_group=False)  # type: ignore[arg-type]

        killpg.assert_not_called()
        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)

    async def test_terminate_process_can_signal_process_group(self) -> None:
        process = _PidProcess()

        with mock.patch("os.killpg") as killpg:
            await claude_code._terminate_process(process, process_group=True)  # type: ignore[arg-type]

        killpg.assert_called_once_with(12345, signal.SIGTERM)
        self.assertFalse(process.terminated)
        self.assertTrue(process.waited)

    async def test_terminate_process_escalates_group_signal_after_timeout(self) -> None:
        process = _PidProcess(wait_timeouts=1)

        with mock.patch("os.killpg") as killpg:
            await claude_code._terminate_process(process, process_group=True)  # type: ignore[arg-type]

        self.assertEqual(
            killpg.mock_calls,
            [mock.call(12345, signal.SIGTERM), mock.call(12345, signal.SIGKILL)],
        )
        self.assertTrue(process.waited)

    async def test_get_claude_code_status_coalesces_concurrent_live_probes(self) -> None:
        calls = 0

        async def fake_probe() -> ClaudeCodeStatus:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return ClaudeCodeStatus(
                installed=True,
                command="/usr/local/bin/claude",
                version="1.0.0",
                has_oauth_token=False,
                has_cli_auth=True,
                auth_method="oauth",
                subscription_type="pro",
                available=True,
                error=None,
            )

        with mock.patch("ragtime.core.claude_code._probe_claude_code_status", fake_probe):
            results = await asyncio.gather(
                claude_code.get_claude_code_status(),
                claude_code.get_claude_code_status(),
                claude_code.get_claude_code_status(),
            )

        self.assertEqual(calls, 1)
        self.assertTrue(all(result.available for result in results))

    async def test_status_requires_logged_in_claude_ai_subscription(self) -> None:
        async def probe_result(*_args: str, **_kwargs: object) -> claude_code._ClaudeCommandResult:
            if _args[1:] == ("--version",):
                return claude_code._ClaudeCommandResult(returncode=0, output="2.1.290 (Claude Code)")
            return claude_code._ClaudeCommandResult(
                returncode=0,
                output='{"loggedIn": false, "authMethod": "claude.ai", "subscriptionType": "pro"}',
            )

        with (
            mock.patch("ragtime.core.claude_code.shutil.which", return_value="/usr/local/bin/claude"),
            mock.patch("ragtime.core.claude_code._run_claude_command", side_effect=probe_result),
            mock.patch.dict(os.environ, {claude_code.CLAUDE_CODE_OAUTH_TOKEN_ENV: ""}, clear=False),
        ):
            status = await claude_code._probe_claude_code_status()

        self.assertFalse(status.has_cli_auth)
        self.assertFalse(status.available)

    async def test_status_rejects_malformed_and_api_key_auth_payloads(self) -> None:
        for payload in ("not-json", '{"loggedIn": true, "authMethod": "apiKey"}'):
            with self.subTest(payload=payload):

                async def probe_result(*_args: str, **_kwargs: object) -> claude_code._ClaudeCommandResult:
                    if _args[1:] == ("--version",):
                        return claude_code._ClaudeCommandResult(returncode=0, output="2.1.290 (Claude Code)")
                    return claude_code._ClaudeCommandResult(returncode=0, output=payload)

                with (
                    mock.patch("ragtime.core.claude_code.shutil.which", return_value="/usr/local/bin/claude"),
                    mock.patch("ragtime.core.claude_code._run_claude_command", side_effect=probe_result),
                    mock.patch.dict(os.environ, {claude_code.CLAUDE_CODE_OAUTH_TOKEN_ENV: ""}, clear=False),
                ):
                    status = await claude_code._probe_claude_code_status()

                self.assertFalse(status.has_cli_auth)
                self.assertFalse(status.available)

    async def test_status_accepts_subscription_auth_or_oauth_environment_token(self) -> None:
        async def subscription_probe(*_args: str, **_kwargs: object) -> claude_code._ClaudeCommandResult:
            if _args[1:] == ("--version",):
                return claude_code._ClaudeCommandResult(returncode=0, output="2.1.290 (Claude Code)")
            return claude_code._ClaudeCommandResult(
                returncode=0,
                output='{"logged_in": true, "auth_method": "claude.ai", "subscription_type": "pro"}',
            )

        with (
            mock.patch("ragtime.core.claude_code.shutil.which", return_value="/usr/local/bin/claude"),
            mock.patch("ragtime.core.claude_code._run_claude_command", side_effect=subscription_probe),
            mock.patch.dict(os.environ, {claude_code.CLAUDE_CODE_OAUTH_TOKEN_ENV: ""}, clear=False),
        ):
            subscription_status = await claude_code._probe_claude_code_status()

        self.assertTrue(subscription_status.has_cli_auth)
        self.assertTrue(subscription_status.available)

    async def test_status_rejects_legacy_cli_version_before_auth_probe(self) -> None:
        async def old_version_probe(*_args: str, **_kwargs: object) -> claude_code._ClaudeCommandResult:
            self.assertEqual(_args[1:], ("--version",))
            return claude_code._ClaudeCommandResult(returncode=0, output="2.1.186 (Claude Code)")

        with (
            mock.patch("ragtime.core.claude_code.shutil.which", return_value="/usr/local/bin/claude"),
            mock.patch("ragtime.core.claude_code._run_claude_command", side_effect=old_version_probe),
            mock.patch.dict(os.environ, {claude_code.CLAUDE_CODE_OAUTH_TOKEN_ENV: "oauth-token"}, clear=False),
        ):
            status = await claude_code._probe_claude_code_status()

        self.assertFalse(status.available)
        self.assertIn("2.1.290 or later", status.error or "")

    async def test_status_accepts_oauth_environment_token_without_cli_login(self) -> None:
        async def token_probe(*_args: str, **_kwargs: object) -> claude_code._ClaudeCommandResult:
            if _args[1:] == ("--version",):
                return claude_code._ClaudeCommandResult(returncode=0, output="2.1.290 (Claude Code)")
            return claude_code._ClaudeCommandResult(returncode=0, output='{"loggedIn": false}')

        with (
            mock.patch("ragtime.core.claude_code.shutil.which", return_value="/usr/local/bin/claude"),
            mock.patch("ragtime.core.claude_code._run_claude_command", side_effect=token_probe),
            mock.patch.dict(os.environ, {claude_code.CLAUDE_CODE_OAUTH_TOKEN_ENV: "oauth-token"}, clear=False),
        ):
            status = await claude_code._probe_claude_code_status()

        self.assertTrue(status.has_oauth_token)
        self.assertFalse(status.has_cli_auth)
        self.assertTrue(status.available)

    def test_subscription_environment_filters_competing_auth_without_mutating_parent(self) -> None:
        parent_environment = {
            "HOME": "/home/ragtime",
            "CLAUDE_CONFIG_DIR": "/home/ragtime/.claude",
            claude_code.CLAUDE_CODE_OAUTH_TOKEN_ENV: "oauth-token",
            "ANTHROPIC_API_KEY": "api-key",
            "ANTHROPIC_AUTH_TOKEN": "auth-token",
            "ANTHROPIC_BASE_URL": "https://gateway.example",
            "ANTHROPIC_PROFILE": "work",
            "ANTHROPIC_FEDERATION_REGION": "example",
            "ANTHROPIC_VERTEX_PROJECT_ID": "project-id",
            "CLAUDE_CODE_BEDROCK_REGION": "us-east-1",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "CLAUDE_CODE_USE_VERTEX": "1",
            "CLAUDE_CODE_USE_FOUNDRY": "1",
            "CLAUDE_CODE_DISABLE_ATTACHMENTS": "0",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "0",
            "DISABLE_TELEMETRY": "0",
            "DISABLE_ERROR_REPORTING": "0",
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "OTEL_LOG_USER_PROMPTS": "1",
            "CLAUDE_CODE_DEBUG_LOGS_DIR": "/tmp/claude-debug",
            "DEBUG": "1",
        }

        environment = claude_code.build_claude_code_subscription_env(parent_environment)

        self.assertEqual(environment["HOME"], parent_environment["HOME"])
        self.assertEqual(environment["CLAUDE_CONFIG_DIR"], parent_environment["CLAUDE_CONFIG_DIR"])
        self.assertEqual(environment[claude_code.CLAUDE_CODE_OAUTH_TOKEN_ENV], "oauth-token")
        self.assertFalse(any(key.startswith("ANTHROPIC_") for key in environment))
        self.assertFalse(any(key.startswith("CLAUDE_CODE_BEDROCK_") for key in environment))
        self.assertFalse(any(key.startswith("CLAUDE_CODE_USE_") for key in environment))
        self.assertEqual(environment["CLAUDE_CODE_DISABLE_ATTACHMENTS"], "1")
        self.assertEqual(environment["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"], "1")
        self.assertEqual(environment["DISABLE_TELEMETRY"], "1")
        self.assertEqual(environment["DISABLE_ERROR_REPORTING"], "1")
        self.assertNotIn("CLAUDE_CODE_ENABLE_TELEMETRY", environment)
        self.assertNotIn("OTEL_LOG_USER_PROMPTS", environment)
        self.assertNotIn("CLAUDE_CODE_DEBUG_LOGS_DIR", environment)
        self.assertNotIn("DEBUG", environment)
        self.assertEqual(parent_environment["ANTHROPIC_API_KEY"], "api-key")
        self.assertEqual(parent_environment["CLAUDE_CODE_DISABLE_ATTACHMENTS"], "0")
        self.assertEqual(parent_environment["DEBUG"], "1")


if __name__ == "__main__":
    unittest.main()
