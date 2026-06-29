"""Claude Code CLI status and adapter scaffolding."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

CLAUDE_CODE_OAUTH_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
CLAUDE_CODE_COMMAND = "claude"
CLAUDE_CODE_AUTH_SESSION_TTL_SECONDS = 10 * 60
_AUTH_URL_RE = re.compile(r"https://claude\.com/cai/oauth/authorize\?\S+")
_CLAUDE_STATUS_PROCESS_TIMEOUT_SECONDS = 5.0
_CLAUDE_AUTH_STATUS_PROCESS_TIMEOUT_SECONDS = 10.0
_PROCESS_TERMINATE_GRACE_SECONDS = 2.0
_TIMEOUT_EXCEPTIONS = (TimeoutError, asyncio.TimeoutError)
_TIMEOUT_OR_CANCEL_EXCEPTIONS = (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError)

# Anthropic REST endpoint and headers for OAuth (Claude Code subscription) calls.
# This is the same public /v1/models endpoint used for API-key auth, but the
# Claude Code subscription token authenticates with a Bearer header plus the
# documented OAuth beta header instead of x-api-key.
ANTHROPIC_API_BASE = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_OAUTH_BETA = "oauth-2025-04-20"
_CLAUDE_CREDENTIALS_FILE = Path.home() / ".claude" / ".credentials.json"
_KEYCHAIN_SERVICE = "Claude Code-credentials"
_STATUS_IN_FLIGHT: tuple[asyncio.AbstractEventLoop, asyncio.Task["ClaudeCodeStatus"]] | None = None


@dataclass(frozen=True)
class ClaudeCodeStatus:
    """Readiness details for the installed Claude Code CLI."""

    installed: bool
    command: str | None
    version: str | None
    has_oauth_token: bool
    has_cli_auth: bool
    auth_method: str | None
    subscription_type: str | None
    available: bool
    error: str | None = None


@dataclass
class ClaudeCodeLoginSession:
    """A running Claude Code login process waiting for an OAuth code."""

    process: asyncio.subprocess.Process
    authorization_url: str
    state: str | None
    expires_at: float


@dataclass(frozen=True)
class ClaudeCodeLoginStart:
    """Non-sensitive details needed by the frontend to complete login."""

    authorization_url: str
    state: str | None
    expires_in: int


@dataclass(frozen=True)
class _ClaudeCommandResult:
    returncode: int | None
    output: str
    timed_out: bool = False
    error: str | None = None


async def _terminate_process(process: asyncio.subprocess.Process, *, process_group: bool = False) -> None:
    """Terminate a CLI process and reap it so status probes cannot leak children."""
    if process.returncode is not None:
        return

    pid = getattr(process, "pid", None)
    if process_group and os.name != "nt" and isinstance(pid, int) and pid > 0:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGTERM)
    else:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()

    try:
        await asyncio.wait_for(process.wait(), timeout=_PROCESS_TERMINATE_GRACE_SECONDS)
        return
    except _TIMEOUT_EXCEPTIONS:
        pass

    if process_group and os.name != "nt" and isinstance(pid, int) and pid > 0:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)
    else:
        with contextlib.suppress(ProcessLookupError):
            process.kill()

    with contextlib.suppress(*_TIMEOUT_EXCEPTIONS):
        await asyncio.wait_for(process.wait(), timeout=_PROCESS_TERMINATE_GRACE_SECONDS)


async def _run_claude_command(command: str, *args: str, timeout: float) -> _ClaudeCommandResult:
    """Run a short Claude CLI probe with hard timeout and guaranteed cleanup."""
    process: asyncio.subprocess.Process | None = None
    subprocess_kwargs: dict[str, Any] = {}
    use_process_group = os.name != "nt"
    if os.name != "nt":
        subprocess_kwargs["start_new_session"] = True

    try:
        process = await asyncio.create_subprocess_exec(
            command,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **subprocess_kwargs,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except _TIMEOUT_EXCEPTIONS:
            await _terminate_process(process, process_group=use_process_group)
            return _ClaudeCommandResult(returncode=process.returncode, output="", timed_out=True)
        output = (stdout or stderr).decode(errors="replace").strip()
        return _ClaudeCommandResult(returncode=process.returncode, output=output)
    except asyncio.CancelledError:
        if process is not None:
            await _terminate_process(process, process_group=use_process_group)
        raise
    except OSError as exc:
        return _ClaudeCommandResult(returncode=None, output="", error=str(exc))


async def get_claude_code_status() -> ClaudeCodeStatus:
    """Return non-sensitive Claude Code CLI/token status.

    Concurrent callers share only the in-flight live probe. There is no TTL
    auth cache, so completed status checks never mask later auth changes.
    """
    global _STATUS_IN_FLIGHT

    loop = asyncio.get_running_loop()
    current = _STATUS_IN_FLIGHT
    if current is not None:
        task_loop, task = current
        if task_loop is loop and not task.done():
            return await asyncio.shield(task)

    task = loop.create_task(_probe_claude_code_status())
    _STATUS_IN_FLIGHT = (loop, task)

    def _clear_in_flight(done_task: asyncio.Task[ClaudeCodeStatus]) -> None:
        global _STATUS_IN_FLIGHT
        if _STATUS_IN_FLIGHT == (loop, done_task):
            _STATUS_IN_FLIGHT = None
        if not done_task.cancelled():
            with contextlib.suppress(Exception):
                done_task.exception()

    task.add_done_callback(_clear_in_flight)
    try:
        return await asyncio.shield(task)
    finally:
        if _STATUS_IN_FLIGHT == (loop, task) and task.done():
            _STATUS_IN_FLIGHT = None


async def _probe_claude_code_status() -> ClaudeCodeStatus:
    """Return non-sensitive Claude Code CLI/token status."""
    command = shutil.which(CLAUDE_CODE_COMMAND)
    has_oauth_token = bool(os.getenv(CLAUDE_CODE_OAUTH_TOKEN_ENV, "").strip())

    if not command:
        return ClaudeCodeStatus(
            installed=False,
            command=None,
            version=None,
            has_oauth_token=has_oauth_token,
            has_cli_auth=False,
            auth_method=None,
            subscription_type=None,
            available=False,
            error="Claude Code CLI is not installed or not on PATH.",
        )

    version: str | None = None
    error: str | None = None
    version_result = await _run_claude_command(command, "--version", timeout=_CLAUDE_STATUS_PROCESS_TIMEOUT_SECONDS)
    if version_result.timed_out:
        error = "claude --version timed out."
    elif version_result.error:
        error = version_result.error
    else:
        output = version_result.output
        version = output.splitlines()[0] if output else None
        if version_result.returncode not in (0, None):
            error = output or f"claude --version exited with status {version_result.returncode}."

    has_cli_auth = False
    auth_method: str | None = None
    subscription_type: str | None = None
    auth_error: str | None = None
    if not error:
        auth_result = await _run_claude_command(command, "auth", "status", "--json", timeout=_CLAUDE_AUTH_STATUS_PROCESS_TIMEOUT_SECONDS)
        if auth_result.timed_out:
            auth_error = "claude auth status timed out."
        elif auth_result.error:
            auth_error = auth_result.error
        else:
            output = auth_result.output
            if auth_result.returncode in (0, None):
                has_cli_auth = True
                try:
                    payload = json.loads(output) if output else {}
                except json.JSONDecodeError:
                    payload = {}
                if isinstance(payload, dict):
                    auth_method = str(payload.get("authMethod") or payload.get("auth_method") or "").strip() or None
                    subscription_type = str(payload.get("subscriptionType") or payload.get("subscription_type") or "").strip() or None
            elif output:
                auth_error = output

    return ClaudeCodeStatus(
        installed=True,
        command=command,
        version=version,
        has_oauth_token=has_oauth_token,
        has_cli_auth=has_cli_auth,
        auth_method=auth_method,
        subscription_type=subscription_type,
        available=bool((has_oauth_token or has_cli_auth) and not error),
        error=error or (None if has_oauth_token else auth_error),
    )


def _token_from_credentials_payload(payload: Any) -> str | None:
    """Extract a non-expired access token from a Claude credentials payload."""
    if not isinstance(payload, dict):
        return None
    oauth = payload.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        return None
    token = str(oauth.get("accessToken") or "").strip()
    if not token:
        return None
    expires_at = oauth.get("expiresAt")
    if isinstance(expires_at, (int, float)) and expires_at > 0:
        # expiresAt is epoch milliseconds; treat as expired with a small skew.
        if expires_at / 1000.0 <= time.time() + 60:
            return None
    return token


def _read_credentials_file_token() -> str | None:
    try:
        if not _CLAUDE_CREDENTIALS_FILE.is_file():
            return None
        payload = json.loads(_CLAUDE_CREDENTIALS_FILE.read_text())
    except (OSError, ValueError):
        return None
    return _token_from_credentials_payload(payload)


async def _read_keychain_token() -> str | None:
    """Read the Claude Code OAuth token from the macOS keychain (dev hosts)."""
    if sys.platform != "darwin":
        return None
    security = shutil.which("security")
    if not security:
        return None
    try:
        process = await asyncio.create_subprocess_exec(
            security,
            "find-generic-password",
            "-s",
            _KEYCHAIN_SERVICE,
            "-w",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5.0)
    except (OSError, TimeoutError):
        return None
    if process.returncode not in (0, None):
        return None
    raw = (stdout or b"").decode(errors="replace").strip()
    if not raw:
        return None
    try:
        return _token_from_credentials_payload(json.loads(raw))
    except ValueError:
        return None


async def get_claude_code_oauth_token() -> str | None:
    """Resolve a usable Claude Code OAuth access token.

    Resolution order mirrors how Claude Code itself stores credentials:
    1. ``CLAUDE_CODE_OAUTH_TOKEN`` environment variable (preferred for
       containers; produced by ``claude setup-token``).
    2. The CLI credentials file (``~/.claude/.credentials.json``).
    3. The macOS keychain entry created by ``claude auth login`` (dev hosts).

    Returns ``None`` when no non-expired token is available.
    """
    env_token = os.getenv(CLAUDE_CODE_OAUTH_TOKEN_ENV, "").strip()
    if env_token:
        return env_token

    file_token = _read_credentials_file_token()
    if file_token:
        return file_token

    return await _read_keychain_token()


def build_claude_code_oauth_headers(token: str) -> dict[str, str]:
    """Build Anthropic REST headers for a Claude Code subscription OAuth token."""
    return {
        "Authorization": f"Bearer {token}",
        "anthropic-version": ANTHROPIC_VERSION,
        "anthropic-beta": ANTHROPIC_OAUTH_BETA,
    }


def extract_claude_code_oauth_code(value: str) -> str:
    """Extract the Claude OAuth code from a pasted callback URL or raw code."""
    raw = (value or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    if parsed.query:
        code = parse_qs(parsed.query).get("code", [""])[0].strip()
        if code:
            return code
    return raw


async def start_claude_code_login() -> tuple[ClaudeCodeLoginStart, ClaudeCodeLoginSession]:
    """Start `claude auth login` and return the authorization URL it prints."""
    command = shutil.which(CLAUDE_CODE_COMMAND)
    if not command:
        raise RuntimeError("Claude Code CLI is not installed or not on PATH.")

    process = await asyncio.create_subprocess_exec(
        command,
        "auth",
        "login",
        "--claudeai",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    output = bytearray()
    authorization_url: str | None = None
    try:
        deadline = asyncio.get_running_loop().time() + 15.0
        while asyncio.get_running_loop().time() < deadline:
            if process.stdout is None:
                break
            try:
                chunk = await asyncio.wait_for(process.stdout.read(1), timeout=max(0.1, deadline - asyncio.get_running_loop().time()))
            except _TIMEOUT_EXCEPTIONS:
                break
            if not chunk:
                break
            output.extend(chunk)
            text = output.decode(errors="replace")
            match = _AUTH_URL_RE.search(text)
            if match:
                authorization_url = match.group(0)
                if "Paste code here" in text:
                    break
        if not authorization_url:
            text = output.decode(errors="replace").strip()
            raise RuntimeError(text or "Claude Code login did not return an authorization URL.")
    except Exception:
        if process.returncode is None:
            process.kill()
            await process.wait()
        raise

    parsed = urlparse(authorization_url)
    state = parse_qs(parsed.query).get("state", [None])[0]
    expires_in = CLAUDE_CODE_AUTH_SESSION_TTL_SECONDS
    session = ClaudeCodeLoginSession(
        process=process,
        authorization_url=authorization_url,
        state=state,
        expires_at=asyncio.get_running_loop().time() + expires_in,
    )
    return ClaudeCodeLoginStart(authorization_url=authorization_url, state=state, expires_in=expires_in), session


async def complete_claude_code_login(session: ClaudeCodeLoginSession, pasted_code_or_url: str) -> str:
    """Feed the OAuth code to a waiting Claude Code login process."""
    code = extract_claude_code_oauth_code(pasted_code_or_url)
    if not code:
        raise ValueError("Paste the Claude callback URL or authorization code.")
    if session.process.returncode is not None:
        raise RuntimeError("Claude Code login session is no longer running. Start authorization again.")
    if session.process.stdin is None:
        raise RuntimeError("Claude Code login session is not accepting input. Start authorization again.")

    try:
        session.process.stdin.write(f"{code}\n".encode())
        await session.process.stdin.drain()
        session.process.stdin.close()
        if session.process.stdout is None:
            await asyncio.wait_for(session.process.wait(), timeout=60.0)
            return ""
        output = await asyncio.wait_for(session.process.stdout.read(), timeout=60.0)
        await asyncio.wait_for(session.process.wait(), timeout=5.0)
    except _TIMEOUT_OR_CANCEL_EXCEPTIONS:
        await _terminate_process(session.process)
        raise
    except Exception:
        await _terminate_process(session.process)
        raise
    text = output.decode(errors="replace").strip()
    if session.process.returncode not in (0, None):
        raise RuntimeError(text or f"Claude Code login exited with status {session.process.returncode}.")
    return text


class ClaudeCodeCLIAdapter:
    """Placeholder boundary for future Claude Code CLI-backed LLM calls."""

    def __init__(self, *, command: str = CLAUDE_CODE_COMMAND, env: dict[str, str] | None = None) -> None:
        self.command = command
        self.env = env or {}

    def build_env(self) -> dict[str, str]:
        """Build a subprocess environment without manufacturing auth state."""
        merged = dict(os.environ)
        merged.update(self.env)
        return merged

    async def ainvoke(self, _messages: list[Any], **_kwargs: Any) -> Any:
        """Future chat adapter hook; intentionally not implemented in Phase 2."""
        raise NotImplementedError("Claude Code chat adapter is scaffolded but not implemented.")
