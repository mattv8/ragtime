from __future__ import annotations

import base64
import threading
import time
from types import SimpleNamespace

import pytest

from ragtime.core.ssh import SSHConfig, _wrap_powershell_command, execute_ssh_command


@pytest.mark.parametrize("timeout,total_timeout,delay,want_writes", [(0, None, 0, ["data"]), (30, 0.01, 0.03, [])])
def test_unlimited_legacy_calls_and_late_exec_payload(monkeypatch, timeout, total_timeout, delay, want_writes) -> None:
    writes: list[str] = []
    channel = SimpleNamespace(
        shutdown_write=lambda: None,
        close=lambda: None,
        exit_status_ready=lambda: True,
        recv_ready=lambda: False,
        recv_stderr_ready=lambda: False,
        recv_exit_status=lambda: 0,
    )

    class Client:
        def exec_command(self, *_args, **_kwargs):
            time.sleep(delay)
            return SimpleNamespace(write=writes.append, channel=channel), SimpleNamespace(channel=channel), SimpleNamespace(channel=channel)

        def close(self):
            return None

    monkeypatch.setattr("ragtime.core.ssh._create_ssh_client", lambda _config: Client())
    result = execute_ssh_command(SSHConfig(host="host", user="user", password="password", timeout=timeout, total_timeout=total_timeout), "command", "data")
    assert writes == want_writes
    assert result.success == (total_timeout is None)


def test_fractional_powershell_deadline_is_not_unlimited() -> None:
    command = _wrap_powershell_command(0.25)
    wrapper = base64.b64decode(command.split()[-1]).decode("utf-16le")
    assert "$deadlineSeconds = 0.25" in wrapper
    assert wrapper.index("$deadline =") < wrapper.index("$payload =")


def test_deadline_rechecked_before_script_write_after_slow_payload_write(monkeypatch) -> None:
    writes: list[str] = []

    def write(data):
        writes.append(data)
        time.sleep(0.03)

    class Client:
        def exec_command(self, *_args, **_kwargs):
            return SimpleNamespace(write=write, channel=SimpleNamespace(settimeout=lambda _value: None)), SimpleNamespace(), SimpleNamespace()

        def close(self):
            return None

    monkeypatch.setattr("ragtime.core.ssh._create_ssh_client", lambda _config: Client())
    result = execute_ssh_command(SSHConfig(host="host", user="user", password="password", shell="powershell", total_timeout=0.01), "wrapper", "script-mutation")
    assert len(writes) == 1
    assert "script-mutation" not in writes
    assert not result.success


def test_missing_exec_ack_is_interrupted_by_transport_deadline(monkeypatch) -> None:
    closed = threading.Event()
    wrote_payload: list[str] = []

    class Client:
        def exec_command(self, *_args, **_kwargs):
            assert closed.wait(timeout=0.5), "missing exec acknowledgment was not interrupted"
            return SimpleNamespace(write=wrote_payload.append), SimpleNamespace(), SimpleNamespace()

        def close(self):
            closed.set()

    monkeypatch.setattr("ragtime.core.ssh._create_ssh_client", lambda _config: Client())
    started = time.monotonic()
    result = execute_ssh_command(SSHConfig(host="host", user="user", password="password", shell="powershell", total_timeout=0.02), "command", "script")
    assert time.monotonic() - started < 0.3
    assert not result.success
    assert wrote_payload == []


def test_total_timeout_prevents_command_after_connection_uses_budget(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    executed = False

    class Client:
        def exec_command(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            nonlocal executed
            executed = True
            raise AssertionError("the command must not start after the total deadline")

        def close(self) -> None:
            return None

    def connect(_config):  # type: ignore[no-untyped-def]
        time.sleep(0.03)
        return Client()

    monkeypatch.setattr("ragtime.core.ssh._create_ssh_client", connect)

    result = execute_ssh_command(SSHConfig(host="host", user="user", password="password", timeout=30, total_timeout=0.01), "host mutation")

    assert not executed
    assert not result.success
    assert "timed out" in result.stderr


def test_total_timeout_bounds_command_after_successful_connection(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    class Channel:
        def shutdown_write(self) -> None:
            return None

        def exit_status_ready(self) -> bool:
            return False

        def recv_ready(self) -> bool:
            return False

        def recv_stderr_ready(self) -> bool:
            return False

        def close(self) -> None:
            return None

        def recv_exit_status(self) -> int:
            return -1

    class Input:
        channel = Channel()

        def write(self, _data: str) -> None:
            return None

    class Output:
        channel = Channel()

    class Client:
        def exec_command(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return Input(), Output(), Output()

        def close(self) -> None:
            return None

    monkeypatch.setattr("ragtime.core.ssh._create_ssh_client", lambda _config: Client())
    started = time.monotonic()
    result = execute_ssh_command(SSHConfig(host="host", user="user", password="password", timeout=30, total_timeout=0.02), "host mutation")

    assert not result.success
    assert time.monotonic() - started < 0.08
