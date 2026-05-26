import unittest
from unittest.mock import patch

from ragtime.core.ssh import SSHConfig, execute_ssh_command


class _FakeChannel:
    def __init__(self, *, exit_status_ready: bool, exit_code: int = 0) -> None:
        self._exit_status_ready = exit_status_ready
        self._exit_code = exit_code
        self.closed = False

    def exit_status_ready(self) -> bool:
        return self._exit_status_ready

    def recv_ready(self) -> bool:
        return False

    def recv_stderr_ready(self) -> bool:
        return False

    def recv(self, _size: int) -> bytes:
        return b""

    def recv_stderr(self, _size: int) -> bytes:
        return b""

    def recv_exit_status(self) -> int:
        return self._exit_code

    def close(self) -> None:
        self.closed = True


class _FakeStream:
    def __init__(self, channel: _FakeChannel) -> None:
        self.channel = channel


class _FakeStdin:
    def __init__(self) -> None:
        self.channel = self
        self.writes: list[str] = []

    def write(self, data: str) -> None:
        self.writes.append(data)

    def shutdown_write(self) -> None:
        return None


class _FakeClient:
    def __init__(self, channel: _FakeChannel) -> None:
        self._channel = channel
        self.commands: list[tuple[str, int | None]] = []
        self.closed = False

    def exec_command(self, command: str, timeout: int):
        self.commands.append((command, timeout))
        stdin = _FakeStdin()
        stream = _FakeStream(self._channel)
        return stdin, stream, stream

    def close(self) -> None:
        self.closed = True


class ExecuteSshCommandTimeoutTests(unittest.TestCase):
    def test_remote_timeout_is_reported_as_failure(self) -> None:
        channel = _FakeChannel(exit_status_ready=True, exit_code=124)
        client = _FakeClient(channel)
        config = SSHConfig(host="host", user="user", password="pw", timeout=60)

        with patch("ragtime.core.ssh._create_ssh_client", return_value=client):
            result = execute_ssh_command(config, "journalctl -f")

        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, 124)
        self.assertIn("timed out after 60 seconds", result.stderr)
        executed_command, timeout = client.commands[0]
        self.assertEqual(timeout, 60)
        self.assertIn("timeout --signal=TERM --kill-after=5s 60s", executed_command)
        self.assertIn('"$command_shell" -lc \'eval "$1" & child=$!;', executed_command)
        self.assertIn("journalctl -f", executed_command)

    def test_timeout_wrapper_preserves_shell_for_multiline_commands(self) -> None:
        command = """set -o pipefail
echo '--- olm journal last 80 ---'
sudo journalctl -u olm.service --no-pager -n 80 || true"""
        channel = _FakeChannel(exit_status_ready=True, exit_code=0)
        client = _FakeClient(channel)
        config = SSHConfig(host="host", user="user", password="pw", timeout=60)

        with patch("ragtime.core.ssh._create_ssh_client", return_value=client):
            result = execute_ssh_command(config, command)

        self.assertTrue(result.success)
        executed_command, _timeout = client.commands[0]
        self.assertIn('command_shell="${SHELL:-/bin/sh}"', executed_command)
        self.assertIn('"$command_shell" -lc', executed_command)
        self.assertNotIn(" sh -lc ", executed_command)
        self.assertIn("set -o pipefail", executed_command)
        self.assertIn("journalctl -u olm.service --no-pager -n 80", executed_command)

    def test_local_timeout_fallback_is_reported_as_failure(self) -> None:
        channel = _FakeChannel(exit_status_ready=False)
        client = _FakeClient(channel)
        config = SSHConfig(host="host", user="user", password="pw", timeout=60)

        with (
            patch("ragtime.core.ssh._create_ssh_client", return_value=client),
            patch(
                "ragtime.core.ssh.logger",
            ),
            patch(
                "ragtime.core.ssh.time.sleep",
                return_value=None,
            ),
            patch(
                "ragtime.core.ssh.time.time",
                side_effect=[0.0, 0.0, 61.0],
            ),
        ):
            result = execute_ssh_command(config, "journalctl -f")

        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, -1)
        self.assertIn("timed out after 60 seconds", result.stderr)
        self.assertTrue(channel.closed)

    def test_unlimited_timeout_does_not_wrap_remote_command(self) -> None:
        channel = _FakeChannel(exit_status_ready=True, exit_code=0)
        client = _FakeClient(channel)
        config = SSHConfig(host="host", user="user", password="pw", timeout=0)

        with patch("ragtime.core.ssh._create_ssh_client", return_value=client):
            result = execute_ssh_command(config, "echo ok")

        self.assertTrue(result.success)
        executed_command, timeout = client.commands[0]
        self.assertEqual(executed_command, "echo ok")
        self.assertIsNone(timeout)


if __name__ == "__main__":
    unittest.main()
