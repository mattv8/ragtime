import unittest
from unittest.mock import patch

from ragtime.core.ssh import RAGTIME_OUTPUT_LIMIT_MARKER, SSHConfig, execute_ssh_command

MiB = 1024 * 1024


class _ChunkedChannel:
    """Serves scripted stdout and stderr data like paramiko: each receive returns at most the requested size."""

    def __init__(self, stdout: list[bytes], stderr: list[bytes] | None = None, *, exit_ready_first: bool = False) -> None:
        self._stdout = memoryview(b"".join(stdout))
        self._stderr = memoryview(b"".join(stderr or []))
        self._exit_ready_first = exit_ready_first
        self.consumed = 0
        self.closed = False

    def exit_status_ready(self) -> bool:
        return self._exit_ready_first or (not self._stdout and not self._stderr)

    def recv_ready(self) -> bool:
        return bool(self._stdout)

    def recv(self, size: int) -> bytes:
        data, self._stdout = bytes(self._stdout[:size]), self._stdout[size:]
        self.consumed += len(data)
        return data

    def recv_stderr_ready(self) -> bool:
        return bool(self._stderr)

    def recv_stderr(self, size: int) -> bytes:
        data, self._stderr = bytes(self._stderr[:size]), self._stderr[size:]
        self.consumed += len(data)
        return data

    def recv_exit_status(self) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


class _FakeStdin:
    def __init__(self) -> None:
        self.channel = self

    def write(self, _data: str) -> None:
        return None

    def shutdown_write(self) -> None:
        return None


class _FakeClient:
    def __init__(self, channel: _ChunkedChannel) -> None:
        self._channel = channel
        self.closed = False

    def exec_command(self, _command: str, timeout: int | None):
        stream = type("Stream", (), {"channel": self._channel})()
        return _FakeStdin(), stream, stream

    def close(self) -> None:
        self.closed = True


class ExecuteSshCommandOutputCapTests(unittest.TestCase):
    def _run(self, channel: _ChunkedChannel, max_output_bytes: int | None):
        client = _FakeClient(channel)
        config = SSHConfig(host="host", user="user", password="pw", timeout=60, max_output_bytes=max_output_bytes)
        with patch("ragtime.core.ssh._create_ssh_client", return_value=client):
            result = execute_ssh_command(config, "echo")
        return result, client

    def test_output_at_or_under_cap_is_returned_unchanged(self) -> None:
        result, _ = self._run(_ChunkedChannel([b"aaaa", b"bbbb"]), max_output_bytes=8)
        self.assertTrue(result.success)
        self.assertEqual(result.stdout, "aaaabbbb")

    def test_one_byte_over_cap_stops_the_command_and_returns_marker(self) -> None:
        channel = _ChunkedChannel([b"aaaa", b"bbbb"])
        result, client = self._run(channel, max_output_bytes=7)
        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, -1)
        self.assertEqual(result.stdout, "")
        self.assertTrue(result.stderr.startswith(RAGTIME_OUTPUT_LIMIT_MARKER))
        self.assertTrue(channel.closed)
        self.assertTrue(client.closed)

    def test_cap_counts_stdout_and_stderr_together(self) -> None:
        result, _ = self._run(_ChunkedChannel([b"aaaa"], [b"bbbb"]), max_output_bytes=7)
        self.assertTrue(result.stderr.startswith(RAGTIME_OUTPUT_LIMIT_MARKER))

    def test_no_cap_keeps_existing_behaviour(self) -> None:
        result, _ = self._run(_ChunkedChannel([b"a" * 9000]), max_output_bytes=None)
        self.assertTrue(result.success)
        self.assertEqual(len(result.stdout), 9000)

    def test_no_cap_drains_everything_after_exit_status(self) -> None:
        result, _ = self._run(_ChunkedChannel([b"a" * (4 * MiB)], exit_ready_first=True), max_output_bytes=None)
        self.assertTrue(result.success)
        self.assertEqual(len(result.stdout), 4 * MiB)

    def test_exit_status_ready_with_buffered_stdout_stops_at_cap_during_drain(self) -> None:
        channel = _ChunkedChannel([b"a" * (4 * MiB)], exit_ready_first=True)
        result, client = self._run(channel, max_output_bytes=MiB)
        self._assert_stopped_at_cap(result, channel, client, MiB)

    def test_exit_status_ready_with_buffered_stderr_stops_at_cap_during_drain(self) -> None:
        channel = _ChunkedChannel([], [b"e" * (4 * MiB)], exit_ready_first=True)
        result, client = self._run(channel, max_output_bytes=MiB)
        self._assert_stopped_at_cap(result, channel, client, MiB)

    def _assert_stopped_at_cap(self, result, channel: _ChunkedChannel, client: _FakeClient, cap: int) -> None:
        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, -1)
        self.assertTrue(result.stderr.startswith(RAGTIME_OUTPUT_LIMIT_MARKER))
        self.assertTrue(channel.closed)
        self.assertTrue(client.closed)
        # Reading stops at the first chunk that crosses the cap, so at most one receive chunk is over it.
        self.assertGreater(channel.consumed, cap)
        self.assertLessEqual(channel.consumed, cap + 4096)


if __name__ == "__main__":
    unittest.main()
