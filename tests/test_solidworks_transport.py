import asyncio
import json
import unittest
from pathlib import Path
from unittest import mock

from ragtime.core.ssh import SSHResult
from ragtime.solidworks.protocol import build_request, encode_request, error_response
from ragtime.solidworks.transport import SolidworksTransportError, SshSolidworksTransport, build_ssh_config


class SolidworksTransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.connection = {
            "host": "windows",
            "port": 2222,
            "user": "alice",
            "password": "secret",
            "key_path": "key",
            "key_content": "content",
            "key_passphrase": "pass",
            "pairing_id": "pair",
            "transport": "tunnel",
            "host_keys": ["key"],
        }
        self.transport = SshSolidworksTransport({"connection_config": self.connection})

    def test_config_mapping(self):
        config = build_ssh_config(self.connection, timeout=17)
        for name, value in self.connection.items():
            self.assertEqual(getattr(config, name), value)
        self.assertEqual(config.shell, "powershell")
        self.assertEqual(config.timeout, 17)

    async def test_call_envelope_wins_and_command_is_exact(self):
        request = build_request("status", {}, {})
        envelope = error_response("status", "host_not_running", "no host")
        result = SSHResult("banner\n" + json.dumps(envelope), "secret", 2, False)
        with mock.patch("ragtime.solidworks.transport.execute_ssh_command", return_value=result) as execute:
            self.assertEqual(await self.transport.call(request, timeout_seconds=30), envelope)
        config, command = execute.call_args.args
        self.assertEqual(config.timeout, 30)
        self.assertEqual(
            command,
            "& (Join-Path $env:LOCALAPPDATA 'RagtimeSolidWorks\\host\\ragtime-solidworks-host.exe') call --request-b64 '" + encode_request(request) + "'",
        )

    async def test_call_classifies_failures_without_exposing_stderr(self):
        cases = [
            ("SSH authentication failed: secret", "ssh_auth_failed"),
            ("SSH connection error: secret", "ssh_unreachable"),
            ("[Command execution timed out after 5 seconds] secret", "timeout"),
            ("[RAGTIME_REMOTE_TIMEOUT] secret", "timeout"),
            ("host produced garbage secret", "no_response"),
        ]
        for stderr, kind in cases:
            with (
                self.subTest(kind=kind),
                mock.patch(
                    "ragtime.solidworks.transport.execute_ssh_command",
                    return_value=SSHResult("", stderr, -1 if kind != "no_response" else 2, False),
                ),
            ):
                with self.assertRaises(SolidworksTransportError) as caught:
                    await self.transport.call(build_request("status", {}, {}), timeout_seconds=5)
                self.assertEqual(caught.exception.kind, kind)
                self.assertNotIn("secret", str(caught.exception))

    def test_config_sets_total_deadline_and_output_cap(self):
        config = build_ssh_config(self.connection, timeout=12)
        self.assertEqual(config.total_timeout, 12)
        self.assertEqual(config.max_output_bytes, 1024 * 1024)

    def test_config_rejects_nonpositive_timeout(self):
        for timeout in (0, -1):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                build_ssh_config(self.connection, timeout=timeout)

    async def test_output_limit_marker_is_reported_as_too_much_output(self):
        result = SSHResult("", "[RAGTIME_OUTPUT_LIMIT] Command output exceeded 10 bytes and was stopped.", -1, False)
        with mock.patch("ragtime.solidworks.transport.execute_ssh_command", return_value=result):
            with self.assertRaises(SolidworksTransportError) as caught:
                await self.transport.call(build_request("status", {}, {}), timeout_seconds=5)
        self.assertEqual(caught.exception.kind, "no_response")
        self.assertTrue(caught.exception.output_limit)
        self.assertEqual(str(caught.exception), "The SolidWorks host returned too much output.")

    async def test_run_powershell_returns_result(self):
        result = SSHResult("ok", "", 0, True)
        with mock.patch("ragtime.solidworks.transport.execute_ssh_command", return_value=result):
            self.assertIs(await self.transport.run_powershell("echo ok", timeout_seconds=3), result)

    async def test_upload_closes_sftp_and_client_on_success_and_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure):
                client = mock.MagicMock()
                sftp = client.open_sftp.return_value
                if failure:
                    sftp.put.side_effect = OSError("failure")
                with mock.patch("ragtime.solidworks.transport._create_ssh_client", return_value=client):
                    if failure:
                        with self.assertRaises(OSError):
                            await self.transport.upload_file(Path("local.zip"), "C:/staging/file.zip", timeout_seconds=2)
                    else:
                        await self.transport.upload_file(Path("local.zip"), "C:/staging/file.zip", timeout_seconds=2)
                sftp.put.assert_called_once_with("local.zip", "C:/staging/file.zip")
                sftp.close.assert_called_once()
                client.close.assert_called()

    async def test_upload_rejects_backslash(self):
        with self.assertRaises(ValueError):
            await self.transport.upload_file(Path("local.zip"), "C:\\file.zip", timeout_seconds=1)

    async def test_upload_closes_client_if_sftp_open_fails(self):
        client = mock.MagicMock()
        client.open_sftp.side_effect = OSError("failure")
        with mock.patch("ragtime.solidworks.transport._create_ssh_client", return_value=client):
            with self.assertRaises(OSError):
                await self.transport.upload_file(Path("a"), "C:/a", timeout_seconds=1)
        client.close.assert_called_once()

    async def test_upload_timeout_is_enforced(self):
        import threading

        client = mock.MagicMock()
        finished = threading.Event()
        client.open_sftp.return_value.put.side_effect = lambda *args: finished.wait(1)
        with mock.patch("ragtime.solidworks.transport._create_ssh_client", return_value=client):
            try:
                with self.assertRaises(asyncio.TimeoutError):
                    await self.transport.upload_file(Path("a"), "C:/a", timeout_seconds=0.02)
            finally:
                finished.set()


if __name__ == "__main__":
    unittest.main()
