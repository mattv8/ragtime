import logging
import socket
import threading
import unittest

from ragtime.core import ssh
from ragtime.core.ssh import SSHConfig, execute_ssh_command


class _SilentBannerServer:
    def __init__(self) -> None:
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self.port = self._listener.getsockname()[1]
        self.accepted = threading.Event()
        self.peer_closed = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            connection, _address = self._listener.accept()
            with connection:
                connection.settimeout(0.05)
                self.accepted.set()
                while not self._stop.is_set():
                    try:
                        if connection.recv(1) == b"":
                            self.peer_closed.set()
                            return
                    except socket.timeout:
                        continue
        finally:
            self._listener.close()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


class _RecordCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class SshConnectionLoggingTests(unittest.TestCase):
    def test_connect_kwargs_bound_all_paramiko_phases(self) -> None:
        kwargs = ssh._build_connect_kwargs(SSHConfig(host="host", user="user", password="pw", timeout=0))

        self.assertEqual(kwargs["timeout"], 30)
        self.assertEqual(kwargs["auth_timeout"], 30)
        self.assertEqual(kwargs["channel_timeout"], 30)
        self.assertLess(kwargs["banner_timeout"], kwargs["timeout"])
        self.assertIs(kwargs["transport_factory"], ssh._RagtimeTransport)

    def test_silent_banner_reports_banner_error_closes_connection_and_logs_one_traceback(self) -> None:
        server = _SilentBannerServer()
        collector = _RecordCollector()
        transport_logger = logging.getLogger("paramiko.transport")
        transport_logger.addHandler(collector)
        try:
            result = execute_ssh_command(
                SSHConfig(host="127.0.0.1", port=server.port, user="user", password="pw", timeout=1),
                "echo unused",
            )

            self.assertTrue(server.accepted.wait(1))
            self.assertIn("Error reading SSH protocol banner", result.stderr)
            self.assertNotIn("No existing session", result.stderr)
            self.assertTrue(server.peer_closed.wait(1))
            traceback_records = [record for record in collector.records if isinstance(record.msg, str) and "Traceback" in record.msg]
            self.assertEqual(len(traceback_records), 1)
            self.assertIn("\n", traceback_records[0].msg)
        finally:
            transport_logger.removeHandler(collector)
            server.close()

    def test_non_list_paramiko_log_message_and_args_are_unchanged(self) -> None:
        transport = ssh._RagtimeTransport(socket.socket())
        collector = _RecordCollector()
        transport.logger.addHandler(collector)
        try:
            transport._log(logging.ERROR, "connection to %s failed", "example.test")
        finally:
            transport.logger.removeHandler(collector)
            transport.close()

        self.assertEqual(len(collector.records), 1)
        self.assertEqual(collector.records[0].msg, "connection to %s failed")
        self.assertEqual(collector.records[0].args, ("example.test",))


if __name__ == "__main__":
    unittest.main()
