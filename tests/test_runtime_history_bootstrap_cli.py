from __future__ import annotations

import io
import ipaddress
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from typing import Protocol
from urllib.parse import urlsplit
from uuid import UUID

from runtime.worker.sqlite_history import bootstrap_cli


class _RequestTransport(Protocol):
    def request(self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]: ...


class _Transport:
    def __init__(self, responses: list[tuple[int, dict[str, object]]]) -> None:
        self.responses = list(responses)
        self.calls: list[SimpleNamespace] = []

    def request(self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]:
        self.calls.append(SimpleNamespace(method=method, url=url, headers=headers, body=body, timeout=timeout))
        status, payload = self.responses.pop(0)
        return status, json.dumps(payload).encode()


class _FailingPostTransport(_Transport):
    def request(self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]:
        self.calls.append(SimpleNamespace(method=method, url=url, headers=headers, body=body, timeout=timeout))
        if method == "POST":
            raise OSError("connection lost")
        status, payload = self.responses.pop(0)
        return status, json.dumps(payload).encode()


class _OutputObservingTransport(_Transport):
    def __init__(self, stdout: io.StringIO, responses: list[tuple[int, dict[str, object]]]) -> None:
        super().__init__(responses)
        self.stdout = stdout
        self.output_before_request = ""

    def request(self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]:
        self.output_before_request = self.stdout.getvalue()
        return super().request(method, url, headers, body, timeout)


class RuntimeHistoryBootstrapCliTests(unittest.TestCase):
    def _run(self, argv: list[str], transport: _RequestTransport, *, environment: dict[str, str] | None = None) -> tuple[int, list[dict[str, object]], str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            exit_code = bootstrap_cli.main(
                argv,
                transport=transport,
                environ=environment or {"RUNTIME_AUTH_TOKEN": "secret-token"},
                stdout=stdout,
                stderr=stderr,
                sleep=lambda _: None,
                monotonic=lambda: 0.0,
            )
        except SystemExit as error:
            self.assertIsNotNone(error.code)
            assert error.code is not None
            exit_code = int(error.code)
        return exit_code, [json.loads(line) for line in stdout.getvalue().splitlines()], stderr.getvalue()

    def test_default_inventory_is_get_only_and_repeats_workspace_filters(self) -> None:
        transport = _Transport([(200, {"version": 1, "workspaces": []})])

        exit_code, output, _ = self._run(["--workspace-id", "one", "--workspace-id", "two"], transport)

        self.assertEqual(0, exit_code)
        self.assertEqual([{"version": 1, "workspaces": []}], output)
        self.assertEqual("GET", transport.calls[0].method)
        self.assertTrue(transport.calls[0].url.endswith("/sqlite-history/bootstrap/inventory?workspace_id=one&workspace_id=two"))
        self.assertIsNone(transport.calls[0].body)

    def test_inventory_and_acceptance_honor_the_configured_request_timeout(self) -> None:
        run_id = "553b6b91-00bb-44d1-8680-0e81533401d1"
        cases = [
            ([], _Transport([(200, {"version": 1})])),
            (["--apply", "--user-id", "local:admin", "--run-id", run_id, "--no-wait"], _Transport([(202, {"run_id": run_id, "status": "accepted"})])),
        ]

        for arguments, transport in cases:
            with self.subTest(arguments=arguments):
                exit_code, _, _ = self._run([*arguments, "--timeout-seconds", "123.5"], transport)
                self.assertEqual(0, exit_code)
                self.assertEqual(123.5, transport.calls[0].timeout)

    def test_apply_posts_expected_payload_then_observes_same_run(self) -> None:
        run_id = "553b6b91-00bb-44d1-8680-0e81533401d1"
        transport = _Transport(
            [
                (202, {"run_id": run_id, "status": "accepted"}),
                (200, {"run_id": run_id, "status": "completed"}),
            ]
        )

        exit_code, output, _ = self._run(["--apply", "--user-id", "local:admin", "--run-id", run_id, "--workspace-id", "one"], transport)

        self.assertEqual(0, exit_code)
        self.assertEqual({"event": "submitting", "run_id": run_id}, output[0])
        self.assertEqual(["accepted", "completed"], [record["status"] for record in output[1:]])
        self.assertEqual(["POST", "GET"], [call.method for call in transport.calls])
        self.assertEqual(
            {"run_id": run_id, "user_id": "local:admin", "workspace_ids": ["one"]},
            json.loads(transport.calls[0].body or b"{}"),
        )

    def test_apply_prints_the_generated_run_id_before_submitting(self) -> None:
        stdout = io.StringIO()
        transport = _OutputObservingTransport(stdout, [(202, {"run_id": "553b6b91-00bb-44d1-8680-0e81533401d1", "status": "accepted"})])

        exit_code = bootstrap_cli.main(
            ["--apply", "--user-id", "local:admin", "--run-id", "553b6b91-00bb-44d1-8680-0e81533401d1", "--no-wait"],
            transport=transport,
            environ={"RUNTIME_AUTH_TOKEN": "secret-token"},
            stdout=stdout,
            stderr=io.StringIO(),
        )

        self.assertEqual(0, exit_code)
        self.assertEqual({"event": "submitting", "run_id": "553b6b91-00bb-44d1-8680-0e81533401d1"}, json.loads(transport.output_before_request))

    def test_resume_retry_failed_posts_only_retry_flag(self) -> None:
        run_id = "553b6b91-00bb-44d1-8680-0e81533401d1"
        transport = _Transport([(200, {"run_id": run_id, "status": "completed"})])

        exit_code, output, _ = self._run(["--resume", run_id, "--retry-failed", "--no-wait"], transport)

        self.assertEqual(0, exit_code)
        self.assertEqual("completed", output[0]["status"])
        self.assertEqual("POST", transport.calls[0].method)
        self.assertEqual({"retry_failed": True}, json.loads(transport.calls[0].body or b"{}"))
        self.assertTrue(transport.calls[0].url.endswith(f"/sqlite-history/bootstrap/{run_id}/resume"))

    def test_auth_token_is_only_sent_as_authorization_header(self) -> None:
        token = "unsafe-token-value"
        transport = _Transport([(200, {"version": 1})])

        exit_code, output, diagnostics = self._run([], transport, environment={"RUNTIME_AUTH_TOKEN": token})

        self.assertEqual(0, exit_code)
        self.assertEqual([{"version": 1}], output)
        self.assertEqual(f"Bearer {token}", transport.calls[0].headers["Authorization"])
        self.assertNotIn(token, transport.calls[0].url)
        self.assertNotIn(token, diagnostics)

    def test_cancel_posts_an_empty_object_to_the_run_endpoint(self) -> None:
        run_id = "553b6b91-00bb-44d1-8680-0e81533401d1"
        transport = _Transport([(200, {"run_id": run_id, "status": "cancelled"})])

        exit_code, output, _ = self._run(["--cancel", run_id], transport)

        self.assertNotEqual(0, exit_code)
        self.assertEqual("cancelled", output[0]["status"])
        self.assertEqual("POST", transport.calls[0].method)
        self.assertEqual({}, json.loads(transport.calls[0].body or b"{}"))
        self.assertTrue(transport.calls[0].url.endswith(f"/sqlite-history/bootstrap/{run_id}/cancel"))

    def test_invalid_http_url_and_conflicting_modes_are_rejected_before_transport(self) -> None:
        transport = _Transport([])

        exit_code, _, _ = self._run(["--url", "http://example.test", "--apply", "--status", "553b6b91-00bb-44d1-8680-0e81533401d1"], transport)

        self.assertNotEqual(0, exit_code)
        self.assertEqual([], transport.calls)

    def test_localhost_http_is_canonicalized_to_a_numeric_loopback_address(self) -> None:
        transport = _Transport([(200, {"version": 1})])

        exit_code, _, _ = self._run(["--url", "http://localhost:8090"], transport)

        self.assertEqual(0, exit_code)
        self.assertTrue(ipaddress.ip_address(urlsplit(transport.calls[0].url).hostname or "").is_loopback)

    def test_run_id_workspace_filters_and_nonfinite_timeouts_are_rejected_outside_apply(self) -> None:
        run_id = "553b6b91-00bb-44d1-8680-0e81533401d1"
        invalid_arguments = [
            ["--run-id", run_id],
            ["--resume", run_id, "--workspace-id", "one"],
            ["--status", run_id, "--workspace-id", "one"],
            ["--cancel", run_id, "--workspace-id", "one"],
            ["--timeout-seconds", "nan"],
            ["--timeout-seconds", "inf"],
            ["--timeout-seconds", "-inf"],
        ]

        for argv in invalid_arguments:
            with self.subTest(argv=argv):
                transport = _Transport([])
                exit_code, _, _ = self._run(argv, transport)
                self.assertNotEqual(0, exit_code)
                self.assertEqual([], transport.calls)

    def test_ambiguous_apply_acceptance_reports_generated_id_without_retrying(self) -> None:
        transport = _FailingPostTransport([])

        exit_code, output, diagnostics = self._run(["--apply", "--user-id", "local:admin", "--no-wait"], transport)

        self.assertNotEqual(0, exit_code)
        self.assertEqual(2, len(output))
        self.assertEqual("submitting", output[0]["event"])
        self.assertEqual("unknown", output[1]["status"])
        self.assertEqual(output[0]["run_id"], output[1]["run_id"])
        UUID(str(output[1]["run_id"]))
        self.assertIn("observe", str(output[1]["instruction"]).lower())
        self.assertEqual(1, len(transport.calls))
        self.assertNotIn("secret-token", diagnostics)

    def test_resume_transport_failure_does_not_issue_a_second_request(self) -> None:
        run_id = "553b6b91-00bb-44d1-8680-0e81533401d1"
        transport = _FailingPostTransport([])

        exit_code, output, _ = self._run(["--resume", run_id, "--no-wait"], transport)

        self.assertNotEqual(0, exit_code)
        self.assertEqual([], output)
        self.assertEqual(1, len(transport.calls))
        self.assertTrue(transport.calls[0].url.endswith(f"/sqlite-history/bootstrap/{run_id}/resume"))

    def test_transport_exception_does_not_echo_the_auth_token(self) -> None:
        class TokenEchoTransport:
            def request(self, method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> tuple[int, bytes]:
                raise OSError(headers["Authorization"])

        token = "do-not-print-this-token"
        exit_code, _, diagnostics = self._run([], TokenEchoTransport(), environment={"RUNTIME_AUTH_TOKEN": token})

        self.assertNotEqual(0, exit_code)
        self.assertNotIn(token, diagnostics)

    def test_redirect_target_never_receives_authorization_header(self) -> None:
        class TargetHandler(BaseHTTPRequestHandler):
            requests: list[dict[str, str]] = []

            def do_GET(self) -> None:  # noqa: N802
                type(self).requests.append(dict(self.headers))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, format: str, *args: object) -> None:
                return

        target = HTTPServer(("127.0.0.1", 0), TargetHandler)

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{target.server_port}/redirect-target")
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                return

        source = HTTPServer(("127.0.0.1", 0), RedirectHandler)
        threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in (source, target)]
        for thread in threads:
            thread.start()
        try:
            token = "redirect-secret"
            stdout = io.StringIO()
            diagnostics = io.StringIO()
            exit_code = bootstrap_cli.main(
                ["--url", f"http://localhost:{source.server_port}"],
                environ={"RUNTIME_AUTH_TOKEN": token},
                stdout=stdout,
                stderr=diagnostics,
            )
            self.assertNotEqual(0, exit_code)
            self.assertEqual([], TargetHandler.requests)
            self.assertNotIn(token, diagnostics.getvalue())
        finally:
            for server in (source, target):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()
