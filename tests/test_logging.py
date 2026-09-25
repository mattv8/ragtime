"""Focused logging hardening tests."""

import logging
import unittest
from unittest import mock

from ragtime.core import logging as logging_module


class AgentAccessLoggingTests(unittest.TestCase):
    def test_colored_formatter_renders_only_safe_content_protection_timing(self) -> None:
        record = logging.LogRecord(
            name="ragtime.content_protection.service",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="content_protection_timing",
            args=(),
            exc_info=None,
        )
        record.content_protection = {
            "direction": "inbound",
            "outcome": "permitted",
            "total_ms": 12.5,
            "payload": "must-not-be-logged",
        }

        formatted = logging_module.ColoredFormatter().format(record)

        self.assertIn('content_protection={"direction":"inbound","outcome":"permitted","total_ms":12.5}', formatted)
        self.assertNotIn("must-not-be-logged", formatted)

    def test_request_correlation_filter_uses_active_request_id_only(self) -> None:
        record = logging.makeLogRecord({"msg": "test"})
        with mock.patch.object(logging_module, "get_request_id", return_value=None):
            logging_module.RequestCorrelationFilter().filter(record)
        self.assertEqual(getattr(record, "request_id"), "-")

        correlated = logging.makeLogRecord({"msg": "test"})
        with mock.patch.object(logging_module, "get_request_id", return_value="generated-request-id"):
            logging_module.RequestCorrelationFilter().filter(correlated)
        self.assertEqual(getattr(correlated, "request_id"), "generated-request-id")

    def test_redact_agent_access_path_only_replaces_bearer_segment(self) -> None:
        self.assertEqual(
            logging_module.redact_agent_access_path("/agent/w/tok-secret/tasks/task-1/reply?full=true"),
            "/agent/w/[redacted]/tasks/task-1/reply?full=true",
        )

    def test_redact_agent_access_path_leaves_non_agent_paths_unchanged(self) -> None:
        self.assertEqual(
            logging_module.redact_agent_access_path("/indexes/userspace/workspaces/ws-1/agent-access"),
            "/indexes/userspace/workspaces/ws-1/agent-access",
        )

    def test_uvicorn_access_filter_redacts_agent_access_token_before_formatting(self) -> None:
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1", "GET", "/agent/w/tok-secret/tasks/task-1", "1.1", 200),
            exc_info=None,
        )

        allowed = logging_module.UvicornAccessFilter().filter(record)

        self.assertTrue(allowed)
        self.assertNotIn("tok-secret", record.getMessage())
        self.assertIn("/agent/w/[redacted]/tasks/task-1", record.getMessage())

    def test_uvicorn_access_filter_downgrades_trailing_slash_quiet_prefixes(self) -> None:
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1", "GET", "/sessions/runtime-1", "1.1", 200),
            exc_info=None,
        )

        with mock.patch.object(logging_module.settings, "debug_mode", False):
            allowed = logging_module.UvicornAccessFilter().filter(record)

        self.assertFalse(allowed)
        self.assertEqual(record.levelno, logging.DEBUG)


if __name__ == "__main__":
    unittest.main()
