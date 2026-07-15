"""Focused logging hardening tests."""

import logging
import unittest
from unittest import mock

from ragtime.core import logging as logging_module


class AgentAccessLoggingTests(unittest.TestCase):
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
