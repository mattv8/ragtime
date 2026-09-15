from __future__ import annotations

import unittest

from runtime.manager.models import RuntimeExecRequest


class RuntimeExecTimeoutTests(unittest.TestCase):
    def test_exec_defaults_to_120_and_accepts_600(self) -> None:
        self.assertEqual(RuntimeExecRequest(command="true").timeout_seconds, 120)
        self.assertEqual(RuntimeExecRequest(command="true", timeout_seconds=600).timeout_seconds, 600)

    def test_exec_rejects_more_than_600(self) -> None:
        with self.assertRaises(Exception):
            RuntimeExecRequest(command="true", timeout_seconds=601)
