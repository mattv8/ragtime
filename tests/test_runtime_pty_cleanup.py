from __future__ import annotations

import importlib
import unittest
from typing import Any

from tests._runtime_cleanup_helpers import assert_process_group_termination

worker_api: Any | None
runtime_import_error: ImportError | None
try:
    worker_api = importlib.import_module("runtime.worker.api")
except ImportError as exc:
    worker_api = None
    runtime_import_error = exc
else:
    runtime_import_error = None


class RuntimePtyCleanupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        if worker_api is None:
            self.skipTest(f"runtime worker unavailable: {runtime_import_error}")

    async def test_terminates_entire_pty_process_group(self) -> None:
        assert worker_api is not None
        await assert_process_group_termination(
            self,
            terminate=worker_api._terminate_pty_process,
            expected_patch_base="runtime.worker.api",
        )

    async def test_escalates_pty_process_group_after_timeout(self) -> None:
        assert worker_api is not None
        await assert_process_group_termination(
            self,
            terminate=worker_api._terminate_pty_process,
            expected_patch_base="runtime.worker.api",
            timeout=0.01,
        )


if __name__ == "__main__":
    unittest.main()
