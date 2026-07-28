from __future__ import annotations

import importlib
import json
import os
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

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
        worker_api._pty_processes.clear()
        worker_api._pty_master_fds.clear()

    async def test_pty_route_spawns_through_sandbox_launcher_without_preexec(self) -> None:
        assert worker_api is not None

        class _DisconnectingWebSocket:
            def __init__(self) -> None:
                self.headers = {"x-pty-token": "token"}
                self.query_params = {}
                self.sent_texts: list[str] = []

            async def accept(self) -> None:
                return None

            async def close(self, *args: Any, **kwargs: Any) -> None:
                return None

            async def send_text(self, text: str) -> None:
                self.sent_texts.append(text)

            async def receive(self) -> dict[str, Any]:
                return {"type": "websocket.disconnect"}

        master_fd, slave_fd = os.openpty()
        process = SimpleNamespace(pid=1234, returncode=None)
        session = SimpleNamespace(
            sandbox_spec=SimpleNamespace(mode="pivot_root"),
            workspace_id="workspace-1",
        )
        service = SimpleNamespace(
            verify_pty_token=mock.AsyncMock(return_value=session),
            build_agent_process_environment=mock.Mock(return_value={"EXISTING": "1"}),
        )
        websocket = _DisconnectingWebSocket()
        spawn = mock.AsyncMock(return_value=process)

        with (
            mock.patch.object(worker_api, "get_worker_service", return_value=service),
            mock.patch.object(worker_api, "ensure_sandbox_ready"),
            mock.patch.object(worker_api, "_write_sandbox_init_file"),
            mock.patch.object(worker_api, "sandbox_env", side_effect=lambda _spec, env: dict(env)),
            mock.patch.object(worker_api.pty_module, "openpty", return_value=(master_fd, slave_fd)),
            mock.patch.object(worker_api, "spawn_sandboxed", spawn, create=True),
            mock.patch.object(worker_api, "_terminate_pty_process", new=mock.AsyncMock()),
        ):
            await worker_api.pty("worker-session-1", websocket)

        spawn.assert_awaited_once()
        self.assertTrue(spawn.await_args.kwargs["pty"])
        self.assertEqual(spawn.await_args.kwargs["stdin"], slave_fd)
        self.assertEqual(spawn.await_args.kwargs["stdout"], slave_fd)
        self.assertEqual(spawn.await_args.kwargs["stderr"], slave_fd)
        self.assertEqual(spawn.await_args.kwargs["env"]["TERM"], "xterm-256color")
        self.assertEqual(spawn.await_args.kwargs["env"]["PROMPT_COMMAND"], "")
        self.assertNotIn("preexec_fn", spawn.await_args.kwargs)
        self.assertEqual(
            json.loads(websocket.sent_texts[0]),
            {
                "type": "status",
                "message": "Runtime PTY bridge online",
                "read_only": False,
            },
        )

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
