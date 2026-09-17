import asyncio
import unittest
from unittest import mock

from ragtime.userspace.object_storage.legacy_migration import workspace_gc_fence
from ragtime.userspace.runtime_service import UserSpaceRuntimeService


class RuntimeGcFenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_fence_blocks_non_controller_runtime_start_path(self) -> None:
        service = UserSpaceRuntimeService()
        entered = asyncio.Event()

        async def start_after_admission(*args: object, **kwargs: object) -> str:
            entered.set()
            return "started"

        with mock.patch.object(service, "_ensure_session_row_unfenced", side_effect=start_after_admission):
            async with workspace_gc_fence("workspace"):
                start = asyncio.create_task(service._ensure_session_row("workspace", "user", auto_start=True))
                await asyncio.sleep(0)
                self.assertFalse(entered.is_set())
            self.assertEqual("started", await start)
        self.assertTrue(entered.is_set())
