from __future__ import annotations

import asyncio
import sys
import types
import unittest

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.userspace.runtime_service import UserSpaceRuntimeService


class _PreviewProbeService(UserSpaceRuntimeService):
    def __init__(self, results: list[bool]) -> None:
        super().__init__()
        self.results = list(results)
        self.calls: list[tuple[str, str]] = []

    async def _probe_public_preview_origin(self, preview_origin: str, probe_url: str) -> bool:
        self.calls.append((preview_origin, probe_url))
        return self.results.pop(0)

    def _log_preview_host_unreachable(self, preview_origin: str, probe_url: str) -> None:
        return None


class _BlockingPreviewProbeService(_PreviewProbeService):
    def __init__(self) -> None:
        super().__init__([True])
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def _probe_public_preview_origin(self, preview_origin: str, probe_url: str) -> bool:
        self.calls.append((preview_origin, probe_url))
        self.started.set()
        await self.release.wait()
        return True


class RuntimePreviewProbeCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_reuses_cached_probe_for_same_workspace_origin(self) -> None:
        service = _PreviewProbeService([False, True])

        first = await service._probe_public_preview_origin_cached(
            "https://workspace-a.ragtime.example.com",
        )
        second = await service._probe_public_preview_origin_cached(
            "https://workspace-a.ragtime.example.com",
        )

        self.assertFalse(first)
        self.assertFalse(second)
        self.assertEqual(len(service.calls), 1)

    async def test_failed_workspace_probe_does_not_poison_other_workspace_on_same_domain(self) -> None:
        service = _PreviewProbeService([False, True])

        first = await service._probe_public_preview_origin_cached(
            "https://workspace-a.ragtime.example.com",
        )
        second = await service._probe_public_preview_origin_cached(
            "https://workspace-b.ragtime.example.com",
        )

        self.assertFalse(first)
        self.assertTrue(second)
        self.assertEqual(
            [call[0] for call in service.calls],
            [
                "https://workspace-a.ragtime.example.com",
                "https://workspace-b.ragtime.example.com",
            ],
        )

    async def test_coalesces_concurrent_probe_for_same_workspace_origin(self) -> None:
        service = _BlockingPreviewProbeService()
        origin = "https://workspace-a.ragtime.example.com"

        first_task = asyncio.create_task(service._probe_public_preview_origin_cached(origin))
        second_task = asyncio.create_task(service._probe_public_preview_origin_cached(origin))
        await asyncio.wait_for(service.started.wait(), timeout=1.0)
        service.release.set()
        first, second = await asyncio.gather(first_task, second_task)

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(len(service.calls), 1)


if __name__ == "__main__":
    unittest.main()
