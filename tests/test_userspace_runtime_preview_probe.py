from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest import mock

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module

from ragtime.config import settings
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


class RuntimePreviewDomainTests(unittest.TestCase):
    def test_derives_preview_domain_from_external_url_before_any_launch(self) -> None:
        with (
            mock.patch.object(settings, "userspace_preview_base_domain", ""),
            mock.patch.object(settings, "external_base_url", "https://ragtime.hammerton.com"),
            mock.patch.object(settings, "debug_mode", False),
        ):
            service = UserSpaceRuntimeService()

            self.assertEqual(service.get_preview_base_domains(), {"ragtime.hammerton.com"})

    def test_prefers_explicit_preview_domain_over_external_url(self) -> None:
        with (
            mock.patch.object(settings, "userspace_preview_base_domain", "preview.example.com"),
            mock.patch.object(settings, "external_base_url", "https://ragtime.hammerton.com"),
            mock.patch.object(settings, "debug_mode", False),
        ):
            service = UserSpaceRuntimeService()

            self.assertEqual(service.get_preview_base_domains(), {"preview.example.com"})


class RuntimePreviewLaunchDescriptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_describe_preview_launch_skips_public_probe_diagnostics(self) -> None:
        with (
            mock.patch.object(settings, "userspace_preview_base_domain", ""),
            mock.patch.object(settings, "external_base_url", "https://ragtime.hammerton.com"),
            mock.patch.object(settings, "debug_mode", False),
        ):
            service = UserSpaceRuntimeService()
            with (
                mock.patch.object(
                    service,
                    "_resolve_preview_host_cached",
                    new=mock.AsyncMock(side_effect=AssertionError("dns probe should not run during launch description")),
                ),
                mock.patch.object(
                    service,
                    "_probe_public_preview_origin_cached",
                    new=mock.AsyncMock(side_effect=AssertionError("public probe should not run during launch description")),
                ),
            ):
                preview_origin, warning = await service._describe_preview_launch(
                    workspace_id="workspace-a",
                    control_plane_origin="https://ragtime.hammerton.com",
                )

        self.assertEqual(preview_origin, "https://workspace-a.ragtime.hammerton.com")
        self.assertIsNone(warning)

    async def test_describe_preview_launch_preserves_dev_domain_warning_without_network_checks(self) -> None:
        with (
            mock.patch.object(settings, "userspace_preview_base_domain", "userspace-preview.lvh.me"),
            mock.patch.object(settings, "external_base_url", "https://ragtime.hammerton.com"),
            mock.patch.object(settings, "debug_mode", False),
        ):
            service = UserSpaceRuntimeService()
            with (
                mock.patch.object(
                    service,
                    "_resolve_preview_host_cached",
                    new=mock.AsyncMock(side_effect=AssertionError("dns probe should not run during launch description")),
                ),
                mock.patch.object(
                    service,
                    "_probe_public_preview_origin_cached",
                    new=mock.AsyncMock(side_effect=AssertionError("public probe should not run during launch description")),
                ),
            ):
                preview_origin, warning = await service._describe_preview_launch(
                    workspace_id="workspace-a",
                    control_plane_origin="https://ragtime.hammerton.com",
                )

        self.assertEqual(preview_origin, "https://workspace-a.userspace-preview.lvh.me")
        self.assertIsNotNone(warning)
        assert warning is not None
        self.assertEqual(warning.issue_code, "preview_dev_domain_outside_debug")


if __name__ == "__main__":
    unittest.main()
