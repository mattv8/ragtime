import unittest
from types import SimpleNamespace
from typing import cast
from unittest import mock

from fastapi import HTTPException
from prisma import models as prisma_models
from pydantic import ValidationError

from ragtime.core.userspace_limits import (
    USERSPACE_EXEC_TIMEOUT_DEFAULT_SECONDS,
    USERSPACE_EXEC_TIMEOUT_MAX_SECONDS,
    resolve_userspace_exec_timeout,
    resolve_userspace_exec_timeout_bounds,
)
from ragtime.indexer import routes as indexer_routes
from ragtime.indexer.models import AppSettings, UpdateSettingsRequest
from ragtime.indexer.repository import IndexerRepository
from tests.test_db_fixtures import FakeDb, make_settings_row


class UserSpaceExecTimeoutLimitsTests(unittest.TestCase):
    def test_bounds_support_empty_mapping_and_attribute_settings(self) -> None:
        self.assertEqual(
            resolve_userspace_exec_timeout_bounds({}),
            (USERSPACE_EXEC_TIMEOUT_DEFAULT_SECONDS, USERSPACE_EXEC_TIMEOUT_MAX_SECONDS),
        )
        self.assertEqual(
            resolve_userspace_exec_timeout_bounds(SimpleNamespace(userspace_exec_timeout_default_seconds=45, userspace_exec_timeout_max_seconds=90)),
            (45, 90),
        )

    def test_bounds_clamp_corrupt_values_in_safe_order(self) -> None:
        self.assertEqual(
            resolve_userspace_exec_timeout_bounds({"userspace_exec_timeout_default_seconds": 9999, "userspace_exec_timeout_max_seconds": 1}),
            (30, 30),
        )
        self.assertEqual(
            resolve_userspace_exec_timeout_bounds({"userspace_exec_timeout_default_seconds": None, "userspace_exec_timeout_max_seconds": True}),
            (120, 600),
        )
        self.assertEqual(
            resolve_userspace_exec_timeout_bounds({"userspace_exec_timeout_default_seconds": 12.5, "userspace_exec_timeout_max_seconds": 10**100}),
            (120, 3600),
        )

    def test_command_resolver_accepts_legacy_compatible_values(self) -> None:
        settings = {"userspace_exec_timeout_default_seconds": 45, "userspace_exec_timeout_max_seconds": 180}
        self.assertEqual(resolve_userspace_exec_timeout(settings), 45)
        self.assertEqual(resolve_userspace_exec_timeout(settings, 120), 120)
        self.assertEqual(resolve_userspace_exec_timeout(settings, "120"), 120)
        self.assertEqual(resolve_userspace_exec_timeout(settings, "४२"), 42)
        self.assertEqual(resolve_userspace_exec_timeout(settings, 120.0), 120)

    def test_command_resolver_rejects_invalid_or_out_of_range_values(self) -> None:
        settings = {"userspace_exec_timeout_default_seconds": 45, "userspace_exec_timeout_max_seconds": 180}
        for value in (True, 12.5, float("nan"), float("inf"), "12.5", "oops", "²", "9" * 5000):
            with self.subTest(value=repr(value)), self.assertRaisesRegex(ValueError, "timeout_seconds must be an integer"):
                resolve_userspace_exec_timeout(settings, value)
        with self.assertRaisesRegex(ValueError, "between 1 and 180"):
            resolve_userspace_exec_timeout(settings, 181)


class UserSpaceExecTimeoutModelTests(unittest.TestCase):
    def test_app_settings_defaults_and_constraints(self) -> None:
        settings = AppSettings()
        self.assertEqual(settings.userspace_exec_timeout_default_seconds, 120)
        self.assertEqual(settings.userspace_exec_timeout_max_seconds, 600)
        with self.assertRaises(ValidationError):
            AppSettings(userspace_exec_timeout_default_seconds=0)
        with self.assertRaises(ValidationError):
            AppSettings(userspace_exec_timeout_max_seconds=29)
        with self.assertRaises(ValidationError):
            AppSettings(userspace_exec_timeout_default_seconds=601, userspace_exec_timeout_max_seconds=600)

    def test_update_settings_rejects_bool_and_fractional_timeout_values(self) -> None:
        for key, value in (("userspace_exec_timeout_default_seconds", True), ("userspace_exec_timeout_max_seconds", 30.5)):
            with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                UpdateSettingsRequest.model_validate({key: value})


class UserSpaceExecTimeoutRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_repository_normalizes_corrupt_response_and_persists_mapping(self) -> None:
        repository = IndexerRepository()
        fake_db = FakeDb(
            make_settings_row(
                userspaceExecTimeoutDefaultSeconds=9000,
                userspaceExecTimeoutMaxSeconds=1,
            )
        )
        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            sanitized = await repository.get_settings()
            self.assertEqual((sanitized.userspace_exec_timeout_default_seconds, sanitized.userspace_exec_timeout_max_seconds), (30, 30))
            updated = await repository.update_settings({"userspace_exec_timeout_default_seconds": 75, "userspace_exec_timeout_max_seconds": 150})

        self.assertEqual((updated.userspace_exec_timeout_default_seconds, updated.userspace_exec_timeout_max_seconds), (75, 150))
        self.assertEqual(
            fake_db.appsettings.last_update_data,
            {"userspaceExecTimeoutDefaultSeconds": 75, "userspaceExecTimeoutMaxSeconds": 150},
        )

    async def test_settings_cache_fallback_contains_timeout_defaults(self) -> None:
        from ragtime.core.app_settings import SettingsCache

        cache = SettingsCache()
        with mock.patch("ragtime.core.app_settings.get_db", mock.AsyncMock(side_effect=RuntimeError("database unavailable"))):
            settings = await cache.get_settings()

        self.assertEqual(settings["userspace_exec_timeout_default_seconds"], 120)
        self.assertEqual(settings["userspace_exec_timeout_max_seconds"], 600)

    async def test_settings_cache_success_normalizes_timeout_fields(self) -> None:
        from ragtime.core.app_settings import SettingsCache

        row = make_settings_row(userspaceExecTimeoutDefaultSeconds=5000, userspaceExecTimeoutMaxSeconds=40)
        fake_db = SimpleNamespace(appsettings=SimpleNamespace(find_unique=mock.AsyncMock(return_value=row)))
        cache = SettingsCache()
        with mock.patch("ragtime.core.app_settings.get_db", mock.AsyncMock(return_value=fake_db)):
            settings = await cache.get_settings()

        self.assertEqual((settings["userspace_exec_timeout_default_seconds"], settings["userspace_exec_timeout_max_seconds"]), (40, 40))
        self.assertIs(settings, await cache.get_settings())


class UserSpaceExecTimeoutRouteTests(unittest.IsolatedAsyncioTestCase):
    async def _update(self, current: AppSettings, request: UpdateSettingsRequest) -> dict:
        updated = current.model_copy(update=request.model_dump(exclude_unset=True, exclude_none=True))
        with (
            mock.patch.object(indexer_routes.repository, "get_settings", mock.AsyncMock(return_value=current)),
            mock.patch.object(indexer_routes.repository, "update_settings", mock.AsyncMock(return_value=updated)) as save,
            mock.patch.object(indexer_routes.rag, "initialize", mock.AsyncMock()),
            mock.patch.object(indexer_routes, "invalidate_settings_cache"),
            mock.patch.object(indexer_routes, "notify_tools_changed"),
        ):
            await indexer_routes.update_settings(
                request,
                cast(prisma_models.User, SimpleNamespace(id="admin-1", role="admin")),
            )
        await_args = save.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        updates = await_args.args[0]
        self.assertIsInstance(updates, dict)
        return updates

    async def test_partial_updates_merge_against_current_pair(self) -> None:
        current = AppSettings(userspace_exec_timeout_default_seconds=120, userspace_exec_timeout_max_seconds=600)
        self.assertEqual(
            await self._update(current, UpdateSettingsRequest(userspace_exec_timeout_default_seconds=240)),
            {"userspace_exec_timeout_default_seconds": 240},
        )
        self.assertEqual(
            await self._update(current, UpdateSettingsRequest(userspace_exec_timeout_max_seconds=240)),
            {"userspace_exec_timeout_max_seconds": 240},
        )

    async def test_explicit_null_is_unchanged_and_invalid_merged_pair_is_rejected(self) -> None:
        current = AppSettings(userspace_exec_timeout_default_seconds=120, userspace_exec_timeout_max_seconds=600)
        self.assertEqual(
            await self._update(current, UpdateSettingsRequest(userspace_exec_timeout_default_seconds=None)),
            {"userspace_exec_timeout_default_seconds": None},
        )
        with (
            mock.patch.object(indexer_routes.repository, "get_settings", mock.AsyncMock(return_value=current)),
            mock.patch.object(indexer_routes.repository, "update_settings", mock.AsyncMock()) as save,
        ):
            with self.assertRaises(HTTPException) as raised:
                await indexer_routes.update_settings(
                    UpdateSettingsRequest(userspace_exec_timeout_max_seconds=100),
                    cast(prisma_models.User, SimpleNamespace(id="admin-1", role="admin")),
                )
        self.assertEqual(raised.exception.status_code, 400)
        save.assert_not_awaited()
