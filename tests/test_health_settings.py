"""Regression coverage for the lightweight authenticated health settings read."""

import unittest
from types import SimpleNamespace
from unittest import mock

from ragtime.api import routes
from ragtime.core import app_settings
from ragtime.core.app_setting_defaults import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER


class _HealthSettingsDb:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows
        self.queries: list[tuple[str, tuple[object, ...]]] = []

    async def query_raw(self, query: str, *params: object) -> list[dict[str, str]]:
        self.queries.append((query, params))
        return self.rows


class HealthSettingsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.cache = app_settings.SettingsCache.get_instance()
        self.cache.invalidate()

    def tearDown(self) -> None:
        self.cache.invalidate()

    async def test_cold_read_selects_only_llm_fields_without_decrypting_or_runtime_hooks(self) -> None:
        db = _HealthSettingsDb([{"llm_model": "small-model", "llm_provider": "local"}])

        with (
            mock.patch.object(app_settings, "get_db", mock.AsyncMock(return_value=db)),
            mock.patch.object(app_settings, "decrypt_secret") as decrypt_secret,
            mock.patch.object(app_settings, "_apply_runtime_setting_hooks") as apply_hooks,
        ):
            result = await app_settings.get_health_llm_settings()

        self.assertEqual(result, {"llm_model": "small-model", "llm_provider": "local"})
        self.assertEqual(
            db.queries,
            [
                (
                    "SELECT llm_model, llm_provider FROM app_settings WHERE id = $1 LIMIT 1",
                    ("default",),
                )
            ],
        )
        decrypt_secret.assert_not_called()
        apply_hooks.assert_not_called()

    async def test_hot_read_reuses_full_settings_cache_without_database_query(self) -> None:
        self.cache._settings = {"llm_model": "cached-model", "llm_provider": "cached-provider"}

        with mock.patch.object(app_settings, "get_db", mock.AsyncMock()) as get_db:
            result = await app_settings.get_health_llm_settings()

        self.assertEqual(result, {"llm_model": "cached-model", "llm_provider": "cached-provider"})
        get_db.assert_not_awaited()

    async def test_missing_settings_returns_established_defaults_without_creating_row(self) -> None:
        db = _HealthSettingsDb([])

        with mock.patch.object(app_settings, "get_db", mock.AsyncMock(return_value=db)):
            result = await app_settings.get_health_llm_settings()

        self.assertEqual(result, {"llm_model": DEFAULT_LLM_MODEL, "llm_provider": DEFAULT_LLM_PROVIDER})
        self.assertEqual(len(db.queries), 1)

    async def test_invalidation_clears_full_cache_for_subsequent_health_read(self) -> None:
        self.cache._settings = {"llm_model": "stale-model", "llm_provider": "stale-provider"}
        self.cache.invalidate()
        db = _HealthSettingsDb([{"llm_model": "fresh-model", "llm_provider": "fresh-provider"}])

        with mock.patch.object(app_settings, "get_db", mock.AsyncMock(return_value=db)):
            result = await app_settings.get_health_llm_settings()

        self.assertEqual(result, {"llm_model": "fresh-model", "llm_provider": "fresh-provider"})
        self.assertEqual(len(db.queries), 1)

    async def test_public_health_stays_minimal_without_settings_read(self) -> None:
        fake_rag = SimpleNamespace(is_ready=True, loading_status={"retrievers_available": ["index-a"]})

        with (
            mock.patch.object(routes, "rag", fake_rag),
            mock.patch.object(routes, "get_health_llm_settings", mock.AsyncMock()) as get_health_llm_settings,
        ):
            response = await routes.health_check(current_user=None)

        self.assertEqual(response.status, "healthy")
        self.assertEqual(response.model, "")
        self.assertEqual(response.llm_provider, "")
        self.assertEqual(response.indexes_loaded, [])
        get_health_llm_settings.assert_not_awaited()

    async def test_authenticated_health_uses_lightweight_llm_settings_and_preserves_details(self) -> None:
        loading_status = {
            "retrievers_available": ["index-a"],
            "indexes_ready": True,
            "indexes_loading": False,
            "indexes_total": 1,
            "indexes_loaded": 1,
            "index_details": [{"name": "index-a", "status": "ready"}],
            "sequential_loading": True,
            "loading_index": None,
        }
        fake_rag = SimpleNamespace(is_ready=True, loading_status=loading_status)
        process = SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=2 * 1024 * 1024, vms=3 * 1024 * 1024),
            memory_percent=lambda: 4.0,
        )
        virtual_memory = SimpleNamespace(available=5 * 1024 * 1024, total=6 * 1024 * 1024)

        with (
            mock.patch.object(routes, "rag", fake_rag),
            mock.patch.object(
                routes,
                "get_health_llm_settings",
                mock.AsyncMock(return_value={"llm_model": "tiny-health", "llm_provider": "local"}),
            ) as get_health_llm_settings,
            mock.patch.object(routes.psutil, "Process", return_value=process),
            mock.patch.object(routes.psutil, "virtual_memory", return_value=virtual_memory),
        ):
            response = await routes.health_check(current_user={"id": "user-1"})

        self.assertEqual(response.model, "tiny-health")
        self.assertEqual(response.llm_provider, "local")
        self.assertEqual(response.indexes_loaded, ["index-a"])
        self.assertTrue(response.indexes_ready)
        self.assertTrue(response.sequential_loading)
        assert response.index_details is not None
        assert response.memory is not None
        self.assertEqual(response.index_details[0].name, "index-a")
        self.assertEqual(response.memory.rss_mb, 2.0)
        get_health_llm_settings.assert_awaited_once_with()
