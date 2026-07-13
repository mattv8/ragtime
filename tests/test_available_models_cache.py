import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from ragtime.indexer.routes import (
    AvailableModel,
    AvailableModelsResponse,
    _available_models_cache_key,
    _get_or_build_available_models,
    _is_available_models_response_cacheable,
)


class AvailableModelsCacheTests(unittest.IsolatedAsyncioTestCase):
    _ORIGINAL_TTL: float | None = None

    def _reset_module_state(self) -> None:
        import ragtime.indexer.routes as routes

        if AvailableModelsCacheTests._ORIGINAL_TTL is None:
            AvailableModelsCacheTests._ORIGINAL_TTL = routes._AVAILABLE_MODELS_CACHE_TTL_SECONDS  # type: ignore[attr-defined]
        routes._AVAILABLE_MODELS_CACHE_TTL_SECONDS = AvailableModelsCacheTests._ORIGINAL_TTL  # type: ignore[attr-defined]
        routes._available_models_cache = None  # type: ignore[attr-defined]
        routes._available_models_inflight = None  # type: ignore[attr-defined]

    def _make_valid_model(self) -> AvailableModel:
        return AvailableModel(id="test-model", name="Test Model", provider="test")

    def _make_response(self, *, models=None, models_loading: bool = False, copilot_refresh_in_progress: bool = False) -> AvailableModelsResponse:
        return AvailableModelsResponse(
            models=models or [],
            models_loading=models_loading,
            copilot_refresh_in_progress=copilot_refresh_in_progress,
        )

    async def asyncSetUp(self) -> None:
        self._reset_module_state()

    async def asyncTearDown(self) -> None:
        # Restore TTL and cache slots so state never leaks into other test classes.
        self._reset_module_state()

    # 1. cache hit: two sequential calls, same key -> builder called once, equal payloads
    async def test_cache_hit_same_key(self) -> None:
        import ragtime.indexer.routes as routes

        calls = 0

        async def builder() -> AvailableModelsResponse:
            nonlocal calls
            calls += 1
            return self._make_response(models=[self._make_valid_model()])

        response1 = await routes._get_or_build_available_models("key-a", builder)
        response2 = await routes._get_or_build_available_models("key-a", builder)

        self.assertEqual(calls, 1)
        self.assertEqual(response1.model_dump(), response2.model_dump())

    # 2. returned object is a copy: mutate first result's models list; second call unaffected
    async def test_returned_object_is_deep_copy(self) -> None:
        import ragtime.indexer.routes as routes

        response = self._make_response(models=[AvailableModel(id="orig", name="Orig", provider="test")])

        async def builder() -> AvailableModelsResponse:
            return response

        r1 = await routes._get_or_build_available_models("key-copy", builder)
        r1.models[0].id = "mutated"  # type: ignore[index]

        r2 = await routes._get_or_build_available_models("key-copy", builder)

        self.assertEqual(r2.models[0].id, "orig")

    # 2b. store-side deep copy: mutating the builder's original object after it
    # was cached must not corrupt subsequent cache hits.
    async def test_stored_object_is_deep_copy(self) -> None:
        import ragtime.indexer.routes as routes

        original = self._make_response(models=[AvailableModel(id="orig", name="Orig", provider="test")])

        async def builder() -> AvailableModelsResponse:
            return original

        await routes._get_or_build_available_models("key-store-copy", builder)
        original.models[0].id = "mutated"

        r2 = await routes._get_or_build_available_models("key-store-copy", builder)

        self.assertEqual(r2.models[0].id, "orig")

    # 3. loading response not cached: builder returns models_loading=True -> second call rebuilds
    async def test_loading_response_not_cached(self) -> None:
        import ragtime.indexer.routes as routes

        calls = 0

        async def builder() -> AvailableModelsResponse:
            nonlocal calls
            calls += 1
            return self._make_response(models=[self._make_valid_model()], models_loading=True)

        await routes._get_or_build_available_models("key-loading", builder)
        await routes._get_or_build_available_models("key-loading", builder)

        self.assertEqual(calls, 2)

    # 4. empty models not cached
    async def test_empty_models_not_cached(self) -> None:
        import ragtime.indexer.routes as routes

        calls = 0

        async def builder() -> AvailableModelsResponse:
            nonlocal calls
            calls += 1
            return self._make_response(models=[])

        await routes._get_or_build_available_models("key-empty", builder)
        await routes._get_or_build_available_models("key-empty", builder)

        self.assertEqual(calls, 2)

    # 5. copilot_refresh_in_progress not cached
    async def test_copilot_refresh_not_cached(self) -> None:
        import ragtime.indexer.routes as routes

        calls = 0

        async def builder() -> AvailableModelsResponse:
            nonlocal calls
            calls += 1
            return self._make_response(
                models=[self._make_valid_model()],
                copilot_refresh_in_progress=True,
            )

        await routes._get_or_build_available_models("key-copilot", builder)
        await routes._get_or_build_available_models("key-copilot", builder)

        self.assertEqual(calls, 2)

    # 6. key change rebuilds: call with key A then key B -> builder called twice
    async def test_key_change_rebuilds(self) -> None:
        import ragtime.indexer.routes as routes

        calls = 0

        async def builder() -> AvailableModelsResponse:
            nonlocal calls
            calls += 1
            return self._make_response(models=[self._make_valid_model()])

        await routes._get_or_build_available_models("key-A", builder)
        await routes._get_or_build_available_models("key-B", builder)

        self.assertEqual(calls, 2)

    # 6b. regression: a new settings key must replace the stale single-slot entry,
    # so the new key's response is itself served from cache afterwards.
    async def test_new_key_replaces_stale_cache_entry(self) -> None:
        import ragtime.indexer.routes as routes

        calls = 0

        async def builder() -> AvailableModelsResponse:
            nonlocal calls
            calls += 1
            return self._make_response(models=[self._make_valid_model()])

        await routes._get_or_build_available_models("key-old", builder)
        await routes._get_or_build_available_models("key-new", builder)
        await routes._get_or_build_available_models("key-new", builder)

        self.assertEqual(calls, 2)

    # 7. TTL expiry rebuilds: patch _AVAILABLE_MODELS_CACHE_TTL_SECONDS to 0 -> rebuild
    async def test_ttl_expiry_rebuilds(self) -> None:
        import ragtime.indexer.routes as routes

        routes._AVAILABLE_MODELS_CACHE_TTL_SECONDS = 0  # type: ignore[attr-defined]

        calls = 0

        async def builder() -> AvailableModelsResponse:
            nonlocal calls
            calls += 1
            return self._make_response(models=[self._make_valid_model()])

        await routes._get_or_build_available_models("key-ttl", builder)
        await routes._get_or_build_available_models("key-ttl", builder)

        self.assertEqual(calls, 2)

    # 8. concurrent dedup: asyncio.gather two calls with slow builder -> builder called once, both get results
    async def test_concurrent_dedup(self) -> None:
        import ragtime.indexer.routes as routes

        calls = 0

        async def builder() -> AvailableModelsResponse:
            nonlocal calls
            await asyncio.sleep(0.05)
            calls += 1
            return self._make_response(models=[self._make_valid_model()])

        results = await asyncio.gather(
            routes._get_or_build_available_models("key-concurrent", builder),
            routes._get_or_build_available_models("key-concurrent", builder),
        )

        self.assertEqual(calls, 1)
        self.assertIsInstance(results[0], AvailableModelsResponse)
        self.assertIsInstance(results[1], AvailableModelsResponse)
        self.assertEqual(results[0].model_dump(), results[1].model_dump())

    # 8b. cross-key inflight non-reuse: an in-flight build for key A must not be
    # served to a concurrent caller with key B; each key builds independently.
    async def test_concurrent_different_keys_build_separately(self) -> None:
        import ragtime.indexer.routes as routes

        built_keys: list[str] = []

        def make_builder(marker: str):
            async def builder() -> AvailableModelsResponse:
                await asyncio.sleep(0.05)
                built_keys.append(marker)
                return self._make_response(models=[AvailableModel(id=marker, name=marker, provider="test")])

            return builder

        result_a, result_b = await asyncio.gather(
            routes._get_or_build_available_models("key-A", make_builder("a")),
            routes._get_or_build_available_models("key-B", make_builder("b")),
        )

        self.assertEqual(sorted(built_keys), ["a", "b"])
        self.assertEqual(result_a.models[0].id, "a")
        self.assertEqual(result_b.models[0].id, "b")

    # 9. builder exception: raises for both concurrent awaiters, nothing cached, next call retries
    async def test_builder_exception_propagates_and_retries(self) -> None:
        import ragtime.indexer.routes as routes

        calls = 0

        async def builder() -> AvailableModelsResponse:
            nonlocal calls
            calls += 1
            raise ValueError("builder failed")

        exceptions = await asyncio.gather(
            routes._get_or_build_available_models("key-ex", builder),
            routes._get_or_build_available_models("key-ex", builder),
            return_exceptions=True,
        )

        self.assertEqual(calls, 1)
        self.assertIsInstance(exceptions[0], ValueError)
        self.assertIsInstance(exceptions[1], ValueError)

        with self.assertRaises(ValueError):
            await routes._get_or_build_available_models("key-ex", builder)

        self.assertEqual(calls, 2)

    def test_is_cacheable_true(self) -> None:
        resp = self._make_response(models=[self._make_valid_model()])
        self.assertTrue(_is_available_models_response_cacheable(resp))

    def test_is_cacheable_false_when_loading(self) -> None:
        resp = self._make_response(models=[self._make_valid_model()], models_loading=True)
        self.assertFalse(_is_available_models_response_cacheable(resp))

    def test_is_cacheable_false_when_refresh_in_progress(self) -> None:
        resp = self._make_response(models=[self._make_valid_model()], copilot_refresh_in_progress=True)
        self.assertFalse(_is_available_models_response_cacheable(resp))

    def test_is_cacheable_false_when_empty_models(self) -> None:
        resp = self._make_response(models=[])
        self.assertFalse(_is_available_models_response_cacheable(resp))

    def test_cache_key_with_updated_at(self) -> None:
        dt = datetime(2026, 7, 12, 10, 30, 0, tzinfo=timezone.utc)
        settings = SimpleNamespace(updated_at=dt)
        self.assertEqual(_available_models_cache_key(settings), dt.isoformat())

    def test_cache_key_without_updated_at(self) -> None:
        settings = SimpleNamespace()
        self.assertEqual(_available_models_cache_key(settings), "")


class GetAvailableChatModelsRouteTests(unittest.IsolatedAsyncioTestCase):
    """Route-level wiring: cache bypass, missing-settings early return, cache reuse."""

    def _reset_module_state(self) -> None:
        import ragtime.indexer.routes as routes

        routes._available_models_cache = None  # type: ignore[attr-defined]
        routes._available_models_inflight = None  # type: ignore[attr-defined]

    async def asyncSetUp(self) -> None:
        self._reset_module_state()

    def _make_built_response(self) -> AvailableModelsResponse:
        return AvailableModelsResponse(
            models=[AvailableModel(id="built", name="Built", provider="test")],
        )

    async def _call_route_twice(self, settings) -> int:
        import ragtime.indexer.routes as routes

        build_calls = 0

        async def fake_build(app_settings) -> AvailableModelsResponse:
            nonlocal build_calls
            build_calls += 1
            return self._make_built_response()

        with (
            mock.patch.object(routes, "ensure_copilot_token_fresh", mock.AsyncMock()),
            mock.patch.object(routes.repository, "get_settings", mock.AsyncMock(return_value=settings)),
            mock.patch.object(routes, "_build_available_models_response", fake_build),
        ):
            r1 = await routes.get_available_chat_models()
            r2 = await routes.get_available_chat_models()
            self.assertEqual(r1.models[0].id, "built")
            self.assertEqual(r2.models[0].id, "built")
        return build_calls

    async def test_missing_settings_returns_empty_response(self) -> None:
        import ragtime.indexer.routes as routes

        build_called = False

        async def fake_build(app_settings) -> AvailableModelsResponse:
            nonlocal build_called
            build_called = True
            return self._make_built_response()

        with (
            mock.patch.object(routes, "ensure_copilot_token_fresh", mock.AsyncMock()),
            mock.patch.object(routes.repository, "get_settings", mock.AsyncMock(return_value=None)),
            mock.patch.object(routes, "_build_available_models_response", fake_build),
        ):
            response = await routes.get_available_chat_models()

        self.assertEqual(response.models, [])
        self.assertFalse(build_called)

    async def test_empty_cache_key_bypasses_cache(self) -> None:
        settings = SimpleNamespace(updated_at=None)
        build_calls = await self._call_route_twice(settings)
        self.assertEqual(build_calls, 2)

    async def test_settings_key_uses_cache(self) -> None:
        settings = SimpleNamespace(updated_at=datetime(2026, 7, 12, 10, 30, 0, tzinfo=timezone.utc))
        build_calls = await self._call_route_twice(settings)
        self.assertEqual(build_calls, 1)

    async def test_builder_updated_settings_key_is_cached(self) -> None:
        import ragtime.indexer.routes as routes

        settings = SimpleNamespace(updated_at=datetime(2026, 7, 12, 10, 30, 0, tzinfo=timezone.utc))
        build_calls = 0

        async def fake_build(app_settings) -> AvailableModelsResponse:
            nonlocal build_calls
            build_calls += 1
            app_settings.updated_at = datetime(2026, 7, 12, 10, 31, 0, tzinfo=timezone.utc)
            return self._make_built_response()

        with (
            mock.patch.object(routes, "ensure_copilot_token_fresh", mock.AsyncMock()),
            mock.patch.object(routes.repository, "get_settings", mock.AsyncMock(return_value=settings)),
            mock.patch.object(routes, "_build_available_models_response", fake_build),
        ):
            await routes.get_available_chat_models()
            await routes.get_available_chat_models()

        self.assertEqual(build_calls, 1)
