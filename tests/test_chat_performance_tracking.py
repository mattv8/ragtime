import unittest

from ragtime.core.performance import PerformanceSettings, SlowRequestMiddleware
from ragtime.indexer import routes
from ragtime.indexer.routes import (
    AvailableModel,
    AvailableModelsResponse,
    LLMModel,
    LLMModelsResponse,
    _get_or_build_available_models,
    _safe_fetch_llm_models_task,
)


class ChatPerformanceTrackingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        routes._available_models_cache = None
        routes._available_models_inflight = None

    async def asyncTearDown(self) -> None:
        routes._available_models_cache = None
        routes._available_models_inflight = None

    async def test_cache_and_provider_stages_attach_without_sensitive_values(self) -> None:
        cache_key = "cache-key-must-not-appear"
        credential = "credential-must-not-appear"
        model_payload = "model-payload-must-not-appear"

        async def builder() -> AvailableModelsResponse:
            await _safe_fetch_llm_models_task(
                "openai",
                _response(
                    LLMModelsResponse(
                        success=True,
                        message="private response",
                        models=[LLMModel(id=model_payload, name="Private")],
                    ),
                    credential,
                ),
            )
            return AvailableModelsResponse(models=[AvailableModel(id=model_payload, name="Private", provider="openai")])

        async def app(scope, receive, send) -> None:
            await _get_or_build_available_models(cache_key, builder)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        async def receive():
            return {"type": "http.disconnect"}

        async def discard_send(message) -> None:
            del message

        middleware = SlowRequestMiddleware(app, settings=PerformanceSettings(0.0, 0.0, 0.0))
        with self.assertLogs("ragtime.performance", level="DEBUG") as logs:
            await middleware({"type": "http", "method": "GET", "headers": []}, receive, discard_send)

        rendered = "\n".join(logs.output)
        self.assertIn("available_models.cache.miss", rendered)
        self.assertIn("available_models.cache.wait", rendered)
        self.assertIn("model_discovery.provider_fetch.openai", rendered)
        self.assertNotIn(cache_key, rendered)
        self.assertNotIn(credential, rendered)
        self.assertNotIn(model_payload, rendered)


async def _response(response: LLMModelsResponse, credential: str) -> LLMModelsResponse:
    del credential
    return response
