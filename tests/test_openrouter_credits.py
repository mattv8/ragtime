import asyncio
import unittest
from collections.abc import Mapping, Sequence
from unittest import mock

import httpx

from ragtime.core import openrouter_credits
from ragtime.indexer.models import AppSettings


class _FakeClient:
    def __init__(self, responses: Sequence[httpx.Response | Exception], calls: list[tuple[str, Mapping[str, str]]]) -> None:
        self._responses = iter(responses)
        self._calls = calls

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(self, url: str, *, headers: Mapping[str, str]) -> httpx.Response:
        self._calls.append((url, headers))
        result = next(self._responses)
        if isinstance(result, Exception):
            raise result
        return result


class OpenRouterCreditTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        openrouter_credits._cached_status = None
        openrouter_credits._cached_at = None
        openrouter_credits._refresh_task = None
        openrouter_credits._monitor_task = None
        openrouter_credits._failure_count = 0
        openrouter_credits._next_refresh_at = None
        openrouter_credits._payment_required_note = None

    async def test_reports_distinct_key_cap_and_wallet_balances(self) -> None:
        calls: list[tuple[str, Mapping[str, str]]] = []
        responses = [
            httpx.Response(200, json={"data": {"limit_remaining": 2.5}}),
            httpx.Response(200, json={"data": {"total_credits": 12, "total_usage": 3}}),
        ]
        settings = AppSettings(
            openrouter_credit_monitor_enabled=True,
            openrouter_low_credit_threshold_usd=5,
            openrouter_api_key="inference-secret",
            openrouter_management_api_key="management-secret",
        )
        with (
            mock.patch.object(openrouter_credits, "get_app_settings", new=mock.AsyncMock(return_value=settings)),
            mock.patch.object(openrouter_credits.httpx, "AsyncClient", return_value=_FakeClient(responses, calls)),
        ):
            status = await openrouter_credits.get_openrouter_credit_status()

        self.assertEqual(status["state"], "low")
        self.assertEqual(status["key_remaining_usd"], 2.5)
        self.assertEqual(status["wallet_remaining_usd"], 9.0)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("secret", str(status))

    async def test_disabled_or_unconfigured_never_calls_provider(self) -> None:
        client = mock.Mock()
        with (
            mock.patch.object(openrouter_credits, "get_app_settings", new=mock.AsyncMock(return_value=AppSettings())),
            mock.patch.object(openrouter_credits.httpx, "AsyncClient", client),
        ):
            disabled = await openrouter_credits.get_openrouter_credit_status()
        self.assertEqual(disabled["state"], "disabled")
        client.assert_not_called()

        with mock.patch.object(
            openrouter_credits,
            "get_app_settings",
            new=mock.AsyncMock(return_value=AppSettings(openrouter_credit_monitor_enabled=True)),
        ):
            unconfigured = await openrouter_credits.get_openrouter_credit_status()
        self.assertEqual(unconfigured["state"], "unconfigured")

    async def test_cache_coalesces_concurrent_refreshes_and_marks_failed_data_stale(self) -> None:
        calls: list[tuple[str, Mapping[str, str]]] = []
        gate = asyncio.Event()

        class DelayedClient(_FakeClient):
            async def get(self, url, *, headers):
                await gate.wait()
                return await super().get(url, headers=headers)

        settings = AppSettings(
            openrouter_credit_monitor_enabled=True,
            openrouter_api_key="inference-secret",
        )
        with (
            mock.patch.object(openrouter_credits, "get_app_settings", new=mock.AsyncMock(return_value=settings)),
            mock.patch.object(
                openrouter_credits.httpx,
                "AsyncClient",
                return_value=DelayedClient([httpx.Response(200, json={"data": {"limit_remaining": 7}})], calls),
            ),
        ):
            first = asyncio.create_task(openrouter_credits.get_openrouter_credit_status(force_refresh=True))
            second = asyncio.create_task(openrouter_credits.get_openrouter_credit_status(force_refresh=True))
            await asyncio.sleep(0)
            gate.set()
            first_status, second_status = await asyncio.gather(first, second)
            self.assertEqual(first_status["state"], "ok")
            self.assertEqual(second_status["state"], "ok")
            self.assertEqual(len(calls), 1)

        with (
            mock.patch.object(openrouter_credits, "get_app_settings", new=mock.AsyncMock(return_value=settings)),
            mock.patch.object(
                openrouter_credits.httpx,
                "AsyncClient",
                return_value=_FakeClient([httpx.ConnectError("provider detail")], []),
            ),
        ):
            stale = await openrouter_credits.get_openrouter_credit_status(force_refresh=True)
        self.assertTrue(stale["stale"])
        self.assertEqual(stale["key_remaining_usd"], 7.0)
        self.assertNotIn("detail", str(stale))

    async def test_payment_note_persists_until_positive_wallet_recovery(self) -> None:
        openrouter_credits.note_openrouter_payment_required()
        settings = AppSettings(
            openrouter_credit_monitor_enabled=True,
            openrouter_api_key="inference-secret",
            openrouter_management_api_key="management-secret",
        )
        calls: list[tuple[str, Mapping[str, str]]] = []
        with (
            mock.patch.object(openrouter_credits, "get_app_settings", new=mock.AsyncMock(return_value=settings)),
            mock.patch.object(
                openrouter_credits.httpx,
                "AsyncClient",
                return_value=_FakeClient(
                    [
                        httpx.Response(200, json={"data": {"limit_remaining": 20}}),
                        httpx.Response(200, json={"data": {"total_credits": 10, "total_usage": 2}}),
                    ],
                    calls,
                ),
            ),
        ):
            status = await openrouter_credits.get_openrouter_credit_status(force_refresh=True)

        self.assertIsNone(status["warning"])
        self.assertIsNone(openrouter_credits.get_openrouter_credit_warning())
