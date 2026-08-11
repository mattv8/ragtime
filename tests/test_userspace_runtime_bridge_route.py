from __future__ import annotations

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from unittest import mock

from fastapi import HTTPException
from starlette.requests import Request
from starlette.routing import Route

import ragtime.userspace.runtime_routes as _RUNTIME_ROUTES
from ragtime.userspace.models import (
    ExecuteComponentRequest,
    ExecuteComponentResponse,
    RuntimeBridgeSqliteMutationOperation,
    RuntimeBridgeSqliteMutationOperationResponse,
    RuntimeBridgeSqliteMutationRequest,
    RuntimeBridgeSqliteMutationResponse,
    RuntimeBridgeSqliteQueryRequest,
    RuntimeBridgeSqliteQueryResponse,
    UserSpaceRuntimeBridgeStatus,
)

runtime_bridge_execute_component = _RUNTIME_ROUTES.runtime_bridge_execute_component
get_runtime_bridge_status = _RUNTIME_ROUTES.get_runtime_bridge_status
refresh_runtime_bridge_credentials = _RUNTIME_ROUTES.refresh_runtime_bridge_credentials


def _build_request(path: str, authorization: str | None = None) -> Request:
    headers = [(b"host", b"ragtime.dev")]
    if authorization is not None:
        headers.append((b"authorization", authorization.encode("utf-8")))
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": b"",
            "headers": headers,
            "scheme": "https",
            "server": ("ragtime.dev", 443),
            "client": ("127.0.0.1", 12345),
        }
    )


def _route(path: str, method: str = "POST") -> Route | None:
    return next(
        (
            cast(Route, route)
            for route in _RUNTIME_ROUTES.router.routes
            if isinstance(route, Route) and route.path == path and method in (route.methods or set())
        ),
        None,
    )


class RuntimeBridgeRouteTests(unittest.IsolatedAsyncioTestCase):
    def _require_route(self, path: str, method: str = "POST") -> Route:
        route = _route(path, method)
        self.assertIsNotNone(route, f"{path} route not found on router")
        return cast(Route, route)

    def test_sqlite_routes_are_rate_limited(self) -> None:
        for path in (
            "/indexes/userspace/runtime-bridge/sqlite/query",
            "/indexes/userspace/runtime-bridge/sqlite/mutate",
        ):
            route = self._require_route(path)
            self.assertTrue(hasattr(route.endpoint, "__wrapped__"), f"{path} is missing the runtime bridge IP rate limit decorator")

    async def test_missing_bearer_returns_401(self) -> None:
        request = _build_request("/indexes/userspace/runtime-bridge/execute-component")
        payload = ExecuteComponentRequest(component_id="tool-1", request={"query": "select 1"})
        runtime_service = SimpleNamespace(
            verify_runtime_bridge_token=mock.AsyncMock(side_effect=HTTPException(status_code=401, detail="Runtime bridge token required"))
        )

        with (
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=runtime_service),
            self.assertRaises(HTTPException) as ctx,
        ):
            await runtime_bridge_execute_component(request, payload)

        self.assertEqual(ctx.exception.status_code, 401)
        runtime_service.verify_runtime_bridge_token.assert_awaited_once_with(None)

    async def test_valid_token_executes_for_claimed_workspace(self) -> None:
        request = _build_request(
            "/indexes/userspace/runtime-bridge/execute-component",
            authorization="Bearer test-token",
        )
        payload = ExecuteComponentRequest(
            component_id="tool-1",
            request={"query": "select 1", "workspace_id": "body-ws"},
        )
        runtime_service = SimpleNamespace(verify_runtime_bridge_token=mock.AsyncMock(return_value={"workspace_id": "ws-1", "session_id": "sess-1"}))
        expected = ExecuteComponentResponse(
            component_id="tool-1",
            rows=[{"id": 1}],
            columns=["id"],
            row_count=1,
        )
        userspace_service = SimpleNamespace(execute_component_from_runtime_bridge=mock.AsyncMock(return_value=expected))

        with (
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=runtime_service),
            mock.patch.object(_RUNTIME_ROUTES, "_userspace_service", return_value=userspace_service),
        ):
            response = await runtime_bridge_execute_component(request, payload)

        self.assertIs(response, expected)
        runtime_service.verify_runtime_bridge_token.assert_awaited_once_with("test-token")
        userspace_service.execute_component_from_runtime_bridge.assert_awaited_once_with(
            "ws-1",
            payload,
            session_id="sess-1",
        )

    async def test_stopped_session_401_propagates(self) -> None:
        request = _build_request(
            "/indexes/userspace/runtime-bridge/execute-component",
            authorization="Bearer stopped-token",
        )
        payload = ExecuteComponentRequest(component_id="tool-1", request={"query": "select 1"})
        runtime_service = SimpleNamespace(
            verify_runtime_bridge_token=mock.AsyncMock(side_effect=HTTPException(status_code=401, detail="Runtime session inactive"))
        )

        with (
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=runtime_service),
            self.assertRaises(HTTPException) as ctx,
        ):
            await runtime_bridge_execute_component(request, payload)

        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, "Runtime session inactive")

    async def test_sqlite_query_missing_bearer_returns_401(self) -> None:
        route = self._require_route("/indexes/userspace/runtime-bridge/sqlite/query")
        request = _build_request("/indexes/userspace/runtime-bridge/sqlite/query")
        payload = RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-ws", sql="select 1")
        runtime_service = SimpleNamespace(
            verify_runtime_bridge_token=mock.AsyncMock(side_effect=HTTPException(status_code=401, detail="Runtime bridge token required"))
        )

        with (
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=runtime_service),
            self.assertRaises(HTTPException) as ctx,
        ):
            await route.endpoint(request, payload)

        self.assertEqual(ctx.exception.status_code, 401)
        runtime_service.verify_runtime_bridge_token.assert_awaited_once_with(None)

    async def test_sqlite_query_requires_verified_workspace_session_and_leased_user(self) -> None:
        route = self._require_route("/indexes/userspace/runtime-bridge/sqlite/query")
        request = _build_request(
            "/indexes/userspace/runtime-bridge/sqlite/query",
            authorization="Bearer test-token",
        )
        payload = RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-ws", sql="select 1")
        runtime_service = SimpleNamespace(verify_runtime_bridge_token=mock.AsyncMock(return_value={"workspace_id": "ws-1", "session_id": "sess-1"}))
        userspace_service = SimpleNamespace(query_runtime_bridge_sqlite=mock.AsyncMock())

        with (
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=runtime_service),
            mock.patch.object(_RUNTIME_ROUTES, "_userspace_service", return_value=userspace_service),
            self.assertRaises(HTTPException) as ctx,
        ):
            await route.endpoint(request, payload)

        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, "Invalid runtime bridge token")
        userspace_service.query_runtime_bridge_sqlite.assert_not_awaited()

    async def test_sqlite_query_delegates_exact_verified_identities(self) -> None:
        route = self._require_route("/indexes/userspace/runtime-bridge/sqlite/query")
        request = _build_request(
            "/indexes/userspace/runtime-bridge/sqlite/query",
            authorization="Bearer test-token",
        )
        payload = RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-ws", sql="select 1")
        runtime_service = SimpleNamespace(
            verify_runtime_bridge_token=mock.AsyncMock(return_value={"workspace_id": "ws-1", "session_id": "sess-1", "leased_by_user_id": "user-1"})
        )
        expected = RuntimeBridgeSqliteQueryResponse(
            target_workspace_id="target-ws",
            database_name="app.sqlite3",
            columns=["value"],
            rows=[{"value": 1}],
            row_count=1,
            truncated=False,
        )
        userspace_service = SimpleNamespace(query_runtime_bridge_sqlite=mock.AsyncMock(return_value=expected))

        with (
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=runtime_service),
            mock.patch.object(_RUNTIME_ROUTES, "_userspace_service", return_value=userspace_service),
        ):
            response = await route.endpoint(request, payload)

        self.assertIs(response, expected)
        userspace_service.query_runtime_bridge_sqlite.assert_awaited_once_with(
            workspace_id="ws-1",
            session_id="sess-1",
            leased_by_user_id="user-1",
            request=payload,
        )

    async def test_sqlite_mutate_delegates_exact_verified_identities_and_bumps_generation_best_effort(self) -> None:
        route = self._require_route("/indexes/userspace/runtime-bridge/sqlite/mutate")
        request = _build_request(
            "/indexes/userspace/runtime-bridge/sqlite/mutate",
            authorization="Bearer test-token",
        )
        payload = RuntimeBridgeSqliteMutationRequest(
            target_workspace_id="target-ws",
            operations=[RuntimeBridgeSqliteMutationOperation(kind="insert", table="items", values={"id": 1})],
        )
        runtime_service = SimpleNamespace(
            verify_runtime_bridge_token=mock.AsyncMock(return_value={"workspace_id": "ws-1", "session_id": "sess-1", "leased_by_user_id": "user-1"}),
            bump_workspace_generation=mock.AsyncMock(side_effect=RuntimeError("generation boom")),
        )
        expected = RuntimeBridgeSqliteMutationResponse(
            target_workspace_id="target-ws",
            database_name="app.sqlite3",
            operations=[RuntimeBridgeSqliteMutationOperationResponse(kind="insert", rowcount=1, lastrowid=7)],
            fingerprint="fp-1",
        )
        userspace_service = SimpleNamespace(mutate_runtime_bridge_sqlite=mock.AsyncMock(return_value=expected))

        with (
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=runtime_service),
            mock.patch.object(_RUNTIME_ROUTES, "_userspace_service", return_value=userspace_service),
            mock.patch.object(_RUNTIME_ROUTES.logger, "exception") as logger_exception,
        ):
            response = await route.endpoint(request, payload)

        self.assertIs(response, expected)
        userspace_service.mutate_runtime_bridge_sqlite.assert_awaited_once_with(
            workspace_id="ws-1",
            session_id="sess-1",
            leased_by_user_id="user-1",
            request=payload,
        )
        runtime_service.bump_workspace_generation.assert_awaited_once()
        bump_args = runtime_service.bump_workspace_generation.await_args.args
        self.assertEqual(bump_args[0], "target-ws")
        logger_exception.assert_called_once()

    async def test_sqlite_query_maps_broker_codes_to_fixed_http_errors(self) -> None:
        route = self._require_route("/indexes/userspace/runtime-bridge/sqlite/query")
        request = _build_request(
            "/indexes/userspace/runtime-bridge/sqlite/query",
            authorization="Bearer test-token",
        )
        payload = RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-ws", sql="select 1")
        runtime_service = SimpleNamespace(
            verify_runtime_bridge_token=mock.AsyncMock(return_value={"workspace_id": "ws-1", "session_id": "sess-1", "leased_by_user_id": "user-1"})
        )

        cases = {
            "invalid_sql": 400,
            "invalid_parameters": 400,
            "sql_too_large": 400,
            "payload_too_large": 400,
            "response_too_large": 400,
            "row_limit_invalid": 400,
            "sql_not_allowed": 403,
            "database_not_found": 404,
            "sqlite_busy": 409,
            "audit_unavailable": 503,
            "query_timeout": 504,
            "query_failed": 500,
            "mutation_failed": 500,
        }
        for code, expected_status in cases.items():
            with self.subTest(code=code):
                userspace_service = SimpleNamespace(query_runtime_bridge_sqlite=mock.AsyncMock(side_effect=_RUNTIME_ROUTES.CrossWorkspaceSqliteError(code)))
                with (
                    mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=runtime_service),
                    mock.patch.object(_RUNTIME_ROUTES, "_userspace_service", return_value=userspace_service),
                    self.assertRaises(HTTPException) as ctx,
                ):
                    await route.endpoint(request, payload)
                self.assertEqual(ctx.exception.status_code, expected_status)
                self.assertEqual(ctx.exception.detail, _RUNTIME_ROUTES.CrossWorkspaceSqliteError(code).safe_message)

    async def test_sqlite_query_preserves_service_http_exceptions(self) -> None:
        route = self._require_route("/indexes/userspace/runtime-bridge/sqlite/query")
        request = _build_request(
            "/indexes/userspace/runtime-bridge/sqlite/query",
            authorization="Bearer test-token",
        )
        payload = RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-ws", sql="select 1")
        runtime_service = SimpleNamespace(
            verify_runtime_bridge_token=mock.AsyncMock(return_value={"workspace_id": "ws-1", "session_id": "sess-1", "leased_by_user_id": "user-1"})
        )
        userspace_service = SimpleNamespace(query_runtime_bridge_sqlite=mock.AsyncMock(side_effect=HTTPException(status_code=429, detail="Rate limited")))

        with (
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=runtime_service),
            mock.patch.object(_RUNTIME_ROUTES, "_userspace_service", return_value=userspace_service),
            self.assertRaises(HTTPException) as ctx,
        ):
            await route.endpoint(request, payload)

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.detail, "Rate limited")

    async def test_sqlite_query_unknown_broker_code_returns_generic_500(self) -> None:
        route = self._require_route("/indexes/userspace/runtime-bridge/sqlite/query")
        request = _build_request(
            "/indexes/userspace/runtime-bridge/sqlite/query",
            authorization="Bearer test-token",
        )
        payload = RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-ws", sql="select 1")
        runtime_service = SimpleNamespace(
            verify_runtime_bridge_token=mock.AsyncMock(return_value={"workspace_id": "ws-1", "session_id": "sess-1", "leased_by_user_id": "user-1"})
        )

        class UnknownCodeError(_RUNTIME_ROUTES.CrossWorkspaceSqliteError):
            def __init__(self) -> None:
                Exception.__init__(self, "boom")
                self.code = "mystery"
                self.safe_message = "leak me"

        userspace_service = SimpleNamespace(query_runtime_bridge_sqlite=mock.AsyncMock(side_effect=UnknownCodeError()))

        with (
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=runtime_service),
            mock.patch.object(_RUNTIME_ROUTES, "_userspace_service", return_value=userspace_service),
            self.assertRaises(HTTPException) as ctx,
        ):
            await route.endpoint(request, payload)

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(ctx.exception.detail, "Runtime bridge SQLite request failed.")

    async def test_sqlite_query_unexpected_exception_returns_generic_500_and_logs(self) -> None:
        route = self._require_route("/indexes/userspace/runtime-bridge/sqlite/query")
        request = _build_request(
            "/indexes/userspace/runtime-bridge/sqlite/query",
            authorization="Bearer test-token",
        )
        payload = RuntimeBridgeSqliteQueryRequest(target_workspace_id="target-ws", sql="select 1")
        runtime_service = SimpleNamespace(
            verify_runtime_bridge_token=mock.AsyncMock(return_value={"workspace_id": "ws-1", "session_id": "sess-1", "leased_by_user_id": "user-1"})
        )
        userspace_service = SimpleNamespace(query_runtime_bridge_sqlite=mock.AsyncMock(side_effect=RuntimeError("secret")))

        with (
            mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=runtime_service),
            mock.patch.object(_RUNTIME_ROUTES, "_userspace_service", return_value=userspace_service),
            mock.patch.object(_RUNTIME_ROUTES.logger, "exception") as logger_exception,
            self.assertRaises(HTTPException) as ctx,
        ):
            await route.endpoint(request, payload)

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(ctx.exception.detail, "Runtime bridge SQLite request failed.")
        logger_exception.assert_called_once()

    async def test_bridge_status_route_returns_service_status(self) -> None:
        expected = UserSpaceRuntimeBridgeStatus(
            state="healthy",
            bridge_url="https://ragtime.example/indexes/userspace/runtime-bridge",
            token_session_id="sess-1",
            current_session_id="sess-1",
            last_success_at=datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc),
        )
        runtime_service = SimpleNamespace(get_runtime_bridge_status=mock.AsyncMock(return_value=expected))
        user = SimpleNamespace(id="user-1")

        with mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=runtime_service):
            response = await get_runtime_bridge_status("ws-1", user)

        self.assertIs(response, expected)
        runtime_service.get_runtime_bridge_status.assert_awaited_once_with("ws-1", "user-1")

    async def test_bridge_refresh_route_returns_refreshed_status(self) -> None:
        expected = UserSpaceRuntimeBridgeStatus(
            state="healthy",
            bridge_url="https://ragtime.example/indexes/userspace/runtime-bridge",
            token_session_id="sess-1",
            current_session_id="sess-1",
        )
        runtime_service = SimpleNamespace(refresh_runtime_bridge_credentials=mock.AsyncMock(return_value=expected))
        user = SimpleNamespace(id="user-1")

        with mock.patch.object(_RUNTIME_ROUTES, "_runtime_service", return_value=runtime_service):
            response = await refresh_runtime_bridge_credentials("ws-1", user)

        self.assertIs(response, expected)
        runtime_service.refresh_runtime_bridge_credentials.assert_awaited_once_with("ws-1", "user-1")


if __name__ == "__main__":
    unittest.main()
