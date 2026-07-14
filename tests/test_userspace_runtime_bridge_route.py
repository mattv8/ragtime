from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from starlette.requests import Request

import ragtime.userspace.runtime_routes as _RUNTIME_ROUTES
from ragtime.userspace.models import ExecuteComponentRequest, ExecuteComponentResponse

runtime_bridge_execute_component = _RUNTIME_ROUTES.runtime_bridge_execute_component


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


class RuntimeBridgeRouteTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
