"""Tests for the public workspace agent HTTP surface."""

import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ragtime.userspace.agent_access import AgentAccessContext


def _build_request(path: str = "/agent/w/tok-abc") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": [(b"host", b"ragtime.example.com")],
            "scheme": "https",
            "query_string": b"",
        }
    )


def _build_http_scope(path: str) -> dict[str, object]:
    return {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [(b"host", b"ragtime.example.com")],
        "scheme": "https",
        "query_string": b"",
    }


def _ctx(allow_task_submission: bool = True) -> AgentAccessContext:
    return AgentAccessContext(
        access_id="acc-1",
        workspace_id="ws-1",
        acting_user_id="user-1",
        acting_user_is_admin=False,
        allow_task_submission=allow_task_submission,
    )


class AgentManifestTests(unittest.IsolatedAsyncioTestCase):
    async def test_manifest_documents_api_and_hides_nothing_sensitive(self) -> None:
        from ragtime.userspace import agent_routes as module
        from ragtime.userspace.service import userspace_service

        with (
            mock.patch.object(module, "resolve_agent_access_token", mock.AsyncMock(return_value=_ctx())),
            mock.patch.object(
                userspace_service,
                "enforce_workspace_role",
                mock.AsyncMock(return_value=SimpleNamespace(id="ws-1", name="Sales Dashboard")),
            ),
            mock.patch.object(
                module,
                "get_browser_matched_origin",
                mock.Mock(return_value="https://ragtime.example.com"),
            ),
        ):
            response = await module.get_agent_manifest("tok-abc", _build_request())

        text = bytes(response.body).decode("utf-8")
        self.assertIn("Sales Dashboard", text)
        self.assertIn("/agent/w/tok-abc/context", text)
        self.assertIn("/agent/w/tok-abc/conversations", text)
        self.assertIn("/agent/w/tok-abc/code-search", text)
        self.assertIn("/agent/w/tok-abc/tasks", text)
        self.assertIn("idempotency_key", text)
        self.assertIn('/tasks/{task_id}/reply   body: {"idempotency_key":', text)
        self.assertIn("direction=forward|backward", text)
        self.assertIn("tool inputs", text)
        self.assertIn("no tool outputs", text)
        self.assertIn("semantic|symbols|hybrid", text)
        self.assertEqual(response.headers.get("cache-control"), "no-store")

    async def test_unknown_token_propagates_404(self) -> None:
        from ragtime.userspace import agent_routes as module

        with mock.patch.object(
            module,
            "resolve_agent_access_token",
            mock.AsyncMock(side_effect=HTTPException(status_code=404, detail="Unknown agent access token")),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await module.get_agent_manifest("bad", _build_request())
        self.assertEqual(ctx.exception.status_code, 404)


class AgentTaskRouteTests(unittest.IsolatedAsyncioTestCase):
    def test_reply_request_requires_idempotency_key(self) -> None:
        from ragtime.userspace import agent_routes as module

        with self.assertRaises(ValidationError):
            module.AgentTaskReplyRequest.model_validate({"message": "Proceed"})

    async def test_task_submission_disabled_returns_403(self) -> None:
        from ragtime.userspace import agent_routes as module
        from ragtime.userspace.agent_briefs import BuildBriefInput

        brief = BuildBriefInput(
            idempotency_key="req-00000001",
            title="t",
            objective="o",
            requirements=["r"],
            acceptance_criteria=["a"],
        )
        with mock.patch.object(module, "resolve_agent_access_token", mock.AsyncMock(return_value=_ctx(allow_task_submission=False))):
            with self.assertRaises(HTTPException) as ctx:
                await module.submit_agent_task("tok-abc", brief, Response())
        self.assertEqual(ctx.exception.status_code, 403)

    async def test_task_submission_delegates_to_service(self) -> None:
        from ragtime.userspace import agent_routes as module
        from ragtime.userspace.agent_briefs import BuildBriefInput

        brief = BuildBriefInput(
            idempotency_key="req-00000001",
            title="t",
            objective="o",
            requirements=["r"],
            acceptance_criteria=["a"],
        )
        started = {"deduplicated": False, "conversation_id": "c1", "task_id": "t1", "status": "pending"}
        with (
            mock.patch.object(module, "resolve_agent_access_token", mock.AsyncMock(return_value=_ctx())),
            mock.patch.object(module.build_task_service, "start_build_task", mock.AsyncMock(return_value=started)) as start_mock,
        ):
            result = await module.submit_agent_task("tok-abc", brief, Response())
        self.assertEqual(result, started)
        start_mock.assert_awaited_once_with("ws-1", "user-1", brief)

    async def test_reply_task_delegates_to_service_with_idempotency_key(self) -> None:
        from ragtime.userspace import agent_routes as module

        body = module.AgentTaskReplyRequest(idempotency_key="reply-001", message="Proceed")
        replied = {"deduplicated": False, "conversation_id": "c1", "task_id": "t2", "status": "pending"}
        with (
            mock.patch.object(module, "resolve_agent_access_token", mock.AsyncMock(return_value=_ctx())),
            mock.patch.object(module.build_task_service, "reply_to_build_task", mock.AsyncMock(return_value=replied)) as reply_mock,
        ):
            result = await module.reply_agent_task("tok-abc", "task-1", body, Response())

        self.assertEqual(result, replied)
        reply_mock.assert_awaited_once_with("ws-1", "user-1", "task-1", "Proceed", "reply-001")


class AgentReadRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_conversations_resolves_token_forwards_args_and_sets_no_store(self) -> None:
        from ragtime.userspace import agent_routes as module

        payload = {"total": 1, "offset": 0, "limit": 50, "conversations": []}
        response = Response()
        with (
            mock.patch.object(module, "resolve_agent_access_token", mock.AsyncMock(return_value=_ctx())) as resolve,
            mock.patch.object(
                module.agent_read_service,
                "list_conversations",
                mock.AsyncMock(return_value=payload),
            ) as service,
        ):
            result = await module.list_agent_conversations("tok-abc", response, offset=0, limit=50)
        self.assertEqual(result, payload)
        resolve.assert_awaited_once_with("tok-abc")
        service.assert_awaited_once_with("ws-1", "user-1", is_admin=False, offset=0, limit=50)
        self.assertEqual(response.headers.get("cache-control"), "no-store")

    async def test_get_conversation_resolves_token_forwards_args_and_sets_no_store(self) -> None:
        from ragtime.userspace import agent_routes as module

        payload = {"conversation": {"id": "conv-1"}, "messages": []}
        response = Response()
        with (
            mock.patch.object(module, "resolve_agent_access_token", mock.AsyncMock(return_value=_ctx())) as resolve,
            mock.patch.object(
                module.agent_read_service,
                "get_conversation_transcript",
                mock.AsyncMock(return_value=payload),
            ) as service,
        ):
            result = await module.get_agent_conversation(
                "tok-abc",
                "conv-1",
                response,
                direction="backward",
                cursor=9,
                limit=7,
            )
        self.assertEqual(result, payload)
        resolve.assert_awaited_once_with("tok-abc")
        service.assert_awaited_once_with(
            "ws-1",
            "user-1",
            "conv-1",
            is_admin=False,
            direction="backward",
            cursor=9,
            limit=7,
        )
        self.assertEqual(response.headers.get("cache-control"), "no-store")

    def test_get_conversation_requires_direction_parameter(self) -> None:
        from ragtime.userspace import agent_routes as module

        parameter = inspect.signature(module.get_agent_conversation).parameters["direction"]
        self.assertIs(parameter.default, inspect.Signature.empty)

    async def test_code_search_resolves_token_forwards_args_and_sets_no_store(self) -> None:
        from ragtime.userspace import agent_routes as module

        payload = {"status": "ready", "mode": "hybrid", "query": "revenue", "result_count": 0, "results": []}
        response = Response()
        with (
            mock.patch.object(module, "resolve_agent_access_token", mock.AsyncMock(return_value=_ctx())) as resolve,
            mock.patch.object(
                module.agent_read_service,
                "search_code",
                mock.AsyncMock(return_value=payload),
            ) as service,
        ):
            result = await module.search_agent_workspace_code(
                "tok-abc",
                response,
                query="revenue",
                mode="hybrid",
                max_results=3,
                max_chars_per_result=400,
            )
        self.assertEqual(result, payload)
        resolve.assert_awaited_once_with("tok-abc")
        service.assert_awaited_once_with(
            "ws-1",
            "user-1",
            is_admin=False,
            query="revenue",
            mode="hybrid",
            max_results=3,
            max_chars_per_result=400,
        )
        self.assertEqual(response.headers.get("cache-control"), "no-store")


class AgentManagementRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_includes_agent_url_when_enabled(self) -> None:
        from ragtime.userspace import agent_routes as module

        status = {
            "workspace_id": "ws-1",
            "enabled": True,
            "allow_task_submission": True,
            "token": "tok-abc",
            "created_at": None,
            "last_used_at": None,
            "hit_count": 3,
        }
        with (
            mock.patch.object(module, "get_agent_access_status", mock.AsyncMock(return_value=status)),
            mock.patch.object(
                module,
                "get_browser_matched_origin",
                mock.Mock(return_value="https://ragtime.example.com"),
            ),
        ):
            response = Response()
            result = await module.get_workspace_agent_access(
                "ws-1",
                _build_request(),
                response,
                user=SimpleNamespace(id="user-1", role="user"),
            )
        self.assertEqual(result["agent_url"], "https://ragtime.example.com/agent/w/tok-abc")
        self.assertEqual(response.headers["cache-control"], "no-store")


class AgentRouteNoStoreMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_agent_path_gets_no_store_for_validation_style_response(self) -> None:
        from ragtime import main

        request = Request(_build_http_scope("/agent/w/not-a-valid-token/files"))

        async def call_next(_: Request) -> Response:
            return Response(status_code=422)

        response = await main._apply_agent_route_no_store(request, call_next)

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.headers.get("cache-control"), "no-store")

    async def test_management_agent_access_path_gets_no_store_for_auth_style_response(self) -> None:
        from ragtime import main

        request = Request(_build_http_scope("/indexes/userspace/workspaces/not-a-workspace/agent-access"))

        async def call_next(_: Request) -> Response:
            return Response(status_code=401)

        response = await main._apply_agent_route_no_store(request, call_next)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.headers.get("cache-control"), "no-store")

    async def test_unrelated_userspace_path_does_not_get_no_store(self) -> None:
        from ragtime import main

        request = Request(_build_http_scope("/indexes/userspace/workspaces/ws-1/files"))

        async def call_next(_: Request) -> Response:
            return Response(status_code=401)

        response = await main._apply_agent_route_no_store(request, call_next)

        self.assertEqual(response.status_code, 401)
        self.assertIsNone(response.headers.get("cache-control"))

    async def test_public_agent_path_unexpected_error_returns_generic_no_store_500(self) -> None:
        from ragtime import main

        request = Request(_build_http_scope("/agent/w/tok-secret/tasks"))

        async def call_next(_: Request) -> Response:
            raise RuntimeError("token leaked tok-secret")

        with mock.patch.object(main.logger, "error") as error_log:
            response = await main._apply_agent_route_no_store(request, call_next)

        self.assertIsInstance(response, JSONResponse)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.headers.get("cache-control"), "no-store")
        self.assertEqual(response.body, b'{"detail":"Internal server error"}')
        logged_args = error_log.call_args.args
        self.assertIn("/agent/w/[redacted]/tasks", logged_args[2])
        self.assertNotIn("tok-secret", logged_args[2])

    async def test_public_agent_path_unexpected_error_logs_only_redacted_path_and_exception_class(self) -> None:
        from ragtime import main

        request = Request(_build_http_scope("/agent/w/tok-secret/tasks"))

        async def call_next(_: Request) -> Response:
            raise RuntimeError("token leaked tok-secret via exception message")

        with mock.patch.object(main.logger, "error") as error_log:
            response = await main._apply_agent_route_no_store(request, call_next)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.headers.get("cache-control"), "no-store")
        error_log.assert_called_once()
        self.assertNotIn("tok-secret", str(error_log.call_args))
        self.assertEqual(error_log.call_args.kwargs, {})
        self.assertEqual(
            error_log.call_args.args,
            (
                "Unexpected agent route error for %s %s (%s)",
                "GET",
                "/agent/w/[redacted]/tasks",
                "RuntimeError",
            ),
        )

    async def test_non_agent_path_unexpected_error_propagates(self) -> None:
        from ragtime import main

        request = Request(_build_http_scope("/api/other"))

        async def call_next(_: Request) -> Response:
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            await main._apply_agent_route_no_store(request, call_next)


class AgentRouteLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_validation_error_log_redacts_agent_access_token(self) -> None:
        from ragtime import main

        request = Request(_build_http_scope("/agent/w/tok-secret/tasks/task-1/reply"))
        exc = RequestValidationError([{"loc": ("body", "idempotency_key"), "msg": "required", "type": "missing"}])

        with mock.patch.object(main.logger, "error") as error_log:
            response = await main.validation_exception_handler(request, exc)

        self.assertEqual(response.status_code, 422)
        logged_args = error_log.call_args.args
        self.assertIn("/agent/w/[redacted]/tasks/task-1/reply", logged_args[0])
        self.assertNotIn("tok-secret", logged_args[0])

    async def test_slow_request_log_redacts_agent_access_token(self) -> None:
        from ragtime import main

        request = Request(_build_http_scope("/agent/w/tok-secret/files"))

        async def call_next(_: Request) -> Response:
            return Response(status_code=200)

        with (
            mock.patch.object(main.time, "perf_counter", side_effect=[0.0, 1.5]),
            mock.patch.object(main.logger, "warning") as warning_log,
        ):
            response = await main._log_slow_requests(request, call_next)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(warning_log.call_args.args[2], "/agent/w/[redacted]/files")
        self.assertNotIn("tok-secret", warning_log.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
