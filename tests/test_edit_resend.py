import importlib
import inspect
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException
from starlette.requests import Request


def _build_request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [(b"host", b"ragtime.dev")],
            "scheme": "https",
        }
    )


def _require_symbol(module_name: str, symbol_name: str):
    """Defer new-contract imports until a test actually exercises them."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise AssertionError(f"Required edit-resend module {module_name} could not be imported") from exc
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise AssertionError(f"Required edit-resend symbol {module_name}.{symbol_name} is missing") from exc


def _edit_resend_route():
    routes = importlib.import_module("ragtime.indexer.routes")
    endpoint = getattr(routes, "edit_resend_conversation_branch", None)
    if endpoint is not None:
        return routes, endpoint

    for route in routes.router.routes:
        if getattr(route, "path", None) == "/indexes/conversations/{conversation_id}/branches/edit-resend":
            return routes, route.endpoint
    raise AssertionError("Required edit-resend route handler is missing")


class EditResendRouteTests(unittest.IsolatedAsyncioTestCase):
    async def _call_edit_resend(
        self,
        outcome: str,
        *,
        validate_error: Exception | None = None,
        start_error: Exception | None = None,
    ):
        routes, endpoint = _edit_resend_route()
        models = importlib.import_module("ragtime.indexer.models")
        request_type = _require_symbol("ragtime.indexer.routes", "EditResendRequest")
        request = request_type(message="replacement", from_message_index=1, auto_snapshot=False)
        user = SimpleNamespace(id="user-1", role="user")
        existing_conversation = models.Conversation(
            id="conversation-1",
            messages=[],
            total_tokens=0,
            active_branch_id=None,
            parent_conversation_id=None,
        )
        branch = SimpleNamespace(id="branch-1")
        updated_conversation = models.Conversation(
            id="conversation-1",
            model="test-model",
            messages=[],
            total_tokens=0,
        )
        task = models.ChatTask(
            id="task-1",
            conversation_id="conversation-1",
            user_message="replacement",
        )
        branch_summary = models.ConversationBranchSummary(
            id="branch-1",
            conversation_id="conversation-1",
            branch_point_index=1,
            message_count=1,
            created_at=datetime.now(timezone.utc),
        )
        repository = SimpleNamespace(
            check_conversation_access=mock.AsyncMock(return_value=True),
            get_conversation=mock.AsyncMock(return_value=existing_conversation),
            create_branch_and_start_edit_resend=mock.AsyncMock(
                return_value=(branch, updated_conversation, task, outcome)
            ),
            cancel_chat_task=mock.AsyncMock(),
            get_conversation_branches=mock.AsyncMock(return_value=[branch_summary]),
        )
        validate_mock = (
            mock.AsyncMock(side_effect=validate_error)
            if validate_error is not None
            else mock.AsyncMock(return_value="test-model")
        )
        start_task = (
            mock.AsyncMock(side_effect=start_error)
            if start_error is not None
            else mock.AsyncMock(return_value=task)
        )
        route_kwargs = {
            "conversation_id": "conversation-1",
            "request": request,
            "workspace_id": None,
            "user": user,
        }
        # Keep direct route calls compatible if the implementation accepts the
        # raw request for logging or request-scoped context.
        if "http_request" in inspect.signature(endpoint).parameters:
            route_kwargs["http_request"] = _build_request("/conversations/conversation-1/branches/edit-resend")

        self._repo_mock = repository
        with (
            mock.patch.object(routes, "repository", repository),
            mock.patch.object(routes, "_assert_workspace_access", mock.AsyncMock()),
            mock.patch.object(routes, "_resolve_workspace_runtime_scope", mock.AsyncMock(return_value=(None, set(), None))),
            mock.patch.object(routes, "_validate_generation_ready_after_user_message", validate_mock),
            mock.patch.object(routes, "_apply_validated_conversation_model", mock.AsyncMock(return_value=updated_conversation)),
            mock.patch.object(routes, "_create_background_chat_task_after_user_message", start_task),
            mock.patch.object(routes, "schedule_title_generation"),
        ):
            result = await endpoint(**route_kwargs)

        return result, repository, start_task

    async def test_edit_resend_returns_branch_conversation_and_task_and_starts_claimed_task(self) -> None:
        result, repository, start_task = await self._call_edit_resend("created")

        self.assertEqual(result.branch.id, "branch-1")
        self.assertEqual(result.conversation.id, "conversation-1")
        self.assertEqual(result.task.id, "task-1")
        self.assertEqual(repository.create_branch_and_start_edit_resend.call_args.args[1], 1)
        self.assertEqual(start_task.call_args.kwargs["existing_task_id"], "task-1")

    async def test_edit_resend_cancels_claimed_task_when_model_validation_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            await self._call_edit_resend("created", validate_error=RuntimeError("model unavailable"))

        self._repo_mock.cancel_chat_task.assert_awaited_once_with("task-1")

    async def test_edit_resend_cancels_claimed_task_when_background_start_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            await self._call_edit_resend("created", start_error=RuntimeError("start failed"))

        self._repo_mock.cancel_chat_task.assert_awaited_once_with("task-1")

    async def test_edit_resend_rejects_active_task(self) -> None:
        with self.assertRaises(HTTPException) as context:
            await self._call_edit_resend("active_task")

        self.assertEqual(context.exception.status_code, 409)

    async def test_edit_resend_rejects_invalid_branch_point(self) -> None:
        with self.assertRaises(HTTPException) as context:
            await self._call_edit_resend("invalid_branch_point")

        self.assertEqual(context.exception.status_code, 400)

    async def test_edit_resend_rejects_missing_conversation(self) -> None:
        with self.assertRaises(HTTPException) as context:
            await self._call_edit_resend("conversation_not_found")

        self.assertEqual(context.exception.status_code, 404)


class EditResendRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_branch_and_edit_resend_uses_server_truncate_append_and_delta_tokens(self) -> None:
        repository_module = importlib.import_module("ragtime.indexer.repository")
        repository_class = _require_symbol("ragtime.indexer.repository", "IndexerRepository")
        try:
            create_edit_resend = getattr(repository_class, "create_branch_and_start_edit_resend")
        except AttributeError as exc:
            raise AssertionError("Required edit-resend repository method is missing") from exc
        estimate_message_tokens = _require_symbol("ragtime.indexer.repository", "_estimate_message_tokens")
        estimate_effective_tokens = _require_symbol(
            "ragtime.indexer.repository",
            "_estimate_effective_conversation_tokens",
        )

        messages = [
            {"role": "user", "content": "keep this", "timestamp": "2026-01-01T00:00:00+00:00"},
            {"role": "assistant", "content": "discard this", "timestamp": "2026-01-01T00:00:01+00:00"},
        ]
        initial_conversation = SimpleNamespace(
            id="conversation-1",
            messages=messages,
            activeTaskId=None,
            activeBranchId=None,
        )
        refreshed_conversation = SimpleNamespace(id="conversation-1", messages=messages[:1])
        tx = SimpleNamespace(
            conversation=SimpleNamespace(
                find_unique=mock.AsyncMock(side_effect=[initial_conversation, refreshed_conversation])
            ),
            conversationbranch=SimpleNamespace(create=mock.AsyncMock(return_value=SimpleNamespace(id="branch-row"))),
            chattask=SimpleNamespace(create=mock.AsyncMock(return_value=SimpleNamespace(id="task-row"))),
            execute_raw=mock.AsyncMock(side_effect=[1, 1]),
        )
        tx.conversationbranch.find_unique = mock.AsyncMock(return_value=SimpleNamespace(id="branch-row"))

        class Transaction:
            async def __aenter__(self):
                return tx

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        db = SimpleNamespace(tx=mock.Mock(return_value=Transaction()))
        repo = repository_class()
        repo._get_db = mock.AsyncMock(return_value=db)
        repo._prisma_branch_to_model = mock.Mock(return_value=SimpleNamespace(id="branch-1"))
        repo._prisma_conversation_to_model = mock.Mock(return_value=SimpleNamespace(id="conversation-1"))
        repo._prisma_task_to_model = mock.Mock(return_value=SimpleNamespace(id="task-1"))

        with mock.patch.object(repository_module.uuid, "uuid4", side_effect=["branch-1", "task-1", "message-1"]):
            branch, conversation, task, outcome = await create_edit_resend(
                repo,
                conversation_id="conversation-1",
                branch_point_index=1,
                user_message="replacement",
                branch_kind=None,
                user_id="user-1",
                parent_branch_id=None,
                associated_snapshot_id=None,
            )

        new_message = {"role": "user", "content": "replacement"}
        expected_total = estimate_effective_tokens(messages[:1]) + estimate_message_tokens(new_message)
        update_statements = [
            call.args[0]
            for call in tx.execute_raw.await_args_list
            if "jsonb_array_elements(messages)" in call.args[0]
        ]

        self.assertEqual((branch.id, conversation.id, task.id, outcome), ("branch-1", "conversation-1", "task-1", "created"))
        self.assertEqual(len(update_statements), 1)
        self.assertIn("WITH ORDINALITY", update_statements[0])
        self.assertIn("WHERE ord <= 1", update_statements[0])
        self.assertIn("||", update_statements[0])
        self.assertIn(f"total_tokens = {expected_total}", update_statements[0])
        self.assertIn("active_branch_id = NULL", update_statements[0])
        self.assertIn("active_task_id", update_statements[0])

    async def test_add_user_message_transaction_uses_extended_timeouts(self) -> None:
        repository_class = _require_symbol("ragtime.indexer.repository", "IndexerRepository")
        add_message = getattr(repository_class, "add_user_message_and_create_chat_task_if_idle")
        repo = repository_class()

        class Transaction:
            async def __aenter__(self):
                return SimpleNamespace(conversation=SimpleNamespace(find_unique=mock.AsyncMock(return_value=None)))

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        db = SimpleNamespace(tx=mock.Mock(return_value=Transaction()))
        repo._get_db = mock.AsyncMock(return_value=db)

        await add_message(repo, "conversation-1", "hello")

        db.tx.assert_called_once_with(max_wait=timedelta(seconds=10), timeout=timedelta(seconds=30))


if __name__ == "__main__":
    unittest.main()
