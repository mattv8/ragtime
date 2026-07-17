import hashlib
import hmac
import json
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import httpx
from fastapi import FastAPI, HTTPException
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request

import ragtime.indexer.routes as indexer_routes
from ragtime.core.rate_limit import limiter as global_limiter
from ragtime.git_webhooks.models import (
    GitWebhookConfigResponse,
    GitWebhookDeliveryResponse,
    GitWebhookDeliveryStatus,
    GitWebhookEnableResponse,
    GitWebhookTarget,
    GitWebhookTargetType,
)
from ragtime.git_webhooks.repository import _webhook_url
from ragtime.git_webhooks.routes import receive_git_webhook
from ragtime.git_webhooks.routes import router as git_webhook_router
from ragtime.indexer.routes import disable_index_webhook, enable_index_webhook, get_index_webhook, list_index_webhook_deliveries, rotate_index_webhook_secret
from ragtime.main import _rate_limit_exceeded_handler
from ragtime.userspace.routes import (
    disable_workspace_scm_webhook,
    enable_workspace_scm_webhook,
    get_workspace_scm_webhook,
    list_workspace_scm_webhook_deliveries,
    rotate_workspace_scm_webhook_secret,
)


def _signature(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _target(
    *,
    provider: str = "github",
    branch: str = "main",
    secret: str = "secret",
    target_type: GitWebhookTargetType = GitWebhookTargetType.GIT_INDEX,
    target_id: str = "target-1",
) -> GitWebhookTarget:
    return GitWebhookTarget(
        target_type=target_type,
        target_id=target_id,
        key=f"{target_type.value}:{target_id}",
        webhook_id="opaque-id",
        secret=secret,
        provider=provider,
        branch=branch,
        name="git-index" if target_type is GitWebhookTargetType.GIT_INDEX else None,
        source="https://example.com/repo.git",
    )


def _delivery(*, status: GitWebhookDeliveryStatus = GitWebhookDeliveryStatus.PENDING) -> GitWebhookDeliveryResponse:
    return GitWebhookDeliveryResponse(
        id="delivery-1",
        target_type=GitWebhookTargetType.GIT_INDEX,
        target_id="target-1",
        provider_delivery_id="provider-delivery-1",
        event_name="push",
        branch="main",
        head_commit="abc123",
        status=status,
        message=None,
        received_at=datetime(2026, 7, 16, tzinfo=timezone.utc),
        started_at=None,
        completed_at=None,
        index_job_id=None,
    )


def _admin() -> SimpleNamespace:
    return SimpleNamespace(id="admin-1", role="admin", username="admin")


def _request(
    body: bytes,
    *,
    headers: dict[str, str] | None = None,
    chunks: list[bytes] | None = None,
    path: str = "/webhooks/git/opaque-id",
    query_string: bytes = b"",
    base_url: str = "https://ragtime.example/",
) -> Request:
    header_items = []
    for key, value in (headers or {}).items():
        header_items.append((key.lower().encode("latin-1"), value.encode("latin-1")))
    if not any(key == b"host" for key, _value in header_items):
        header_items.append((b"host", b"ragtime.example"))
    payload_chunks = list(chunks if chunks is not None else [body])
    if not payload_chunks:
        payload_chunks = [b""]
    index = 0

    async def receive() -> dict[str, object]:
        nonlocal index
        if index < len(payload_chunks):
            chunk = payload_chunks[index]
            index += 1
            return {"type": "http.request", "body": chunk, "more_body": index < len(payload_chunks)}
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "raw_path": path.encode("utf-8"),
            "query_string": query_string,
            "headers": header_items,
            "scheme": base_url.split(":", 1)[0],
            "server": ("ragtime.example", 443),
            "client": ("203.0.113.10", 12345),
        },
        receive,
    )


class GitWebhookPublicRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_indexer_models_import_cleanly_in_fresh_subprocess(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import ragtime.indexer.models"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    async def test_public_webhook_accepts_valid_configured_branch(self) -> None:
        body = b'{"ref":"refs/heads/main","after":"abc123"}'
        request = _request(
            body,
            headers={
                "x-github-event": "push",
                "x-github-delivery": "delivery-1",
                "x-hub-signature-256": _signature("secret", body),
            },
        )
        target = _target(provider="github", branch="main", secret="secret")
        with (
            mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.resolve_target", mock.AsyncMock(return_value=target)),
            mock.patch("ragtime.git_webhooks.service.git_webhook_service.accept_push", mock.AsyncMock(return_value=_delivery())),
        ):
            response = await receive_git_webhook("opaque-id", request)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(json.loads(bytes(response.body)), {"status": "accepted"})

    async def test_public_webhook_rejects_unknown_or_disabled_target(self) -> None:
        request = _request(b"{}")
        with mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.resolve_target", mock.AsyncMock(return_value=None)):
            with self.assertRaises(HTTPException) as raised:
                await receive_git_webhook("opaque-id", request)
        self.assertEqual(raised.exception.status_code, 404)
        self.assertNotIn("opaque-id", str(raised.exception.detail))

    async def test_public_webhook_rejects_invalid_signature_without_recording_delivery(self) -> None:
        body = b'{"ref":"refs/heads/main","after":"abc123"}'
        request = _request(
            body,
            headers={
                "x-github-event": "push",
                "x-hub-signature-256": "sha256=00",
            },
        )
        target = _target(secret="secret")
        accept_push = mock.AsyncMock()
        record_ignored = mock.AsyncMock()
        with (
            mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.resolve_target", mock.AsyncMock(return_value=target)),
            mock.patch("ragtime.git_webhooks.service.git_webhook_service.accept_push", accept_push),
            mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.record_ignored", record_ignored),
        ):
            with self.assertRaises(HTTPException) as raised:
                await receive_git_webhook("opaque-id", request)
        self.assertEqual(raised.exception.status_code, 401)
        accept_push.assert_not_awaited()
        record_ignored.assert_not_awaited()

    async def test_public_webhook_rejects_malformed_json_after_authentication(self) -> None:
        body = b'  {"ref": "refs/heads/main",  '
        request = _request(
            body,
            headers={
                "x-github-event": "push",
                "x-hub-signature-256": _signature("secret", body),
            },
        )
        target = _target(secret="secret")
        with mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.resolve_target", mock.AsyncMock(return_value=target)):
            with self.assertRaises(HTTPException) as raised:
                await receive_git_webhook("opaque-id", request)
        self.assertEqual(raised.exception.status_code, 400)

    async def test_public_webhook_rejects_oversize_body_before_json_decode(self) -> None:
        body = b"x" * (1024 * 1024 + 1)
        request = _request(body, headers={"content-length": "1"}, chunks=[b"x" * (1024 * 1024), b"x"])
        target = _target(secret="secret")
        with mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.resolve_target", mock.AsyncMock(return_value=target)):
            response = await receive_git_webhook("opaque-id", request)
        self.assertEqual(response.status_code, 413)

    async def test_public_webhook_returns_accepted_for_ping_event_and_records_ignored(self) -> None:
        body = b'{"zen":"pong"}'
        request = _request(
            body,
            headers={
                "x-github-event": "ping",
                "x-hub-signature-256": _signature("secret", body),
            },
        )
        target = _target(secret="secret")
        accept_push = mock.AsyncMock()
        record_ignored = mock.AsyncMock(return_value=_delivery(status=GitWebhookDeliveryStatus.IGNORED))
        with (
            mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.resolve_target", mock.AsyncMock(return_value=target)),
            mock.patch("ragtime.git_webhooks.service.git_webhook_service.accept_push", accept_push),
            mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.record_ignored", record_ignored),
        ):
            response = await receive_git_webhook("opaque-id", request)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(json.loads(bytes(response.body)), {"status": "accepted"})
        accept_push.assert_not_awaited()
        record_ignored.assert_awaited_once()

    async def test_public_webhook_returns_accepted_for_wrong_branch_and_records_ignored(self) -> None:
        body = b'{"ref":"refs/heads/develop","after":"abc123"}'
        request = _request(
            body,
            headers={
                "x-github-event": "push",
                "x-github-delivery": "delivery-1",
                "x-hub-signature-256": _signature("secret", body),
            },
        )
        target = _target(branch="main", secret="secret")
        accept_push = mock.AsyncMock()
        record_ignored = mock.AsyncMock(return_value=_delivery(status=GitWebhookDeliveryStatus.IGNORED))
        with (
            mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.resolve_target", mock.AsyncMock(return_value=target)),
            mock.patch("ragtime.git_webhooks.service.git_webhook_service.accept_push", accept_push),
            mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.record_ignored", record_ignored),
        ):
            response = await receive_git_webhook("opaque-id", request)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(json.loads(bytes(response.body)), {"status": "accepted"})
        accept_push.assert_not_awaited()
        record_ignored.assert_awaited_once()

    async def test_public_webhook_returns_same_body_for_duplicate_delivery(self) -> None:
        body = b'{"ref":"refs/heads/main","after":"abc123"}'
        request = _request(
            body,
            headers={
                "x-github-event": "push",
                "x-github-delivery": "delivery-1",
                "x-hub-signature-256": _signature("secret", body),
            },
        )
        target = _target(secret="secret")
        with (
            mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.resolve_target", mock.AsyncMock(return_value=target)),
            mock.patch(
                "ragtime.git_webhooks.service.git_webhook_service.accept_push",
                mock.AsyncMock(return_value=_delivery(status=GitWebhookDeliveryStatus.SKIPPED)),
            ),
        ):
            response = await receive_git_webhook("opaque-id", request)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(json.loads(bytes(response.body)), {"status": "accepted"})

    async def test_public_webhook_selects_matching_branch_before_ignored_delivery(self) -> None:
        body = json.dumps(
            {
                "push": {
                    "changes": [
                        {"new": {"type": "branch", "name": "develop", "target": {"hash": "111"}}},
                        {"new": {"type": "branch", "name": "main", "target": {"hash": "222"}}},
                    ]
                }
            }
        ).encode("utf-8")
        signature = hmac.new(b"secret", body, hashlib.sha256).hexdigest()
        request = _request(
            body,
            headers={
                "x-event-key": "repo:push",
                "x-request-uuid": "delivery-9",
                "x-gitea-signature": signature,
            },
        )
        target = _target(provider="generic", branch="main", secret="secret")
        accept_push = mock.AsyncMock(return_value=_delivery())
        record_ignored = mock.AsyncMock()
        with (
            mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.resolve_target", mock.AsyncMock(return_value=target)),
            mock.patch("ragtime.git_webhooks.service.git_webhook_service.accept_push", accept_push),
            mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.record_ignored", record_ignored),
        ):
            response = await receive_git_webhook("opaque-id", request)
        self.assertEqual(response.status_code, 202)
        accept_push_call = accept_push.await_args
        assert accept_push_call is not None
        accepted_event = accept_push_call.args[1]
        self.assertEqual(accepted_event.branch, "main")
        record_ignored.assert_not_awaited()

    async def test_public_webhook_rejects_odd_length_signature_without_recording_delivery(self) -> None:
        body = b'{"ref":"refs/heads/main","after":"abc123"}'
        request = _request(
            body,
            headers={
                "x-github-event": "push",
                "x-hub-signature-256": "sha256=abc",
            },
        )
        target = _target(secret="secret")
        record_ignored = mock.AsyncMock()
        with (
            mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.resolve_target", mock.AsyncMock(return_value=target)),
            mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.record_ignored", record_ignored),
        ):
            with self.assertRaises(HTTPException) as raised:
                await receive_git_webhook("opaque-id", request)
        self.assertEqual(raised.exception.status_code, 401)
        record_ignored.assert_not_awaited()

    async def test_public_webhook_url_contract_matches_mounted_prefix(self) -> None:
        self.assertEqual(_webhook_url("https://ragtime.example/", "opaque-id"), "https://ragtime.example/webhooks/git/opaque-id")
        self.assertTrue(any(getattr(route, "path", None) == "/webhooks/git/{webhook_id}" for route in git_webhook_router.routes))

    async def test_public_webhook_rate_limit_applies_in_non_debug_mode(self) -> None:
        # The shared limiter bypasses limits in debug mode; this test forces the
        # production path and exercises the real decorated ASGI route.
        app = FastAPI()
        test_limiter = Limiter(key_func=get_remote_address, default_limits=[], enabled=True)
        app.state.limiter = test_limiter
        app.add_exception_handler(RateLimitExceeded, cast(Any, _rate_limit_exceeded_handler))
        original_enabled = global_limiter.enabled
        global_limiter.enabled = True
        try:
            app.include_router(git_webhook_router)
            transport = httpx.ASGITransport(app=app, client=("198.51.100.9", 12345))
            async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
                with mock.patch(
                    "ragtime.git_webhooks.routes.git_webhook_repository.resolve_target",
                    mock.AsyncMock(return_value=None),
                ):
                    for _ in range(60):
                        response = await client.post("/webhooks/git/opaque-id", content=b"{}")
                        self.assertEqual(response.status_code, 404)
                    response = await client.post("/webhooks/git/opaque-id", content=b"{}")
            self.assertEqual(response.status_code, 429)
        finally:
            global_limiter.enabled = original_enabled


class GitWebhookConfigurationRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_index_enable_http_response_omits_secret_when_unavailable(self) -> None:
        app = FastAPI()
        app.include_router(indexer_routes.router)
        app.dependency_overrides[indexer_routes.require_admin] = lambda: _admin()
        try:
            with (
                mock.patch(
                    "ragtime.indexer.routes.repository.get_index_metadata",
                    mock.AsyncMock(return_value=SimpleNamespace(sourceType="git", source="https://git.example/repo.git")),
                ),
                mock.patch.object(
                    indexer_routes.git_webhook_repository,
                    "enable_index",
                    mock.AsyncMock(
                        return_value=GitWebhookEnableResponse(
                            enabled=True,
                            webhook_id="opaque-id",
                            webhook_url="https://ragtime.example/webhooks/git/opaque-id",
                            provider="github",
                            branch="main",
                            secret=None,
                        )
                    ),
                ),
            ):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="https://ragtime.example") as client:
                    response = await client.post("/indexes/git-index/webhook")
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("secret", response.json())
        finally:
            app.dependency_overrides.clear()

    async def test_index_webhook_enable_rejects_non_git_index(self) -> None:
        metadata = SimpleNamespace(sourceType="upload", source=None)
        with mock.patch(
            "ragtime.indexer.routes.repository.get_index_metadata",
            mock.AsyncMock(return_value=metadata),
        ):
            with self.assertRaises(HTTPException) as raised:
                await enable_index_webhook("git-index", _request(b"", path="/indexes/git-index/webhook"), _admin())
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("Git", str(raised.exception.detail))

    async def test_workspace_webhook_enable_rejects_disconnected_workspace(self) -> None:
        user = SimpleNamespace(id="owner-1", role="user")
        workspace_record = SimpleNamespace(id="workspace-1", scmGitUrl=None, scmRemoteRole=None)
        db = SimpleNamespace(workspace=SimpleNamespace(find_unique=mock.AsyncMock(return_value=workspace_record)))
        with (
            mock.patch("ragtime.userspace.routes.userspace_service.enforce_workspace_role", mock.AsyncMock()),
            mock.patch("ragtime.userspace.routes.get_db", mock.AsyncMock(return_value=db)),
        ):
            with self.assertRaises(HTTPException) as raised:
                await enable_workspace_scm_webhook(
                    "workspace-1",
                    _request(b"", path="/indexes/userspace/workspaces/workspace-1/scm/webhook"),
                    user,
                )
        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn("connect", str(raised.exception.detail).lower())

    async def test_workspace_webhook_rotate_rejects_non_upstream_workspace(self) -> None:
        user = SimpleNamespace(id="owner-1", role="user")
        workspace_record = SimpleNamespace(id="workspace-1", scmGitUrl="https://git.example/repo.git", scmRemoteRole="publish")
        db = SimpleNamespace(workspace=SimpleNamespace(find_unique=mock.AsyncMock(return_value=workspace_record)))
        with (
            mock.patch("ragtime.userspace.routes.userspace_service.enforce_workspace_role", mock.AsyncMock()),
            mock.patch("ragtime.userspace.routes.get_db", mock.AsyncMock(return_value=db)),
        ):
            with self.assertRaises(HTTPException) as raised:
                await rotate_workspace_scm_webhook_secret(
                    "workspace-1",
                    _request(b"", path="/indexes/userspace/workspaces/workspace-1/scm/webhook/rotate"),
                    user,
                )
        self.assertEqual(raised.exception.status_code, 409)
        self.assertIn("upstream", str(raised.exception.detail).lower())

    async def test_get_config_does_not_return_secret(self) -> None:
        config = GitWebhookConfigResponse(
            enabled=True,
            webhook_id="opaque-id",
            webhook_url="https://example/webhooks/git/opaque-id",
            provider="github",
            branch="main",
        )
        with mock.patch(
            "ragtime.indexer.routes.git_webhook_repository.get_index_config",
            mock.AsyncMock(return_value=config),
        ):
            response = await get_index_webhook("git-index", _request(b"", path="/indexes/git-index/webhook"), _admin())
        self.assertNotIn("secret", response.model_dump())

    async def test_workspace_editor_cannot_enable_scm_webhook(self) -> None:
        user = SimpleNamespace(id="editor-1", role="user")
        with mock.patch(
            "ragtime.userspace.routes.userspace_service.enforce_workspace_role",
            mock.AsyncMock(side_effect=HTTPException(status_code=403, detail="Owner access required")),
        ):
            with self.assertRaises(HTTPException) as raised:
                await enable_workspace_scm_webhook("workspace-1", _request(b"", path="/indexes/userspace/workspaces/workspace-1/scm/webhook"), user)
        self.assertEqual(raised.exception.status_code, 403)

    async def test_index_routes_strip_trailing_slash_from_base_url(self) -> None:
        request = _request(b"", path="/indexes/git-index/webhook", base_url="https://ragtime.example/")
        with (
            mock.patch(
                "ragtime.indexer.routes.repository.get_index_metadata",
                mock.AsyncMock(return_value=SimpleNamespace(sourceType="git", source="https://git.example/repo.git")),
            ),
            mock.patch(
                "ragtime.indexer.routes.git_webhook_repository.enable_index",
                mock.AsyncMock(
                    return_value=GitWebhookEnableResponse(enabled=True, webhook_url="https://ragtime.example/webhooks/git/opaque-id", secret="secret")
                ),
            ) as enable_index_mock,
        ):
            await enable_index_webhook("git-index", request, _admin())
        enable_index_call = enable_index_mock.await_args
        assert enable_index_call is not None
        self.assertEqual(enable_index_call.args, ("git-index", "https://ragtime.example"))

    async def test_index_disable_uses_pre_disable_target_snapshot(self) -> None:
        target = _target(target_type=GitWebhookTargetType.GIT_INDEX, target_id="index-1")
        with (
            mock.patch("ragtime.indexer.routes.git_webhook_repository.resolve_index_target", mock.AsyncMock(return_value=target)),
            mock.patch("ragtime.indexer.routes.git_webhook_repository.disable_index", mock.AsyncMock()),
            mock.patch("ragtime.git_webhooks.service.git_webhook_service.disable_target") as disable_target,
        ):
            response = await disable_index_webhook("git-index", _request(b"", path="/indexes/git-index/webhook"), _admin())
        self.assertEqual(response, {"status": "disabled"})
        disable_target.assert_called_once_with(target)

    async def test_workspace_admin_routes_enforce_owner_with_admin_flag(self) -> None:
        user = SimpleNamespace(id="admin-1", role="admin")
        config = GitWebhookConfigResponse(enabled=False, webhook_url=None, provider=None, branch=None)
        with (
            mock.patch("ragtime.userspace.routes.userspace_service.enforce_workspace_role", mock.AsyncMock()) as enforce_role,
            mock.patch("ragtime.userspace.routes.git_webhook_repository.get_workspace_config", mock.AsyncMock(return_value=config)),
        ):
            await get_workspace_scm_webhook(
                "workspace-1",
                _request(b"", path="/indexes/userspace/workspaces/workspace-1/scm/webhook"),
                user,
            )
        enforce_role.assert_awaited_once_with("workspace-1", "admin-1", "owner", is_admin=True)

    async def test_workspace_disable_uses_pre_disable_target_snapshot(self) -> None:
        user = SimpleNamespace(id="owner-1", role="user")
        target = _target(target_type=GitWebhookTargetType.WORKSPACE_SCM, target_id="workspace-1")
        with (
            mock.patch("ragtime.userspace.routes.userspace_service.enforce_workspace_role", mock.AsyncMock()),
            mock.patch("ragtime.userspace.routes.git_webhook_repository.resolve_workspace_target", mock.AsyncMock(return_value=target)),
            mock.patch("ragtime.userspace.routes.git_webhook_repository.disable_workspace", mock.AsyncMock()),
            mock.patch("ragtime.git_webhooks.service.git_webhook_service.disable_target") as disable_target,
        ):
            response = await disable_workspace_scm_webhook(
                "workspace-1",
                _request(b"", path="/indexes/userspace/workspaces/workspace-1/scm/webhook"),
                user,
            )
        self.assertEqual(response, {"status": "disabled"})
        disable_target.assert_called_once_with(target)

    async def test_delivery_history_routes_clamp_limit_and_return_rows(self) -> None:
        deliveries = [_delivery()]
        with (
            mock.patch("ragtime.indexer.routes.git_webhook_repository.get_index_target_id", mock.AsyncMock(return_value="index-1")),
            mock.patch(
                "ragtime.indexer.routes.git_webhook_repository.list_deliveries",
                mock.AsyncMock(return_value=deliveries),
            ) as list_mock,
        ):
            response = await list_index_webhook_deliveries("git-index", limit=20, _user=_admin())
        self.assertEqual(response, deliveries)
        list_call = list_mock.await_args
        assert list_call is not None
        self.assertEqual(list_call.kwargs["limit"], 20)

    async def test_workspace_delivery_history_enforces_owner_access(self) -> None:
        user = SimpleNamespace(id="owner-1", role="user")
        with (
            mock.patch("ragtime.userspace.routes.userspace_service.enforce_workspace_role", mock.AsyncMock()) as enforce_role,
            mock.patch("ragtime.userspace.routes.git_webhook_repository.get_workspace_target_id", mock.AsyncMock(return_value="workspace-1")),
            mock.patch(
                "ragtime.userspace.routes.git_webhook_repository.list_deliveries",
                mock.AsyncMock(return_value=[_delivery()]),
            ),
        ):
            await list_workspace_scm_webhook_deliveries("workspace-1", limit=20, user=user)
        enforce_role.assert_awaited_once_with("workspace-1", "owner-1", "owner", is_admin=False)

    async def test_rotate_workspace_scm_webhook_secret_returns_enable_response(self) -> None:
        user = SimpleNamespace(id="owner-1", role="user")
        response_model = GitWebhookEnableResponse(
            enabled=True,
            webhook_id="opaque-id",
            webhook_url="https://ragtime.example/webhooks/git/opaque-id",
            provider="github",
            branch="main",
            secret="secret-1",
        )
        with (
            mock.patch("ragtime.userspace.routes.userspace_service.enforce_workspace_role", mock.AsyncMock()),
            mock.patch(
                "ragtime.userspace.routes.get_db",
                mock.AsyncMock(
                    return_value=SimpleNamespace(
                        workspace=SimpleNamespace(
                            find_unique=mock.AsyncMock(
                                return_value=SimpleNamespace(
                                    id="workspace-1",
                                    scmGitUrl="https://git.example/repo.git",
                                    scmRemoteRole="upstream",
                                )
                            )
                        )
                    )
                ),
            ),
            mock.patch(
                "ragtime.userspace.routes.git_webhook_repository.rotate_workspace_secret",
                mock.AsyncMock(return_value=response_model),
            ),
        ):
            response = await rotate_workspace_scm_webhook_secret(
                "workspace-1",
                _request(b"", path="/indexes/userspace/workspaces/workspace-1/scm/webhook/rotate"),
                user,
            )
        self.assertEqual(response.secret, "secret-1")
