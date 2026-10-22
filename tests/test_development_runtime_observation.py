import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

from ragtime.userspace.development_access import DevelopmentPrincipal
from ragtime.userspace.development_service import DevelopmentService
from ragtime.userspace.models import UserSpaceRuntimeObservation, UserSpaceRuntimeSession, UserSpaceRuntimeSessionResponse, UserSpaceRuntimeStatusResponse


class DevelopmentRuntimeObservationTests(unittest.IsolatedAsyncioTestCase):
    async def test_runtime_status_preserves_session_and_exposes_nonrecovering_observation(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        status = UserSpaceRuntimeStatusResponse(
            workspace_id="workspace-1",
            session_state="running",
            session_id="session-1",
            devserver_running=False,
            devserver_port=3000,
            preview_url="http://preview.test",
            observation=UserSpaceRuntimeObservation(
                state="unavailable",
                devserver_running=None,
                warning="Runtime provider status is unavailable; recorded runtime state is not current evidence.",
            ),
        )
        session = UserSpaceRuntimeSessionResponse(
            workspace_id="workspace-1",
            session=UserSpaceRuntimeSession(
                id="session-1",
                workspace_id="workspace-1",
                leased_by_user_id="user-1",
                state="running",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            ),
        )
        service = DevelopmentService()
        with (
            mock.patch.object(service, "_workspace", new=mock.AsyncMock()),
            mock.patch.object(service, "_audit_operation", new=mock.AsyncMock()),
            mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service.get_devserver_status", new=mock.AsyncMock(return_value=status)) as get_status,
            mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service.get_runtime_session", new=mock.AsyncMock(return_value=session)),
        ):
            result = await service._execute_unprotected(principal, "workspace-1", "runtime_status", {})  # pyright: ignore[reportPrivateUsage]

        get_status.assert_awaited_once_with("workspace-1", "user-1", allow_recovery=False)
        self.assertEqual(result["workspace_id"], "workspace-1")
        self.assertEqual(result["session"]["id"], "session-1")
        self.assertEqual(result["session"]["state"], "running")
        self.assertNotIn("session", result["session"])
        self.assertEqual(result["observation"]["state"], "unavailable")
        self.assertFalse(result["session_record"]["liveness_authoritative"])

    async def test_runtime_status_preserves_a_missing_session_as_none(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        status = UserSpaceRuntimeStatusResponse(workspace_id="workspace-1", session_state="stopped")
        service = DevelopmentService()
        with (
            mock.patch.object(service, "_workspace", new=mock.AsyncMock()),
            mock.patch.object(service, "_audit_operation", new=mock.AsyncMock()),
            mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service.get_devserver_status", new=mock.AsyncMock(return_value=status)),
            mock.patch(
                "ragtime.userspace.runtime_service.userspace_runtime_service.get_runtime_session",
                new=mock.AsyncMock(return_value=UserSpaceRuntimeSessionResponse(workspace_id="workspace-1")),
            ),
        ):
            result = await service._execute_unprotected(principal, "workspace-1", "runtime_status", {})  # pyright: ignore[reportPrivateUsage]

        self.assertIsNone(result["session"])

    async def test_http_resources_keeps_the_raw_catalog_only_when_paging_is_omitted(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        service = DevelopmentService()
        raw_catalog = {"tools": [{"component_id": "tool-1"}], "indexes": []}
        bounded_catalog = {"authorized_tools": [{"component_id": "tool-1"}], "authorized_indexes": []}
        with (
            mock.patch.object(service, "_workspace", new=mock.AsyncMock()),
            mock.patch.object(service, "_audit_operation", new=mock.AsyncMock()),
            mock.patch.object(service, "_resources", new=mock.AsyncMock(return_value=raw_catalog)) as resources,
            mock.patch.object(service, "get_resource_catalog", new=mock.AsyncMock(return_value=bounded_catalog)) as get_catalog,
        ):
            raw = await service._execute_unprotected(principal, "workspace-1", "resources", {})  # pyright: ignore[reportPrivateUsage]
            page = await service._execute_unprotected(principal, "workspace-1", "resources", {"offset": 0, "limit": 1})  # pyright: ignore[reportPrivateUsage]

        self.assertEqual(raw, raw_catalog)
        self.assertEqual(page["items"][0]["component_id"], "tool-1")
        resources.assert_awaited_once()
        get_catalog.assert_awaited_once_with(principal, "workspace-1")

    async def test_management_custom_selection_preserves_empty_lists_and_rejects_non_strings(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read", "write"}), auth_source="user")
        service = DevelopmentService()
        with (
            mock.patch.object(service, "_workspace", new=mock.AsyncMock()),
            mock.patch.object(service, "_audit_operation", new=mock.AsyncMock()),
            mock.patch("ragtime.userspace.development_service.execute_management_operation", new=mock.AsyncMock(return_value={"selected_tool_ids": []})) as execute,
        ):
            result = await service._execute_unprotected(  # pyright: ignore[reportPrivateUsage]
                principal,
                "workspace-1",
                "workspace_tools_set",
                {"mode": "custom", "selected_tool_ids": []},
            )
            with self.assertRaisesRegex(Exception, "invalid type"):
                await service._execute_unprotected(  # pyright: ignore[reportPrivateUsage]
                    principal,
                    "workspace-1",
                    "workspace_tools_set",
                    {"mode": "custom", "selected_tool_ids": ["tool-1", 1]},
                )

        self.assertEqual(result["selected_tool_ids"], [])
        execute.assert_awaited_once()

    async def test_context_labels_recorded_blockers_when_provider_observation_is_unavailable(self) -> None:
        principal = DevelopmentPrincipal(user_id="user-1", is_admin=False, scopes=frozenset({"read"}))
        service = DevelopmentService()
        status = UserSpaceRuntimeStatusResponse(
            workspace_id="workspace-1",
            session_state="error",
            last_error="startup failed",
            live_data_warning="query timeout",
            observation=None,
        )
        db = SimpleNamespace(user=mock.AsyncMock(find_unique=mock.AsyncMock(return_value=SimpleNamespace(username="Ada", displayName="Ada", role="user"))))
        with (
            mock.patch("ragtime.userspace.development_service.planning_service.get_workspace_context", new=mock.AsyncMock(return_value={"workspace": {"id": "workspace-1"}})),
            mock.patch("ragtime.userspace.development_service.get_db", new=mock.AsyncMock(return_value=db)),
            mock.patch("ragtime.userspace.development_service.content_protection_service.access_guidance", new=mock.AsyncMock(return_value={})),
            mock.patch.object(service, "_resources", new=mock.AsyncMock(return_value={"tools": [], "indexes": []})),
            mock.patch("ragtime.userspace.development_service.userspace_service.list_workspace_mounts", new=mock.AsyncMock(return_value=[])),
            mock.patch("ragtime.userspace.development_service.userspace_service.list_mountable_sources", new=mock.AsyncMock(return_value=[])),
            mock.patch("ragtime.userspace.development_service.userspace_service.list_workspace_env_var_summaries", new=mock.AsyncMock(return_value=[])),
            mock.patch("ragtime.userspace.development_service.userspace_service.list_workspace_preview_diagnostic_summary", new=mock.AsyncMock(return_value=[])),
            mock.patch("ragtime.userspace.development_service.userspace_service.get_workspace_object_storage_summary", new=mock.AsyncMock(return_value=None)),
            mock.patch("ragtime.userspace.development_service.userspace_service.list_accessible_cross_workspace_sqlite_targets", new=mock.AsyncMock(return_value=[])),
            mock.patch("ragtime.userspace.runtime_service.userspace_runtime_service.get_devserver_status", new=mock.AsyncMock(return_value=status)),
        ):
            bundle = await service._context(principal, "workspace-1")  # pyright: ignore[reportPrivateUsage]

        self.assertIn("Recorded runtime blocker", bundle["turn_instructions"])
        self.assertIn("Live data query TIMEOUT", bundle["turn_instructions"])
        self.assertEqual(bundle["facts"]["runtime_observation"]["state"], "unavailable")
