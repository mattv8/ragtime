import json
import sys
import tempfile
import types
import unittest
from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

fake_copilot_auth = types.ModuleType("ragtime.core.copilot_auth")


async def _fake_ensure_copilot_token_fresh(*_args, **_kwargs):
    return None


setattr(fake_copilot_auth, "ensure_copilot_token_fresh", _fake_ensure_copilot_token_fresh)
sys.modules.setdefault("ragtime.core.copilot_auth", fake_copilot_auth)

from ragtime.rag.components import (
    RAGComponents,
    load_workspace_server_delegation_sources,
    validate_server_delegated_live_data,
)
from ragtime.userspace.models import UpsertWorkspaceFileRequest, UserSpaceLiveDataCheck, UserSpaceLiveDataConnection
from ragtime.userspace.service import userspace_service

SERVER_SRC = """
const BRIDGE = process.env.RAGTIME_BRIDGE_URL;
app.get("/api/cash", async (req, res) => {
  const r = await fetch(`${BRIDGE}/execute-component`, {
    method: "POST",
    headers: { Authorization: `Bearer ${process.env.RAGTIME_BRIDGE_TOKEN}`,
               "Content-Type": "application/json" },
    body: JSON.stringify({ component_id: "comp-1", request: { query: "SELECT 1 LIMIT 1" } }),
  });
  res.json(await r.json());
});
"""

ENTRYPOINT_CONFIG = json.dumps({"command": "node server.js"})
DASHBOARD_FETCH_SRC = 'export async function render(_container, _context) { const res = await fetch("/api/cash"); return await res.json(); }'


class ServerDelegatedContractTests(unittest.TestCase):
    def test_server_source_with_component_id_satisfies_delegation(self) -> None:
        ok, missing = validate_server_delegated_live_data({"server.js": SERVER_SRC}, ["comp-1"])
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_component_id_absent_from_server_sources_fails(self) -> None:
        ok, missing = validate_server_delegated_live_data({"server.js": SERVER_SRC}, ["comp-2"])
        self.assertFalse(ok)
        self.assertEqual(missing, ["comp-2"])

    def test_bridge_reference_without_component_id_fails(self) -> None:
        src = 'fetch(process.env.RAGTIME_BRIDGE_URL + "/execute-component")'
        ok, missing = validate_server_delegated_live_data({"server.js": src}, ["comp-1"])
        self.assertFalse(ok)
        self.assertEqual(missing, ["comp-1"])


class RuntimeBridgeContractIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.workspace_id = "workspace-bridge"
        self.user_id = "user-1"
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workspaces_dir = Path(self.temp_dir.name) / "workspaces"
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir = self.workspaces_dir / self.workspace_id / "files"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        userspace_service._execution_proofs.pop(self.workspace_id, None)
        self.addCleanup(lambda: userspace_service._execution_proofs.pop(self.workspace_id, None))

    def _workspace(self) -> SimpleNamespace:
        return SimpleNamespace(
            sqlite_persistence_mode="exclude",
            tool_selection_mode="custom",
            selected_tool_ids=["comp-1"],
            selected_tool_group_ids=[],
        )

    def _write_workspace_file(self, relative_path: str, content: str) -> None:
        target = self.files_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def _workspace_file_rows(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(path=str(path.relative_to(self.files_dir))) for path in self.files_dir.rglob("*") if path.is_file()]

    @contextmanager
    def _service_environment(self, *extra_patchers: AbstractContextManager[object]) -> Iterator[None]:
        with ExitStack() as stack:
            base_patchers: tuple[AbstractContextManager[object], ...] = (
                mock.patch.object(userspace_service, "_base_dir", Path(self.temp_dir.name)),
                mock.patch.object(userspace_service, "_workspaces_dir", self.workspaces_dir),
                mock.patch.object(userspace_service, "_enforce_workspace_access", new=mock.AsyncMock(return_value=self._workspace())),
                mock.patch.object(userspace_service, "_ensure_workspace_git_repo", new=mock.AsyncMock()),
                mock.patch.object(
                    userspace_service,
                    "ensure_workspace_path_not_in_disabled_mount",
                    new=mock.AsyncMock(side_effect=lambda _workspace_id, normalized_path: normalized_path),
                ),
                mock.patch.object(userspace_service, "resolve_workspace_mounts_for_runtime", new=mock.AsyncMock(return_value=[])),
                mock.patch.object(
                    userspace_service,
                    "list_workspace_files",
                    new=mock.AsyncMock(side_effect=lambda *_args, **_kwargs: self._workspace_file_rows()),
                ),
                mock.patch.object(userspace_service, "_resolve_effective_workspace_tool_ids", new=mock.AsyncMock(return_value=["comp-1"])),
                mock.patch.object(userspace_service, "clear_workspace_changed_file_acknowledgements_for_paths_for_all_users", new=mock.AsyncMock()),
                mock.patch.object(userspace_service, "_touch_workspace", new=mock.AsyncMock()),
                mock.patch.object(userspace_service, "_mark_workspace_code_index_dirty", new=mock.AsyncMock()),
                mock.patch.object(userspace_service, "enforce_workspace_role", new=mock.AsyncMock()),
                mock.patch.object(userspace_service, "get_workspace", new=mock.AsyncMock(return_value=self._workspace())),
                mock.patch("ragtime.rag.components.repository.list_healthy_enabled_tool_ids", new=mock.AsyncMock(return_value=["comp-1"])),
                mock.patch("ragtime.rag.components.repository.get_tool_ids_for_groups", new=mock.AsyncMock(return_value=[])),
            )
            for patcher in base_patchers:
                stack.enter_context(patcher)
            for patcher in extra_patchers:
                stack.enter_context(patcher)
            yield

    def _dashboard_fetch_request(self) -> UpsertWorkspaceFileRequest:
        return UpsertWorkspaceFileRequest(
            content=DASHBOARD_FETCH_SRC,
            artifact_type="module_ts",
            live_data_requested=True,
            live_data_connections=[
                UserSpaceLiveDataConnection(
                    component_kind="tool_config",
                    component_id="comp-1",
                    request={"query": "SELECT 1 LIMIT 1"},
                )
            ],
            live_data_checks=[
                UserSpaceLiveDataCheck(
                    component_id="comp-1",
                    connection_check_passed=True,
                    transformation_check_passed=True,
                )
            ],
        )

    async def test_upsert_tool_accepts_dashboard_fetch_when_server_delegates_and_proofs_exist(self) -> None:
        self._write_workspace_file(".ragtime/runtime-entrypoint.json", ENTRYPOINT_CONFIG)
        self._write_workspace_file("server.js", SERVER_SRC)
        userspace_service.record_execution_proof(self.workspace_id, "comp-1", 1, "SELECT 1 LIMIT 1")

        with self._service_environment():
            tool = next(
                tool for tool in await RAGComponents()._create_userspace_file_tools(self.workspace_id, self.user_id) if tool.name == "upsert_userspace_file"
            )
            coroutine = tool.coroutine
            assert coroutine is not None

            raw = await coroutine(
                path="dashboard/main.ts",
                content=DASHBOARD_FETCH_SRC,
                artifact_type="module_ts",
                live_data_requested=True,
                live_data_connections=[
                    {
                        "component_kind": "tool_config",
                        "component_id": "comp-1",
                        "request": {"query": "SELECT 1 LIMIT 1"},
                    }
                ],
                live_data_checks=[
                    {
                        "component_id": "comp-1",
                        "connection_check_passed": True,
                        "transformation_check_passed": True,
                    }
                ],
            )

        payload = json.loads(raw)
        self.assertEqual(payload["status"], "persisted")
        self.assertFalse(payload.get("contract_violations"))

    async def test_load_server_delegation_sources_skips_non_404_import_read_failures(self) -> None:
        async def fake_get_workspace_file(workspace_id: str, path: str, user_id: str) -> SimpleNamespace:
            self.assertEqual(workspace_id, self.workspace_id)
            self.assertEqual(user_id, self.user_id)
            if path == ".ragtime/runtime-entrypoint.json":
                return SimpleNamespace(content=ENTRYPOINT_CONFIG)
            if path == "server.js":
                return SimpleNamespace(content='import bridge from "./bridge";\nexport const ok = bridge;\n')
            if path == "bridge.js":
                raise HTTPException(status_code=415, detail="Unsupported")
            if path == "bridge.ts":
                raise HTTPException(status_code=404, detail="Not found")
            raise AssertionError(f"unexpected path: {path}")

        with mock.patch.object(userspace_service, "get_workspace_file", new=mock.AsyncMock(side_effect=fake_get_workspace_file)):
            result = await load_workspace_server_delegation_sources(self.workspace_id, self.user_id)

        self.assertEqual(
            result,
            {"server.js": 'import bridge from "./bridge";\nexport const ok = bridge;\n'},
        )

    async def test_service_rejects_dashboard_fetch_without_server_delegation(self) -> None:
        self._write_workspace_file(".ragtime/runtime-entrypoint.json", ENTRYPOINT_CONFIG)
        userspace_service.record_execution_proof(self.workspace_id, "comp-1", 1, "SELECT 1 LIMIT 1")

        with self._service_environment(
            mock.patch(
                "ragtime.rag.components.validate_live_data_binding",
                new=mock.AsyncMock(
                    return_value={
                        "ok": True,
                        "validator_available": True,
                        "has_execute_calls": False,
                        "found_component_ids": [],
                        "has_local_imports": False,
                        "has_context_components_access": False,
                        "missing_component_ids": ["comp-1"],
                    }
                ),
            ),
            mock.patch(
                "ragtime.rag.components.load_workspace_server_delegation_sources",
                new=mock.AsyncMock(return_value={}),
            ),
        ):
            with self.assertRaises(HTTPException) as exc_info:
                await userspace_service.upsert_workspace_file(
                    self.workspace_id,
                    "dashboard/main.ts",
                    self._dashboard_fetch_request(),
                    self.user_id,
                )

        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertIn("Live data binding not found in module source", str(exc_info.exception.detail))

    async def test_service_rejects_delegated_dashboard_fetch_without_execution_proof(self) -> None:
        self._write_workspace_file(".ragtime/runtime-entrypoint.json", ENTRYPOINT_CONFIG)
        self._write_workspace_file("server.js", SERVER_SRC)

        with self._service_environment():
            with self.assertRaises(HTTPException) as exc_info:
                await userspace_service.upsert_workspace_file(
                    self.workspace_id,
                    "dashboard/main.ts",
                    self._dashboard_fetch_request(),
                    self.user_id,
                )

        detail = str(exc_info.exception.detail)
        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertIn("No server-verified execution proof", detail)
        self.assertNotIn("Live data binding not found in module source", detail)

    async def test_service_accepts_dashboard_fetch_when_server_delegates_and_proofs_exist(self) -> None:
        self._write_workspace_file(".ragtime/runtime-entrypoint.json", ENTRYPOINT_CONFIG)
        self._write_workspace_file("server.js", SERVER_SRC)
        userspace_service.record_execution_proof(self.workspace_id, "comp-1", 1, "SELECT 1 LIMIT 1")

        with self._service_environment():
            result = await userspace_service.upsert_workspace_file(
                self.workspace_id,
                "dashboard/main.ts",
                self._dashboard_fetch_request(),
                self.user_id,
            )

        self.assertEqual(result.path, "dashboard/main.ts")
        self.assertEqual(result.content, DASHBOARD_FETCH_SRC)
        self.assertEqual(result.artifact_type, "module_ts")
        self.assertEqual(result.live_data_connections, self._dashboard_fetch_request().live_data_connections)
        self.assertEqual(result.live_data_checks, self._dashboard_fetch_request().live_data_checks)
