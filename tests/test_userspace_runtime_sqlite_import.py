"""Control-plane runtime import transport contracts."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

from ragtime.userspace import service as service_module


class RuntimeSqliteImportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "upload.sqlite3"
        with sqlite3.connect(self.source) as database:
            database.execute("CREATE TABLE uploads (id INTEGER PRIMARY KEY)")
        self.context = SimpleNamespace(
            owner_workspace_id="workspace-1",
            owner_workspace_name="Workspace",
            ownership="owner",
            access_mode="read_write",
            persistence_mode="include",
        )

    def _service(self) -> service_module.UserSpaceService:
        service = SimpleNamespace()
        service._resolve_sqlite_inspector_database = mock.AsyncMock(return_value=(self.root, False, self.context, None))
        service._build_sqlite_inspector_database_summary = mock.Mock(return_value=SimpleNamespace(owner_workspace_id="workspace-1"))
        service._record_linked_sqlite_mutation_best_effort = mock.AsyncMock()
        return cast(service_module.UserSpaceService, service)

    async def test_active_import_uses_runtime_transport_without_local_fence(self) -> None:
        service = self._service()
        history = SimpleNamespace(runtime_history_active=mock.AsyncMock(return_value=True))
        response = SimpleNamespace(status_code=200, text="", json=lambda: {"size_bytes": 113})
        client = mock.AsyncMock()
        client.post.return_value = response
        client_context = mock.MagicMock()
        client_context.__aenter__ = mock.AsyncMock(return_value=client)
        client_context.__aexit__ = mock.AsyncMock(return_value=False)

        with (
            mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history),
            mock.patch.object(
                service_module, "get_runtime_manager_request_config", return_value=SimpleNamespace(base_url="http://runtime", headers={}, timeout_seconds=5)
            ),
            mock.patch.object(service_module.httpx, "AsyncClient", return_value=client_context),
            mock.patch.object(service_module, "sqlite_workspace_access", side_effect=AssertionError("local fence entered")),
        ):
            await service_module.UserSpaceService.import_sqlite_database(service, "workspace-1", "user-1", "app.sqlite3", self.source)

        self.assertTrue(client.post.await_args.args[0].endswith("/sqlite-history/import-database/app.sqlite3"))
        history.runtime_history_active.assert_awaited_once()

    async def test_passive_import_fences_captures_and_publishes_locally(self) -> None:
        service = self._service()
        files_dir = self.root / "files"
        files_dir.mkdir()
        history = SimpleNamespace(
            runtime_history_active=mock.AsyncMock(return_value=False),
            capture_workspace_databases=mock.AsyncMock(),
        )
        entered = False

        @asynccontextmanager
        async def fence(workspace_id: str, *, maintenance: bool):
            nonlocal entered
            self.assertEqual("workspace-1", workspace_id)
            self.assertTrue(maintenance)
            entered = True
            yield files_dir

        published: list[tuple[Path, str, Path, str]] = []

        def publish(source_root: Path, source_name: str, target_root: Path, target_name: str) -> None:
            published.append((source_root, source_name, target_root, target_name))
            destination = target_root / target_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((source_root / source_name).read_bytes())

        with (
            mock.patch("ragtime.userspace.sqlite_history.get_sqlite_history_service", return_value=history),
            mock.patch.object(service_module, "sqlite_workspace_access", fence),
            mock.patch("runtime.core.secure_files.publish_regular_file", publish),
        ):
            await service_module.UserSpaceService.import_sqlite_database(service, "workspace-1", "user-1", "app.sqlite3", self.source)

        self.assertTrue(entered)
        history.capture_workspace_databases.assert_awaited_once()
        self.assertTrue(history.capture_workspace_databases.await_args.kwargs["mandatory"])
        self.assertEqual("pre_restore", history.capture_workspace_databases.await_args.kwargs["trigger"])
        self.assertEqual(1, len(published))
