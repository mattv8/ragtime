import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from fastapi import HTTPException

from ragtime.indexer.models import UpdateSettingsRequest, WorkspaceCodeIndexJobResponse, WorkspaceCodeIndexJobStatus
from ragtime.indexer.repository import IndexerRepository
from ragtime.indexer.tool_presentation import HIDDEN_TOOL_VISIBILITY, normalize_tool_presentation
from ragtime.userspace.routes import (
    cancel_workspace_code_index_job,
    delete_workspace_code_index_admin,
    list_workspace_code_index_jobs,
    list_workspace_code_indexes,
    reconcile_workspace_code_indexes_now,
    reindex_workspace_code_index,
    workspace_code_index_service,
)


def _settings_row(**overrides: Any) -> SimpleNamespace:
    values = {
        "id": "default",
        "serverName": "Ragtime",
        "defaultThemePack": "default",
        "openaiApiKey": "",
        "anthropicApiKey": "",
        "ollamaProtocol": "http",
        "ollamaHost": "localhost",
        "ollamaPort": 11434,
        "ollamaBaseUrl": "http://localhost:11434",
        "allowedChatModels": [],
        "enabledTools": [],
        "postgresHost": "",
        "postgresUser": "",
        "postgresPassword": "",
        "postgresDb": "",
        "enableWriteOps": False,
        "updatedAt": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _FakeAppSettingsClient:
    def __init__(self, row: SimpleNamespace) -> None:
        self._row = row
        self.last_update_data: dict[str, Any] | None = None

    async def find_unique(self, *, where: dict[str, str]) -> SimpleNamespace:
        return self._row

    async def update(self, *, where: dict[str, str], data: dict[str, Any]) -> SimpleNamespace:
        self.last_update_data = data
        for key, value in data.items():
            setattr(self._row, key, value)
        return self._row


class _FakeDb:
    def __init__(self, row: SimpleNamespace) -> None:
        self.appsettings = _FakeAppSettingsClient(row)


class WorkspaceCodeIndexServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_collect_indexable_files_uses_only_canonical_workspace_files(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        with tempfile.TemporaryDirectory() as tmpdir:
            files_root = Path(tmpdir) / "files"
            (files_root / "src").mkdir(parents=True)
            (files_root / "src" / "app.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
            (files_root / "mounted" / "src").mkdir(parents=True)
            (files_root / "mounted" / "src" / "external.py").write_text("def skip_me():\n    pass\n", encoding="utf-8")
            (files_root / ".ragtime").mkdir()
            (files_root / ".ragtime" / "runtime-entrypoint.json").write_text("{}", encoding="utf-8")
            (files_root / "node_modules" / "pkg").mkdir(parents=True)
            (files_root / "node_modules" / "pkg" / "index.js").write_text("console.log('skip')\n", encoding="utf-8")

            class FakeUserSpaceService:
                def _workspace_files_dir(self, workspace_id: str) -> Path:
                    self.workspace_id = workspace_id
                    return files_root

                async def _resolve_workspace_tree_file_path(self, *_args, **_kwargs):  # pragma: no cover - must not be called
                    raise AssertionError("workspace code indexing must not resolve mounted tree paths")

                async def _list_workspace_mount_target_repo_paths(self, workspace_id: str) -> list[str]:
                    self.mount_workspace_id = workspace_id
                    return ["mounted"]

                def _workspace_path_matches_mount_prefix(self, path: str, prefix: str) -> bool:
                    return path == prefix or path.startswith(f"{prefix}/")

            service = WorkspaceCodeIndexService(userspace_service=FakeUserSpaceService())

            files = await service.collect_indexable_files("workspace-1")

        self.assertEqual([path.relative_to(files_root).as_posix() for path in files], ["src/app.py"])

    def test_dirty_operation_coalescing_preserves_latest_file_state(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        service = WorkspaceCodeIndexService()

        self.assertEqual(service.coalesce_dirty_operation("upsert", "upsert"), "upsert")
        self.assertEqual(service.coalesce_dirty_operation("upsert", "delete"), "delete")
        self.assertEqual(service.coalesce_dirty_operation("delete", "upsert"), "upsert")
        self.assertEqual(service.coalesce_dirty_operation("reindex", "delete"), "reindex")

    async def test_reconcile_stale_workspaces_schedules_persisted_dirty_rows(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        class FakeDb:
            def __init__(self) -> None:
                self.query = ""

            async def query_raw(self, query: str):
                self.query = query
                return [{"workspace_id": "workspace-1"}]

        fake_db = FakeDb()
        service = WorkspaceCodeIndexService()
        scheduled: list[str] = []

        def schedule_workspace(workspace_id: str) -> bool:
            scheduled.append(workspace_id)
            return True

        with (
            mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=fake_db),
            mock.patch.object(service, "schedule_workspace", side_effect=schedule_workspace),
        ):
            result = await service.reconcile_stale_workspaces()

        self.assertIn("attempt_count < 3", fake_db.query)
        self.assertEqual(scheduled, ["workspace-1"])
        self.assertEqual(result, ["workspace-1"])

    async def test_reconcile_with_missing_workspaces_queues_baseline_reindex(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        class FakeDb:
            def __init__(self) -> None:
                self.queries: list[str] = []

            async def query_raw(self, query: str):
                self.queries.append(query)
                if "FROM workspace_code_index_dirty_paths" in query:
                    return []
                return [{"workspace_id": "workspace-missing"}]

        fake_db = FakeDb()
        service = WorkspaceCodeIndexService()
        service.mark_dirty = mock.AsyncMock()  # type: ignore[method-assign]

        with mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=fake_db):
            result = await service.reconcile_stale_workspaces(include_missing=True)

        service.mark_dirty.assert_awaited_once_with("workspace-missing", operation="reindex")
        self.assertEqual(result, ["workspace-missing"])
        self.assertTrue(any("workspace_code_index_states" in query for query in fake_db.queries))

    async def test_mark_dirty_is_noop_when_code_indexing_disabled(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        service = WorkspaceCodeIndexService()

        with (
            mock.patch(
                "ragtime.userspace.workspace_code_index_service.get_app_settings",
                mock.AsyncMock(return_value={"userspace_code_index_enabled": False}),
            ),
            mock.patch("ragtime.userspace.workspace_code_index_service.get_db", mock.AsyncMock()) as get_db,
        ):
            await service.mark_dirty("workspace-1", "src/app.py", "upsert")

        get_db.assert_not_called()

    def test_schedule_workspace_does_not_cancel_active_worker(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        class FakeTask:
            cancelled = False

            def done(self) -> bool:
                return False

            def cancel(self) -> None:
                self.cancelled = True

        service = WorkspaceCodeIndexService()
        active_task = FakeTask()
        service._workspace_tasks["workspace-1"] = active_task  # type: ignore[assignment]

        with mock.patch("ragtime.userspace.workspace_code_index_service.asyncio.create_task") as create_task:
            service.schedule_workspace("workspace-1")

        self.assertFalse(active_task.cancelled)
        create_task.assert_not_called()

    async def test_mark_dirty_resets_failed_attempt_count_for_fresh_edits(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        class FakeDb:
            def __init__(self) -> None:
                self.executed: list[str] = []

            async def query_raw(self, _query: str):
                return [{"operation": "upsert"}]

            async def execute_raw(self, query: str):
                self.executed.append(query)
                return 1

        fake_db = FakeDb()
        service = WorkspaceCodeIndexService()
        scheduled: list[str] = []

        def schedule_workspace(workspace_id: str) -> bool:
            scheduled.append(workspace_id)
            return True

        with (
            mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=fake_db),
            mock.patch.object(service, "schedule_workspace", side_effect=schedule_workspace),
        ):
            await service.mark_dirty("workspace-1", "src/app.py", "upsert")

        dirty_upsert_sql = next(query for query in fake_db.executed if "workspace_code_index_dirty_paths" in query and "ON CONFLICT" in query)
        self.assertIn("attempt_count = 0", dirty_upsert_sql)
        self.assertEqual(scheduled, ["workspace-1"])

    async def test_index_file_does_not_mark_workspace_ready_mid_reindex(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        with tempfile.TemporaryDirectory() as tmpdir:
            files_root = Path(tmpdir)
            (files_root / "src").mkdir()
            (files_root / "src" / "app.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")

            class FakeUserSpaceService:
                def _workspace_files_dir(self, _workspace_id: str) -> Path:
                    return files_root

            class FakeBackend:
                async def store_embeddings(self, *_args, **_kwargs):
                    return None

            service = WorkspaceCodeIndexService(userspace_service=FakeUserSpaceService())
            service.ensure_state = mock.AsyncMock(return_value="userspace_workspace_1")  # type: ignore[method-assign]
            service._workspace_mount_prefixes = mock.AsyncMock(return_value=[])  # type: ignore[method-assign]
            service.delete_file_from_index = mock.AsyncMock()  # type: ignore[method-assign]
            service._filesystem_indexer._load_and_chunk_file = mock.AsyncMock(return_value=["chunk"])  # type: ignore[method-assign]
            service._upsert_file_metadata = mock.AsyncMock()  # type: ignore[method-assign]
            service._replace_symbols = mock.AsyncMock()  # type: ignore[method-assign]
            service._refresh_state_counts = mock.AsyncMock()  # type: ignore[method-assign]

            with (
                mock.patch("ragtime.userspace.workspace_code_index_service.get_app_settings", mock.AsyncMock(return_value={})),
                mock.patch("ragtime.userspace.workspace_code_index_service.get_embeddings_model", mock.AsyncMock(return_value=object())),
                mock.patch("ragtime.userspace.workspace_code_index_service.embed_documents_subbatched", mock.AsyncMock(return_value=[[0.1]])),
                mock.patch("ragtime.userspace.workspace_code_index_service.get_pgvector_backend", return_value=FakeBackend()),
                mock.patch.object(
                    IndexerRepository,
                    "get_settings",
                    mock.AsyncMock(return_value=SimpleNamespace(get_embedding_config_hash=lambda: "hash-1")),
                ),
            ):
                result = await service.index_file("workspace-1", "src/app.py")

        service._refresh_state_counts.assert_not_awaited()
        self.assertEqual(result, (1, 1))

    def test_workspace_code_index_job_progress_is_phase_aware(self) -> None:
        base: dict[str, Any] = {
            "id": "job-1",
            "workspace_id": "ws-1",
            "workspace_name": "Workspace",
            "index_name": "userspace_workspace_ws_1",
            "status": "indexing",
            "total_files": 10,
            "processed_files": 5,
            "total_chunks": 100,
            "processed_chunks": 25,
            "current_file": "src/app.py",
            "error_message": None,
            "created_at": "2026-07-01T00:00:00Z",
            "started_at": "2026-07-01T00:00:01Z",
            "completed_at": None,
        }

        cases: list[tuple[dict[str, Any], float]] = [
            ({"phase": "collecting"}, 0.0),
            ({"phase": "loading_files"}, 35.0),
            ({"phase": "loading_files", "total_files": 0}, 0.0),
            ({"phase": "chunking"}, 75.0),
            ({"phase": "embedding"}, 84.75),
            ({"phase": "embedding", "total_chunks": 0}, 80.0),
            ({"phase": "embedding", "processed_chunks": 200}, 99.0),
            ({"phase": "indexing_symbols"}, 98.0),
            ({"phase": "finalizing"}, 99.0),
            ({"phase": "finalizing", "status": "completed"}, 100.0),
            ({"phase": "embedding", "status": "failed"}, 0.0),
            ({"phase": "loading_files", "status": "pending"}, 0.0),
        ]

        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                data: dict[str, Any] = {**base, **overrides}
                job = WorkspaceCodeIndexJobResponse(**data)
                self.assertEqual(job.progress_percent, expected)

    async def test_update_job_progress_persists_phase(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexJobPhase, WorkspaceCodeIndexService

        class FakeDb:
            def __init__(self) -> None:
                self.executed: list[str] = []

            async def execute_raw(self, query: str):
                self.executed.append(query)
                return 1

        fake_db = FakeDb()
        service = WorkspaceCodeIndexService()

        with mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=fake_db):
            await service._update_job_progress(
                "job-1",
                total_files=12,
                processed_files=3,
                current_file="src/app.py",
                phase=WorkspaceCodeIndexJobPhase.EMBEDDING,
            )

        self.assertEqual(len(fake_db.executed), 1)
        self.assertIn("phase = 'embedding'", fake_db.executed[0])
        self.assertIn("total_files = 12", fake_db.executed[0])
        self.assertIn("processed_files = 3", fake_db.executed[0])

    def test_settings_response_includes_userspace_code_index_concurrency_default(self) -> None:
        from ragtime.indexer.models import SettingsResponse

        settings = SettingsResponse()

        self.assertEqual(settings.userspace_code_index_max_concurrency, 1)

    def test_workspace_code_index_job_response_includes_queue_fields(self) -> None:
        from datetime import datetime, timezone

        from ragtime.indexer.models import WorkspaceCodeIndexJobResponse

        response = WorkspaceCodeIndexJobResponse(
            id="job-2",
            workspace_id="workspace-2",
            workspace_name="Workspace 2",
            index_name="userspace_workspace_2",
            status=WorkspaceCodeIndexJobStatus.PENDING,
            waiting_for_job_id="job-1",
            cancel_requested=True,
            created_at=datetime.now(timezone.utc),
        )

        self.assertEqual(response.waiting_for_job_id, "job-1")
        self.assertTrue(response.cancel_requested)

    async def test_queue_relink_sql_orders_pending_jobs(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        class FakeDb:
            def __init__(self) -> None:
                self.executed: list[str] = []

            async def query_raw(self, query: str):
                if "WHERE status = 'pending'" in query:
                    return [{"id": "job-1"}, {"id": "job-2"}]
                return []

            async def execute_raw(self, query: str):
                self.executed.append(query)
                return 1

        fake_db = FakeDb()
        service = WorkspaceCodeIndexService()

        with mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=fake_db):
            await service._relink_pending_jobs()

        self.assertTrue(any("waiting_for_job_id = NULL" in query for query in fake_db.executed))
        self.assertTrue(any("waiting_for_job_id = 'job-1'" in query for query in fake_db.executed))

    async def test_relink_pending_jobs_points_head_at_running_job(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        class FakeDb:
            def __init__(self) -> None:
                self.executed: list[str] = []

            async def query_raw(self, query: str):
                if "status = 'indexing'" in query:
                    return [{"id": "job-running"}]
                if "WHERE status = 'pending'" in query:
                    return [{"id": "job-1"}, {"id": "job-2"}]
                return []

            async def execute_raw(self, query: str):
                self.executed.append(query)
                return 1

        fake_db = FakeDb()
        service = WorkspaceCodeIndexService()

        with mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=fake_db):
            await service._relink_pending_jobs()

        self.assertTrue(any("waiting_for_job_id = 'job-running'" in query for query in fake_db.executed))
        self.assertTrue(any("waiting_for_job_id = 'job-1'" in query for query in fake_db.executed))
        self.assertFalse(any("waiting_for_job_id = NULL" in query for query in fake_db.executed))

    async def test_delete_workspace_index_flags_active_job_cancelled(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        class FakeDb:
            def __init__(self) -> None:
                self.executed: list[str] = []

            async def query_raw(self, query: str):
                if "status = 'indexing'" in query and "workspace-1" in query:
                    return [{"id": "job-9"}]
                return []

            async def execute_raw(self, query: str):
                self.executed.append(query)
                return 1

        class FakeBackend:
            async def delete_index(self, _index_name: str) -> None:
                return None

        fake_db = FakeDb()
        service = WorkspaceCodeIndexService()

        with (
            mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=fake_db),
            mock.patch("ragtime.userspace.workspace_code_index_service.get_pgvector_backend", return_value=FakeBackend()),
        ):
            await service.delete_workspace_index("workspace-1")

        self.assertTrue(service._cancellation_flags.get("job-9"))
        self.assertTrue(any("DELETE FROM workspace_code_index_jobs" in query for query in fake_db.executed))

    async def test_workspace_queue_concurrency_setting_defaults_to_one(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        service = WorkspaceCodeIndexService()

        with mock.patch(
            "ragtime.userspace.workspace_code_index_service.get_app_settings",
            mock.AsyncMock(return_value={}),
        ):
            self.assertEqual(await service._max_concurrent_jobs(), 1)

    async def test_claim_next_pending_job_does_not_depend_on_waiting_link(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        class FakeDb:
            def __init__(self) -> None:
                self.claim_query = ""

            async def query_raw(self, query: str):
                self.claim_query = query
                return [{"id": "job-2", "workspace_id": "workspace-2"}]

        fake_db = FakeDb()
        service = WorkspaceCodeIndexService()

        with mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=fake_db):
            claimed = await service._claim_next_pending_job()

        self.assertEqual(claimed, ("job-2", "workspace-2"))
        self.assertNotIn("waiting_for_job_id IS NULL", fake_db.claim_query)

    async def test_claim_next_pending_job_skips_active_workspaces(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        class FakeDb:
            def __init__(self) -> None:
                self.claim_query = ""

            async def query_raw(self, query: str):
                self.claim_query = query
                return [{"id": "job-2", "workspace_id": "workspace-2"}]

        fake_db = FakeDb()
        service = WorkspaceCodeIndexService()
        service._active_workspace_ids = {"workspace-1"}

        with mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=fake_db):
            claimed = await service._claim_next_pending_job(service._active_workspace_ids)

        self.assertEqual(claimed, ("job-2", "workspace-2"))
        self.assertIn("workspace_id NOT IN ('workspace-1')", fake_db.claim_query)
        self.assertNotIn("workspace_id NOT IN ()", fake_db.claim_query)

    async def test_reindex_job_failure_does_not_double_fail(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        class FakeDb:
            def __init__(self) -> None:
                self.queries: list[str] = []

            async def query_raw(self, query: str):
                self.queries.append(query)
                if "workspace_code_index_dirty_paths" in query and "SELECT" in query:
                    return [{"id": "dirty-1", "path": "", "operation": "reindex"}]
                return []

            async def execute_raw(self, query: str):
                self.queries.append(query)
                return 1

        fake_db = FakeDb()
        service = WorkspaceCodeIndexService()
        service.ensure_state = mock.AsyncMock(return_value="userspace_workspace_ws_1")  # type: ignore[method-assign]
        service._refresh_state_counts = mock.AsyncMock()  # type: ignore[method-assign]
        service.reindex_workspace = mock.AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
        service._fail_job = mock.AsyncMock()  # type: ignore[method-assign]

        with mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=fake_db):
            await service._process_dirty_workspace_for_job("ws-1", "job-1")

        service._fail_job.assert_awaited_once_with("job-1", "boom")

    async def test_cancelled_job_does_not_enqueue_follow_up(self) -> None:
        from ragtime.userspace.workspace_code_index_service import (
            WorkspaceCodeIndexCancelled,
            WorkspaceCodeIndexService,
        )

        service = WorkspaceCodeIndexService()
        service._process_dirty_workspace_for_job = mock.AsyncMock(side_effect=WorkspaceCodeIndexCancelled)  # type: ignore[method-assign]
        service._cancel_job = mock.AsyncMock()  # type: ignore[method-assign]
        service._relink_pending_jobs = mock.AsyncMock()  # type: ignore[method-assign]
        service._maybe_enqueue_follow_up = mock.AsyncMock()  # type: ignore[method-assign]

        await service._run_claimed_job("ws-1", "job-1")

        service._maybe_enqueue_follow_up.assert_not_awaited()
        service._cancel_job.assert_awaited_once_with("job-1", "ws-1")

    async def test_recover_orphaned_jobs_wakes_queue_runner(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        class FakeDb:
            async def execute_raw(self, _query: str):
                return 1

            async def query_raw(self, query: str):
                if "WHERE status = 'pending'" in query:
                    return []
                return []

        service = WorkspaceCodeIndexService()

        with mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=FakeDb()):
            await service._recover_orphaned_jobs()

        self.assertTrue(service._queue_event.is_set())

    async def test_cancel_job_returns_false_for_missing_job(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        class FakeDb:
            async def query_raw(self, _query: str):
                return []

        service = WorkspaceCodeIndexService()

        with mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=FakeDb()):
            result = await service.cancel_job("job-missing")

        self.assertFalse(result)

    async def test_cancel_job_returns_false_for_terminal_jobs(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        service = WorkspaceCodeIndexService()

        for status in ("completed", "failed", "cancelled"):
            with self.subTest(status=status):

                class FakeDb:
                    def __init__(self, job_status: str) -> None:
                        self.job_status = job_status

                    async def query_raw(self, _query: str):
                        return [{"id": "job-1", "workspace_id": "ws-1", "status": self.job_status}]

                fake_db = FakeDb(status)
                with mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=fake_db):
                    result = await service.cancel_job("job-1")
                self.assertFalse(result)

    async def test_cancel_job_cancels_pending_job_and_relinks_queue(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        class FakeDb:
            def __init__(self) -> None:
                self.queries: list[str] = []
                self.job_status = "pending"

            async def query_raw(self, query: str):
                if "status = 'indexing'" in query:
                    # No running job in this scenario; relink seeds from nothing.
                    return []
                if "workspace_code_index_jobs" in query and "SELECT" in query:
                    return [{"id": "job-1", "workspace_id": "ws-1", "status": self.job_status}]
                if "WHERE status = 'pending'" in query:
                    return [{"id": "job-2"}]
                return []

            async def execute_raw(self, query: str):
                self.queries.append(query)
                if "SET status = 'cancelled'" in query:
                    self.job_status = "cancelled"
                return 1

        fake_db = FakeDb()
        service = WorkspaceCodeIndexService()

        with mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=fake_db):
            result = await service.cancel_job("job-1")

        self.assertTrue(result)
        self.assertTrue(any("status = 'cancelled'" in q for q in fake_db.queries))
        self.assertTrue(any("current_file = NULL" in q for q in fake_db.queries))
        self.assertTrue(any("completed_at = NOW()" in q for q in fake_db.queries))
        self.assertTrue(any("cancel_requested = FALSE" in q for q in fake_db.queries))
        self.assertTrue(any("waiting_for_job_id = NULL" in q for q in fake_db.queries))
        self.assertTrue(service._queue_event.is_set())

    async def test_cancel_job_flags_indexing_job_and_updates_db(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        class FakeDb:
            def __init__(self) -> None:
                self.queries: list[str] = []

            async def query_raw(self, _query: str):
                return [{"id": "job-1", "workspace_id": "ws-1", "status": "indexing"}]

            async def execute_raw(self, query: str):
                self.queries.append(query)
                return 1

        fake_db = FakeDb()
        service = WorkspaceCodeIndexService()

        with mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=fake_db):
            result = await service.cancel_job("job-1")

        self.assertTrue(result)
        self.assertTrue(service._cancellation_flags.get("job-1"))
        self.assertTrue(any("cancel_requested = TRUE" in q for q in fake_db.queries))
        self.assertFalse(any("status = 'cancelled'" in q for q in fake_db.queries))

    async def test_cancel_job_handles_pending_to_indexing_race(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        class FakeDb:
            def __init__(self) -> None:
                self.queries: list[str] = []
                self.select_count = 0

            async def query_raw(self, query: str) -> list[dict[str, Any]]:
                self.queries.append(query)
                if "workspace_code_index_jobs" in query and "SELECT" in query:
                    self.select_count += 1
                    status = "pending" if self.select_count == 1 else "indexing"
                    return [{"id": "job-1", "workspace_id": "ws-1", "status": status}]
                return []

            async def execute_raw(self, query: str) -> int:
                self.queries.append(query)
                if "status = 'cancelled'" in query:
                    return 0
                return 1

        fake_db = FakeDb()
        service = WorkspaceCodeIndexService()

        with (
            mock.patch("ragtime.userspace.workspace_code_index_service.get_db", return_value=fake_db),
            mock.patch.object(service, "_relink_pending_jobs") as relink,
        ):
            result = await service.cancel_job("job-1")

        self.assertTrue(result)
        self.assertTrue(service._cancellation_flags.get("job-1"))
        self.assertTrue(any("status = 'cancelled'" in q and "AND status = 'pending'" in q for q in fake_db.queries))
        self.assertTrue(any("cancel_requested = TRUE" in q for q in fake_db.queries))
        relink.assert_not_awaited()

    def test_userspace_code_context_tools_are_hidden_from_ui_presentation(self) -> None:
        for tool_name in ("search_userspace_code", "assay_userspace_code"):
            with self.subTest(tool_name=tool_name):
                self.assertEqual(
                    normalize_tool_presentation(tool_name),
                    {"visibility": HIDDEN_TOOL_VISIBILITY},
                )


class WorkspaceCodeIndexSettingsTests(unittest.IsolatedAsyncioTestCase):
    async def test_service_reads_dynamic_settings_from_app_settings(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        service = WorkspaceCodeIndexService()
        fake_settings = {
            "userspace_code_index_enabled": True,
            "userspace_code_index_debounce_seconds": 5,
            "userspace_code_index_reconcile_interval_seconds": 600,
            "userspace_code_index_max_attempts": 7,
        }

        with mock.patch(
            "ragtime.userspace.workspace_code_index_service.get_app_settings",
            mock.AsyncMock(return_value=fake_settings),
        ):
            self.assertTrue(await service._enabled())
            self.assertEqual(await service._debounce_seconds(), 5)
            self.assertEqual(await service._reconcile_interval_seconds(), 600)
            self.assertEqual(await service._max_dirty_attempts(), 7)

    async def test_service_clamps_out_of_range_settings(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        service = WorkspaceCodeIndexService()
        fake_settings = {
            "userspace_code_index_debounce_seconds": -1,
            "userspace_code_index_reconcile_interval_seconds": 5,
            "userspace_code_index_max_attempts": 100,
        }

        with mock.patch(
            "ragtime.userspace.workspace_code_index_service.get_app_settings",
            mock.AsyncMock(return_value=fake_settings),
        ):
            self.assertEqual(await service._debounce_seconds(), 0)
            self.assertEqual(await service._reconcile_interval_seconds(), 10)
            self.assertEqual(await service._max_dirty_attempts(), 20)

    async def test_service_uses_constructor_overrides_when_provided(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        service = WorkspaceCodeIndexService(
            debounce_seconds=0.5,
            reconcile_interval_seconds=60,
            max_attempts=1,
        )
        fake_settings = {
            "userspace_code_index_debounce_seconds": 5,
            "userspace_code_index_reconcile_interval_seconds": 600,
            "userspace_code_index_max_attempts": 7,
        }

        with mock.patch(
            "ragtime.userspace.workspace_code_index_service.get_app_settings",
            mock.AsyncMock(return_value=fake_settings),
        ):
            self.assertEqual(await service._debounce_seconds(), 0.5)
            self.assertEqual(await service._reconcile_interval_seconds(), 60)
            self.assertEqual(await service._max_dirty_attempts(), 1)

    async def test_service_falls_back_to_defaults_when_settings_unavailable(self) -> None:
        from ragtime.userspace.workspace_code_index_service import WorkspaceCodeIndexService

        service = WorkspaceCodeIndexService()

        with mock.patch(
            "ragtime.userspace.workspace_code_index_service.get_app_settings",
            mock.AsyncMock(side_effect=RuntimeError("database unavailable")),
        ):
            self.assertEqual(await service._debounce_seconds(), 2)
            self.assertEqual(await service._reconcile_interval_seconds(), 300)
            self.assertEqual(await service._max_dirty_attempts(), 3)

    async def test_repository_get_settings_maps_code_index_fields(self) -> None:
        repository = IndexerRepository()
        fake_db = _FakeDb(
            _settings_row(
                userspaceCodeIndexEnabled=False,
                userspaceCodeIndexDebounceSeconds=10,
                userspaceCodeIndexReconcileIntervalSeconds=1200,
                userspaceCodeIndexMaxAttempts=5,
            )
        )

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            settings = await repository.get_settings()

        self.assertFalse(settings.userspace_code_index_enabled)
        self.assertEqual(settings.userspace_code_index_debounce_seconds, 10)
        self.assertEqual(settings.userspace_code_index_reconcile_interval_seconds, 1200)
        self.assertEqual(settings.userspace_code_index_max_attempts, 5)

    async def test_repository_update_settings_maps_code_index_fields(self) -> None:
        repository = IndexerRepository()
        fake_db = _FakeDb(_settings_row())

        with mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)):
            await repository.update_settings(
                {
                    "userspace_code_index_enabled": False,
                    "userspace_code_index_debounce_seconds": 7,
                    "userspace_code_index_reconcile_interval_seconds": 900,
                    "userspace_code_index_max_attempts": 4,
                }
            )

        last_update_data = fake_db.appsettings.last_update_data
        self.assertIsNotNone(last_update_data)
        assert last_update_data is not None
        self.assertFalse(last_update_data["userspaceCodeIndexEnabled"])
        self.assertEqual(last_update_data["userspaceCodeIndexDebounceSeconds"], 7)
        self.assertEqual(last_update_data["userspaceCodeIndexReconcileIntervalSeconds"], 900)
        self.assertEqual(last_update_data["userspaceCodeIndexMaxAttempts"], 4)

    def test_update_settings_request_preserves_code_index_enabled_toggle(self) -> None:
        request = UpdateSettingsRequest(userspace_code_index_enabled=False)

        self.assertEqual(
            request.model_dump(exclude_unset=True),
            {"userspace_code_index_enabled": False},
        )

    async def test_orphan_embedding_cleanup_preserves_hidden_workspace_code_indexes(self) -> None:
        repository = IndexerRepository()

        class FakeDb:
            def __init__(self) -> None:
                self.executed: list[str] = []

            async def execute_raw(self, query: str) -> int:
                self.executed.append(query)
                return 0

        fake_db = FakeDb()

        with (
            mock.patch.object(repository, "_get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch.object(repository, "list_tool_configs", mock.AsyncMock(return_value=[])),
        ):
            await repository.cleanup_orphaned_embeddings()

        self.assertIn(
            "DELETE FROM filesystem_embeddings WHERE index_name NOT LIKE 'userspace_workspace_%'",
            fake_db.executed,
        )
        self.assertIn(
            "DELETE FROM filesystem_file_metadata WHERE index_name NOT LIKE 'userspace_workspace_%'",
            fake_db.executed,
        )
        self.assertNotIn("DELETE FROM filesystem_embeddings", fake_db.executed)
        self.assertNotIn("DELETE FROM filesystem_file_metadata", fake_db.executed)


class WorkspaceCodeIndexAdminRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_workspace_code_indexes_returns_admin_summaries(self) -> None:
        class FakeDb:
            async def query_raw(self, _query: str) -> list[dict[str, Any]]:
                return [
                    {
                        "workspace_id": "ws-1",
                        "workspace_name": "Test Workspace",
                        "index_name": "userspace_workspace_ws_1",
                        "status": "ready",
                        "file_count": 12,
                        "chunk_count": 45,
                        "symbol_count": 8,
                        "dirty_path_count": 2,
                        "last_indexed_at": None,
                        "last_reconciled_at": None,
                        "last_error": None,
                        "created_at": "2026-07-01T00:00:00Z",
                        "updated_at": "2026-07-01T00:00:00Z",
                    }
                ]

        fake_admin = SimpleNamespace(role="admin")
        with mock.patch("ragtime.userspace.routes.get_db", mock.AsyncMock(return_value=FakeDb())):
            result = await list_workspace_code_indexes(_user=fake_admin)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].workspace_id, "ws-1")
        self.assertEqual(result[0].workspace_name, "Test Workspace")
        self.assertEqual(result[0].dirty_path_count, 2)
        self.assertFalse(hasattr(result[0], "progress_percent"))
        self.assertFalse(hasattr(result[0], "total_files"))
        self.assertFalse(hasattr(result[0], "processed_files"))
        self.assertFalse(hasattr(result[0], "current_file"))

    async def test_list_workspace_code_index_jobs_returns_canonical_progress_rows(self) -> None:
        class FakeDb:
            async def query_raw(self, _query: str) -> list[dict[str, Any]]:
                return [
                    {
                        "id": "job-1",
                        "workspace_id": "ws-1",
                        "workspace_name": "Test Workspace",
                        "index_name": "userspace_workspace_ws_1",
                        "status": "indexing",
                        "phase": "embedding",
                        "total_files": 20,
                        "processed_files": 20,
                        "total_chunks": 100,
                        "processed_chunks": 25,
                        "current_file": "src/app.py",
                        "error_message": None,
                        "created_at": "2026-07-01T00:00:00Z",
                        "started_at": "2026-07-01T00:00:01Z",
                        "completed_at": None,
                        "waiting_for_job_id": "job-0",
                        "cancel_requested": True,
                    },
                    {
                        "id": "job-2",
                        "workspace_id": "ws-1",
                        "workspace_name": "Test Workspace",
                        "index_name": "userspace_workspace_ws_1",
                        "status": "failed",
                        "phase": "embedding",
                        "total_files": 20,
                        "processed_files": 20,
                        "total_chunks": 100,
                        "processed_chunks": 25,
                        "current_file": None,
                        "error_message": "boom",
                        "created_at": "2026-07-01T00:00:00Z",
                        "started_at": "2026-07-01T00:00:01Z",
                        "completed_at": "2026-07-01T00:00:02Z",
                        "waiting_for_job_id": None,
                        "cancel_requested": False,
                    },
                ]

        fake_admin = SimpleNamespace(role="admin")
        with mock.patch("ragtime.userspace.routes.get_db", mock.AsyncMock(return_value=FakeDb())):
            result = await list_workspace_code_index_jobs(_user=fake_admin)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].id, "job-1")
        self.assertEqual(result[0].workspace_id, "ws-1")
        self.assertEqual(result[0].workspace_name, "Test Workspace")
        self.assertEqual(result[0].status, "indexing")
        self.assertEqual(result[0].phase, "embedding")
        self.assertEqual(result[0].progress_percent, 84.75)
        self.assertEqual(result[0].total_files, 20)
        self.assertEqual(result[0].processed_files, 20)
        self.assertEqual(result[0].current_file, "src/app.py")
        self.assertEqual(result[0].waiting_for_job_id, "job-0")
        self.assertTrue(result[0].cancel_requested)
        self.assertEqual(result[1].status, "failed")
        self.assertEqual(result[1].progress_percent, 0.0)
        self.assertIsNone(result[1].waiting_for_job_id)
        self.assertFalse(result[1].cancel_requested)

    async def test_reindex_endpoint_marks_workspace_dirty(self) -> None:
        fake_admin = SimpleNamespace(role="admin")
        with mock.patch.object(
            workspace_code_index_service,
            "mark_dirty",
            mock.AsyncMock(),
        ) as mark_dirty:
            result = await reindex_workspace_code_index("ws-1", _user=fake_admin)

        mark_dirty.assert_awaited_once_with("ws-1", operation="reindex")
        self.assertTrue(result.success)
        self.assertEqual(result.workspace_id, "ws-1")

    async def test_delete_endpoint_deletes_workspace_index(self) -> None:
        fake_admin = SimpleNamespace(role="admin")
        with mock.patch.object(
            workspace_code_index_service,
            "delete_workspace_index",
            mock.AsyncMock(),
        ) as delete_index:
            result = await delete_workspace_code_index_admin("ws-1", _user=fake_admin)

        delete_index.assert_awaited_once_with("ws-1")
        self.assertTrue(result.success)
        self.assertEqual(result.workspace_id, "ws-1")

    async def test_reconcile_now_endpoint_schedules_dirty_workspaces(self) -> None:
        fake_admin = SimpleNamespace(role="admin")
        with mock.patch.object(
            workspace_code_index_service,
            "reconcile_stale_workspaces",
            mock.AsyncMock(return_value=["ws-1", "ws-2"]),
        ) as reconcile:
            result = await reconcile_workspace_code_indexes_now(_user=fake_admin)

        reconcile.assert_awaited_once_with(include_missing=True)
        self.assertEqual(result.scheduled_count, 2)

    async def test_cancel_job_endpoint_returns_success_when_cancellable(self) -> None:
        fake_admin = SimpleNamespace(role="admin")
        with mock.patch.object(
            workspace_code_index_service,
            "cancel_job",
            mock.AsyncMock(return_value=True),
        ) as cancel_job:
            result = await cancel_workspace_code_index_job("job-1", _user=fake_admin)

        cancel_job.assert_awaited_once_with("job-1")
        self.assertTrue(result["success"])
        self.assertEqual(result["job_id"], "job-1")

    async def test_cancel_job_endpoint_returns_400_when_not_cancellable(self) -> None:
        fake_admin = SimpleNamespace(role="admin")
        with mock.patch.object(
            workspace_code_index_service,
            "cancel_job",
            mock.AsyncMock(return_value=False),
        ) as cancel_job:
            with self.assertRaises(HTTPException) as ctx:
                await cancel_workspace_code_index_job("job-1", _user=fake_admin)

        cancel_job.assert_awaited_once_with("job-1")
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
