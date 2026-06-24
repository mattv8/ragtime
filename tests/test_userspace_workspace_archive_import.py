import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, patch

from ragtime.userspace.models import UserSpaceWorkspaceArchiveExportRequest
from ragtime.userspace.service import UserSpaceService

if "ragtime.rag.prompts" not in sys.modules:
    fake_rag_package = types.ModuleType("ragtime.rag")
    fake_prompts_module = types.ModuleType("ragtime.rag.prompts")
    setattr(fake_prompts_module, "build_workspace_scm_setup_prompt", lambda *args, **kwargs: "")
    setattr(fake_rag_package, "prompts", fake_prompts_module)
    sys.modules.setdefault("ragtime.rag", fake_rag_package)
    sys.modules["ragtime.rag.prompts"] = fake_prompts_module


class _FakeToolConfigTable:
    def __init__(self, records: list[SimpleNamespace]) -> None:
        self._by_id = {str(record.id): record for record in records}

    async def find_unique(self, *, where: dict[str, str]) -> Optional[SimpleNamespace]:
        return self._by_id.get(str(where.get("id") or ""))


class _FakeToolGroupTable:
    def __init__(self, records: list[SimpleNamespace]) -> None:
        self._by_id = {str(record.id): record for record in records}

    async def find_unique(self, *, where: dict[str, str]) -> Optional[SimpleNamespace]:
        return self._by_id.get(str(where.get("id") or ""))


class _CaptureTable:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.updated: list[dict[str, object]] = []

    async def create(self, *, data: dict[str, object]) -> SimpleNamespace:
        self.created.append(data)
        return SimpleNamespace(**data)

    async def update(
        self,
        *,
        where: dict[str, object],
        data: dict[str, object],
    ) -> SimpleNamespace:
        self.updated.append({"where": where, "data": data})
        return SimpleNamespace(**data)


class WorkspaceArchiveImportTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_guarded_workspace_archive_task_respects_admin_access(self) -> None:
        service = UserSpaceService()
        captured: list[bool] = []

        async def _fake_enforce(
            workspace_id: str,
            user_id: str,
            required_role: str | None = None,
            is_admin: bool = False,
        ) -> SimpleNamespace:
            captured.append(is_admin)
            return SimpleNamespace(id=workspace_id)

        task_body = AsyncMock()
        on_failure = AsyncMock()

        with patch.object(service, "_enforce_workspace_access", new=_fake_enforce):
            await service._run_guarded_workspace_archive_task(
                "workspace-1",
                "user-1",
                task_body,
                on_failure,
                log_message="archive task failed",
                persist_failure_message="persist failure",
                is_admin=True,
            )

        self.assertEqual(captured, [True])
        task_body.assert_awaited_once()
        on_failure.assert_not_awaited()

    async def test_run_workspace_archive_export_task_passes_admin_to_mount_listing(self) -> None:
        service = UserSpaceService()
        task_id = "task-1"

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            workspace_files = tmp / "workspace-files"
            workspace_files.mkdir()
            (workspace_files / "notes.txt").write_text("hello\n", encoding="utf-8")

            async def _fake_run_guarded(
                workspace_id: str,
                user_id: str,
                task_body,
                on_failure,
                **_: object,
            ) -> None:
                await task_body(
                    SimpleNamespace(
                        name="Workspace",
                        description=None,
                        sqlite_persistence_mode="include",
                        selected_tool_ids=[],
                        selected_tool_group_ids=[],
                        scm=None,
                    )
                )

            def _fake_write_archive(
                source_root: Path,
                archive_path: Path,
                archive_format: str,
                manifest: dict[str, object],
                ignored_prefixes: list[str],
                extra_files: dict[str, Path],
                progress_callback,
            ) -> None:
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                archive_path.write_bytes(b"archive")

            with (
                patch.object(service, "_run_guarded_workspace_archive_task", new=_fake_run_guarded),
                patch.object(service, "list_workspace_mounts", new=AsyncMock(return_value=[])) as list_mounts,
                patch.object(service, "_update_workspace_archive_export_task_phase", new=AsyncMock()),
                patch.object(service, "_serialize_workspace_env_var_placeholders", new=AsyncMock(return_value=[])),
                patch.object(service, "export_workspace_audit_identity_manifest", new=AsyncMock(return_value={})),
                patch.object(service, "_workspace_files_dir", return_value=workspace_files),
                patch.object(service, "_workspace_archive_task_dir", return_value=tmp / task_id),
                patch.object(service, "_write_workspace_archive_sync", new=_fake_write_archive),
            ):
                await service._run_workspace_archive_export_task(
                    task_id,
                    "workspace-1",
                    UserSpaceWorkspaceArchiveExportRequest(
                        archive_format="zip",
                        include_snapshots=False,
                        include_chat_history=False,
                    ),
                    "user-1",
                    is_admin=True,
                )

        await_args = list_mounts.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertTrue(await_args.kwargs["is_admin"])

    async def test_apply_workspace_archive_manifest_passes_admin_to_workspace_updates(
        self,
    ) -> None:
        service = UserSpaceService()
        manifest = {
            "workspace": {
                "description": "Imported workspace",
                "sqlite_persistence_mode": "include",
                "selected_tool_ids": [],
                "selected_tool_group_ids": [],
            }
        }

        with (
            patch.object(service, "import_workspace_audit_identity_manifest", new=AsyncMock()),
            patch.object(service, "_resolve_workspace_archive_selection_id_sets", new=AsyncMock(return_value=(set(), set(), []))),
            patch.object(service, "update_workspace", new=AsyncMock()) as update_workspace,
            patch.object(service, "_import_workspace_env_var_placeholders", new=AsyncMock(return_value=(0, 0))),
            patch.object(service, "_import_workspace_mount_placeholders", new=AsyncMock(return_value=[])),
            patch.object(service, "_restore_workspace_archive_scm_metadata", new=AsyncMock()),
        ):
            warnings, imported_snapshot_count, imported_chat_count = await service._apply_workspace_archive_manifest(
                "workspace-1",
                "user-1",
                manifest,
                include_snapshots=False,
                include_chat_history=False,
                extract_dir=Path("/tmp/unused"),
                is_admin=True,
            )

        self.assertEqual(warnings, [])
        self.assertEqual(imported_snapshot_count, 0)
        self.assertEqual(imported_chat_count, 0)
        await_args = update_workspace.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertTrue(await_args.kwargs["is_admin"])

    async def test_resolve_workspace_archive_selection_id_sets_keeps_exact_ids_only(
        self,
    ) -> None:
        service = UserSpaceService()
        fake_db = SimpleNamespace(
            toolconfig=_FakeToolConfigTable(
                [
                    SimpleNamespace(id="tool-a", enabled=True),
                    SimpleNamespace(id="tool-disabled", enabled=False),
                ]
            ),
            toolgroup=_FakeToolGroupTable([SimpleNamespace(id="group-a")]),
        )
        manifest = {
            "workspace": {
                "selected_tool_ids": ["tool-a", "tool-disabled", "tool-missing"],
                "selected_tool_group_ids": ["group-a", "group-missing"],
            },
            "chats": [
                {
                    "tool_config_ids": ["tool-a", "tool-disabled", "tool-missing"],
                    "tool_group_ids": ["group-a", "group-missing"],
                }
            ],
        }

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            (
                allowed_tool_ids,
                allowed_tool_group_ids,
                warnings,
            ) = await service._resolve_workspace_archive_selection_id_sets(manifest)

        self.assertEqual(allowed_tool_ids, {"tool-a"})
        self.assertEqual(allowed_tool_group_ids, {"group-a"})
        self.assertTrue(any("Skipped 2 archived tool selection references" in warning for warning in warnings))
        self.assertTrue(any("Skipped 1 archived tool group reference" in warning for warning in warnings))

    async def test_import_workspace_chat_payloads_filters_to_allowed_exact_ids(
        self,
    ) -> None:
        service = UserSpaceService()
        fake_db = SimpleNamespace(
            conversation=_CaptureTable(),
            conversationtoolselection=_CaptureTable(),
            conversationtoolgroupselection=_CaptureTable(),
            conversationbranch=_CaptureTable(),
        )
        chat_payloads = [
            {
                "title": "Imported chat",
                "messages": [],
                "tool_config_ids": ["tool-a", "tool-missing", "tool-a"],
                "tool_group_ids": ["group-a", "group-missing", "group-a"],
                "branches": [],
            }
        ]

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            imported_count = await service._import_workspace_chat_payloads(
                "workspace-1",
                "user-1",
                chat_payloads,
                allowed_tool_config_ids={"tool-a"},
                allowed_tool_group_ids={"group-a"},
            )

        self.assertEqual(imported_count, 1)
        self.assertEqual(len(fake_db.conversation.created), 1)
        self.assertEqual(
            [row["toolConfigId"] for row in fake_db.conversationtoolselection.created],
            ["tool-a"],
        )
        self.assertEqual(
            [row["toolGroupId"] for row in fake_db.conversationtoolgroupselection.created],
            ["group-a"],
        )


if __name__ == "__main__":
    unittest.main()
