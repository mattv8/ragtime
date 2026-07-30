import sys
import tarfile
import types
import unittest
import zipfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Generator, Optional
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from ragtime.userspace.models import UserSpaceWorkspaceArchiveExportRequest
from ragtime.userspace.service import UserSpaceService, _GitCommandResult

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


class _FindManyTable:
    def __init__(self, records: list[SimpleNamespace]) -> None:
        self._records = records

    async def find_many(self, **_: object) -> list[SimpleNamespace]:
        return self._records


class _ConversationFkCaptureTable(_CaptureTable):
    async def create(self, *, data: dict[str, object]) -> SimpleNamespace:
        parent_id = data.get("parentConversationId")
        if parent_id and parent_id not in {row.get("id") for row in self.created}:
            raise AssertionError("parent conversation must exist before child insert")
        return await super().create(data=data)


class WorkspaceArchiveImportTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    async def _snapshot_query_raw_result(query: str) -> list[dict[str, object]]:
        if "FROM workspaces" in query:
            return [
                {
                    "current_snapshot_id": "snapshot-1",
                    "current_snapshot_branch_id": "branch-1",
                }
            ]
        if "FROM userspace_snapshot_branches" in query:
            return [
                {
                    "id": "branch-1",
                    "name": "Main",
                    "git_ref_name": "refs/heads/main",
                    "base_snapshot_id": None,
                    "branched_from_snapshot_id": None,
                    "is_active": True,
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        if "FROM userspace_snapshots" in query:
            return [
                {
                    "id": "snapshot-1",
                    "branch_id": "branch-1",
                    "git_commit_hash": "abc123",
                    "message": "Initial",
                    "remote_commit_hash": None,
                    "file_count": 1,
                    "parent_snapshot_id": None,
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        return []

    @contextmanager
    def _fake_run_git_capture_calls(
        self,
        service: UserSpaceService,
        git_calls: list[list[str]] | None = None,
        *,
        returncode: int = 1,
        stderr: str = "unexpected git command",
    ) -> Generator[None, None, None]:
        captured = git_calls if git_calls is not None else []

        async def _fake_run_git(
            workspace_id: str,
            args: list[str],
            check: bool = True,
            env: dict[str, str] | None = None,
        ) -> _GitCommandResult:
            captured.append(args)
            if args[:2] == ["bundle", "create"]:
                Path(args[2]).write_bytes(b"bundle")
                return _GitCommandResult(0, "", "")
            return _GitCommandResult(returncode, "", stderr)

        with patch.object(service, "_run_git", new=_fake_run_git):
            yield

    async def _run_build_workspace_snapshot_archive_payload(
        self,
        service: UserSpaceService,
        changed_paths: list[str] | Exception,
        *,
        is_admin: bool = False,
        git_calls: list[list[str]] | None = None,
    ) -> tuple[dict[str, object] | None, Path | None, list[str], AsyncMock]:
        fake_db = SimpleNamespace()
        fake_db.query_raw = self._snapshot_query_raw_result

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        if isinstance(changed_paths, Exception):
            list_changed = AsyncMock(side_effect=changed_paths)
        else:
            list_changed = AsyncMock(return_value=changed_paths)

        with TemporaryDirectory() as tmpdir:
            with (
                patch("ragtime.userspace.service.get_db", new=_fake_get_db),
                self._fake_run_git_capture_calls(service, git_calls=git_calls),
                patch.object(
                    service,
                    "list_workspace_changed_file_paths",
                    new=list_changed,
                ),
            ):
                return (
                    await service._build_workspace_snapshot_archive_payload(
                        "workspace-1",
                        Path(tmpdir),
                        "user-1",
                        is_admin=is_admin,
                    )
                ) + (list_changed,)

    def _write_zip_archive(self, path: Path, files: dict[str, bytes]) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", b'{"version":1,"workspace":{}}')
            for name, content in files.items():
                archive.writestr(name, content)

    def _write_zip_archive_with_dirs(self, path: Path, dirs: list[str]) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("manifest.json", b'{"version":1,"workspace":{}}')
            for name in dirs:
                archive.writestr(name.rstrip("/") + "/", b"")

    def _write_tar_archive(self, path: Path, files: dict[str, bytes]) -> None:
        with tarfile.open(path, "w:gz") as archive:
            manifest = b'{"version":1,"workspace":{}}'
            manifest_info = tarfile.TarInfo("manifest.json")
            manifest_info.size = len(manifest)
            archive.addfile(manifest_info, BytesIO(manifest))
            for name, content in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, BytesIO(content))

    def _write_tar_archive_with_dirs(self, path: Path, dirs: list[str]) -> None:
        with tarfile.open(path, "w:gz") as archive:
            manifest = b'{"version":1,"workspace":{}}'
            manifest_info = tarfile.TarInfo("manifest.json")
            manifest_info.size = len(manifest)
            archive.addfile(manifest_info, BytesIO(manifest))
            for name in dirs:
                info = tarfile.TarInfo(name.rstrip("/"))
                info.type = tarfile.DIRTYPE
                archive.addfile(info)

    def test_zip_workspace_archive_import_rejects_too_many_entries(self) -> None:
        service = UserSpaceService()
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            archive_path = tmp / "workspace.zip"
            self._write_zip_archive(archive_path, {"files/a.txt": b"a", "files/b.txt": b"b"})

            with self.assertRaises(HTTPException) as ctx:
                service._extract_workspace_archive_sync(
                    archive_path,
                    tmp / "extract",
                    max_entries=1,
                    max_bytes=1024,
                )

        self.assertEqual(ctx.exception.status_code, 413)
        self.assertIn("entry", str(ctx.exception.detail).lower())

    def test_tar_workspace_archive_import_rejects_too_many_entries(self) -> None:
        service = UserSpaceService()
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            archive_path = tmp / "workspace.tar.gz"
            self._write_tar_archive(archive_path, {"files/a.txt": b"a", "files/b.txt": b"b"})

            with self.assertRaises(HTTPException) as ctx:
                service._extract_workspace_archive_sync(
                    archive_path,
                    tmp / "extract",
                    max_entries=1,
                    max_bytes=1024,
                )

        self.assertEqual(ctx.exception.status_code, 413)
        self.assertIn("entry", str(ctx.exception.detail).lower())

    def test_zip_workspace_archive_import_counts_directory_entries(self) -> None:
        service = UserSpaceService()
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            archive_path = tmp / "workspace.zip"
            self._write_zip_archive_with_dirs(archive_path, ["files/a", "files/b"])

            with self.assertRaises(HTTPException) as ctx:
                service._extract_workspace_archive_sync(
                    archive_path,
                    tmp / "extract",
                    max_entries=1,
                    max_bytes=1024,
                )

        self.assertEqual(ctx.exception.status_code, 413)
        self.assertIn("entry", str(ctx.exception.detail).lower())

    def test_tar_workspace_archive_import_counts_directory_entries(self) -> None:
        service = UserSpaceService()
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            archive_path = tmp / "workspace.tar.gz"
            self._write_tar_archive_with_dirs(archive_path, ["files/a", "files/b"])

            with self.assertRaises(HTTPException) as ctx:
                service._extract_workspace_archive_sync(
                    archive_path,
                    tmp / "extract",
                    max_entries=1,
                    max_bytes=1024,
                )

        self.assertEqual(ctx.exception.status_code, 413)
        self.assertIn("entry", str(ctx.exception.detail).lower())

    def test_zip_workspace_archive_import_rejects_extracted_size_limit(self) -> None:
        service = UserSpaceService()
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            archive_path = tmp / "workspace.zip"
            self._write_zip_archive(archive_path, {"files/large.txt": b"x" * 128})

            with self.assertRaises(HTTPException) as ctx:
                service._extract_workspace_archive_sync(
                    archive_path,
                    tmp / "extract",
                    max_entries=10,
                    max_bytes=64,
                )

        self.assertEqual(ctx.exception.status_code, 413)
        self.assertIn("size", str(ctx.exception.detail).lower())

    def test_tar_workspace_archive_import_rejects_extracted_size_limit(self) -> None:
        service = UserSpaceService()
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            archive_path = tmp / "workspace.tar.gz"
            self._write_tar_archive(archive_path, {"files/large.txt": b"x" * 128})

            with self.assertRaises(HTTPException) as ctx:
                service._extract_workspace_archive_sync(
                    archive_path,
                    tmp / "extract",
                    max_entries=10,
                    max_bytes=64,
                )

        self.assertEqual(ctx.exception.status_code, 413)
        self.assertIn("size", str(ctx.exception.detail).lower())

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
                        tool_options={},
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

    async def test_serialize_workspace_chat_payloads_includes_subagent_linkage(
        self,
    ) -> None:
        service = UserSpaceService()
        fake_db = SimpleNamespace(
            conversation=_FindManyTable(
                [
                    SimpleNamespace(
                        id="parent-chat",
                        title="Parent",
                        model="gpt-4.1",
                        messages=[],
                        totalTokens=0,
                        toolOutputMode="default",
                        activeBranchId=None,
                        subagentsEnabled=True,
                        parentConversationId=None,
                        subagentRole=None,
                        subagentIndex=None,
                    ),
                    SimpleNamespace(
                        id="child-chat",
                        title="Worker",
                        model="gpt-4.1",
                        messages=[],
                        totalTokens=0,
                        toolOutputMode="default",
                        activeBranchId=None,
                        subagentsEnabled=False,
                        parentConversationId="parent-chat",
                        subagentRole="frontend",
                        subagentIndex=2,
                    ),
                ]
            ),
            conversationtoolselection=_FindManyTable([]),
            conversationtoolgroupselection=_FindManyTable([]),
            conversationbranch=_FindManyTable([]),
        )

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            payloads = await service._serialize_workspace_chat_payloads("workspace-1")

        child_payload = next(payload for payload in payloads if payload["id"] == "child-chat")
        self.assertEqual(child_payload["parent_conversation_id"], "parent-chat")
        self.assertEqual(child_payload["subagent_role"], "frontend")
        self.assertEqual(child_payload["subagent_index"], 2)
        self.assertFalse(child_payload["subagents_enabled"])

    async def test_serialize_workspace_chat_payloads_includes_loaded_tool_skill_ids(
        self,
    ) -> None:
        service = UserSpaceService()
        fake_db = SimpleNamespace(
            conversation=_FindManyTable(
                [
                    SimpleNamespace(
                        id="chat-1",
                        title="Parent",
                        model="gpt-4.1",
                        messages=[],
                        totalTokens=0,
                        toolOutputMode="default",
                        activeBranchId=None,
                        subagentsEnabled=True,
                        parentConversationId=None,
                        subagentRole=None,
                        subagentIndex=None,
                        loadedToolSkillIds=["skill.alpha", "skill.beta"],
                    )
                ]
            ),
            conversationtoolselection=_FindManyTable([]),
            conversationtoolgroupselection=_FindManyTable([]),
            conversationbranch=_FindManyTable([]),
        )

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            payloads = await service._serialize_workspace_chat_payloads("workspace-1")

        self.assertEqual(payloads[0]["loaded_tool_skill_ids"], ["skill.alpha", "skill.beta"])

    async def test_import_workspace_chat_payloads_remaps_subagent_parent_linkage(
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
                "id": "source-parent",
                "title": "Parent",
                "messages": [],
                "subagents_enabled": True,
                "branches": [],
            },
            {
                "id": "source-child",
                "title": "Worker",
                "messages": [],
                "subagents_enabled": False,
                "parent_conversation_id": "source-parent",
                "subagent_role": "frontend",
                "subagent_index": 1,
                "branches": [],
            },
        ]

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            imported_count = await service._import_workspace_chat_payloads(
                "workspace-1",
                "user-1",
                chat_payloads,
            )

        self.assertEqual(imported_count, 2)
        parent_row, child_row = fake_db.conversation.created
        self.assertEqual(child_row["parentConversationId"], parent_row["id"])
        self.assertEqual(child_row["subagentRole"], "frontend")
        self.assertEqual(child_row["subagentIndex"], 1)
        self.assertFalse(child_row["subagentsEnabled"])

    async def test_import_workspace_chat_payloads_links_subagent_after_parent_exists(
        self,
    ) -> None:
        service = UserSpaceService()
        fake_db = SimpleNamespace(
            conversation=_ConversationFkCaptureTable(),
            conversationtoolselection=_CaptureTable(),
            conversationtoolgroupselection=_CaptureTable(),
            conversationbranch=_CaptureTable(),
        )
        chat_payloads = [
            {
                "id": "source-child",
                "title": "Worker",
                "messages": [],
                "parent_conversation_id": "source-parent",
                "branches": [],
            },
            {
                "id": "source-parent",
                "title": "Parent",
                "messages": [],
                "branches": [],
            },
        ]

        async def _fake_get_db() -> SimpleNamespace:
            return fake_db

        with patch("ragtime.userspace.service.get_db", new=_fake_get_db):
            imported_count = await service._import_workspace_chat_payloads(
                "workspace-1",
                "user-1",
                chat_payloads,
            )

        self.assertEqual(imported_count, 2)
        child_row, parent_row = fake_db.conversation.created
        self.assertIsNone(child_row["parentConversationId"])
        self.assertEqual(
            fake_db.conversation.updated[-1],
            {
                "where": {"id": child_row["id"]},
                "data": {"parentConversationId": parent_row["id"]},
            },
        )

    async def test_import_workspace_chat_payloads_ignores_invalid_subagent_index(
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
                "id": "source-child",
                "title": "Worker",
                "messages": [],
                "subagent_index": "not-a-number",
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
            )

        self.assertEqual(imported_count, 1)
        self.assertIsNone(fake_db.conversation.created[0]["subagentIndex"])

    async def test_import_workspace_chat_payloads_persists_loaded_tool_skill_ids(
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
                "id": "source-chat",
                "title": "Worker",
                "messages": [],
                "loaded_tool_skill_ids": [" skill.alpha ", "", "skill.alpha", "skill.beta"],
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
            )

        self.assertEqual(imported_count, 1)
        self.assertEqual(
            fake_db.conversation.created[0]["loadedToolSkillIds"].data,
            ["skill.alpha", "skill.beta"],
        )

    async def test_build_workspace_snapshot_archive_payload_bundles_dirty_worktree(
        self,
    ) -> None:
        service = UserSpaceService()
        git_calls: list[list[str]] = []

        snapshot_manifest, bundle_path, warnings, list_changed = await self._run_build_workspace_snapshot_archive_payload(
            service,
            changed_paths=["app.py"],
            git_calls=git_calls,
        )

        self.assertIsNotNone(snapshot_manifest)
        self.assertIsNotNone(bundle_path)
        self.assertTrue(
            any("Snapshot history includes committed snapshots only" in warning and "current uncommitted workspace changes" in warning for warning in warnings)
        )
        list_changed.assert_awaited_once_with("workspace-1", "user-1", is_admin=False)
        self.assertTrue(any(call[:2] == ["bundle", "create"] for call in git_calls))

    async def test_build_workspace_snapshot_archive_payload_no_warning_for_clean_worktree(
        self,
    ) -> None:
        service = UserSpaceService()

        snapshot_manifest, bundle_path, warnings, list_changed = await self._run_build_workspace_snapshot_archive_payload(
            service,
            changed_paths=[],
        )

        self.assertIsNotNone(snapshot_manifest)
        self.assertIsNotNone(bundle_path)
        self.assertEqual(warnings, [])
        list_changed.assert_awaited_once_with("workspace-1", "user-1", is_admin=False)

    async def test_build_workspace_snapshot_archive_payload_passes_is_admin_to_dirty_check(
        self,
    ) -> None:
        service = UserSpaceService()

        _, _, _, list_changed = await self._run_build_workspace_snapshot_archive_payload(
            service,
            changed_paths=[],
            is_admin=True,
        )

        list_changed.assert_awaited_once_with("workspace-1", "user-1", is_admin=True)

    async def test_build_workspace_snapshot_archive_payload_warns_on_dirty_check_failure(
        self,
    ) -> None:
        service = UserSpaceService()

        snapshot_manifest, bundle_path, warnings, _list_changed = await self._run_build_workspace_snapshot_archive_payload(
            service,
            changed_paths=RuntimeError("git status failed"),
        )

        self.assertIsNotNone(snapshot_manifest)
        self.assertIsNotNone(bundle_path)
        self.assertTrue(any("Could not determine whether the workspace has uncommitted changes" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
