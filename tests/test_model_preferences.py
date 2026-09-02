import unittest
from types import SimpleNamespace
from unittest import mock


class _FakeUserTable:
    def __init__(self, default_model_by_user_id: dict[str, str | None]) -> None:
        self.default_model_by_user_id = default_model_by_user_id
        self.update_calls: list[tuple[dict[str, str], dict[str, str | None]]] = []
        self.update_many_calls: list[tuple[dict[str, object], dict[str, str | None]]] = []
        self.before_update_many = None

    async def find_unique(self, where: dict[str, str]) -> SimpleNamespace | None:
        user_id = where["id"]
        if user_id not in self.default_model_by_user_id:
            return None
        return SimpleNamespace(id=user_id, defaultChatModel=self.default_model_by_user_id[user_id])

    async def update(self, *, where: dict[str, str], data: dict[str, str | None]) -> SimpleNamespace | None:
        self.update_calls.append((where, data))
        user_id = where["id"]
        if user_id not in self.default_model_by_user_id:
            return None
        self.default_model_by_user_id[user_id] = data.get("defaultChatModel")
        return SimpleNamespace(id=user_id, defaultChatModel=self.default_model_by_user_id[user_id])

    async def update_many(self, *, where: dict[str, object], data: dict[str, str | None]) -> SimpleNamespace:
        self.update_many_calls.append((where, data))
        if self.before_update_many is not None:
            self.before_update_many(where, data)
        user_id = str(where["id"])
        expected_model = where.get("defaultChatModel")
        updated = 0
        if self.default_model_by_user_id.get(user_id) == expected_model:
            self.default_model_by_user_id[user_id] = data.get("defaultChatModel")
            updated = 1
        return SimpleNamespace(count=updated)


class _FakeWorkspacePreferenceTable:
    def __init__(self, default_model_by_key: dict[tuple[str, str], str]) -> None:
        self.default_model_by_key = default_model_by_key
        self.create_calls: list[dict[str, str]] = []
        self.update_calls: list[tuple[dict[str, str], dict[str, str]]] = []
        self.delete_calls: list[dict[str, str]] = []
        self.delete_many_calls: list[dict[str, str]] = []
        self.upsert_calls: list[tuple[dict[str, object], dict[str, str], dict[str, str]]] = []
        self.before_delete_many = None

    async def find_first(self, where: dict[str, str]) -> SimpleNamespace | None:
        key = (where["workspaceId"], where["userId"])
        value = self.default_model_by_key.get(key)
        if value is None:
            return None
        return SimpleNamespace(id=f"{key[0]}:{key[1]}", workspaceId=key[0], userId=key[1], defaultChatModel=value)

    async def create(self, *, data: dict[str, str]) -> SimpleNamespace:
        self.create_calls.append(data)
        key = (data["workspaceId"], data["userId"])
        self.default_model_by_key[key] = data["defaultChatModel"]
        return SimpleNamespace(id=f"{key[0]}:{key[1]}", workspaceId=key[0], userId=key[1], defaultChatModel=data["defaultChatModel"])

    async def update(self, *, where: dict[str, str], data: dict[str, str]) -> SimpleNamespace | None:
        self.update_calls.append((where, data))
        workspace_id, user_id = where["id"].split(":", 1)
        key = (workspace_id, user_id)
        if key not in self.default_model_by_key:
            return None
        self.default_model_by_key[key] = data["defaultChatModel"]
        return SimpleNamespace(id=where["id"], workspaceId=workspace_id, userId=user_id, defaultChatModel=data["defaultChatModel"])

    async def delete(self, *, where: dict[str, str]) -> None:
        self.delete_calls.append(where)
        workspace_id, user_id = where["id"].split(":", 1)
        self.default_model_by_key.pop((workspace_id, user_id), None)

    async def delete_many(self, *, where: dict[str, str]) -> SimpleNamespace:
        self.delete_many_calls.append(where)
        if self.before_delete_many is not None:
            self.before_delete_many(where)
        key = (where["workspaceId"], where["userId"])
        deleted = 0
        if self.default_model_by_key.get(key) == where.get("defaultChatModel"):
            self.default_model_by_key.pop(key, None)
            deleted = 1
        return SimpleNamespace(count=deleted)

    async def upsert(
        self,
        *,
        where: dict[str, object],
        create: dict[str, str],
        update: dict[str, str],
    ) -> SimpleNamespace:
        self.upsert_calls.append((where, create, update))
        compound = where["workspaceId_userId"]
        key = (compound["workspaceId"], compound["userId"])
        if key in self.default_model_by_key:
            self.default_model_by_key[key] = update["defaultChatModel"]
            value = update["defaultChatModel"]
        else:
            self.default_model_by_key[key] = create["defaultChatModel"]
            value = create["defaultChatModel"]
        return SimpleNamespace(id=f"{key[0]}:{key[1]}", workspaceId=key[0], userId=key[1], defaultChatModel=value)


class ModelPreferenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_normalize_default_model_trims_and_normalizes_provider_prefix(self) -> None:
        from ragtime.indexer.model_preferences import normalize_default_model

        self.assertIsNone(normalize_default_model(None))
        self.assertIsNone(normalize_default_model("  "))
        self.assertEqual(normalize_default_model("  Anthropic::claude-3.7-sonnet  "), "anthropic::claude-3.7-sonnet")
        self.assertEqual(normalize_default_model(" gpt-4o-mini "), "gpt-4o-mini")

    async def test_user_default_model_reads_and_writes_normalized_values(self) -> None:
        from ragtime.indexer import model_preferences

        fake_db = SimpleNamespace(user=_FakeUserTable({"user-1": " openai::gpt-4o-mini "}))

        with mock.patch.object(model_preferences, "get_db", mock.AsyncMock(return_value=fake_db)):
            current = await model_preferences.get_user_default_model("user-1")
            updated = await model_preferences.set_user_default_model("user-1", " Anthropic::claude-3-7-sonnet ")
            cleared = await model_preferences.set_user_default_model("user-1", None)

        self.assertEqual(current, "openai::gpt-4o-mini")
        self.assertEqual(updated, "anthropic::claude-3-7-sonnet")
        self.assertIsNone(cleared)
        self.assertEqual(
            fake_db.user.update_calls,
            [
                ({"id": "user-1"}, {"defaultChatModel": "anthropic::claude-3-7-sonnet"}),
                ({"id": "user-1"}, {"defaultChatModel": None}),
            ],
        )

    async def test_workspace_user_default_model_upserts_and_deletes(self) -> None:
        from ragtime.indexer import model_preferences

        fake_db = SimpleNamespace(workspaceuserpreference=_FakeWorkspacePreferenceTable({("workspace-1", "user-1"): "openai::gpt-4o"}))

        with mock.patch.object(model_preferences, "get_db", mock.AsyncMock(return_value=fake_db)):
            current = await model_preferences.get_workspace_user_default_model("user-1", "workspace-1")
            updated = await model_preferences.set_workspace_user_default_model("user-1", "workspace-1", " openai::gpt-4.1 ")
            created = await model_preferences.set_workspace_user_default_model("user-2", "workspace-2", " anthropic::claude-3-5-haiku ")
            cleared = await model_preferences.set_workspace_user_default_model("user-1", "workspace-1", None)

        self.assertEqual(current, "openai::gpt-4o")
        self.assertEqual(updated, "openai::gpt-4.1")
        self.assertEqual(created, "anthropic::claude-3-5-haiku")
        self.assertIsNone(cleared)
        self.assertEqual(fake_db.workspaceuserpreference.update_calls, [])
        self.assertEqual(fake_db.workspaceuserpreference.create_calls, [])
        self.assertEqual(
            fake_db.workspaceuserpreference.upsert_calls,
            [
                (
                    {"workspaceId_userId": {"workspaceId": "workspace-1", "userId": "user-1"}},
                    {"workspaceId": "workspace-1", "userId": "user-1", "defaultChatModel": "openai::gpt-4.1"},
                    {"defaultChatModel": "openai::gpt-4.1"},
                ),
                (
                    {"workspaceId_userId": {"workspaceId": "workspace-2", "userId": "user-2"}},
                    {"workspaceId": "workspace-2", "userId": "user-2", "defaultChatModel": "anthropic::claude-3-5-haiku"},
                    {"defaultChatModel": "anthropic::claude-3-5-haiku"},
                ),
            ],
        )
        self.assertEqual(fake_db.workspaceuserpreference.delete_calls, [])
        self.assertEqual(
            fake_db.workspaceuserpreference.delete_many_calls,
            [{"workspaceId": "workspace-1", "userId": "user-1"}],
        )

    async def test_clear_matching_personal_defaults_only_clears_matching_current_values(self) -> None:
        from ragtime.indexer import model_preferences

        fake_db = SimpleNamespace(
            user=_FakeUserTable({"user-1": "anthropic::claude-3-7-sonnet", "user-2": "openai::gpt-4o"}),
            workspaceuserpreference=_FakeWorkspacePreferenceTable(
                {
                    ("workspace-1", "user-1"): "anthropic::claude-3-7-sonnet",
                    ("workspace-2", "user-2"): "openai::gpt-4.1",
                }
            ),
        )

        with mock.patch.object(model_preferences, "get_db", mock.AsyncMock(return_value=fake_db)):
            await model_preferences.clear_matching_personal_defaults("user-1", "workspace-1", "anthropic::claude-3-7-sonnet")
            await model_preferences.clear_matching_personal_defaults("user-2", "workspace-2", "openai::gpt-4o")

        self.assertEqual(fake_db.workspaceuserpreference.delete_calls, [])
        self.assertEqual(
            fake_db.workspaceuserpreference.delete_many_calls,
            [
                {"workspaceId": "workspace-1", "userId": "user-1", "defaultChatModel": "anthropic::claude-3-7-sonnet"},
                {"workspaceId": "workspace-2", "userId": "user-2", "defaultChatModel": "openai::gpt-4o"},
            ],
        )
        self.assertEqual(
            fake_db.user.update_calls,
            [],
        )
        self.assertEqual(
            fake_db.user.update_many_calls,
            [
                ({"id": "user-1", "defaultChatModel": "anthropic::claude-3-7-sonnet"}, {"defaultChatModel": None}),
                ({"id": "user-2", "defaultChatModel": "openai::gpt-4o"}, {"defaultChatModel": None}),
            ],
        )
        self.assertEqual(fake_db.workspaceuserpreference.default_model_by_key[("workspace-2", "user-2")], "openai::gpt-4.1")
        self.assertIsNone(fake_db.user.default_model_by_user_id["user-2"])

    async def test_clear_matching_personal_defaults_preserves_concurrent_workspace_change(self) -> None:
        from ragtime.indexer import model_preferences

        workspace_table = _FakeWorkspacePreferenceTable({("workspace-1", "user-1"): "openai::gpt-4o"})
        workspace_table.before_delete_many = lambda _where: workspace_table.default_model_by_key.__setitem__(("workspace-1", "user-1"), "openai::gpt-4.1")
        fake_db = SimpleNamespace(
            user=_FakeUserTable({"user-1": "anthropic::claude-3-7-sonnet"}),
            workspaceuserpreference=workspace_table,
        )

        with mock.patch.object(model_preferences, "get_db", mock.AsyncMock(return_value=fake_db)):
            await model_preferences.clear_matching_personal_defaults("user-1", "workspace-1", "openai::gpt-4o")

        self.assertEqual(
            workspace_table.delete_many_calls,
            [{"workspaceId": "workspace-1", "userId": "user-1", "defaultChatModel": "openai::gpt-4o"}],
        )
        self.assertEqual(workspace_table.default_model_by_key[("workspace-1", "user-1")], "openai::gpt-4.1")

    async def test_clear_matching_personal_defaults_preserves_concurrent_user_change(self) -> None:
        from ragtime.indexer import model_preferences

        user_table = _FakeUserTable({"user-1": "openai::gpt-4o"})
        user_table.before_update_many = lambda _where, _data: user_table.default_model_by_user_id.__setitem__("user-1", "openai::gpt-4.1")
        fake_db = SimpleNamespace(
            user=user_table,
            workspaceuserpreference=_FakeWorkspacePreferenceTable({}),
        )

        with mock.patch.object(model_preferences, "get_db", mock.AsyncMock(return_value=fake_db)):
            await model_preferences.clear_matching_personal_defaults("user-1", None, "openai::gpt-4o")

        self.assertEqual(
            user_table.update_many_calls,
            [({"id": "user-1", "defaultChatModel": "openai::gpt-4o"}, {"defaultChatModel": None})],
        )
        self.assertEqual(user_table.default_model_by_user_id["user-1"], "openai::gpt-4.1")

    async def test_resolve_new_conversation_model_honors_precedence_and_clears_stale_workspace_default(self) -> None:
        from ragtime.indexer import model_preferences

        fake_db = SimpleNamespace(
            user=_FakeUserTable({"user-1": "anthropic::claude-3-7-sonnet"}),
            workspaceuserpreference=_FakeWorkspacePreferenceTable({("workspace-1", "user-1"): "openai::gpt-4o"}),
        )
        availability = model_preferences.ModelAvailabilitySnapshot(
            available_model_ids=frozenset({"anthropic::claude-3-7-sonnet"}),
            authoritative_providers=frozenset({"openai", "anthropic"}),
        )

        with (
            mock.patch.object(model_preferences, "get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch.object(model_preferences, "_resolve_default_conversation_model", return_value="global::fallback"),
        ):
            explicit = await model_preferences.resolve_new_conversation_model(
                SimpleNamespace(),
                user_id="user-1",
                workspace_id="workspace-1",
                explicit_model=" manual::choice ",
                availability=availability,
            )
            resolved = await model_preferences.resolve_new_conversation_model(
                SimpleNamespace(),
                user_id="user-1",
                workspace_id="workspace-1",
                availability=availability,
            )

        self.assertEqual(explicit, " manual::choice ")
        self.assertEqual(resolved, "anthropic::claude-3-7-sonnet")
        self.assertEqual(fake_db.workspaceuserpreference.delete_calls, [])
        self.assertEqual(
            fake_db.workspaceuserpreference.delete_many_calls,
            [{"workspaceId": "workspace-1", "userId": "user-1", "defaultChatModel": "openai::gpt-4o"}],
        )
        self.assertEqual(fake_db.user.update_calls, [])
        self.assertEqual(
            fake_db.user.update_many_calls,
            [({"id": "user-1", "defaultChatModel": "openai::gpt-4o"}, {"defaultChatModel": None})],
        )

    async def test_resolve_new_conversation_model_clears_stale_user_default_and_falls_back_to_global(self) -> None:
        from ragtime.indexer import model_preferences

        fake_db = SimpleNamespace(
            user=_FakeUserTable({"user-1": "openai::gpt-4o"}),
            workspaceuserpreference=_FakeWorkspacePreferenceTable({}),
        )
        availability = model_preferences.ModelAvailabilitySnapshot(
            available_model_ids=frozenset({"anthropic::claude-3-7-sonnet"}),
            authoritative_providers=frozenset({"openai"}),
        )

        with (
            mock.patch.object(model_preferences, "get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch.object(model_preferences, "_resolve_default_conversation_model", return_value="global::fallback"),
        ):
            resolved = await model_preferences.resolve_new_conversation_model(
                SimpleNamespace(),
                user_id="user-1",
                availability=availability,
            )

        self.assertEqual(resolved, "global::fallback")
        self.assertEqual(fake_db.user.update_calls, [])
        self.assertEqual(
            fake_db.user.update_many_calls,
            [({"id": "user-1", "defaultChatModel": "openai::gpt-4o"}, {"defaultChatModel": None})],
        )

    async def test_resolve_new_conversation_model_preserves_non_authoritative_provider_errors(self) -> None:
        from ragtime.indexer import model_preferences

        fake_db = SimpleNamespace(
            user=_FakeUserTable({"user-1": "openai::gpt-4o"}),
            workspaceuserpreference=_FakeWorkspacePreferenceTable({("workspace-1", "user-1"): "anthropic::claude-3-7-sonnet"}),
        )
        availability = model_preferences.ModelAvailabilitySnapshot(
            available_model_ids=frozenset(),
            authoritative_providers=frozenset({"openai"}),
        )

        with (
            mock.patch.object(model_preferences, "get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch.object(model_preferences, "_resolve_default_conversation_model", return_value="global::fallback"),
        ):
            resolved = await model_preferences.resolve_new_conversation_model(
                SimpleNamespace(),
                user_id="user-1",
                workspace_id="workspace-1",
                availability=availability,
            )

        self.assertEqual(resolved, "anthropic::claude-3-7-sonnet")
        self.assertEqual(fake_db.workspaceuserpreference.delete_calls, [])
        self.assertEqual(fake_db.user.update_calls, [])
        self.assertEqual(fake_db.workspaceuserpreference.delete_many_calls, [])
        self.assertEqual(fake_db.user.update_many_calls, [])
