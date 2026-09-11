"""Opt-in installed CLI -> Postgres -> HTTP API chat fixture round trips."""

import copy
import json
import os
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any

import httpx

from ragtime.core.auth import create_access_token, hash_token
from ragtime.fixtures.common import database_cursor, json_parameter


@unittest.skipUnless(os.environ.get("CHAT_FIXTURE_CONTAINER_TESTS") == "1", "local container fixture integration opt-in")
class ChatFixtureRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.source_owner = str(uuid.uuid4())
        self.target_owner = str(uuid.uuid4())
        self.username = f"fixture-import-{self.target_owner}"
        self.source_id = str(uuid.uuid4())
        self.title = f"Fixture roundtrip {self.source_id}"
        self.branch_ids = [str(uuid.uuid4()) for _ in range(3)]
        self.task_id = str(uuid.uuid4())
        self.session_id = str(uuid.uuid4())
        self.token = create_access_token(self.target_owner, self.username, "admin", mfa_verified=True)
        self.messages = [
            {"role": "user", "content": "Shared prompt", "message_id": "shared-user"},
            {"role": "assistant", "content": "Shared answer", "message_id": "shared-assistant"},
            {"role": "user", "content": "New prompt with 'quotes' and Unicode café", "message_id": "new-user"},
            {"role": "assistant", "content": "New answer", "message_id": "new-assistant", "events": [{"type": "content", "content": "New answer"}]},
        ]
        self.old_messages = self.messages[:2] + [
            {"role": "user", "content": "Old prompt", "message_id": "old-user"},
            {"role": "assistant", "content": "Old answer", "message_id": "old-assistant"},
        ]
        with database_cursor(read_only=False) as cursor:
            for user_id, username in [(self.source_owner, f"fixture-source-{self.source_owner}"), (self.target_owner, self.username)]:
                cursor.execute("INSERT INTO users (id, username, role, created_at, updated_at) VALUES (%s,%s,'admin',NOW(),NOW())", (user_id, username))
            cursor.execute(
                "INSERT INTO sessions (id,user_id,token_hash,expires_at,mfa_verified_at,created_at) VALUES (%s,%s,%s,NOW()+interval '5 minutes',NOW(),NOW())",
                (self.session_id, self.target_owner, hash_token(self.token)),
            )
            cursor.execute(
                "INSERT INTO conversations (id,title,model,messages,total_tokens,user_id,active_branch_id,tool_output_mode,created_at,updated_at) "
                "VALUES (%s,%s,'fixture-model',%s::jsonb,42,%s,%s,'show',NOW(),NOW())",
                (self.source_id, self.title, json_parameter(self.messages), self.source_owner, self.branch_ids[2]),
            )
            for branch_id, parent, point, kind, payload in [
                (self.branch_ids[0], None, 0, "replay", self.messages),
                (self.branch_ids[1], self.branch_ids[0], 2, "edit", self.old_messages),
                (self.branch_ids[2], self.branch_ids[0], 2, None, self.messages),
            ]:
                cursor.execute(
                    "INSERT INTO conversation_branches (id,conversation_id,parent_branch_id,branch_point_index,branch_kind,"
                    "base_messages,preserved_messages,created_by_user_id,created_at,updated_at) "
                    'VALUES (%s,%s,%s,%s,%s::"ConversationBranchKind",%s::jsonb,%s::jsonb,%s,NOW(),NOW())',
                    (branch_id, self.source_id, parent, point, kind, json_parameter(payload[:point]), json_parameter(payload[point:]), self.source_owner),
                )
            cursor.execute(
                "INSERT INTO chat_tasks (id,conversation_id,status,user_message,response_content,streaming_state,created_at,completed_at,last_update_at) "
                "VALUES (%s,%s,'completed','historical prompt','historical answer',%s::jsonb,NOW(),NOW(),NOW())",
                (
                    self.task_id,
                    self.source_id,
                    json_parameter({"content": "historical answer", "events": [{"type": "content", "content": "historical answer"}]}),
                ),
            )
        self.addCleanup(self.cleanup_records)

    def cleanup_records(self):
        with database_cursor(read_only=False) as cursor:
            cursor.execute("DELETE FROM conversations WHERE user_id = ANY(%s)", ([self.source_owner, self.target_owner],))
            cursor.execute("DELETE FROM users WHERE id = ANY(%s)", ([self.source_owner, self.target_owner],))

    def command(self, action: str, *arguments: str, payload: str | None = None) -> dict[str, Any]:
        completed = subprocess.run([f"/usr/local/bin/{action}", *arguments], input=payload, capture_output=True, text=True, timeout=40)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        result = json.loads(completed.stdout)
        if not isinstance(result, dict):
            self.fail(f"Expected JSON object, received {type(result).__name__}")
        return result

    def rejected_command(self, action: str, *arguments: str, payload: str | None = None) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run([f"/usr/local/bin/{action}", *arguments], input=payload, capture_output=True, text=True, timeout=40)
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, "")
        self.assertTrue(completed.stderr)
        return completed

    def export_source(self) -> dict[str, Any]:
        return self.command("export", "chat", self.source_id)

    def clone_count(self) -> int:
        with database_cursor(read_only=True) as cursor:
            cursor.execute("SELECT count(*) FROM conversations WHERE user_id = %s", (self.target_owner,))
            return cursor.fetchone()[0]

    def test_stdin_export_import_reexport_and_branch_api_round_trip(self):
        source = self.command("export", "chat", self.source_id[:12])
        self.assertEqual(source["conversation"]["id"], self.source_id)
        self.assertEqual(len(source["branches"]), 3)
        self.assertEqual(len(source["completed_tasks"]), 1)
        result = self.command("import", "chat", "--owner", self.username, payload=json.dumps(source))
        clone_id = result["conversation_id"]
        self.assertNotEqual(clone_id, self.source_id)
        self.assertEqual((result["message_count"], result["branch_count"], result["completed_task_count"]), (4, 3, 1))
        clone = self.command("export", "chat", clone_id)
        self.assertEqual(clone["conversation"]["messages"], source["conversation"]["messages"])
        self.assertEqual(clone["conversation"]["user_id"], self.target_owner)
        self.assertEqual(clone["conversation"]["tool_output_mode"], "show")
        for field in ("active_task_id", "workspace_id", "parent_conversation_id", "subagent_role", "subagent_index"):
            self.assertIsNone(clone["conversation"][field])
        self.assertEqual(clone["conversation"]["tool_selection_mode"], "custom")
        self.assertEqual(clone["conversation"]["loaded_tool_skill_ids"], [])
        self.assertFalse(clone["conversation"]["subagents_enabled"])
        new_by_kind = {branch["branch_kind"]: branch for branch in clone["branches"]}
        self.assertEqual(clone["conversation"]["active_branch_id"], new_by_kind[None]["id"])
        for branch in source["branches"]:
            copied = new_by_kind[branch["branch_kind"]]
            self.assertNotIn(copied["id"], self.branch_ids)
            self.assertEqual(copied["conversation_id"], clone_id)
            self.assertEqual(copied["base_messages"], branch["base_messages"])
            self.assertEqual(copied["preserved_messages"], branch["preserved_messages"])
            self.assertEqual(copied["created_by_user_id"], self.target_owner)
            self.assertEqual(copied["parent_branch_id"], new_by_kind["replay"]["id"] if branch["parent_branch_id"] else None)
        task = clone["completed_tasks"][0]
        self.assertNotEqual(task["id"], self.task_id)
        for field in ("user_message", "response_content", "streaming_state"):
            self.assertEqual(task[field], source["completed_tasks"][0][field])
        with database_cursor(read_only=True) as cursor:
            cursor.execute("SELECT status FROM chat_tasks WHERE conversation_id = %s", (clone_id,))
            self.assertEqual(cursor.fetchall(), [("completed",)])
            for table in ("conversation_tool_selections", "conversation_tool_group_selections", "conversation_members", "conversation_shares"):
                cursor.execute(f"SELECT count(*) FROM {table} WHERE conversation_id = %s", (clone_id,))
                self.assertEqual(cursor.fetchone()[0], 0)
        with httpx.Client(base_url="http://localhost:8000", headers={"Authorization": f"Bearer {self.token}"}, timeout=30) as client:
            for kind, expected in [("edit", self.old_messages), (None, self.messages), ("replay", self.messages), (None, self.messages)]:
                response = client.post(f"/indexes/conversations/{clone_id}/branches/switch", json={"branch_id": new_by_kind[kind]["id"]})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual([message["content"] for message in response.json()["messages"]], [message["content"] for message in expected])
        self.assertEqual(self.export_source(), source)
        print(f"CLI round trip verified: 4 messages, 3 remapped branches, 1 historical task; API branch switches passed ({clone_id})")

    def test_file_import_accepts_legacy_v1_and_title_override(self):
        source = self.export_source()
        source.pop("fixture_type", None)
        for branch in source["branches"]:
            branch.pop("base_messages", None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat fixture.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            result = self.command("import", "chat", str(path), "--owner", self.username, "--title", "Custom fixture title")
        self.assertEqual(result["title"], "Custom fixture title")
        clone = self.command("export", "chat", result["conversation_id"])
        self.assertTrue(all(branch["base_messages"] is None for branch in clone["branches"]))
        self.assertEqual(clone["conversation"]["messages"], source["conversation"]["messages"])

    def test_rejected_imports_and_database_failure_leave_no_partial_clone(self):
        source = self.export_source()
        self.rejected_command("import", "chat", "--owner", "missing-" + self.username, payload=json.dumps(source))
        invalid = copy.deepcopy(source)
        invalid["branches"][1]["parent_branch_id"] = "missing-parent"
        self.rejected_command("import", "chat", "--owner", self.username, payload=json.dumps(invalid))
        # Valid graph, but PostgreSQL rejects a NUL in a later branch JSON value.
        # This exercises rollback after the conversation INSERT has succeeded.
        invalid = copy.deepcopy(source)
        invalid["branches"][1]["preserved_messages"][0]["content"] = "\u0000"
        self.rejected_command("import", "chat", "--owner", self.username, payload=json.dumps(invalid))
        self.assertEqual(self.clone_count(), 0)
        self.assertEqual(self.export_source(), source)
