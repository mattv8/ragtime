"""Database regressions for the durable remote-pairing lifecycle."""

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
from prisma.enums import ToolAccessLevel

from ragtime.core.database import connect_db, disconnect_db, get_db
from ragtime.indexer.models import ToolConfig, ToolType
from ragtime.indexer.repository import repository as tool_repository
from ragtime.remote_client.repository import remote_pairing_repository
from ragtime.remote_client.service import remote_pairing_service

TEST_DATABASE_URL_ENV = "RAGTIME_TEST_DATABASE_URL"


def test_remote_pairing_lifecycle_database_regressions() -> None:
    test_database_url = os.environ.get(TEST_DATABASE_URL_ENV)
    if not test_database_url or os.environ.get("DATABASE_URL") != test_database_url:
        pytest.skip("requires DATABASE_URL to match the opt-in RAGTIME_TEST_DATABASE_URL disposable database")
    asyncio.run(_exercise_lifecycle())


async def _exercise_lifecycle() -> None:
    await connect_db()
    created_tool_ids: list[str] = []
    try:
        await _assert_saved_tools_survive_abandoned_pairing(created_tool_ids)
        original, token, _ = await remote_pairing_service.create(ToolType.SSH_SHELL)
        tool_id = str(original["toolId"])
        created_tool_ids.append(tool_id)
        await tool_repository.update_tool_config(tool_id, {"name": f"Remote pairing DB verify {original['id']}"})
        facts = {
            "os": "linux",
            "username": "verify",
            "hostname": "verify-host",
            "addresses": ["10.0.0.10"],
            "sshd_port": 22,
            "host_keys": ["ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIexample"],
        }
        claimed = await remote_pairing_repository.redeem(token, "a" * 16, facts, "direct")
        assert claimed is not None
        replay = await remote_pairing_repository.redeem(token, "a" * 16, {**facts, "hostname": "changed"}, "direct")
        assert replay is not None and replay["facts"]["hostname"] == "verify-host"
        assert await remote_pairing_repository.redeem(token, "b" * 16, facts, "direct") is None

        config = {"host": "10.0.0.10", "port": 22, "user": "verify", "host_keys": facts["host_keys"], "transport": "direct"}
        paired = await remote_pairing_repository.mark_paired(str(original["id"]), config)
        assert paired is not None and paired["status"] == "paired"

        await tool_repository.update_tool_config(tool_id, {"enabled": True})
        await tool_repository.update_tool_config(tool_id, {"enabled": False})
        db = await get_db()
        await db.execute_raw(
            "UPDATE remote_pairings SET last_activity_at = $2::timestamp WHERE id = $1",
            str(original["id"]),
            (datetime.now(timezone.utc) - timedelta(days=2)).replace(tzinfo=None).isoformat(),
        )
        assert await remote_pairing_repository.gc_drafts() == 0
        assert await tool_repository.get_tool_config(tool_id) is not None

        regenerated = await remote_pairing_service.regenerate(str(original["id"]))
        assert regenerated is not None
        replacement, replacement_token, _ = regenerated
        assert replacement["id"] != original["id"] and replacement["toolId"] == original["toolId"]
        replacement_claim = await remote_pairing_repository.redeem(replacement_token, "c" * 16, facts, "direct")
        assert replacement_claim is not None
        swapped = await remote_pairing_repository.mark_paired(str(replacement["id"]), config)
        assert swapped is not None and swapped["status"] == "paired"
        revoked_original = await remote_pairing_repository.get(str(original["id"]))
        assert revoked_original is not None and revoked_original["status"] == "revoked"
        assert await remote_pairing_repository.mark_paired(str(original["id"]), config) is None
        revoked_replacement = await remote_pairing_repository.revoke(str(replacement["id"]))
        assert revoked_replacement is not None and revoked_replacement["status"] == "revoked"
        assert await remote_pairing_service.regenerate(str(replacement["id"])) is None

        expired, expired_token, _ = await remote_pairing_service.create(ToolType.SSH_SHELL)
        created_tool_ids.append(str(expired["toolId"]))
        await tool_repository.update_tool_config(
            str(expired["toolId"]),
            {"name": f"Remote pairing DB verify {expired['id']}"},
        )
        await db.execute_raw(
            "UPDATE remote_pairings SET enrollment_expires_at = $2::timestamp WHERE id = $1",
            str(expired["id"]),
            (datetime.now(timezone.utc) - timedelta(minutes=1)).replace(tzinfo=None).isoformat(),
        )
        assert await remote_pairing_repository.redeem(expired_token, "d" * 16, facts, "direct") is None
        assert await remote_pairing_repository.redeem(expired_token, "e" * 16, facts, "direct") is None
        await tool_repository.delete_tool_config(str(expired["toolId"]))
    finally:
        for created_tool_id in created_tool_ids:
            await tool_repository.delete_tool_config(created_tool_id)
        await disconnect_db()


async def _assert_saved_tools_survive_abandoned_pairing(created_tool_ids: list[str]) -> None:
    db = await get_db()
    for enabled in (True, False):
        tool = await tool_repository.create_tool_config(
            ToolConfig(
                name=f"Saved remote pairing DB verify {enabled}",
                tool_type=ToolType.SSH_SHELL,
                enabled=enabled,
                connection_config={"host": "manual.example", "port": 2222, "user": "manual"},
            )
        )
        assert tool.id
        created_tool_ids.append(tool.id)
        policy = await db.toolaccesspolicy.upsert(
            where={"toolConfigId": tool.id},
            data={
                "create": {
                    "toolConfig": {"connect": {"id": tool.id}},
                    "defaultChatAccess": ToolAccessLevel.read,
                    "defaultWorkspaceAccess": ToolAccessLevel.read_write,
                },
                "update": {"defaultChatAccess": ToolAccessLevel.read, "defaultWorkspaceAccess": ToolAccessLevel.read_write},
            },
        )
        pairing, _, _ = await remote_pairing_service.create(ToolType.SSH_SHELL, tool.id)
        assert pairing["isDraft"] is False
        persisted = await tool_repository.get_tool_config(tool.id)
        assert persisted is not None and persisted.connection_config == tool.connection_config
        preserved_policy = await db.toolaccesspolicy.find_unique(where={"id": policy.id})
        assert preserved_policy is not None
        assert preserved_policy.defaultChatAccess == "read"
        assert preserved_policy.defaultWorkspaceAccess == "read_write"
        await db.execute_raw(
            "UPDATE remote_pairings SET last_activity_at = $2::timestamp WHERE id = $1",
            str(pairing["id"]),
            (datetime.now(timezone.utc) - timedelta(days=2)).replace(tzinfo=None).isoformat(),
        )
    assert await remote_pairing_repository.gc_drafts() == 0
    for tool_id in created_tool_ids:
        assert await tool_repository.get_tool_config(tool_id) is not None
