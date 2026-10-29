"""Persistence for remote pairings, using parameterized PostgreSQL operations."""

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from prisma.enums import ToolAccessLevel

from ragtime.core.database import get_db
from ragtime.core.encryption import CONNECTION_CONFIG_PASSWORD_FIELDS, decrypt_secret, encrypt_json_passwords, encrypt_secret
from ragtime.indexer.models import ToolConfig, ToolType
from ragtime.indexer.repository import repository as tool_repository

ENROLLMENT_TTL = timedelta(minutes=10)
RECOVERY_TTL = timedelta(minutes=10)
DRAFT_TTL = timedelta(hours=24)


def credential_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pairing_row(row: Any) -> dict[str, Any]:
    """Normalize raw SQL column names to the model's stable Python contract."""
    raw: dict[str, Any] = dict(row)
    aliases = {
        "tool_id": "toolId",
        "enrollment_token_hash": "enrollmentTokenHash",
        "enrollment_expires_at": "enrollmentExpiresAt",
        "enrollment_nonce": "enrollmentNonce",
        "encrypted_private_key": "encryptedPrivateKey",
        "public_key": "publicKey",
        "tunnel_credential_hash": "tunnelCredentialHash",
        "recovery_expires_at": "recoveryExpiresAt",
        "is_draft": "isDraft",
        "last_activity_at": "lastActivityAt",
        "created_at": "createdAt",
        "updated_at": "updatedAt",
    }
    normalized = {aliases.get(key, key): value for key, value in raw.items()}
    for key in ("enrollmentExpiresAt", "recoveryExpiresAt", "lastActivityAt", "createdAt", "updatedAt"):
        value = normalized.get(key)
        if isinstance(value, str):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            normalized[key] = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed
    return normalized


class RemotePairingRepository:
    async def create(
        self, *, tool_type: ToolType, token: str, tunnel_credential: str, private_key: str, public_key: str, tool_id: str | None = None
    ) -> dict[str, Any]:
        tool = await tool_repository.get_tool_config(tool_id) if tool_id else None
        if tool_id and tool is None:
            raise ValueError("Existing tool was not found")
        if tool and tool.tool_type != tool_type:
            raise ValueError("Existing tool type does not match remote pairing type")
        if tool is None:
            tool = await tool_repository.create_tool_config(
                ToolConfig(
                    name="Remote client (pending)", tool_type=tool_type, enabled=False, description="Remote client enrollment draft", connection_config={}
                )
            )
        if not tool.id:
            raise RuntimeError("Remote draft was created without an id")
        is_draft = tool_id is None
        db = await get_db()
        now = datetime.now(timezone.utc)
        expires_at = now + ENROLLMENT_TTL
        try:
            row = await db.remotepairing.create(
                data={
                    "id": str(uuid.uuid4()),
                    "toolId": tool.id,
                    "enrollmentTokenHash": credential_hash(token),
                    "enrollmentExpiresAt": expires_at,
                    "encryptedPrivateKey": encrypt_secret(private_key),
                    "publicKey": public_key,
                    "tunnelCredentialHash": credential_hash(tunnel_credential),
                    "isDraft": is_draft,
                }
            )
            await db.remotepairing.update(
                where={"id": row.id},
                data={"encryptedEnrollmentResponse": encrypt_secret(tunnel_credential)},
            )
            if is_draft:
                await db.toolaccesspolicy.upsert(
                    where={"toolConfigId": tool.id},
                    data={
                        "create": {
                            "id": str(uuid.uuid4()),
                            "toolConfig": {"connect": {"id": tool.id}},
                            "defaultChatAccess": ToolAccessLevel.deny,
                            "defaultWorkspaceAccess": ToolAccessLevel.deny,
                        },
                        "update": {"defaultChatAccess": ToolAccessLevel.deny, "defaultWorkspaceAccess": ToolAccessLevel.deny},
                    },
                )
        except Exception:
            if not tool_id:
                await tool_repository.delete_tool_config(tool.id)
            raise
        return await self.get(str(row.id)) or _pairing_row(row.__dict__)

    async def enrollment_credential(self, pairing_id: str) -> str | None:
        db = await get_db()
        rows = await db.query_raw("SELECT encrypted_enrollment_response FROM remote_pairings WHERE id = $1", pairing_id)
        pairing = await self.get(pairing_id)
        if (
            not rows
            or not rows[0].get("encrypted_enrollment_response")
            or not pairing
            or not pairing.get("recoveryExpiresAt")
            or pairing["recoveryExpiresAt"] <= datetime.now(timezone.utc)
        ):
            return None
        return decrypt_secret(str(rows[0]["encrypted_enrollment_response"]))

    async def get_by_token(self, token: str) -> dict[str, Any] | None:
        db = await get_db()
        row = await db.remotepairing.find_unique(where={"enrollmentTokenHash": credential_hash(token)})
        return _pairing_row(row.__dict__) if row else None

    async def get(self, pairing_id: str) -> dict[str, Any] | None:
        db = await get_db()
        row = await db.remotepairing.find_unique(where={"id": pairing_id})
        return _pairing_row(row.__dict__) if row else None

    async def touch(self, pairing_id: str) -> None:
        db = await get_db()
        await db.execute_raw("UPDATE remote_pairings SET last_activity_at = CURRENT_TIMESTAMP WHERE id = $1", pairing_id)

    async def get_by_tunnel_credential(self, credential: str) -> dict[str, Any] | None:
        db = await get_db()
        row = await db.remotepairing.find_unique(where={"tunnelCredentialHash": credential_hash(credential)})
        return _pairing_row(row.__dict__) if row else None

    async def regenerate(self, pairing_id: str, token: str, tunnel_credential: str, private_key: str, public_key: str) -> dict[str, Any] | None:
        db = await get_db()
        async with db.tx() as tx:
            rows = await tx.query_raw("SELECT * FROM remote_pairings WHERE id = $1 FOR UPDATE", pairing_id)
            if not rows:
                return None
            current = _pairing_row(rows[0])
            if str(getattr(current["status"], "value", current["status"])) == "revoked":
                return None
            replacement = await tx.remotepairing.create(
                data={
                    "id": str(uuid.uuid4()),
                    "toolId": str(current["toolId"]),
                    "enrollmentTokenHash": credential_hash(token),
                    "enrollmentExpiresAt": datetime.now(timezone.utc) + ENROLLMENT_TTL,
                    "encryptedPrivateKey": encrypt_secret(private_key),
                    "publicKey": public_key,
                    "tunnelCredentialHash": credential_hash(tunnel_credential),
                    "encryptedEnrollmentResponse": encrypt_secret(tunnel_credential),
                    "isDraft": bool(current.get("isDraft", True)),
                }
            )
            await tx.execute_raw(
                "UPDATE remote_pairings SET status = 'revoked', updated_at = CURRENT_TIMESTAMP WHERE tool_id = $1 AND id <> $2 AND status IN ('pending', 'consumed')",
                current["toolId"],
                str(replacement.id),
            )
        return await self.get(str(replacement.id))

    async def redeem(self, token: str, nonce: str, facts: dict[str, Any], transport: str) -> dict[str, Any] | None:
        """Atomically claim a token; the same nonce returns its durable result."""
        db = await get_db()
        token_digest = credential_hash(token)
        async with db.tx() as tx:
            rows = await tx.query_raw("SELECT * FROM remote_pairings WHERE enrollment_token_hash = $1 FOR UPDATE", token_digest)
            if not rows:
                return None
            pairing = _pairing_row(rows[0])
            status = str(getattr(pairing["status"], "value", pairing["status"]))
            if status == "revoked":
                return None
            if pairing.get("enrollmentNonce"):
                return pairing if pairing["enrollmentNonce"] == nonce else None
            if status != "pending" or pairing["enrollmentExpiresAt"] < datetime.now(timezone.utc):
                await tx.execute_raw("UPDATE remote_pairings SET status = 'expired', updated_at = CURRENT_TIMESTAMP WHERE id = $1", pairing["id"])
                return None
            await tx.execute_raw(
                "UPDATE remote_pairings SET status = 'consumed', enrollment_nonce = $2, facts = $3::jsonb, transport = $4, recovery_expires_at = CURRENT_TIMESTAMP + INTERVAL '10 minutes', last_activity_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                pairing["id"],
                nonce,
                json.dumps(facts),
                transport,
            )
            pairing.update(status="consumed", enrollment_nonce=nonce, facts=facts, transport=transport)
            return pairing

    async def redeem_offline(self, pairing_id: str, nonce: str, facts: dict[str, Any]) -> dict[str, Any] | None:
        """Accept offline facts only under an authenticated administrator action."""
        db = await get_db()
        async with db.tx() as tx:
            rows = await tx.query_raw("SELECT * FROM remote_pairings WHERE id = $1 FOR UPDATE", pairing_id)
            if not rows:
                return None
            pairing = _pairing_row(rows[0])
            status = str(getattr(pairing["status"], "value", pairing["status"]))
            if status in {"revoked", "expired"}:
                return None
            if pairing.get("enrollmentNonce") and pairing["enrollmentNonce"] != nonce:
                return None
            if not pairing.get("enrollmentNonce"):
                await tx.execute_raw(
                    "UPDATE remote_pairings SET status = 'consumed', enrollment_nonce = $2, facts = $3::jsonb, transport = 'direct', recovery_expires_at = CURRENT_TIMESTAMP + INTERVAL '10 minutes', "
                    "last_activity_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                    pairing_id,
                    nonce,
                    json.dumps(facts),
                )
                pairing.update(status="consumed", enrollment_nonce=nonce, facts=facts, transport="direct")
            return pairing

    async def mark_paired(self, pairing_id: str, connection_config: dict[str, Any]) -> dict[str, Any] | None:
        db = await get_db()
        async with db.tx() as tx:
            rows = await tx.query_raw("SELECT * FROM remote_pairings WHERE id = $1 FOR UPDATE", pairing_id)
            if not rows:
                return None
            pairing = _pairing_row(rows[0])
            status = str(getattr(pairing["status"], "value", pairing["status"]))
            if status == "revoked":
                return None
            if status == "paired":
                return pairing
            if status != "consumed":
                return None
            tool_rows = await tx.query_raw("SELECT connection_config FROM tool_configs WHERE id = $1 FOR UPDATE", pairing["toolId"])
            if not tool_rows:
                return None
            config = dict(tool_rows[0]["connection_config"] or {})
            config.update(connection_config)
            config["key_content"] = decrypt_secret(str(pairing["encryptedPrivateKey"]))
            config["pairing_id"] = pairing_id
            encrypted_config = encrypt_json_passwords(config, CONNECTION_CONFIG_PASSWORD_FIELDS)
            await tx.execute_raw(
                "UPDATE tool_configs SET connection_config = $2::jsonb, updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                pairing["toolId"],
                json.dumps(encrypted_config),
            )
            revoked_rows = await tx.query_raw(
                "UPDATE remote_pairings SET status = 'revoked', updated_at = CURRENT_TIMESTAMP "
                "WHERE tool_id = $1 AND id <> $2 AND status IN ('pending', 'consumed', 'paired') RETURNING id",
                pairing["toolId"],
                pairing_id,
            )
            await tx.execute_raw(
                "UPDATE remote_pairings SET status = 'paired', transport = $2, last_activity_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                pairing_id,
                str(connection_config.get("transport") or "direct"),
            )
            pairing["status"] = "paired"
            pairing["transport"] = str(connection_config.get("transport") or "direct")
            pairing["revokedPairingIds"] = [str(row["id"]) for row in revoked_rows]
            return pairing

    async def revoke(self, pairing_id: str) -> dict[str, Any] | None:
        db = await get_db()
        async with db.tx() as tx:
            rows = await tx.query_raw("SELECT * FROM remote_pairings WHERE id = $1 FOR UPDATE", pairing_id)
            if not rows:
                return None
            pairing = _pairing_row(rows[0])
            revoked_rows = await tx.query_raw(
                "UPDATE remote_pairings SET status = 'revoked', updated_at = CURRENT_TIMESTAMP WHERE tool_id = $1 AND status <> 'revoked' RETURNING id",
                pairing["toolId"],
            )
            tool_rows = await tx.query_raw("SELECT connection_config FROM tool_configs WHERE id = $1 FOR UPDATE", pairing["toolId"])
            if tool_rows:
                safe_config = dict(tool_rows[0]["connection_config"] or {})
                for key in ("key_content", "key_path", "key_passphrase", "pairing_id", "transport", "host_keys"):
                    safe_config.pop(key, None)
                await tx.execute_raw(
                    "UPDATE tool_configs SET enabled = false, connection_config = $2::jsonb, updated_at = CURRENT_TIMESTAMP WHERE id = $1",
                    pairing["toolId"],
                    json.dumps(safe_config),
                )
        pairing["status"] = "revoked"
        pairing["revokedPairingIds"] = [str(row["id"]) for row in revoked_rows]
        return pairing

    async def select_tunnel(self, pairing_id: str) -> dict[str, Any] | None:
        db = await get_db()
        await db.execute_raw(
            "UPDATE remote_pairings SET transport = 'tunnel', updated_at = CURRENT_TIMESTAMP WHERE id = $1 AND status = 'consumed'", pairing_id
        )
        return await self.get(pairing_id)

    async def fail(self, pairing_id: str, error: str) -> dict[str, Any] | None:
        db = await get_db()
        await db.execute_raw(
            "UPDATE remote_pairings SET status = 'failed', error = $2, updated_at = CURRENT_TIMESTAMP WHERE id = $1 AND status = 'consumed'",
            pairing_id,
            error[:512],
        )
        return await self.get(pairing_id)

    async def delete_draft(self, pairing_id: str) -> bool:
        pairing = await self.get(pairing_id)
        if pairing is None:
            return False
        await tool_repository.delete_tool_config(str(pairing["toolId"]))
        return True

    async def gc_drafts(self) -> int:
        db = await get_db()
        cutoff = datetime.now(timezone.utc) - DRAFT_TTL
        rows = await db.query_raw(
            "SELECT DISTINCT p.tool_id FROM remote_pairings p JOIN tool_configs t ON t.id = p.tool_id "
            "WHERE t.enabled = false AND NOT EXISTS (SELECT 1 FROM remote_pairings saved WHERE saved.tool_id = p.tool_id AND saved.is_draft = false) "
            "AND p.last_activity_at < $1::timestamp",
            cutoff.replace(tzinfo=None).isoformat(),
        )
        for row in rows:
            await tool_repository.delete_tool_config(str(row["tool_id"]))
        return len(rows)


remote_pairing_repository = RemotePairingRepository()
