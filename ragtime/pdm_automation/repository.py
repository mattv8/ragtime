from __future__ import annotations

import secrets
from datetime import datetime
from typing import Any

from ragtime.core.database import get_db
from ragtime.core.datetimes import coerce_utc_datetime, parse_utc_iso_datetime, utc_now
from ragtime.core.encryption import decrypt_secret, encrypt_secret
from ragtime.core.sql import sql_quote_literal
from ragtime.core.webhooks import constant_time_token_matches
from ragtime.pdm_automation.models import PdmWebhookConfigResponse, PdmWebhookEnableResponse


def _lock(tool_id: str) -> str:
    return f"SELECT pg_advisory_xact_lock(hashtext({sql_quote_literal('pdm-index:' + tool_id)}))"


def _response(row: dict[str, Any] | None, base_url: str = "") -> PdmWebhookConfigResponse:
    row = row or {}
    webhook_id = str(row.get("webhook_id") or "") or None
    enabled = bool(webhook_id and row.get("webhook_secret"))
    return PdmWebhookConfigResponse(
        enabled=enabled,
        paused=bool(enabled and row.get("webhook_paused")),
        webhook_id=webhook_id,
        webhook_url=f"{base_url.rstrip('/')}/webhooks/pdm/{webhook_id}" if webhook_id and base_url else None,
        created_at=row.get("webhook_created_at"),
        last_received_at=row.get("last_received_at"),
        pending=bool(row.get("pending_webhook") or row.get("pending_schedule")),
        active_job_id=str(row.get("active_job_id") or "") or None,
        last_attempt_at=row.get("last_attempt_at"),
        last_success_at=row.get("last_success_at"),
        last_error=str(row.get("last_error") or "") or None,
    )


class PdmAutomationRepository:
    async def _state(self, db: Any, tool_id: str) -> dict[str, Any] | None:
        rows = await db.query_raw(f"SELECT * FROM pdm_automation_state WHERE tool_config_id={sql_quote_literal(tool_id)}")
        return rows[0] if rows else None

    async def get_status(self, tool_id: str, base_url: str = "") -> PdmWebhookConfigResponse:
        return _response(await self._state(await get_db(), tool_id), base_url)

    async def resolve_webhook(self, webhook_id: str) -> dict[str, Any] | None:
        rows = await (await get_db()).query_raw(
            f"SELECT s.*,t.enabled,t.tool_type FROM pdm_automation_state s JOIN tool_configs t ON t.id=s.tool_config_id WHERE s.webhook_id={sql_quote_literal(webhook_id)}"
        )
        return rows[0] if rows else None

    async def enable_webhook(self, tool_id: str, base_url: str) -> PdmWebhookEnableResponse:
        db = await get_db()
        async with db.tx() as tx:
            await tx.execute_raw(_lock(tool_id))
            row = await self._state(tx, tool_id)
            if row and row.get("webhook_id") and row.get("webhook_secret"):
                await tx.execute_raw(f"UPDATE pdm_automation_state SET webhook_paused=FALSE WHERE tool_config_id={sql_quote_literal(tool_id)}")
                row["webhook_paused"] = False
                return PdmWebhookEnableResponse(**_response(row, base_url).model_dump(), secret=None)
            secret, webhook_id, now = secrets.token_urlsafe(32), secrets.token_urlsafe(24), utc_now().isoformat()
            await tx.execute_raw(f"""INSERT INTO pdm_automation_state(tool_config_id,webhook_id,webhook_secret,webhook_created_at)
                VALUES({sql_quote_literal(tool_id)},{sql_quote_literal(webhook_id)},{sql_quote_literal(encrypt_secret(secret))},{sql_quote_literal(now)}::timestamp)
                ON CONFLICT(tool_config_id) DO UPDATE SET webhook_id=EXCLUDED.webhook_id,webhook_secret=EXCLUDED.webhook_secret,webhook_paused=FALSE,webhook_created_at=EXCLUDED.webhook_created_at""")
            return PdmWebhookEnableResponse(**_response(await self._state(tx, tool_id), base_url).model_dump(), secret=secret)

    async def rotate_webhook(self, tool_id: str, base_url: str) -> PdmWebhookEnableResponse:
        db = await get_db()
        async with db.tx() as tx:
            await tx.execute_raw(_lock(tool_id))
            secret = secrets.token_urlsafe(32)
            await tx.execute_raw(
                f"UPDATE pdm_automation_state SET webhook_secret={sql_quote_literal(encrypt_secret(secret))} WHERE tool_config_id={sql_quote_literal(tool_id)}"
            )
            return PdmWebhookEnableResponse(**_response(await self._state(tx, tool_id), base_url).model_dump(), secret=secret)

    async def set_paused(self, tool_id: str, paused: bool, base_url: str) -> PdmWebhookConfigResponse:
        db = await get_db()
        async with db.tx() as tx:
            await tx.execute_raw(_lock(tool_id))
            suffix = ",pending_webhook=FALSE" if paused else ""
            await tx.execute_raw(
                f"UPDATE pdm_automation_state SET webhook_paused={'TRUE' if paused else 'FALSE'}{suffix} WHERE tool_config_id={sql_quote_literal(tool_id)}"
            )
            return _response(await self._state(tx, tool_id), base_url)

    async def disable_webhook(self, tool_id: str, base_url: str) -> PdmWebhookConfigResponse:
        db = await get_db()
        async with db.tx() as tx:
            await tx.execute_raw(_lock(tool_id))
            await tx.execute_raw(
                f"UPDATE pdm_automation_state SET webhook_id=NULL,webhook_secret=NULL,webhook_paused=FALSE,webhook_created_at=NULL,pending_webhook=FALSE WHERE tool_config_id={sql_quote_literal(tool_id)}"
            )
            return _response(await self._state(tx, tool_id), base_url)

    async def accept_authenticated_webhook(self, webhook_id: str, provided_token: str | None, event_id: str | None) -> str:
        db = await get_db()
        async with db.tx() as tx:
            rows = await tx.query_raw(
                f"SELECT s.*,t.enabled,t.tool_type FROM pdm_automation_state s JOIN tool_configs t ON t.id=s.tool_config_id WHERE s.webhook_id={sql_quote_literal(webhook_id)}"
            )
            if not rows:
                return "missing"
            target = rows[0]
            tool_id = str(target["tool_config_id"])
            await tx.execute_raw(_lock(tool_id))
            # Rotation/pause/disable may have happened after the first lookup.
            rows = await tx.query_raw(
                f"SELECT s.*,t.enabled,t.tool_type FROM pdm_automation_state s JOIN tool_configs t ON t.id=s.tool_config_id WHERE s.webhook_id={sql_quote_literal(webhook_id)} FOR UPDATE"
            )
            if not rows:
                return "missing"
            target = rows[0]
            if not constant_time_token_matches(decrypt_secret(str(target.get("webhook_secret") or "")), provided_token):
                return "unauthorized"
            if target.get("tool_type") != "solidworks_pdm":
                return "missing"
            if not target.get("enabled") or target.get("webhook_paused"):
                return "ignored"
            state = await self._state(tx, tool_id)
            if state and event_id and state.get("last_event_id") == event_id:
                return "accepted"
            now = utc_now().isoformat()
            await tx.execute_raw(
                f"UPDATE pdm_automation_state SET pending_webhook=TRUE,first_pending_at=COALESCE(first_pending_at,{sql_quote_literal(now)}::timestamp),last_received_at={sql_quote_literal(now)}::timestamp,last_event_id={sql_quote_literal(event_id)},pending_generation=pending_generation+1 WHERE tool_config_id={sql_quote_literal(tool_id)}"
            )
            return "accepted"

    async def pending_tool_ids(self) -> list[str]:
        rows = await (await get_db()).query_raw("SELECT tool_config_id FROM pdm_automation_state WHERE pending_webhook OR pending_schedule")
        return [str(row["tool_config_id"]) for row in rows]

    async def mark_schedule_pending(self, tool_id: str) -> None:
        await (await get_db()).execute_raw(f"""INSERT INTO pdm_automation_state(tool_config_id,pending_schedule,first_pending_at,pending_generation)
            VALUES({sql_quote_literal(tool_id)},TRUE,NOW(),1) ON CONFLICT(tool_config_id) DO UPDATE SET pending_schedule=TRUE,first_pending_at=COALESCE(pdm_automation_state.first_pending_at,NOW()),pending_generation=pdm_automation_state.pending_generation+1""")

    async def clear_schedule_pending(self, tool_id: str) -> None:
        await (await get_db()).execute_raw(f"UPDATE pdm_automation_state SET pending_schedule=FALSE WHERE tool_config_id={sql_quote_literal(tool_id)}")

    async def ready_generation(self, tool_id: str) -> int | None:
        """Return a durable ready-work snapshot before external preflight."""
        rows = await (await get_db()).query_raw(f"""SELECT pending_generation FROM pdm_automation_state
            WHERE tool_config_id={sql_quote_literal(tool_id)}
              AND (pending_schedule OR (pending_webhook AND (
                  NOW() >= last_received_at + interval '60 seconds'
                  OR NOW() >= first_pending_at + interval '300 seconds')))
        """)
        return int(rows[0]["pending_generation"]) if rows else None

    async def claim_for_admission(self, tx: Any, tool_id: str, job_id: str) -> int | None:
        """Claim and link an automated job inside PDM's admission transaction."""
        rows = await tx.query_raw(f"SELECT * FROM pdm_automation_state WHERE tool_config_id={sql_quote_literal(tool_id)} FOR UPDATE")
        if not rows or not (rows[0].get("pending_webhook") or rows[0].get("pending_schedule")):
            return None
        state = rows[0]
        if state.get("pending_webhook") and not state.get("pending_schedule"):
            due = await tx.query_raw(
                f"SELECT NOW() >= last_received_at + interval '60 seconds' OR NOW() >= first_pending_at + interval '300 seconds' AS due FROM pdm_automation_state WHERE tool_config_id={sql_quote_literal(tool_id)}"
            )
            if not due or not due[0].get("due"):
                return None
        generation = int(state.get("pending_generation") or 0)
        await tx.execute_raw(
            f"UPDATE pdm_automation_state SET claimed_generation=pending_generation,pending_webhook=FALSE,pending_schedule=FALSE,first_pending_at=NULL,last_attempt_at=NOW(),last_error=NULL,active_job_id={sql_quote_literal(job_id)} WHERE tool_config_id={sql_quote_literal(tool_id)}"
        )
        return generation

    async def finish_job(self, tool_id: str, job_id: str, success: bool, error: str | None = None) -> None:
        await (await get_db()).execute_raw(
            f"UPDATE pdm_automation_state SET active_job_id=NULL,last_attempt_at=NOW(),last_success_at={'NOW()' if success else 'last_success_at'},last_error={sql_quote_literal((error or '')[:1000] or None)} WHERE tool_config_id={sql_quote_literal(tool_id)} AND active_job_id={sql_quote_literal(job_id)}"
        )

    async def record_preflight_failure(self, tool_id: str, generation: int, message: str) -> None:
        # Pending flags were cleared by the atomic claim. Do not touch them here:
        # an event arriving during preflight is a later generation.
        await (await get_db()).execute_raw(
            f"UPDATE pdm_automation_state SET last_attempt_at=NOW(),last_error={sql_quote_literal(message[:1000])} WHERE tool_config_id={sql_quote_literal(tool_id)} AND claimed_generation={int(generation)}"
        )

    async def consume_preflight_failure(self, tool_id: str, generation: int, message: str) -> None:
        """Consume only the observed generation after out-of-transaction preflight.

        A webhook that arrives while a provider check is in progress increments
        ``pending_generation``.  It must remain pending for the next dispatch.
        """
        db = await get_db()
        async with db.tx() as tx:
            await tx.execute_raw(_lock(tool_id))
            state = await self._state(tx, tool_id)
            if not state or int(state.get("pending_generation") or 0) < generation:
                return
            await tx.execute_raw(f"""UPDATE pdm_automation_state SET
                pending_webhook=CASE WHEN pending_generation={generation} THEN FALSE ELSE pending_webhook END,
                pending_schedule=CASE WHEN pending_generation={generation} THEN FALSE ELSE pending_schedule END,
                first_pending_at=CASE WHEN pending_generation={generation} THEN NULL ELSE first_pending_at END,
                last_attempt_at=NOW(),last_error={sql_quote_literal(message[:1000])}
                WHERE tool_config_id={sql_quote_literal(tool_id)}""")

    async def record_preflight_failure_tx(self, tx: Any, tool_id: str, generation: int, message: str) -> None:
        await tx.execute_raw(
            f"UPDATE pdm_automation_state SET last_attempt_at=NOW(),last_error={sql_quote_literal(message[:1000])} WHERE tool_config_id={sql_quote_literal(tool_id)} AND claimed_generation={int(generation)}"
        )

    async def last_attempt_at(self, tool_id: str) -> datetime | None:
        rows = await (await get_db()).query_raw(f"SELECT last_attempt_at FROM pdm_automation_state WHERE tool_config_id={sql_quote_literal(tool_id)}")
        value = rows[0].get("last_attempt_at") if rows else None
        if value is None:
            return None
        if isinstance(value, datetime):
            return coerce_utc_datetime(value)
        return parse_utc_iso_datetime(value)

    async def reconcile(self) -> None:
        """Release stale/missing links after PDM stale-job cleanup."""
        db = await get_db()
        await db.execute_raw(
            """UPDATE pdm_automation_state s SET active_job_id=NULL,last_attempt_at=COALESCE(last_attempt_at,NOW()),last_error=COALESCE(last_error,'Automatic PDM indexing was interrupted by restart.') WHERE active_job_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM pdm_index_jobs j WHERE j.id::text=s.active_job_id AND j.status IN ('pending','indexing'))"""
        )

    async def reset_index_state(self, tool_id: str) -> None:
        await (await get_db()).execute_raw(
            f"UPDATE pdm_automation_state SET pending_webhook=FALSE,pending_schedule=FALSE,first_pending_at=NULL,claimed_generation=NULL,active_job_id=NULL,last_attempt_at=NULL,last_success_at=NULL,last_error=NULL WHERE tool_config_id={sql_quote_literal(tool_id)}"
        )


pdm_automation_repository = PdmAutomationRepository()
