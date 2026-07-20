from __future__ import annotations

import importlib
import secrets
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any, cast

from ragtime.core.database import get_db
from ragtime.core.datetimes import utc_now
from ragtime.core.encryption import decrypt_secret, encrypt_secret
from ragtime.core.git import parse_git_url
from ragtime.core.sql import sql_quote_literal as _sql_quote_literal
from ragtime.git_webhooks.models import (
    GitPushEvent,
    GitWebhookConfigResponse,
    GitWebhookDelivery,
    GitWebhookDeliveryResponse,
    GitWebhookDeliveryStatus,
    GitWebhookEnableResponse,
    GitWebhookTarget,
    GitWebhookTargetType,
    format_git_webhook_target_key,
)
from ragtime.indexer.models import IndexConfig, OcrMode, OcrProvider


def _unique_violation_error_type() -> type[Exception] | None:
    try:
        error_type = getattr(importlib.import_module("prisma.errors"), "UniqueViolationError", None)
    except Exception:
        return None
    return error_type if isinstance(error_type, type) else None


def _is_unique_violation(exc: Exception) -> bool:
    error_type = _unique_violation_error_type()
    return isinstance(error_type, type) and isinstance(exc, error_type)


def _target_column(target_type: GitWebhookTargetType) -> str:
    return "workspace_id" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "index_metadata_id"


def _target_field(target_type: GitWebhookTargetType) -> str:
    return "workspaceId" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "indexMetadataId"


def _paused_field(target_type: GitWebhookTargetType) -> str:
    return "scmWebhookPaused" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "webhookPaused"


def _row_target_id(row: dict[str, Any]) -> str:
    return str(row.get("workspace_id") or row.get("index_metadata_id") or "")


def _delivery_columns() -> str:
    return (
        "id, target_type, index_metadata_id, workspace_id, provider_delivery_id, "
        "event_name, branch, head_commit, status, index_job_id, message, "
        "received_at, started_at, completed_at"
    )


def _delivery_from_row(row: dict[str, Any]) -> GitWebhookDelivery:
    return GitWebhookDelivery(
        id=str(row.get("id") or ""),
        target_type=GitWebhookTargetType(str(row.get("target_type") or GitWebhookTargetType.GIT_INDEX.value)),
        target_id=_row_target_id(row),
        provider_delivery_id=(str(row.get("provider_delivery_id")) if row.get("provider_delivery_id") is not None else None),
        event_name=str(row.get("event_name") or ""),
        branch=(str(row.get("branch")) if row.get("branch") is not None else None),
        head_commit=(str(row.get("head_commit")) if row.get("head_commit") is not None else None),
        status=GitWebhookDeliveryStatus(str(row.get("status") or GitWebhookDeliveryStatus.PENDING.value)),
        index_job_id=(str(row.get("index_job_id")) if row.get("index_job_id") is not None else None),
        message=(str(row.get("message")) if row.get("message") is not None else None),
        received_at=row.get("received_at") or utc_now(),
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
    )


def _delivery_response_from_row(row: dict[str, Any]) -> GitWebhookDeliveryResponse:
    return GitWebhookDeliveryResponse.model_validate(_delivery_from_row(row).model_dump())


def _webhook_url(base_url: str, webhook_id: str | None) -> str | None:
    cleaned_base_url = (base_url or "").rstrip("/")
    cleaned_id = (webhook_id or "").strip()
    if not cleaned_base_url or not cleaned_id:
        return None
    return f"{cleaned_base_url}/webhooks/git/{cleaned_id}"


def build_stored_git_index_config(*, name: str, description: str | None, config_snapshot: dict[str, Any] | None) -> IndexConfig:
    config_data = config_snapshot if isinstance(config_snapshot, dict) else {}
    ocr_mode_str = str(config_data.get("ocr_mode") or "disabled")
    if ocr_mode_str == "disabled" and config_data.get("enable_ocr", False):
        ocr_mode_str = "tesseract"
    return IndexConfig(
        name=name,
        description=description or "",
        file_patterns=config_data.get("file_patterns", ["**/*"]),
        exclude_patterns=config_data.get("exclude_patterns", ["**/test/**", "**/tests/**", "**/__pycache__/**"]),
        chunk_size=config_data.get("chunk_size", 1000),
        chunk_overlap=config_data.get("chunk_overlap", 200),
        max_file_size_kb=config_data.get("max_file_size_kb", 500),
        ocr_mode=OcrMode(ocr_mode_str),
        ocr_provider=(OcrProvider(config_data["ocr_provider"]) if config_data.get("ocr_provider") else None),
        ocr_vision_model=config_data.get("ocr_vision_model"),
        git_clone_timeout_minutes=config_data.get("git_clone_timeout_minutes", 5),
        git_history_depth=config_data.get("git_history_depth", 1),
        reindex_interval_hours=config_data.get("reindex_interval_hours", 0),
        reindex_start_minute=config_data.get("reindex_start_minute"),
        reindex_timezone=config_data.get("reindex_timezone"),
    )


def _lock_key(target_type: GitWebhookTargetType, target_id: str) -> str:
    return f"git-webhook:{format_git_webhook_target_key(target_type, target_id)}"


def _maybe_find_unique(delegate: Any, where: dict[str, Any]) -> Awaitable[Any | None]:
    finder = cast(Callable[..., Awaitable[Any | None]] | None, getattr(delegate, "find_unique", None))
    if callable(finder):
        return finder(where=where)
    finder = cast(Callable[..., Awaitable[Any | None]] | None, getattr(delegate, "find_first", None))
    if callable(finder):
        return finder(where=where)
    raise AttributeError("Prisma delegate does not support find_unique/find_first")


class GitWebhookRepository:
    async def resolve_target(
        self,
        webhook_id_or_target_type: str | GitWebhookTargetType,
        target_id: str | None = None,
    ) -> GitWebhookTarget | None:
        db = await get_db()
        if target_id is not None:
            target_type = (
                webhook_id_or_target_type
                if isinstance(webhook_id_or_target_type, GitWebhookTargetType)
                else GitWebhookTargetType(str(webhook_id_or_target_type))
            )
            return await self._resolve_dispatch_target(db, target_type=target_type, target_id=target_id)
        webhook_id = str(webhook_id_or_target_type)
        index_row = await _maybe_find_unique(db.indexmetadata, {"webhookId": webhook_id})
        if index_row is not None and getattr(index_row, "webhookSecret", None):
            return self._index_target_from_row(index_row)
        workspace_row = await _maybe_find_unique(db.workspace, {"scmWebhookId": webhook_id})
        if workspace_row is None or not getattr(workspace_row, "scmWebhookSecret", None):
            return None
        return self._workspace_target_from_row(workspace_row)

    async def get_index_config(self, name: str, base_url: str) -> GitWebhookConfigResponse:
        db = await get_db()
        row = await db.indexmetadata.find_unique(where={"name": name})
        return self._config_from_row(row, base_url, target_type=GitWebhookTargetType.GIT_INDEX)

    async def get_index_target_id(self, name: str) -> str | None:
        db = await get_db()
        row = await db.indexmetadata.find_unique(where={"name": name})
        if row is None:
            return None
        return str(row.id)

    async def get_workspace_config(self, workspace_id: str, base_url: str) -> GitWebhookConfigResponse:
        db = await get_db()
        row = await db.workspace.find_unique(where={"id": workspace_id})
        return self._config_from_row(row, base_url, target_type=GitWebhookTargetType.WORKSPACE_SCM)

    async def get_workspace_target_id(self, workspace_id: str) -> str | None:
        db = await get_db()
        row = await db.workspace.find_unique(where={"id": workspace_id})
        if row is None:
            return None
        return str(row.id)

    async def resolve_index_target(self, name: str) -> GitWebhookTarget | None:
        db = await get_db()
        row = await db.indexmetadata.find_unique(where={"name": name})
        if row is None or not getattr(row, "webhookSecret", None):
            return None
        return self._index_target_from_row(row)

    async def resolve_workspace_target(self, workspace_id: str) -> GitWebhookTarget | None:
        db = await get_db()
        row = await db.workspace.find_unique(where={"id": workspace_id})
        if row is None or not getattr(row, "scmWebhookSecret", None):
            return None
        return self._workspace_target_from_row(row)

    async def enable_index(self, name: str, base_url: str) -> GitWebhookEnableResponse:
        db = await get_db()
        row = await db.indexmetadata.find_unique(where={"name": name})
        if row is None:
            return GitWebhookEnableResponse(enabled=False, webhook_url=None)
        return await self._enable_target(db=db, row=row, base_url=base_url, target_type=GitWebhookTargetType.GIT_INDEX, lookup={"name": name})

    async def enable_workspace(self, workspace_id: str, base_url: str) -> GitWebhookEnableResponse:
        db = await get_db()
        row = await db.workspace.find_unique(where={"id": workspace_id})
        if row is None:
            return GitWebhookEnableResponse(enabled=False, webhook_url=None)
        return await self._enable_target(db=db, row=row, base_url=base_url, target_type=GitWebhookTargetType.WORKSPACE_SCM, lookup={"id": workspace_id})

    async def rotate_index_secret(self, name: str, base_url: str) -> GitWebhookEnableResponse:
        db = await get_db()
        row = await db.indexmetadata.find_unique(where={"name": name})
        if row is None:
            return GitWebhookEnableResponse(enabled=False, webhook_url=None)
        return await self._rotate_target_secret(db=db, row=row, base_url=base_url, target_type=GitWebhookTargetType.GIT_INDEX, lookup={"name": name})

    async def rotate_workspace_secret(self, workspace_id: str, base_url: str) -> GitWebhookEnableResponse:
        db = await get_db()
        row = await db.workspace.find_unique(where={"id": workspace_id})
        if row is None:
            return GitWebhookEnableResponse(enabled=False, webhook_url=None)
        return await self._rotate_target_secret(db=db, row=row, base_url=base_url, target_type=GitWebhookTargetType.WORKSPACE_SCM, lookup={"id": workspace_id})

    async def disable_index(self, name: str) -> None:
        db = await get_db()
        row = await db.indexmetadata.find_unique(where={"name": name})
        if row is None:
            return
        await self._disable_target(db=db, row=row, target_type=GitWebhookTargetType.GIT_INDEX, lookup={"name": name})

    async def pause_index(self, name: str, base_url: str) -> GitWebhookConfigResponse:
        db = await get_db()
        row = await db.indexmetadata.find_unique(where={"name": name})
        if row is None:
            return GitWebhookConfigResponse(enabled=False, paused=False, webhook_url=None)
        return await self._pause_target(db=db, row=row, base_url=base_url, target_type=GitWebhookTargetType.GIT_INDEX, lookup={"name": name})

    async def resume_index(self, name: str, base_url: str) -> GitWebhookConfigResponse:
        db = await get_db()
        row = await db.indexmetadata.find_unique(where={"name": name})
        if row is None:
            return GitWebhookConfigResponse(enabled=False, paused=False, webhook_url=None)
        return await self._resume_target(db=db, row=row, base_url=base_url, target_type=GitWebhookTargetType.GIT_INDEX, lookup={"name": name})

    async def disable_workspace(self, workspace_id: str) -> None:
        db = await get_db()
        row = await db.workspace.find_unique(where={"id": workspace_id})
        if row is None:
            return
        await self._disable_target(db=db, row=row, target_type=GitWebhookTargetType.WORKSPACE_SCM, lookup={"id": workspace_id})

    async def pause_workspace(self, workspace_id: str, base_url: str) -> GitWebhookConfigResponse:
        db = await get_db()
        row = await db.workspace.find_unique(where={"id": workspace_id})
        if row is None:
            return GitWebhookConfigResponse(enabled=False, paused=False, webhook_url=None)
        return await self._pause_target(
            db=db,
            row=row,
            base_url=base_url,
            target_type=GitWebhookTargetType.WORKSPACE_SCM,
            lookup={"id": workspace_id},
        )

    async def resume_workspace(self, workspace_id: str, base_url: str) -> GitWebhookConfigResponse:
        db = await get_db()
        row = await db.workspace.find_unique(where={"id": workspace_id})
        if row is None:
            return GitWebhookConfigResponse(enabled=False, paused=False, webhook_url=None)
        return await self._resume_target(
            db=db,
            row=row,
            base_url=base_url,
            target_type=GitWebhookTargetType.WORKSPACE_SCM,
            lookup={"id": workspace_id},
        )

    async def record_ignored(self, target: GitWebhookTarget, event: GitPushEvent, message: str) -> GitWebhookDelivery:
        db = await get_db()
        async with db.tx() as tx:
            await tx.execute_raw(f"SELECT pg_advisory_xact_lock(hashtext({_sql_quote_literal(_lock_key(target.target_type, target.target_id))}))")
            existing = await self._read_existing_delivery(
                tx,
                target_type=target.target_type,
                target_id=target.target_id,
                provider_delivery_id=event.provider_delivery_id,
            )
            if existing is not None:
                return existing
            try:
                rows = await tx.query_raw(
                    self._insert_delivery_sql(
                        delivery_id=secrets.token_urlsafe(24),
                        target_type=target.target_type,
                        target_id=target.target_id,
                        provider_delivery_id=event.provider_delivery_id,
                        event_name=event.event_name,
                        branch=event.branch,
                        head_commit=event.head_commit,
                        status=GitWebhookDeliveryStatus.IGNORED,
                        received_at=utc_now(),
                        message=message,
                    )
                )
            except Exception as exc:
                if not _is_unique_violation(exc):
                    raise
                winner = await self._read_existing_delivery(
                    tx,
                    target_type=target.target_type,
                    target_id=target.target_id,
                    provider_delivery_id=event.provider_delivery_id,
                )
                if winner is not None:
                    return winner
                raise
            await self._prune_terminal_deliveries(tx, target_type=target.target_type, target_id=target.target_id)
            return _delivery_from_row(rows[0])

    async def enqueue_push(
        self,
        *,
        target_type: GitWebhookTargetType,
        target_id: str,
        provider_delivery_id: str | None,
        event_name: str,
        branch: str,
        head_commit: str | None,
    ) -> GitWebhookDelivery:
        db = await get_db()
        async with db.tx() as tx:
            await tx.execute_raw(f"SELECT pg_advisory_xact_lock(hashtext({_sql_quote_literal(_lock_key(target_type, target_id))}))")
            existing = await self._read_existing_delivery(tx, target_type=target_type, target_id=target_id, provider_delivery_id=provider_delivery_id)
            if existing is not None:
                return existing
            await tx.query_raw(self._skip_pending_sql(target_type=target_type, target_id=target_id))
            now = utc_now()
            try:
                rows = await tx.query_raw(
                    self._insert_delivery_sql(
                        delivery_id=secrets.token_urlsafe(24),
                        target_type=target_type,
                        target_id=target_id,
                        provider_delivery_id=provider_delivery_id,
                        event_name=event_name,
                        branch=branch,
                        head_commit=head_commit,
                        status=GitWebhookDeliveryStatus.PENDING,
                        received_at=now,
                    )
                )
            except Exception as exc:
                if not _is_unique_violation(exc):
                    raise
                winner = await self._read_existing_delivery(tx, target_type=target_type, target_id=target_id, provider_delivery_id=provider_delivery_id)
                if winner is not None:
                    return winner
                rows = await tx.query_raw(self._latest_pending_sql(target_type=target_type, target_id=target_id))
            await self._prune_terminal_deliveries(tx, target_type=target_type, target_id=target_id)
            return _delivery_from_row(rows[0])

    async def link_index_job(self, delivery_id: str, index_job_id: str) -> None:
        db = await get_db()
        await db.execute_raw(
            f"""
            UPDATE git_webhook_deliveries
            SET index_job_id = {_sql_quote_literal(index_job_id)}
            WHERE id = {_sql_quote_literal(delivery_id)}
              AND status = {_sql_quote_literal(GitWebhookDeliveryStatus.PROCESSING.value)}
            """
        )

    async def has_pending(self, target_type: GitWebhookTargetType, target_id: str) -> bool:
        db = await get_db()
        rows = await db.query_raw(
            f"""
            SELECT 1 AS exists
            FROM git_webhook_deliveries
            WHERE {_target_column(target_type)} = {_sql_quote_literal(target_id)}
              AND status = {_sql_quote_literal(GitWebhookDeliveryStatus.PENDING.value)}
            LIMIT 1
            """
        )
        return bool(rows)

    async def claim_latest_pending(self, target_type: GitWebhookTargetType, target_id: str) -> GitWebhookDelivery | None:
        db = await get_db()
        async with db.tx() as tx:
            await tx.execute_raw(f"SELECT pg_advisory_xact_lock(hashtext({_sql_quote_literal(_lock_key(target_type, target_id))}))")
            rows = await tx.query_raw(
                f"""
                UPDATE git_webhook_deliveries AS d
                SET status = 'processing', started_at = {_sql_quote_literal(utc_now().isoformat())}::timestamp
                WHERE d.id = (
                    SELECT latest.id
                    FROM git_webhook_deliveries AS latest
                    WHERE latest.{_target_column(target_type)} = {_sql_quote_literal(target_id)}
                      AND latest.status = 'pending'
                      AND NOT EXISTS (
                        SELECT 1 FROM git_webhook_deliveries AS active
                        WHERE active.{_target_column(target_type)} = latest.{_target_column(target_type)}
                          AND active.status = 'processing'
                      )
                    ORDER BY latest.received_at DESC, latest.id DESC
                    LIMIT 1
                )
                RETURNING {_delivery_columns()}
                """
            )
            if not rows:
                return None
            return _delivery_from_row(rows[0])

    async def defer_claim(self, delivery_id: str) -> GitWebhookDeliveryStatus:
        db = await get_db()
        async with db.tx() as tx:
            rows = await tx.query_raw(f"SELECT {_delivery_columns()} FROM git_webhook_deliveries WHERE id = {_sql_quote_literal(delivery_id)} LIMIT 1")
            if not rows:
                return GitWebhookDeliveryStatus.SKIPPED
            delivery = rows[0]
            target_type = GitWebhookTargetType(str(delivery.get("target_type") or GitWebhookTargetType.GIT_INDEX.value))
            target_id = _row_target_id(delivery)
            await tx.execute_raw(f"SELECT pg_advisory_xact_lock(hashtext({_sql_quote_literal(_lock_key(target_type, target_id))}))")
            pending_rows = await tx.query_raw(
                f"""
                SELECT {_delivery_columns()}
                FROM git_webhook_deliveries
                WHERE {_target_column(target_type)} = {_sql_quote_literal(target_id)}
                  AND status = 'pending'
                  AND id <> {_sql_quote_literal(delivery_id)}
                LIMIT 1
                """
            )
            if pending_rows:
                await tx.execute_raw(
                    f"UPDATE git_webhook_deliveries SET status = 'skipped', completed_at = {_sql_quote_literal(utc_now().isoformat())}::timestamp WHERE id = {_sql_quote_literal(delivery_id)}"
                )
                await self._prune_terminal_deliveries(tx, target_type=target_type, target_id=target_id)
                return GitWebhookDeliveryStatus.SKIPPED
            try:
                rows = await tx.query_raw(
                    f"""
                    UPDATE git_webhook_deliveries
                    SET status = 'pending', started_at = NULL, completed_at = NULL
                    WHERE id = {_sql_quote_literal(delivery_id)}
                    RETURNING status
                    """
                )
            except Exception as exc:
                if not _is_unique_violation(exc):
                    raise
                pending_rows = await tx.query_raw(
                    f"""
                    SELECT {_delivery_columns()}
                    FROM git_webhook_deliveries
                    WHERE {_target_column(target_type)} = {_sql_quote_literal(target_id)}
                      AND status = 'pending'
                      AND id <> {_sql_quote_literal(delivery_id)}
                    LIMIT 1
                    """
                )
                if pending_rows:
                    await tx.execute_raw(
                        f"UPDATE git_webhook_deliveries SET status = 'skipped', completed_at = {_sql_quote_literal(utc_now().isoformat())}::timestamp WHERE id = {_sql_quote_literal(delivery_id)}"
                    )
                    await self._prune_terminal_deliveries(tx, target_type=target_type, target_id=target_id)
                    return GitWebhookDeliveryStatus.SKIPPED
                rows = await tx.query_raw(f"SELECT status FROM git_webhook_deliveries WHERE id = {_sql_quote_literal(delivery_id)} LIMIT 1")
                if rows:
                    return GitWebhookDeliveryStatus(str(rows[0].get("status") or GitWebhookDeliveryStatus.SKIPPED.value))
                return GitWebhookDeliveryStatus.SKIPPED
            if rows:
                return GitWebhookDeliveryStatus(str(rows[0].get("status") or GitWebhookDeliveryStatus.PENDING.value))
            return GitWebhookDeliveryStatus.PENDING

    async def complete(
        self,
        delivery_id: str,
        *,
        status: GitWebhookDeliveryStatus,
        message: str,
        index_job_id: str | None = None,
    ) -> None:
        db = await get_db()
        async with db.tx() as tx:
            rows = await tx.query_raw(
                f"""
                SELECT {_delivery_columns()}
                FROM git_webhook_deliveries
                WHERE id = {_sql_quote_literal(delivery_id)}
                LIMIT 1
                """
            )
            if not rows:
                return
            delivery = rows[0]
            target_type = GitWebhookTargetType(str(delivery.get("target_type") or GitWebhookTargetType.GIT_INDEX.value))
            target_id = _row_target_id(delivery)
            await tx.execute_raw(f"SELECT pg_advisory_xact_lock(hashtext({_sql_quote_literal(_lock_key(target_type, target_id))}))")
            await tx.execute_raw(
                f"""
                UPDATE git_webhook_deliveries
                SET status = {_sql_quote_literal(status.value)},
                    message = {_sql_quote_literal(message)},
                    index_job_id = {_sql_quote_literal(index_job_id)},
                    completed_at = {_sql_quote_literal(utc_now().isoformat())}::timestamp
                WHERE id = {_sql_quote_literal(delivery_id)}
                  AND status = {_sql_quote_literal(GitWebhookDeliveryStatus.PROCESSING.value)}
                """
            )
            await self._prune_terminal_deliveries(tx, target_type=target_type, target_id=target_id)

    async def list_deliveries(self, target_type: GitWebhookTargetType, target_id: str, limit: int = 20) -> list[GitWebhookDeliveryResponse]:
        db = await get_db()
        rows = await db.query_raw(
            f"""
            SELECT {_delivery_columns()}
            FROM git_webhook_deliveries
            WHERE {_target_column(target_type)} = {_sql_quote_literal(target_id)}
            ORDER BY received_at DESC
            LIMIT {int(limit)}
            """
        )
        return [_delivery_response_from_row(row) for row in rows]

    async def list_recoverable(self) -> list[GitWebhookDelivery]:
        db = await get_db()
        rows = await db.query_raw(
            f"""
            SELECT {_delivery_columns()}
            FROM git_webhook_deliveries
            WHERE status = {_sql_quote_literal(GitWebhookDeliveryStatus.PROCESSING.value)}
            ORDER BY received_at ASC
            """
        )
        return [_delivery_from_row(row) for row in rows]

    async def list_pending_targets(self) -> list[GitWebhookTarget]:
        db = await get_db()
        rows = await db.query_raw(
            f"""
            SELECT DISTINCT target_type, index_metadata_id, workspace_id
            FROM git_webhook_deliveries
            WHERE status = {_sql_quote_literal(GitWebhookDeliveryStatus.PENDING.value)}
            ORDER BY target_type ASC, index_metadata_id ASC, workspace_id ASC
            """
        )
        targets: list[GitWebhookTarget] = []
        for row in rows:
            target_type = GitWebhookTargetType(str(row.get("target_type") or GitWebhookTargetType.GIT_INDEX.value))
            target_id = _row_target_id(row)
            if not target_id:
                continue
            target = await self.resolve_target(target_type, target_id)
            if target is not None:
                targets.append(target)
        return targets

    async def reset_processing_to_pending(self, delivery_id: str) -> GitWebhookDeliveryStatus:
        db = await get_db()
        async with db.tx() as tx:
            rows = await tx.query_raw(f"SELECT {_delivery_columns()} FROM git_webhook_deliveries WHERE id = {_sql_quote_literal(delivery_id)} LIMIT 1")
            if not rows:
                return GitWebhookDeliveryStatus.SKIPPED
            delivery = rows[0]
            target_type = GitWebhookTargetType(str(delivery.get("target_type") or GitWebhookTargetType.GIT_INDEX.value))
            target_id = _row_target_id(delivery)
            await tx.execute_raw(f"SELECT pg_advisory_xact_lock(hashtext({_sql_quote_literal(_lock_key(target_type, target_id))}))")
            pending_rows = await tx.query_raw(
                f"""
                SELECT {_delivery_columns()}
                FROM git_webhook_deliveries
                WHERE {_target_column(target_type)} = {_sql_quote_literal(target_id)}
                  AND status = 'pending'
                  AND id <> {_sql_quote_literal(delivery_id)}
                LIMIT 1
                """
            )
            if pending_rows:
                await tx.execute_raw(
                    f"UPDATE git_webhook_deliveries SET status = 'skipped', completed_at = {_sql_quote_literal(utc_now().isoformat())}::timestamp WHERE id = {_sql_quote_literal(delivery_id)}"
                )
                await self._prune_terminal_deliveries(tx, target_type=target_type, target_id=target_id)
                return GitWebhookDeliveryStatus.SKIPPED
            try:
                rows = await tx.query_raw(
                    f"""
                    UPDATE git_webhook_deliveries
                    SET status = 'pending',
                        started_at = NULL,
                        completed_at = NULL,
                        index_job_id = NULL,
                        message = NULL
                    WHERE id = {_sql_quote_literal(delivery_id)}
                    RETURNING status
                    """
                )
            except Exception as exc:
                if not _is_unique_violation(exc):
                    raise
                pending_rows = await tx.query_raw(
                    f"""
                    SELECT {_delivery_columns()}
                    FROM git_webhook_deliveries
                    WHERE {_target_column(target_type)} = {_sql_quote_literal(target_id)}
                      AND status = 'pending'
                      AND id <> {_sql_quote_literal(delivery_id)}
                    LIMIT 1
                    """
                )
                if pending_rows:
                    await tx.execute_raw(
                        f"UPDATE git_webhook_deliveries SET status = 'skipped', completed_at = {_sql_quote_literal(utc_now().isoformat())}::timestamp WHERE id = {_sql_quote_literal(delivery_id)}"
                    )
                    await self._prune_terminal_deliveries(tx, target_type=target_type, target_id=target_id)
                    return GitWebhookDeliveryStatus.SKIPPED
                rows = await tx.query_raw(f"SELECT status FROM git_webhook_deliveries WHERE id = {_sql_quote_literal(delivery_id)} LIMIT 1")
                if rows:
                    return GitWebhookDeliveryStatus(str(rows[0].get("status") or GitWebhookDeliveryStatus.SKIPPED.value))
                return GitWebhookDeliveryStatus.SKIPPED
            if rows:
                return GitWebhookDeliveryStatus(str(rows[0].get("status") or GitWebhookDeliveryStatus.PENDING.value))
            return GitWebhookDeliveryStatus.PENDING

    def _config_from_row(self, row: Any | None, base_url: str, *, target_type: GitWebhookTargetType) -> GitWebhookConfigResponse:
        webhook_id_field = "scmWebhookId" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "webhookId"
        secret_field = "scmWebhookSecret" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "webhookSecret"
        paused_field = _paused_field(target_type)
        created_at_field = "scmWebhookCreatedAt" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "webhookCreatedAt"
        webhook_id = str(getattr(row, webhook_id_field, "") or "") or None
        enabled = bool(webhook_id and getattr(row, secret_field, None))
        paused = bool(enabled and row is not None and getattr(row, paused_field, False))
        provider: str | None = None
        branch: str | None = None
        if row is not None:
            if target_type is GitWebhookTargetType.WORKSPACE_SCM:
                provider = str(getattr(row, "scmProvider", "") or "") or None
                branch = str(getattr(row, "scmGitBranch", "") or "") or None
            else:
                source = str(getattr(row, "source", "") or "")
                encrypted_token = getattr(row, "gitToken", None)
                parsed = parse_git_url(source, token=(decrypt_secret(encrypted_token) if encrypted_token else None)) if source else None
                provider = parsed.provider.value if parsed is not None else None
                branch = str(getattr(row, "gitBranch", "") or "") or None
        return GitWebhookConfigResponse(
            enabled=enabled,
            paused=paused,
            webhook_id=webhook_id,
            webhook_url=_webhook_url(base_url, webhook_id),
            provider=provider,
            branch=branch,
            created_at=getattr(row, created_at_field, None) if row is not None else None,
        )

    async def _resolve_dispatch_target(self, db: Any, *, target_type: GitWebhookTargetType, target_id: str) -> GitWebhookTarget | None:
        if target_type is GitWebhookTargetType.WORKSPACE_SCM:
            row = await db.workspace.find_unique(where={"id": target_id})
            if row is None or not getattr(row, "scmWebhookSecret", None):
                return None
            return self._workspace_target_from_row(row)
        row = await db.indexmetadata.find_unique(where={"id": target_id})
        if row is None or not getattr(row, "webhookSecret", None):
            return None
        return self._index_target_from_row(row)

    def _index_target_from_row(self, row: Any) -> GitWebhookTarget:
        encrypted_token = getattr(row, "gitToken", None)
        source = str(getattr(row, "source", "") or "")
        parsed = parse_git_url(source, token=(decrypt_secret(encrypted_token) if encrypted_token else None))
        return GitWebhookTarget(
            target_type=GitWebhookTargetType.GIT_INDEX,
            target_id=str(row.id),
            key=format_git_webhook_target_key(GitWebhookTargetType.GIT_INDEX, str(row.id)),
            webhook_id=str(row.webhookId),
            secret=decrypt_secret(str(row.webhookSecret or "")),
            provider=parsed.provider.value if parsed is not None else "generic",
            branch=str(getattr(row, "gitBranch", "") or "") or "main",
            paused=bool(getattr(row, "webhookPaused", False)),
            created_at=getattr(row, "webhookCreatedAt", None),
            name=str(getattr(row, "name", "") or "") or None,
            description=str(getattr(row, "description", "") or "") or None,
            source=source or None,
            git_token=(decrypt_secret(encrypted_token) if encrypted_token else None),
            config_snapshot=(getattr(row, "configSnapshot", None) if isinstance(getattr(row, "configSnapshot", None), dict) else None),
        )

    def _workspace_target_from_row(self, row: Any) -> GitWebhookTarget:
        return GitWebhookTarget(
            target_type=GitWebhookTargetType.WORKSPACE_SCM,
            target_id=str(row.id),
            key=format_git_webhook_target_key(GitWebhookTargetType.WORKSPACE_SCM, str(row.id)),
            webhook_id=str(row.scmWebhookId),
            secret=decrypt_secret(str(row.scmWebhookSecret or "")),
            provider=str(getattr(row, "scmProvider", "") or "") or "generic",
            branch=str(getattr(row, "scmGitBranch", "") or "") or "main",
            paused=bool(getattr(row, "scmWebhookPaused", False)),
            created_at=getattr(row, "scmWebhookCreatedAt", None),
            source=str(getattr(row, "scmGitUrl", "") or "") or None,
        )

    async def _enable_target(
        self,
        *,
        db: Any,
        row: Any,
        base_url: str,
        target_type: GitWebhookTargetType,
        lookup: dict[str, Any],
    ) -> GitWebhookEnableResponse:
        async with db.tx() as tx:
            await tx.execute_raw(f"SELECT pg_advisory_xact_lock(hashtext({_sql_quote_literal(_lock_key(target_type, str(row.id)))}))")
            current = await (
                db.workspace.find_unique(where=lookup) if target_type is GitWebhookTargetType.WORKSPACE_SCM else db.indexmetadata.find_unique(where=lookup)
            )
            current_id_field = "scmWebhookId" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "webhookId"
            current_secret_field = "scmWebhookSecret" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "webhookSecret"
            current_paused_field = _paused_field(target_type)
            current_created_at_field = "scmWebhookCreatedAt" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "webhookCreatedAt"
            webhook_id = str(getattr(current, current_id_field, "") or "") or None
            encrypted_secret = getattr(current, current_secret_field, None)
            if webhook_id and encrypted_secret:
                if getattr(current, current_paused_field, False):
                    current = await (
                        db.workspace.update(where=lookup, data={current_paused_field: False})
                        if target_type is GitWebhookTargetType.WORKSPACE_SCM
                        else db.indexmetadata.update(where=lookup, data={current_paused_field: False})
                    )
                return GitWebhookEnableResponse(
                    **self._config_from_row(current, base_url, target_type=target_type).model_dump(),
                    secret=None,
                )
            secret = secrets.token_urlsafe(32)
            update_data = {
                current_id_field: webhook_id or secrets.token_urlsafe(24),
                current_secret_field: encrypt_secret(secret),
                current_paused_field: False,
                current_created_at_field: utc_now(),
            }
            updated = await (
                db.workspace.update(where=lookup, data=update_data)
                if target_type is GitWebhookTargetType.WORKSPACE_SCM
                else db.indexmetadata.update(where=lookup, data=update_data)
            )
            enabled = self._config_from_row(updated, base_url, target_type=target_type)
            return GitWebhookEnableResponse(**enabled.model_dump(), secret=secret)

    async def _rotate_target_secret(
        self,
        *,
        db: Any,
        row: Any,
        base_url: str,
        target_type: GitWebhookTargetType,
        lookup: dict[str, Any],
    ) -> GitWebhookEnableResponse:
        async with db.tx() as tx:
            await tx.execute_raw(f"SELECT pg_advisory_xact_lock(hashtext({_sql_quote_literal(_lock_key(target_type, str(row.id)))}))")
            current_id_field = "scmWebhookId" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "webhookId"
            current_secret_field = "scmWebhookSecret" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "webhookSecret"
            current_created_at_field = "scmWebhookCreatedAt" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "webhookCreatedAt"
            secret = secrets.token_urlsafe(32)
            update_data = {
                current_id_field: str(getattr(row, current_id_field, "") or "") or secrets.token_urlsafe(24),
                current_secret_field: encrypt_secret(secret),
                _paused_field(target_type): bool(getattr(row, _paused_field(target_type), False)),
                current_created_at_field: utc_now(),
            }
            updated = await (
                db.workspace.update(where=lookup, data=update_data)
                if target_type is GitWebhookTargetType.WORKSPACE_SCM
                else db.indexmetadata.update(where=lookup, data=update_data)
            )
            enabled = self._config_from_row(updated, base_url, target_type=target_type)
            return GitWebhookEnableResponse(**enabled.model_dump(), secret=secret)

    async def _disable_target(self, *, db: Any, row: Any, target_type: GitWebhookTargetType, lookup: dict[str, Any]) -> None:
        async with db.tx() as tx:
            await tx.execute_raw(f"SELECT pg_advisory_xact_lock(hashtext({_sql_quote_literal(_lock_key(target_type, str(row.id)))}))")
            await tx.execute_raw(
                f"UPDATE git_webhook_deliveries SET status = 'skipped', completed_at = {_sql_quote_literal(utc_now().isoformat())}::timestamp WHERE {_target_column(target_type)} = {_sql_quote_literal(str(row.id))} AND status = 'pending'"
            )
            await self._prune_terminal_deliveries(tx, target_type=target_type, target_id=str(row.id))
            clear_data = {
                ("scmWebhookId" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "webhookId"): None,
                ("scmWebhookSecret" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "webhookSecret"): None,
                _paused_field(target_type): False,
                ("scmWebhookCreatedAt" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "webhookCreatedAt"): None,
            }
            if target_type is GitWebhookTargetType.WORKSPACE_SCM:
                await db.workspace.update(where=lookup, data=clear_data)
            else:
                await db.indexmetadata.update(where=lookup, data=clear_data)

    async def _pause_target(
        self,
        *,
        db: Any,
        row: Any,
        base_url: str,
        target_type: GitWebhookTargetType,
        lookup: dict[str, Any],
    ) -> GitWebhookConfigResponse:
        id_field = "scmWebhookId" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "webhookId"
        secret_field = "scmWebhookSecret" if target_type is GitWebhookTargetType.WORKSPACE_SCM else "webhookSecret"
        paused_field = _paused_field(target_type)
        async with db.tx() as tx:
            await tx.execute_raw(f"SELECT pg_advisory_xact_lock(hashtext({_sql_quote_literal(_lock_key(target_type, str(row.id)))}))")
            current = await (
                db.workspace.find_unique(where=lookup) if target_type is GitWebhookTargetType.WORKSPACE_SCM else db.indexmetadata.find_unique(where=lookup)
            )
            if current is None or not getattr(current, id_field, None) or not getattr(current, secret_field, None):
                return self._config_from_row(current, base_url, target_type=target_type)
            await tx.execute_raw(
                f"UPDATE git_webhook_deliveries SET status = 'skipped', message = 'Webhook paused before processing.', completed_at = {_sql_quote_literal(utc_now().isoformat())}::timestamp WHERE {_target_column(target_type)} = {_sql_quote_literal(str(row.id))} AND status = 'pending'"
            )
            await self._prune_terminal_deliveries(tx, target_type=target_type, target_id=str(row.id))
            updated = await (
                db.workspace.update(where=lookup, data={paused_field: True})
                if target_type is GitWebhookTargetType.WORKSPACE_SCM
                else db.indexmetadata.update(where=lookup, data={paused_field: True})
            )
        return self._config_from_row(updated, base_url, target_type=target_type)

    async def _resume_target(
        self,
        *,
        db: Any,
        row: Any,
        base_url: str,
        target_type: GitWebhookTargetType,
        lookup: dict[str, Any],
    ) -> GitWebhookConfigResponse:
        paused_field = _paused_field(target_type)
        async with db.tx() as tx:
            await tx.execute_raw(f"SELECT pg_advisory_xact_lock(hashtext({_sql_quote_literal(_lock_key(target_type, str(row.id)))}))")
            updated = await (
                db.workspace.update(where=lookup, data={paused_field: False})
                if target_type is GitWebhookTargetType.WORKSPACE_SCM
                else db.indexmetadata.update(where=lookup, data={paused_field: False})
            )
        return self._config_from_row(updated, base_url, target_type=target_type)

    async def _read_existing_delivery(
        self,
        tx: Any,
        *,
        target_type: GitWebhookTargetType,
        target_id: str,
        provider_delivery_id: str | None,
    ) -> GitWebhookDelivery | None:
        if provider_delivery_id:
            rows = await tx.query_raw(
                f"""
                SELECT {_delivery_columns()}
                FROM git_webhook_deliveries
                WHERE {_target_column(target_type)} = {_sql_quote_literal(target_id)}
                  AND provider_delivery_id = {_sql_quote_literal(provider_delivery_id)}
                LIMIT 1
                """
            )
            if rows:
                return _delivery_from_row(rows[0])
        return None

    def _skip_pending_sql(self, *, target_type: GitWebhookTargetType, target_id: str) -> str:
        return f"""
            UPDATE git_webhook_deliveries
            SET status = 'skipped', completed_at = {_sql_quote_literal(utc_now().isoformat())}::timestamp
            WHERE id IN (
                SELECT id
                FROM git_webhook_deliveries
                WHERE {_target_column(target_type)} = {_sql_quote_literal(target_id)}
                  AND status = 'pending'
                ORDER BY received_at DESC, id DESC
                LIMIT 1
            )
            RETURNING id
        """

    def _latest_pending_sql(self, *, target_type: GitWebhookTargetType, target_id: str) -> str:
        return f"""
            SELECT {_delivery_columns()}
            FROM git_webhook_deliveries
            WHERE {_target_column(target_type)} = {_sql_quote_literal(target_id)}
              AND status = 'pending'
            ORDER BY received_at DESC, id DESC
            LIMIT 1
        """

    def _insert_delivery_sql(
        self,
        *,
        delivery_id: str,
        target_type: GitWebhookTargetType,
        target_id: str,
        provider_delivery_id: str | None,
        event_name: str,
        branch: str | None,
        head_commit: str | None,
        status: GitWebhookDeliveryStatus,
        received_at: datetime,
        message: str | None = None,
    ) -> str:
        target_column = _target_column(target_type)
        return f"""
            INSERT INTO git_webhook_deliveries (
                id, target_type, {target_column}, provider_delivery_id, event_name, branch, head_commit, status, message, received_at
            ) VALUES (
                {_sql_quote_literal(delivery_id)},
                {_sql_quote_literal(target_type.value)},
                {_sql_quote_literal(target_id)},
                {_sql_quote_literal(provider_delivery_id)},
                {_sql_quote_literal(event_name)},
                {_sql_quote_literal(branch)},
                {_sql_quote_literal(head_commit)},
                {_sql_quote_literal(status.value)},
                {_sql_quote_literal(message)},
                {_sql_quote_literal(received_at.isoformat())}::timestamp
            )
            RETURNING {_delivery_columns()}
        """

    async def _prune_terminal_deliveries(self, tx: Any, *, target_type: GitWebhookTargetType, target_id: str) -> None:
        await tx.execute_raw(
            f"""
            DELETE FROM git_webhook_deliveries
            WHERE id IN (
                SELECT stale.id
                FROM git_webhook_deliveries AS stale
                WHERE stale.{_target_column(target_type)} = {_sql_quote_literal(target_id)}
                  AND stale.status NOT IN ('pending', 'processing')
                  AND stale.id IN (
                      SELECT older.id
                      FROM git_webhook_deliveries AS older
                      WHERE older.{_target_column(target_type)} = {_sql_quote_literal(target_id)}
                      ORDER BY older.received_at DESC, older.id DESC
                      OFFSET 50
                  )
            )
            """
        )


git_webhook_repository = GitWebhookRepository()
