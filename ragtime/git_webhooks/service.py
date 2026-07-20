from __future__ import annotations

import asyncio
from typing import Any

from ragtime.core.logging import get_logger
from ragtime.git_webhooks.models import (
    GitPushEvent,
    GitWebhookDelivery,
    GitWebhookDeliveryStatus,
    GitWebhookTarget,
    GitWebhookTargetType,
    format_git_webhook_target_key,
)
from ragtime.git_webhooks.repository import build_stored_git_index_config, git_webhook_repository
from ragtime.indexer.models import IndexStatus
from ragtime.indexer.repository import repository as index_job_repository
from ragtime.indexer.service import indexer

logger = get_logger(__name__)


class _LazyUserSpaceServiceProxy:
    def __getattr__(self, name: str) -> Any:
        from ragtime.userspace.service import userspace_service

        return getattr(userspace_service, name)


class GitWebhookService:
    def __init__(
        self,
        *,
        repo: Any = git_webhook_repository,
        indexer: Any = indexer,
        userspace: Any | None = None,
        index_jobs: Any = index_job_repository,
        poll_seconds: float = 1.0,
    ) -> None:
        if userspace is None:
            userspace = _LazyUserSpaceServiceProxy()
        self._repo = repo
        self._indexer = indexer
        self._userspace = userspace
        self._index_jobs = index_jobs
        self._poll_seconds = poll_seconds
        self._target_tasks: dict[str, asyncio.Task[None]] = {}
        self._target_events: dict[str, asyncio.Event] = {}
        self._stopping = False

    async def start(self) -> None:
        self._stopping = False
        await self.reconcile_processing()
        for target in await self._repo.list_pending_targets():
            self.schedule_target(target)

    async def stop(self) -> None:
        self._stopping = True
        for event in self._target_events.values():
            event.set()
        tasks = list(self._target_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._target_tasks.clear()
        self._target_events.clear()

    async def accept_push(self, target: GitWebhookTarget, event: GitPushEvent) -> GitWebhookDelivery:
        delivery = await self._repo.enqueue_push(
            target_type=target.target_type,
            target_id=target.target_id,
            provider_delivery_id=event.provider_delivery_id,
            event_name=event.event_name,
            branch=event.branch or target.branch or "main",
            head_commit=event.head_commit,
        )
        self.schedule_target(target)
        return delivery

    def schedule_target(self, target: GitWebhookTarget) -> None:
        if target.paused:
            return
        self._ensure_task(target.target_type, target.target_id, target.key or format_git_webhook_target_key(target.target_type, target.target_id))

    def disable_target(self, target: GitWebhookTarget) -> None:
        self.wake_target(target.key or format_git_webhook_target_key(target.target_type, target.target_id))

    def wake_target(self, key: str) -> None:
        event = self._target_events.get(key)
        if event is not None:
            event.set()

    async def reconcile_processing(self) -> None:
        for delivery in await self._repo.list_recoverable():
            if delivery.target_type is GitWebhookTargetType.WORKSPACE_SCM:
                await self._repo.reset_processing_to_pending(delivery.id)
                continue
            if not delivery.index_job_id:
                await self._repo.reset_processing_to_pending(delivery.id)
                continue
            job = await self._index_jobs.get_job(delivery.index_job_id)
            if job is None:
                await self._repo.reset_processing_to_pending(delivery.id)
                continue
            if self._job_terminal_status(job) is not None:
                await self._complete_git_delivery(delivery.id, job=job, index_job_id=delivery.index_job_id)
                continue
            self._ensure_task(
                delivery.target_type,
                delivery.target_id,
                format_git_webhook_target_key(delivery.target_type, delivery.target_id),
                recovery_delivery_id=delivery.id,
                recovery_job_id=delivery.index_job_id,
            )

    def _ensure_task(
        self,
        target_type: GitWebhookTargetType,
        target_id: str,
        key: str,
        *,
        recovery_delivery_id: str | None = None,
        recovery_job_id: str | None = None,
    ) -> None:
        event = self._target_events.setdefault(key, asyncio.Event())
        event.set()
        task = self._target_tasks.get(key)
        if task is not None and not task.done():
            return
        self._target_tasks[key] = asyncio.create_task(
            self._drain_target(
                target_type,
                target_id,
                key,
                recovery_delivery_id=recovery_delivery_id,
                recovery_job_id=recovery_job_id,
            ),
            name=f"git-webhook:{key}",
        )

    async def _drain_target(
        self,
        target_type: GitWebhookTargetType,
        target_id: str,
        key: str,
        *,
        recovery_delivery_id: str | None = None,
        recovery_job_id: str | None = None,
    ) -> None:
        event = self._target_events.setdefault(key, asyncio.Event())
        try:
            if recovery_delivery_id and recovery_job_id:
                await self._monitor_git_job(recovery_delivery_id, recovery_job_id)
            while not self._stopping:
                try:
                    progressed = await (
                        self._drain_git_target(target_id) if target_type is GitWebhookTargetType.GIT_INDEX else self._drain_workspace_target(target_id)
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Git webhook drain iteration failed for %s:%s", target_type.value, target_id)
                    await asyncio.sleep(self._poll_seconds)
                    continue
                if progressed:
                    continue
                try:
                    event.clear()
                    pending_exists = await self._repo.has_pending(target_type, target_id)
                    if event.is_set():
                        continue
                    if pending_exists:
                        try:
                            await asyncio.wait_for(event.wait(), timeout=self._poll_seconds)
                        except asyncio.TimeoutError:
                            pass
                        continue
                    await event.wait()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Git webhook wait failed for %s:%s", target_type.value, target_id)
                    await asyncio.sleep(self._poll_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Git webhook drain failed for %s:%s", target_type.value, target_id)
        finally:
            self._target_tasks.pop(key, None)
            self._target_events.pop(key, None)

    async def _drain_git_target(self, target_id: str) -> bool:
        target = await self._repo.resolve_target(GitWebhookTargetType.GIT_INDEX, target_id)
        if target is None or not target.name or not target.source:
            delivery = await self._repo.claim_latest_pending(GitWebhookTargetType.GIT_INDEX, target_id)
            if delivery is None:
                return False
            await self._retry_complete_delivery(
                delivery.id,
                status=GitWebhookDeliveryStatus.SKIPPED,
                message="Webhook target no longer exists.",
            )
            return True
        if target.paused:
            return False
        if await self._index_jobs.get_active_job_for_index(target.name) is not None:
            return False
        delivery = await self._repo.claim_latest_pending(GitWebhookTargetType.GIT_INDEX, target_id)
        if delivery is None:
            return False
        config = build_stored_git_index_config(
            name=target.name,
            description=target.description,
            config_snapshot=target.config_snapshot,
        )
        try:
            job = await self._indexer.try_create_index_from_git(
                target.source,
                target.branch or "main",
                config,
                target.git_token,
            )
        except Exception as exc:
            await self._retry_complete_delivery(
                delivery.id,
                status=GitWebhookDeliveryStatus.FAILED,
                message=str(exc),
            )
            return True
        if job is None:
            await self._repo.defer_claim(delivery.id)
            return True
        await self._retry_link_index_job(delivery.id, str(job.id))
        await self._monitor_git_job(delivery.id, str(job.id))
        return True

    async def _drain_workspace_target(self, workspace_id: str) -> bool:
        if not await self._repo.has_pending(GitWebhookTargetType.WORKSPACE_SCM, workspace_id):
            return False
        async with self._userspace.workspace_scm_operation(workspace_id):
            target = await self._repo.resolve_target(GitWebhookTargetType.WORKSPACE_SCM, workspace_id)
            if target is None:
                delivery = await self._repo.claim_latest_pending(GitWebhookTargetType.WORKSPACE_SCM, workspace_id)
                if delivery is None:
                    return False
                await self._retry_complete_delivery(
                    delivery.id,
                    status=GitWebhookDeliveryStatus.SKIPPED,
                    message="Webhook target no longer exists.",
                )
                return True
            if target.paused:
                return False
            delivery = await self._repo.claim_latest_pending(GitWebhookTargetType.WORKSPACE_SCM, workspace_id)
            if delivery is None:
                return False
            try:
                outcome = await self._userspace.run_workspace_scm_webhook_pull_locked(workspace_id)
            except Exception as exc:
                await self._retry_complete_delivery(
                    delivery.id,
                    status=GitWebhookDeliveryStatus.FAILED,
                    message=str(exc),
                )
                return True
            status, message = self._workspace_outcome_status(outcome)
            await self._retry_complete_delivery(delivery.id, status=status, message=message)
            return True

    async def _monitor_git_job(self, delivery_id: str, job_id: str) -> None:
        while not self._stopping:
            try:
                job = await self._index_jobs.get_job(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Git webhook job lookup failed for delivery %s job %s", delivery_id, job_id)
                await asyncio.sleep(self._poll_seconds)
                continue
            if job is None:
                await self._retry_complete_delivery(
                    delivery_id,
                    status=GitWebhookDeliveryStatus.FAILED,
                    message="Webhook-triggered git indexing job disappeared before completion.",
                    index_job_id=job_id,
                )
                return
            terminal_status = self._job_terminal_status(job)
            if terminal_status is not None:
                await self._complete_git_delivery(delivery_id, job=job, index_job_id=job_id)
                return
            await asyncio.sleep(self._poll_seconds)

    async def _complete_git_delivery(self, delivery_id: str, *, job: Any, index_job_id: str) -> None:
        status = self._job_terminal_status(job)
        if status is GitWebhookDeliveryStatus.COMPLETED:
            message = "Git webhook push indexed successfully."
        else:
            error_message = str(getattr(job, "error_message", "") or "").strip()
            message = error_message or "Git webhook push indexing failed."
        await self._retry_complete_delivery(
            delivery_id,
            status=status or GitWebhookDeliveryStatus.FAILED,
            message=message,
            index_job_id=index_job_id,
        )

    async def _retry_link_index_job(self, delivery_id: str, index_job_id: str) -> None:
        await self._retry_repo_write(
            operation=f"git webhook link delivery {delivery_id} to job {index_job_id}",
            action=lambda: self._repo.link_index_job(delivery_id, index_job_id),
        )

    async def _retry_complete_delivery(
        self,
        delivery_id: str,
        *,
        status: GitWebhookDeliveryStatus,
        message: str,
        index_job_id: str | None = None,
    ) -> None:
        await self._retry_repo_write(
            operation=f"git webhook complete delivery {delivery_id} as {status.value}",
            action=lambda: self._repo.complete(
                delivery_id,
                status=status,
                message=message,
                index_job_id=index_job_id,
            ),
        )

    async def _retry_repo_write(self, *, operation: str, action: Any) -> None:
        while not self._stopping:
            try:
                await action()
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Retrying after transient repository failure during %s", operation)
                await asyncio.sleep(self._poll_seconds)

    def _job_terminal_status(self, job: Any) -> GitWebhookDeliveryStatus | None:
        raw_status = getattr(job, "status", None)
        status_value = str(getattr(raw_status, "value", raw_status) or "")
        if status_value == IndexStatus.COMPLETED.value:
            return GitWebhookDeliveryStatus.COMPLETED
        if status_value == IndexStatus.FAILED.value:
            return GitWebhookDeliveryStatus.FAILED
        return None

    def _workspace_outcome_status(self, outcome: Any) -> tuple[GitWebhookDeliveryStatus, str]:
        state = str(getattr(outcome, "state", "") or "")
        summary = str(getattr(outcome, "summary", getattr(outcome, "message", "")) or "")
        if state == "imported":
            return GitWebhookDeliveryStatus.COMPLETED, summary or "Workspace imported from webhook push."
        if state == "up_to_date":
            return GitWebhookDeliveryStatus.COMPLETED, summary or "Workspace was already up to date."
        return GitWebhookDeliveryStatus.FAILED, summary or f"Workspace webhook pull finished with state '{state}'."


git_webhook_service = GitWebhookService()
