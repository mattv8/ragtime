import asyncio
import unittest
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest import mock

from ragtime.git_webhooks.models import (
    GitPushEvent,
    GitWebhookDelivery,
    GitWebhookDeliveryStatus,
    GitWebhookTarget,
    GitWebhookTargetType,
)
from ragtime.git_webhooks.repository import format_git_webhook_target_key
from ragtime.git_webhooks.service import GitWebhookService
from ragtime.indexer.models import IndexConfig


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _git_target(
    *,
    target_id: str = "index-1",
    name: str = "git-index",
    source: str = "https://git.example/owner/repo.git",
    git_token: str | None = "token-1",
    branch: str = "main",
    config_snapshot: dict[str, Any] | None = None,
) -> GitWebhookTarget:
    return GitWebhookTarget(
        target_type=GitWebhookTargetType.GIT_INDEX,
        target_id=target_id,
        webhook_id=f"wh-{target_id}",
        secret="secret",
        provider="github",
        branch=branch,
        created_at=_utcnow(),
        name=name,
        key=format_git_webhook_target_key(GitWebhookTargetType.GIT_INDEX, target_id),
        source=source,
        git_token=git_token,
        paused=False,
        config_snapshot=config_snapshot or {"name": name, "git_history_depth": 1},
    )


def _workspace_target(*, workspace_id: str = "workspace-1", branch: str = "main") -> GitWebhookTarget:
    return GitWebhookTarget(
        target_type=GitWebhookTargetType.WORKSPACE_SCM,
        target_id=workspace_id,
        webhook_id=f"wh-{workspace_id}",
        secret="secret",
        provider="github",
        branch=branch,
        created_at=_utcnow(),
        key=format_git_webhook_target_key(GitWebhookTargetType.WORKSPACE_SCM, workspace_id),
        paused=False,
    )


def _push(head_commit: str, *, branch: str = "main", provider_delivery_id: str | None = None) -> GitPushEvent:
    return GitPushEvent(
        provider_delivery_id=provider_delivery_id or f"delivery-{head_commit}",
        event_name="push",
        branch=branch,
        head_commit=head_commit,
    )


@dataclass
class FakeIndexJob:
    id: str
    name: str
    git_url: str
    git_branch: str
    git_token: str | None
    config: dict[str, Any]
    status: str = "processing"
    error_message: str | None = None


def _config_name(config: IndexConfig) -> str:
    return str(config.name or "git-index")


class FakeIndexJobRepository:
    def __init__(self, *, active_git_job: bool = False) -> None:
        self.active_git_job = active_git_job
        self.jobs: dict[str, FakeIndexJob] = {}
        self.active_job_by_name: dict[str, FakeIndexJob] = {}
        self.active_lookup_calls: list[str] = []
        self.get_job_calls: list[str] = []
        self._job_events: dict[str, asyncio.Event] = {}

    async def get_active_job_for_index(self, name: str) -> FakeIndexJob | None:
        self.active_lookup_calls.append(name)
        if self.active_git_job:
            return self.active_job_by_name.get(name) or FakeIndexJob(
                id=f"manual-{name}",
                name=name,
                git_url="manual",
                git_branch="main",
                git_token=None,
                config={"name": name},
            )
        return self.active_job_by_name.get(name)

    async def get_job(self, job_id: str) -> FakeIndexJob | None:
        self.get_job_calls.append(job_id)
        return self.jobs.get(job_id)

    def create_job(self, job: FakeIndexJob) -> FakeIndexJob:
        self.jobs[job.id] = job
        self.active_job_by_name[job.name] = job
        self._job_events.setdefault(job.id, asyncio.Event()).set()
        return job

    def complete_job(self, job_id: str, *, failed: bool = False, error_message: str | None = None) -> None:
        job = self.jobs[job_id]
        job.status = "failed" if failed else "completed"
        job.error_message = error_message
        self.active_job_by_name.pop(job.name, None)


class FakeIndexer:
    def __init__(self, repo: "FakeWebhookRepository", jobs: FakeIndexJobRepository) -> None:
        self.repo = repo
        self.jobs = jobs
        self.started_commits: list[str] = []
        self.created_jobs: list[FakeIndexJob] = []

    async def try_create_index_from_git(
        self,
        git_url: str,
        branch: str,
        config: IndexConfig,
        git_token: str | None = None,
    ) -> FakeIndexJob | None:
        name = _config_name(config)
        if self.jobs.active_git_job:
            return None
        commit = str(self.repo.processing_commit or "unknown")
        self.started_commits.append(commit)
        job = FakeIndexJob(
            id=f"job-{len(self.created_jobs) + 1}",
            name=name,
            git_url=git_url,
            git_branch=branch,
            git_token=git_token,
            config=config.model_dump(),
        )
        self.created_jobs.append(job)
        self.jobs.create_job(job)
        self.jobs.complete_job(job.id)
        return job


class BlockingIndexer(FakeIndexer):
    def __init__(self, repo: "FakeWebhookRepository", jobs: FakeIndexJobRepository) -> None:
        super().__init__(repo, jobs)
        self._releases: dict[str, asyncio.Event] = {}
        self._blocked_once = False

    async def try_create_index_from_git(self, git_url: str, branch: str, config: IndexConfig, git_token: str | None = None) -> FakeIndexJob | None:
        job = await super().try_create_index_from_git(git_url, branch, config, git_token)
        if job is None:
            return None
        if self._blocked_once:
            return job
        self._blocked_once = True
        self.jobs.jobs[job.id].status = "processing"
        release = self._releases.setdefault(self.repo.processing_commit or "unknown", asyncio.Event())
        await release.wait()
        self.jobs.complete_job(job.id)
        return job

    def release(self, commit: str, *, failed: bool = False, error_message: str | None = None) -> None:
        release = self._releases.setdefault(commit, asyncio.Event())
        for job in self.created_jobs:
            if job.id in self.jobs.jobs and self.jobs.jobs[job.id].status == "processing":
                self.jobs.complete_job(job.id, failed=failed, error_message=error_message)
        release.set()


class FailingOnceIndexer(FakeIndexer):
    def __init__(self, repo: "FakeWebhookRepository", jobs: FakeIndexJobRepository) -> None:
        super().__init__(repo, jobs)
        self.failed = False

    async def try_create_index_from_git(self, git_url: str, branch: str, config: IndexConfig, git_token: str | None = None) -> FakeIndexJob | None:
        job = await super().try_create_index_from_git(git_url, branch, config, git_token)
        if job is None:
            return None
        if not self.failed:
            self.failed = True
            self.jobs.jobs[job.id].status = "failed"
            self.jobs.jobs[job.id].error_message = "boom"
            self.jobs.active_job_by_name.pop(job.name, None)
        return job


class DeferredRaceIndexer(FakeIndexer):
    def __init__(self, repo: "FakeWebhookRepository", jobs: FakeIndexJobRepository) -> None:
        super().__init__(repo, jobs)
        self._returned_none = False
        self._release = asyncio.Event()

    async def try_create_index_from_git(self, git_url: str, branch: str, config: IndexConfig, git_token: str | None = None) -> FakeIndexJob | None:
        if not self._returned_none:
            self._returned_none = True
            await self._release.wait()
            return None
        return await super().try_create_index_from_git(git_url, branch, config, git_token)

    def release_race(self) -> None:
        self._release.set()


class ImmediatePendingIndexer(FakeIndexer):
    async def try_create_index_from_git(
        self,
        git_url: str,
        branch: str,
        config: IndexConfig,
        git_token: str | None = None,
    ) -> FakeIndexJob | None:
        if self.jobs.active_git_job:
            return None
        commit = str(self.repo.processing_commit or "unknown")
        self.started_commits.append(commit)
        job = FakeIndexJob(
            id=f"job-{len(self.created_jobs) + 1}",
            name=_config_name(config),
            git_url=git_url,
            git_branch=branch,
            git_token=git_token,
            config=config.model_dump(),
        )
        self.created_jobs.append(job)
        self.jobs.create_job(job)
        return job


class FlakyGetJobRepository(FakeIndexJobRepository):
    def __init__(self) -> None:
        super().__init__()
        self.fail_once = True

    async def get_job(self, job_id: str) -> FakeIndexJob | None:
        self.get_job_calls.append(job_id)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("temporary get_job failure")
        return self.jobs.get(job_id)


class BlockingGetJobRepository(FakeIndexJobRepository):
    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def get_job(self, job_id: str) -> FakeIndexJob | None:
        self.get_job_calls.append(job_id)
        job = self.jobs.get(job_id)
        if job is not None and job.status == "processing":
            await self.release.wait()
        return job

    def complete_and_release(self, job_id: str, *, failed: bool = False, error_message: str | None = None) -> None:
        self.complete_job(job_id, failed=failed, error_message=error_message)
        self.release.set()


class RaisingOnceIndexer(FakeIndexer):
    def __init__(self, repo: "FakeWebhookRepository", jobs: FakeIndexJobRepository) -> None:
        super().__init__(repo, jobs)
        self.raised = False

    async def try_create_index_from_git(self, git_url: str, branch: str, config: IndexConfig, git_token: str | None = None) -> FakeIndexJob | None:
        if not self.raised:
            self.raised = True
            raise RuntimeError("dispatcher exploded")
        return await super().try_create_index_from_git(git_url, branch, config, git_token)


class FakeWebhookRepository:
    def __init__(self) -> None:
        self.targets: dict[tuple[str, str], GitWebhookTarget | None] = {}
        self.deliveries: dict[str, GitWebhookDelivery] = {}
        self.deliveries_by_target: dict[tuple[str, str], list[str]] = {}
        self.completed: list[tuple[str, str, str | None]] = []
        self.defer_calls: list[str] = []
        self.resolve_calls: list[tuple[str, str]] = []
        self.claim_calls: list[tuple[str, str]] = []
        self.reset_calls: list[str] = []
        self.link_calls: list[tuple[str, str]] = []
        self.complete_calls: list[tuple[str, str, str | None]] = []
        self.has_pending_calls: list[tuple[str, str]] = []
        self.fail_link_once_for_delivery_ids: set[str] = set()
        self.fail_complete_once_for_delivery_ids: set[str] = set()
        self.processing_commit: str | None = None
        self._status_events: dict[tuple[str, str], asyncio.Event] = {}
        self._counter = 0

    def set_target(self, target: GitWebhookTarget | None) -> None:
        if target is None:
            return
        self.targets[(target.target_type.value, target.target_id)] = target

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
        key = (target_type.value, target_id)
        for delivery_id in reversed(self.deliveries_by_target.get(key, [])):
            delivery = self.deliveries[delivery_id]
            if delivery.status is GitWebhookDeliveryStatus.PENDING:
                delivery.status = GitWebhookDeliveryStatus.SKIPPED
                delivery.completed_at = _utcnow()
                self._signal(delivery.id, delivery.status)
                break
        self._counter += 1
        delivery = GitWebhookDelivery(
            id=f"delivery-{self._counter}",
            target_type=target_type,
            target_id=target_id,
            provider_delivery_id=provider_delivery_id,
            event_name=event_name,
            branch=branch,
            head_commit=head_commit,
            status=GitWebhookDeliveryStatus.PENDING,
            received_at=_utcnow(),
            started_at=None,
            completed_at=None,
        )
        self.deliveries[delivery.id] = delivery
        self.deliveries_by_target.setdefault(key, []).append(delivery.id)
        self._signal(delivery.id, delivery.status)
        return delivery

    async def claim_latest_pending(self, target_type: GitWebhookTargetType, target_id: str) -> GitWebhookDelivery | None:
        self.claim_calls.append((target_type.value, target_id))
        key = (target_type.value, target_id)
        deliveries = [self.deliveries[delivery_id] for delivery_id in self.deliveries_by_target.get(key, [])]
        if any(delivery.status is GitWebhookDeliveryStatus.PROCESSING for delivery in deliveries):
            return None
        pending = [delivery for delivery in deliveries if delivery.status is GitWebhookDeliveryStatus.PENDING]
        if not pending:
            return None
        claimed = pending[-1]
        claimed.status = GitWebhookDeliveryStatus.PROCESSING
        claimed.started_at = _utcnow()
        self.processing_commit = claimed.head_commit
        self._signal(claimed.id, claimed.status)
        return claimed

    async def defer_claim(self, delivery_id: str) -> GitWebhookDeliveryStatus:
        self.defer_calls.append(delivery_id)
        delivery = self.deliveries[delivery_id]
        key = (delivery.target_type.value, delivery.target_id)
        newer_pending = any(
            self.deliveries[other_id].status is GitWebhookDeliveryStatus.PENDING and other_id != delivery_id
            for other_id in self.deliveries_by_target.get(key, [])
        )
        if newer_pending:
            delivery.status = GitWebhookDeliveryStatus.SKIPPED
            delivery.completed_at = _utcnow()
            self._signal(delivery.id, delivery.status)
            return GitWebhookDeliveryStatus.SKIPPED
        delivery.status = GitWebhookDeliveryStatus.PENDING
        delivery.started_at = None
        delivery.completed_at = None
        self._signal(delivery.id, delivery.status)
        return GitWebhookDeliveryStatus.PENDING

    async def complete(
        self,
        delivery_id: str,
        *,
        status: GitWebhookDeliveryStatus,
        message: str,
        index_job_id: str | None = None,
    ) -> None:
        delivery = self.deliveries[delivery_id]
        self.complete_calls.append((delivery_id, status.value, index_job_id))
        if delivery_id in self.fail_complete_once_for_delivery_ids:
            self.fail_complete_once_for_delivery_ids.remove(delivery_id)
            raise RuntimeError("temporary complete failure")
        if delivery.status is not GitWebhookDeliveryStatus.PROCESSING:
            raise RuntimeError(f"complete requires processing delivery, got {delivery.status.value}")
        delivery.status = status
        delivery.message = message
        delivery.index_job_id = index_job_id
        delivery.completed_at = _utcnow()
        self.completed.append((delivery_id, status.value, index_job_id))
        self._signal(delivery.id, delivery.status)

    async def link_index_job(self, delivery_id: str, index_job_id: str) -> None:
        delivery = self.deliveries[delivery_id]
        if delivery_id in self.fail_link_once_for_delivery_ids:
            self.fail_link_once_for_delivery_ids.remove(delivery_id)
            raise RuntimeError("temporary link failure")
        delivery.index_job_id = index_job_id
        self.link_calls.append((delivery_id, index_job_id))

    async def has_pending(self, target_type: GitWebhookTargetType, target_id: str) -> bool:
        self.has_pending_calls.append((target_type.value, target_id))
        return any(
            self.deliveries[delivery_id].status is GitWebhookDeliveryStatus.PENDING
            for delivery_id in self.deliveries_by_target.get((target_type.value, target_id), [])
        )

    async def list_recoverable(self) -> list[GitWebhookDelivery]:
        return [delivery for delivery in self.deliveries.values() if delivery.status is GitWebhookDeliveryStatus.PROCESSING]

    async def list_pending_targets(self) -> list[GitWebhookTarget]:
        targets: list[GitWebhookTarget] = []
        for key, delivery_ids in self.deliveries_by_target.items():
            if any(self.deliveries[delivery_id].status is GitWebhookDeliveryStatus.PENDING for delivery_id in delivery_ids):
                target = self.targets.get(key)
                if target is not None:
                    targets.append(target)
        return targets

    async def resolve_target(self, target_type: GitWebhookTargetType, target_id: str) -> GitWebhookTarget | None:
        self.resolve_calls.append((target_type.value, target_id))
        return self.targets.get((target_type.value, target_id))

    async def reset_processing_to_pending(self, delivery_id: str) -> GitWebhookDeliveryStatus:
        self.reset_calls.append(delivery_id)
        delivery = self.deliveries[delivery_id]
        key = (delivery.target_type.value, delivery.target_id)
        if any(
            self.deliveries[other_id].status is GitWebhookDeliveryStatus.PENDING and other_id != delivery_id
            for other_id in self.deliveries_by_target.get(key, [])
        ):
            delivery.status = GitWebhookDeliveryStatus.SKIPPED
            delivery.completed_at = _utcnow()
            self._signal(delivery.id, delivery.status)
            return GitWebhookDeliveryStatus.SKIPPED
        delivery.status = GitWebhookDeliveryStatus.PENDING
        delivery.started_at = None
        delivery.index_job_id = None
        delivery.completed_at = None
        self._signal(delivery.id, delivery.status)
        return GitWebhookDeliveryStatus.PENDING

    def add_delivery(self, delivery: GitWebhookDelivery) -> None:
        self.deliveries[delivery.id] = delivery
        key = (delivery.target_type.value, delivery.target_id)
        self.deliveries_by_target.setdefault(key, []).append(delivery.id)
        self._signal(delivery.id, delivery.status)

    def status(self, delivery_id: str) -> str:
        return self.deliveries[delivery_id].status.value

    async def wait_for_status(self, delivery_id: str, status: str) -> None:
        event = self._status_events.setdefault((delivery_id, status), asyncio.Event())
        if self.status(delivery_id) == status:
            return
        await asyncio.wait_for(event.wait(), timeout=2)

    def _signal(self, delivery_id: str, status: GitWebhookDeliveryStatus) -> None:
        event = self._status_events.setdefault((delivery_id, status.value), asyncio.Event())
        event.set()


@dataclass
class FakeWorkspaceOutcome:
    state: str
    message: str


class FakeUserSpace:
    def __init__(self, outcome: FakeWorkspaceOutcome | None = None) -> None:
        self.outcome = outcome or FakeWorkspaceOutcome("imported", "Imported from webhook")
        self.calls: list[str] = []
        self._entered = asyncio.Event()
        self._release = asyncio.Event()
        self.block_inside_lock = False
        self.raise_once: Exception | None = None

    @asynccontextmanager
    async def workspace_scm_operation(self, workspace_id: str):
        self.calls.append(f"enter:{workspace_id}")
        self._entered.set()
        if self.block_inside_lock:
            await self._release.wait()
        try:
            yield
        finally:
            self.calls.append(f"exit:{workspace_id}")

    async def run_workspace_scm_webhook_pull_locked(self, workspace_id: str) -> FakeWorkspaceOutcome:
        self.calls.append(f"pull:{workspace_id}")
        if self.raise_once is not None:
            exc = self.raise_once
            self.raise_once = None
            raise exc
        return self.outcome

    async def wait_until_entered(self) -> None:
        await asyncio.wait_for(self._entered.wait(), timeout=2)

    def release(self) -> None:
        self._release.set()


class GitWebhookServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_link_index_job_retry_reaches_terminal_state_and_newer_push_proceeds(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target()
        repo.set_target(target)
        jobs = FakeIndexJobRepository()
        indexer = BlockingIndexer(repo, jobs)
        service = GitWebhookService(repo=repo, indexer=indexer, userspace=FakeUserSpace(), index_jobs=jobs, poll_seconds=0.01)

        await service.start()
        self.addAsyncCleanup(service.stop)

        first = await service.accept_push(target, _push("commit-1"))
        repo.fail_link_once_for_delivery_ids.add(first.id)
        await repo.wait_for_status(first.id, "processing")
        trailing = await service.accept_push(target, _push("commit-2"))

        indexer.release("commit-1")
        await repo.wait_for_status(first.id, "completed")
        await repo.wait_for_status(trailing.id, "completed")

        self.assertEqual(repo.deliveries[first.id].index_job_id, "job-1")
        self.assertEqual(repo.link_calls, [(first.id, "job-1"), (trailing.id, "job-2")])
        self.assertEqual(indexer.started_commits, ["commit-1", "commit-2"])

    async def test_complete_retry_after_git_terminal_state_reaches_terminal_and_newer_push_proceeds(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target()
        repo.set_target(target)
        jobs = FakeIndexJobRepository()
        indexer = BlockingIndexer(repo, jobs)
        service = GitWebhookService(repo=repo, indexer=indexer, userspace=FakeUserSpace(), index_jobs=jobs, poll_seconds=0.01)

        await service.start()
        self.addAsyncCleanup(service.stop)

        first = await service.accept_push(target, _push("commit-1"))
        await repo.wait_for_status(first.id, "processing")
        repo.fail_complete_once_for_delivery_ids.add(first.id)
        trailing = await service.accept_push(target, _push("commit-2"))

        indexer.release("commit-1")
        await repo.wait_for_status(first.id, "completed")
        await repo.wait_for_status(trailing.id, "completed")

        self.assertGreaterEqual(
            [call for call in repo.complete_calls if call[0] == first.id and call[1] == "completed"].__len__(),
            2,
        )
        self.assertEqual(indexer.started_commits, ["commit-1", "commit-2"])

    async def test_git_job_is_linked_before_monitoring(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target()
        repo.set_target(target)
        jobs = BlockingGetJobRepository()
        indexer = ImmediatePendingIndexer(repo, jobs)
        service = GitWebhookService(repo=repo, indexer=indexer, userspace=FakeUserSpace(), index_jobs=jobs, poll_seconds=0.01)

        await service.start()
        self.addAsyncCleanup(service.stop)

        delivery = await service.accept_push(target, _push("commit-1"))
        await repo.wait_for_status(delivery.id, "processing")

        self.assertEqual(repo.deliveries[delivery.id].index_job_id, "job-1")
        self.assertEqual(repo.link_calls, [(delivery.id, "job-1")])

        jobs.complete_and_release("job-1")
        await repo.wait_for_status(delivery.id, "completed")

    async def test_git_dispatch_exception_does_not_terminate_worker_or_block_newer_push(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target()
        repo.set_target(target)
        jobs = FakeIndexJobRepository()
        indexer = RaisingOnceIndexer(repo, jobs)
        service = GitWebhookService(repo=repo, indexer=indexer, userspace=FakeUserSpace(), index_jobs=jobs, poll_seconds=0.01)

        await service.start()
        self.addAsyncCleanup(service.stop)

        failed = await service.accept_push(target, _push("commit-1"))
        await asyncio.wait_for(repo.wait_for_status(failed.id, "failed"), timeout=0.5)

        trailing = await service.accept_push(target, _push("commit-2"))
        await asyncio.wait_for(repo.wait_for_status(trailing.id, "completed"), timeout=0.5)

        self.assertEqual(repo.status(failed.id), "failed")
        self.assertEqual(repo.status(trailing.id), "completed")
        self.assertEqual(indexer.started_commits, ["commit-2"])

    async def test_idle_workspace_worker_waits_without_reacquiring_scm_operation(self) -> None:
        repo = FakeWebhookRepository()
        target = _workspace_target()
        repo.set_target(target)
        userspace = FakeUserSpace()
        service = GitWebhookService(
            repo=repo,
            indexer=FakeIndexer(repo, FakeIndexJobRepository()),
            userspace=userspace,
            index_jobs=FakeIndexJobRepository(),
            poll_seconds=0.01,
        )

        await service.start()
        self.addAsyncCleanup(service.stop)

        delivery = await service.accept_push(target, _push("commit-1"))
        await repo.wait_for_status(delivery.id, "completed")
        await asyncio.sleep(0.05)

        self.assertEqual(userspace.calls, ["enter:workspace-1", "pull:workspace-1", "exit:workspace-1"])
        self.assertGreaterEqual(len(repo.has_pending_calls), 1)

    async def test_workspace_non_import_outcomes_have_truthful_statuses(self) -> None:
        cases = {
            "imported": GitWebhookDeliveryStatus.COMPLETED,
            "up_to_date": GitWebhookDeliveryStatus.COMPLETED,
            "conflict": GitWebhookDeliveryStatus.FAILED,
            "paused": GitWebhookDeliveryStatus.FAILED,
            "not_upstream": GitWebhookDeliveryStatus.FAILED,
            "disconnected": GitWebhookDeliveryStatus.FAILED,
            "missing_branch": GitWebhookDeliveryStatus.FAILED,
        }

        for state, expected_status in cases.items():
            with self.subTest(state=state):
                service = GitWebhookService()
                status, message = service._workspace_outcome_status(FakeWorkspaceOutcome(state, f"message for {state}"))
                self.assertEqual(status, expected_status)
                self.assertIn(state, message)

    async def test_only_latest_pending_push_runs_after_active_git_job(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target()
        repo.set_target(target)
        jobs = FakeIndexJobRepository(active_git_job=True)
        service = GitWebhookService(repo=repo, indexer=FakeIndexer(repo, jobs), userspace=FakeUserSpace(), index_jobs=jobs)

        await service.start()
        self.addAsyncCleanup(service.stop)

        first = await service.accept_push(target, _push("commit-1"))
        second = await service.accept_push(target, _push("commit-2"))
        third = await service.accept_push(target, _push("commit-3"))
        self.assertEqual(repo.status(first.id), "skipped")
        self.assertEqual(repo.status(second.id), "skipped")
        self.assertEqual(repo.status(third.id), "pending")
        self.assertEqual(len(service._target_tasks), 1)
        self.assertEqual(len(service._target_events), 1)

        jobs.active_git_job = False
        assert target.key is not None
        service.wake_target(target.key)
        await repo.wait_for_status(third.id, "completed")
        self.assertEqual(service._indexer.started_commits, ["commit-3"])

    async def test_push_during_processing_creates_one_trailing_run(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target()
        repo.set_target(target)
        jobs = FakeIndexJobRepository()
        indexer = BlockingIndexer(repo, jobs)
        service = GitWebhookService(repo=repo, indexer=indexer, userspace=FakeUserSpace(), index_jobs=jobs)

        await service.start()
        self.addAsyncCleanup(service.stop)

        running = await service.accept_push(target, _push("commit-1"))
        await repo.wait_for_status(running.id, "processing")
        skipped = await service.accept_push(target, _push("commit-2"))
        trailing = await service.accept_push(target, _push("commit-3"))
        self.assertEqual(repo.status(skipped.id), "skipped")

        indexer.release("commit-1")
        await repo.wait_for_status(trailing.id, "completed")
        self.assertEqual(indexer.started_commits, ["commit-1", "commit-3"])

    async def test_failed_run_does_not_block_latest_pending_push(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target()
        repo.set_target(target)
        jobs = FakeIndexJobRepository()
        indexer = FailingOnceIndexer(repo, jobs)
        service = GitWebhookService(repo=repo, indexer=indexer, userspace=FakeUserSpace(), index_jobs=jobs)

        await service.start()
        self.addAsyncCleanup(service.stop)

        failed = await service.accept_push(target, _push("commit-1"))
        await repo.wait_for_status(failed.id, "processing")
        trailing = await service.accept_push(target, _push("commit-2"))
        await repo.wait_for_status(trailing.id, "completed")
        self.assertEqual(repo.status(failed.id), "failed")
        self.assertEqual(indexer.started_commits, ["commit-1", "commit-2"])

    async def test_post_claim_manual_git_job_race_defers_to_newest_pending(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target()
        repo.set_target(target)
        jobs = FakeIndexJobRepository()
        indexer = DeferredRaceIndexer(repo, jobs)
        service = GitWebhookService(repo=repo, indexer=indexer, userspace=FakeUserSpace(), index_jobs=jobs)

        await service.start()
        self.addAsyncCleanup(service.stop)

        first = await service.accept_push(target, _push("commit-1"))
        await repo.wait_for_status(first.id, "processing")
        trailing = await service.accept_push(target, _push("commit-2"))
        indexer.release_race()
        await repo.wait_for_status(trailing.id, "completed")

        self.assertEqual(repo.defer_calls, [first.id])
        self.assertEqual(repo.status(first.id), "skipped")
        self.assertEqual(repo.status(trailing.id), "completed")
        self.assertEqual(indexer.started_commits, ["commit-2"])

    async def test_monitors_webhook_git_jobs_by_id_only(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target()
        repo.set_target(target)
        jobs = FakeIndexJobRepository()
        indexer = BlockingIndexer(repo, jobs)
        service = GitWebhookService(repo=repo, indexer=indexer, userspace=FakeUserSpace(), index_jobs=jobs, poll_seconds=0.01)

        await service.start()
        self.addAsyncCleanup(service.stop)

        delivery = await service.accept_push(target, _push("commit-1"))
        await repo.wait_for_status(delivery.id, "processing")
        jobs.active_job_by_name[target.name or "git-index"] = FakeIndexJob(
            id="manual-replacement",
            name=target.name or "git-index",
            git_url="manual",
            git_branch="main",
            git_token=None,
            config={"name": target.name or "git-index"},
        )
        indexer.release("commit-1")
        await repo.wait_for_status(delivery.id, "completed")

        self.assertEqual(repo.deliveries[delivery.id].index_job_id, "job-1")
        self.assertIn("job-1", jobs.get_job_calls)
        self.assertNotIn("manual-replacement", jobs.get_job_calls)

    async def test_monitor_git_job_retries_one_transient_lookup_failure_without_replacement_work(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target()
        repo.set_target(target)
        jobs = FlakyGetJobRepository()
        indexer = ImmediatePendingIndexer(repo, jobs)
        service = GitWebhookService(repo=repo, indexer=indexer, userspace=FakeUserSpace(), index_jobs=jobs, poll_seconds=0.01)

        await service.start()
        self.addAsyncCleanup(service.stop)

        delivery = await service.accept_push(target, _push("commit-1"))
        await repo.wait_for_status(delivery.id, "processing")
        self.assertEqual(repo.deliveries[delivery.id].index_job_id, "job-1")

        await asyncio.sleep(0.05)
        self.assertEqual(repo.status(delivery.id), "processing")
        self.assertEqual(repo.deliveries[delivery.id].index_job_id, "job-1")

        jobs.complete_job("job-1")
        await repo.wait_for_status(delivery.id, "completed")

        self.assertEqual(indexer.started_commits, ["commit-1"])
        self.assertEqual(len(indexer.created_jobs), 1)
        self.assertGreaterEqual(jobs.get_job_calls.count("job-1"), 2)

    async def test_workspace_pull_acquires_operation_context_before_claim_and_conflict_is_terminalized(self) -> None:
        repo = FakeWebhookRepository()
        target = _workspace_target()
        repo.set_target(target)
        userspace = FakeUserSpace(FakeWorkspaceOutcome("conflict", "Remote has conflicts"))
        service = GitWebhookService(
            repo=repo,
            indexer=FakeIndexer(repo, FakeIndexJobRepository()),
            userspace=userspace,
            index_jobs=FakeIndexJobRepository(),
        )

        await service.start()
        self.addAsyncCleanup(service.stop)

        delivery = await service.accept_push(target, _push("commit-1"))
        await repo.wait_for_status(delivery.id, "failed")

        self.assertEqual(userspace.calls[:2], ["enter:workspace-1", "pull:workspace-1"])
        self.assertEqual(repo.status(delivery.id), "failed")
        self.assertIn("Remote has conflicts", repo.deliveries[delivery.id].message or "")

    async def test_workspace_exception_fails_processing_and_allows_newer_pending_progress(self) -> None:
        repo = FakeWebhookRepository()
        target = _workspace_target()
        repo.set_target(target)
        userspace = FakeUserSpace()
        userspace.raise_once = RuntimeError("preview exploded")
        service = GitWebhookService(
            repo=repo,
            indexer=FakeIndexer(repo, FakeIndexJobRepository()),
            userspace=userspace,
            index_jobs=FakeIndexJobRepository(),
        )

        await service.start()
        self.addAsyncCleanup(service.stop)

        first = await service.accept_push(target, _push("commit-1"))
        await repo.wait_for_status(first.id, "failed")
        second = await service.accept_push(target, _push("commit-2"))
        await repo.wait_for_status(second.id, "completed")

        self.assertEqual(repo.status(first.id), "failed")
        self.assertEqual(repo.status(second.id), "completed")

    async def test_pre_claim_manual_work_interleaving_claims_newest_pending_after_manual_completion(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target()
        repo.set_target(target)
        jobs = FakeIndexJobRepository(active_git_job=True)
        service = GitWebhookService(repo=repo, indexer=FakeIndexer(repo, jobs), userspace=FakeUserSpace(), index_jobs=jobs, poll_seconds=0.01)

        await service.start()
        self.addAsyncCleanup(service.stop)

        first = await service.accept_push(target, _push("commit-1"))
        second = await service.accept_push(target, _push("commit-2"))
        self.assertEqual(repo.status(first.id), "skipped")
        self.assertEqual(repo.status(second.id), "pending")

        jobs.active_git_job = False
        assert target.key is not None
        service.wake_target(target.key)
        await repo.wait_for_status(second.id, "completed")

        self.assertEqual(service._indexer.started_commits, ["commit-2"])

    async def test_paused_push_is_ignored_without_queueing(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target().model_copy(update={"paused": True})
        event = _push("commit-paused")
        service = GitWebhookService(
            repo=repo, indexer=FakeIndexer(repo, FakeIndexJobRepository()), userspace=FakeUserSpace(), index_jobs=FakeIndexJobRepository()
        )

        record_ignored = mock.AsyncMock(
            return_value=GitWebhookDelivery(
                id="ignored-1",
                target_type=target.target_type,
                target_id=target.target_id,
                provider_delivery_id=event.provider_delivery_id,
                event_name=event.event_name,
                branch=event.branch,
                head_commit=event.head_commit,
                status=GitWebhookDeliveryStatus.IGNORED,
                message="Webhook push ignored while paused.",
                received_at=_utcnow(),
                started_at=None,
                completed_at=_utcnow(),
            )
        )
        with mock.patch("ragtime.git_webhooks.routes.git_webhook_repository.record_ignored", record_ignored):
            from ragtime.git_webhooks.routes import _accept_matching_events

            await _accept_matching_events(target, [event])

        record_ignored.assert_awaited_once()
        ignored_call = record_ignored.await_args
        assert ignored_call is not None
        ignored_message = ignored_call.args[2]
        self.assertIn("paused", ignored_message.lower())

    async def test_resume_does_not_replay_pending_rows_skipped_by_pause(self) -> None:
        repo = FakeWebhookRepository()
        active_target = _git_target()
        repo.set_target(active_target)
        jobs = FakeIndexJobRepository(active_git_job=True)
        service = GitWebhookService(repo=repo, indexer=FakeIndexer(repo, jobs), userspace=FakeUserSpace(), index_jobs=jobs, poll_seconds=0.01)

        await service.start()
        self.addAsyncCleanup(service.stop)

        delivery = await service.accept_push(active_target, _push("commit-1"))
        self.assertEqual(repo.status(delivery.id), "pending")
        paused_target = active_target.model_copy(update={"paused": True})
        repo.set_target(paused_target)

        pause_delivery = repo.deliveries[delivery.id]
        pause_delivery.status = GitWebhookDeliveryStatus.SKIPPED
        pause_delivery.message = "Webhook paused before processing."
        pause_delivery.completed_at = _utcnow()
        repo._signal(pause_delivery.id, pause_delivery.status)

        resumed_target = active_target.model_copy(update={"paused": False})
        repo.set_target(resumed_target)
        service.schedule_target(resumed_target)
        await asyncio.sleep(0.05)

        self.assertEqual(repo.claim_calls, [])
        self.assertEqual(repo.status(delivery.id), "skipped")
        self.assertEqual(service._indexer.started_commits, [])

    async def test_start_ignores_paused_pending_targets(self) -> None:
        repo = FakeWebhookRepository()
        paused_target = _git_target(target_id="paused-index").model_copy(update={"paused": True})
        repo.set_target(paused_target)
        repo.add_delivery(
            GitWebhookDelivery(
                id="pending-paused",
                target_type=paused_target.target_type,
                target_id=paused_target.target_id,
                provider_delivery_id="delivery-paused",
                event_name="push",
                branch="main",
                head_commit="commit-paused",
                status=GitWebhookDeliveryStatus.PENDING,
                received_at=_utcnow(),
                started_at=None,
                completed_at=None,
            )
        )
        service = GitWebhookService(
            repo=repo, indexer=FakeIndexer(repo, FakeIndexJobRepository()), userspace=FakeUserSpace(), index_jobs=FakeIndexJobRepository(), poll_seconds=0.01
        )

        await service.start()
        self.addAsyncCleanup(service.stop)
        await asyncio.sleep(0.05)

        self.assertEqual(service._target_tasks, {})
        self.assertEqual(repo.claim_calls, [])

    async def test_disable_during_active_git_work_still_terminalizes_processing_delivery(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target(git_token="token-before-disable")
        repo.set_target(target)
        jobs = FakeIndexJobRepository()
        indexer = BlockingIndexer(repo, jobs)
        service = GitWebhookService(repo=repo, indexer=indexer, userspace=FakeUserSpace(), index_jobs=jobs)

        await service.start()
        self.addAsyncCleanup(service.stop)

        delivery = await service.accept_push(target, _push("commit-1"))
        await repo.wait_for_status(delivery.id, "processing")
        repo.targets[(target.target_type.value, target.target_id)] = None

        indexer.release("commit-1")
        await repo.wait_for_status(delivery.id, "completed")
        self.assertEqual(repo.status(delivery.id), "completed")
        self.assertEqual(indexer.created_jobs[0].git_token, "token-before-disable")

    async def test_stop_cancels_drain_tasks_cleanly(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target()
        repo.set_target(target)
        jobs = FakeIndexJobRepository(active_git_job=True)
        service = GitWebhookService(
            repo=repo,
            indexer=FakeIndexer(repo, jobs),
            userspace=FakeUserSpace(),
            index_jobs=jobs,
            poll_seconds=1.0,
        )

        await service.start()
        await service.accept_push(target, _push("commit-1"))
        await asyncio.sleep(0)

        await service.stop()

        self.assertEqual(service._target_tasks, {})
        self.assertEqual(service._target_events, {})

    async def test_start_reconciles_processing_before_scheduling_pending_targets(self) -> None:
        repo = FakeWebhookRepository()
        git_target = _git_target()
        workspace_target = _workspace_target()
        repo.set_target(git_target)
        repo.set_target(workspace_target)
        processing_git = GitWebhookDelivery(
            id="processing-git",
            target_type=GitWebhookTargetType.GIT_INDEX,
            target_id=git_target.target_id,
            provider_delivery_id="delivery-processing-git",
            event_name="push",
            branch="main",
            head_commit="commit-processing",
            status=GitWebhookDeliveryStatus.PROCESSING,
            index_job_id="job-recovered",
            received_at=_utcnow(),
            started_at=_utcnow(),
            completed_at=None,
        )
        trailing_git = GitWebhookDelivery(
            id="pending-git",
            target_type=GitWebhookTargetType.GIT_INDEX,
            target_id=git_target.target_id,
            provider_delivery_id="delivery-pending-git",
            event_name="push",
            branch="main",
            head_commit="commit-trailing",
            status=GitWebhookDeliveryStatus.PENDING,
            received_at=_utcnow(),
            started_at=None,
            completed_at=None,
        )
        processing_workspace = GitWebhookDelivery(
            id="processing-workspace",
            target_type=GitWebhookTargetType.WORKSPACE_SCM,
            target_id=workspace_target.target_id,
            provider_delivery_id="delivery-processing-workspace",
            event_name="push",
            branch="main",
            head_commit="commit-workspace",
            status=GitWebhookDeliveryStatus.PROCESSING,
            received_at=_utcnow(),
            started_at=_utcnow(),
            completed_at=None,
        )
        repo.add_delivery(processing_git)
        repo.add_delivery(trailing_git)
        repo.add_delivery(processing_workspace)
        jobs = FakeIndexJobRepository()
        assert git_target.source is not None
        jobs.create_job(
            FakeIndexJob(
                id="job-recovered",
                name=git_target.name or "git-index",
                git_url=git_target.source,
                git_branch=git_target.branch or "main",
                git_token=git_target.git_token,
                config=dict(git_target.config_snapshot or {}),
                status="completed",
            )
        )
        jobs.complete_job("job-recovered")
        userspace = FakeUserSpace()
        service = GitWebhookService(repo=repo, indexer=FakeIndexer(repo, jobs), userspace=userspace, index_jobs=jobs)

        await service.start()
        self.addAsyncCleanup(service.stop)

        await repo.wait_for_status("pending-git", "completed")
        await repo.wait_for_status("processing-workspace", "completed")

        self.assertEqual(repo.status("processing-git"), "completed")
        self.assertEqual(repo.status("pending-git"), "completed")
        self.assertEqual(repo.status("processing-workspace"), "completed")
        self.assertLess(repo.claim_calls.index((GitWebhookTargetType.GIT_INDEX.value, git_target.target_id)), len(repo.claim_calls))

    async def test_start_recovery_missing_linked_git_job_skips_stale_processing_when_pending_winner_exists(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target()
        repo.set_target(target)
        processing = GitWebhookDelivery(
            id="processing-git",
            target_type=GitWebhookTargetType.GIT_INDEX,
            target_id=target.target_id,
            provider_delivery_id="delivery-processing",
            event_name="push",
            branch="main",
            head_commit="commit-old",
            status=GitWebhookDeliveryStatus.PROCESSING,
            index_job_id="missing-job",
            received_at=_utcnow(),
            started_at=_utcnow(),
            completed_at=None,
        )
        pending = GitWebhookDelivery(
            id="pending-git",
            target_type=GitWebhookTargetType.GIT_INDEX,
            target_id=target.target_id,
            provider_delivery_id="delivery-pending",
            event_name="push",
            branch="main",
            head_commit="commit-new",
            status=GitWebhookDeliveryStatus.PENDING,
            received_at=_utcnow(),
            started_at=None,
            completed_at=None,
        )
        repo.add_delivery(processing)
        repo.add_delivery(pending)
        jobs = FakeIndexJobRepository()
        service = GitWebhookService(repo=repo, indexer=FakeIndexer(repo, jobs), userspace=FakeUserSpace(), index_jobs=jobs)

        await service.start()
        self.addAsyncCleanup(service.stop)

        await repo.wait_for_status("pending-git", "completed")
        self.assertEqual(repo.status("processing-git"), "skipped")
        self.assertEqual(repo.reset_calls, ["processing-git"])

    async def test_start_recovery_null_git_job_id_skips_stale_processing_when_pending_winner_exists(self) -> None:
        repo = FakeWebhookRepository()
        target = _git_target()
        repo.set_target(target)
        processing = GitWebhookDelivery(
            id="processing-git",
            target_type=GitWebhookTargetType.GIT_INDEX,
            target_id=target.target_id,
            provider_delivery_id="delivery-processing",
            event_name="push",
            branch="main",
            head_commit="commit-old",
            status=GitWebhookDeliveryStatus.PROCESSING,
            index_job_id=None,
            received_at=_utcnow(),
            started_at=_utcnow(),
            completed_at=None,
        )
        pending = GitWebhookDelivery(
            id="pending-git",
            target_type=GitWebhookTargetType.GIT_INDEX,
            target_id=target.target_id,
            provider_delivery_id="delivery-pending",
            event_name="push",
            branch="main",
            head_commit="commit-new",
            status=GitWebhookDeliveryStatus.PENDING,
            received_at=_utcnow(),
            started_at=None,
            completed_at=None,
        )
        repo.add_delivery(processing)
        repo.add_delivery(pending)
        jobs = FakeIndexJobRepository()
        service = GitWebhookService(repo=repo, indexer=FakeIndexer(repo, jobs), userspace=FakeUserSpace(), index_jobs=jobs)

        await service.start()
        self.addAsyncCleanup(service.stop)

        await repo.wait_for_status("pending-git", "completed")
        self.assertEqual(repo.status("processing-git"), "skipped")


if __name__ == "__main__":
    unittest.main()
