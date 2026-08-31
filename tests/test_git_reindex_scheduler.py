from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

import ragtime.indexer.background_tasks as background_tasks


def _git_metadata(
    name: str,
    *,
    webhook_id: str | None = None,
    webhook_secret: str | None = None,
    webhook_paused: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        sourceType="git",
        source="https://example.com/repo.git",
        gitBranch="main",
        gitToken=None,
        description="",
        webhookId=webhook_id,
        webhookSecret=webhook_secret,
        webhookPaused=webhook_paused,
        lastModified=datetime(2026, 1, 1, tzinfo=timezone.utc) - timedelta(days=2),
        configSnapshot={
            "reindex_interval_hours": 24,
            "file_patterns": ["**/*"],
            "exclude_patterns": [],
        },
    )


class GitReindexSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_due_scheduled_git_index_without_webhook_credentials_creates_job(self) -> None:
        service = background_tasks.BackgroundTaskService()
        metadata = _git_metadata("scheduled-only")
        fake_db = SimpleNamespace(
            indexmetadata=SimpleNamespace(find_many=mock.AsyncMock(return_value=[metadata])),
            indexjob=SimpleNamespace(find_first=mock.AsyncMock(return_value=None)),
        )

        with (
            mock.patch.object(background_tasks.repository, "_get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch.object(background_tasks.indexer, "create_index_from_git", mock.AsyncMock()) as create_index,
            mock.patch.object(background_tasks, "utc_now", return_value=datetime(2026, 1, 4, tzinfo=timezone.utc)),
        ):
            await service._check_and_trigger_git_reindex()

        create_index.assert_awaited_once()

    async def test_due_scheduled_git_index_with_complete_webhook_credentials_skips_job(self) -> None:
        service = background_tasks.BackgroundTaskService()
        metadata = _git_metadata(
            "active-webhook",
            webhook_id="wh_123",
            webhook_secret="secret-123",
        )
        fake_db = SimpleNamespace(
            indexmetadata=SimpleNamespace(find_many=mock.AsyncMock(return_value=[metadata])),
            indexjob=SimpleNamespace(find_first=mock.AsyncMock(return_value=None)),
        )

        with (
            mock.patch.object(background_tasks.repository, "_get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch.object(background_tasks.indexer, "create_index_from_git", mock.AsyncMock()) as create_index,
            mock.patch.object(background_tasks, "utc_now", return_value=datetime(2026, 1, 4, tzinfo=timezone.utc)),
        ):
            await service._check_and_trigger_git_reindex()

        create_index.assert_not_awaited()

    async def test_due_scheduled_git_index_with_paused_webhook_credentials_still_skips_job(self) -> None:
        service = background_tasks.BackgroundTaskService()
        metadata = _git_metadata(
            "paused-webhook",
            webhook_id="wh_456",
            webhook_secret="secret-456",
            webhook_paused=True,
        )
        fake_db = SimpleNamespace(
            indexmetadata=SimpleNamespace(find_many=mock.AsyncMock(return_value=[metadata])),
            indexjob=SimpleNamespace(find_first=mock.AsyncMock(return_value=None)),
        )

        with (
            mock.patch.object(background_tasks.repository, "_get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch.object(background_tasks.indexer, "create_index_from_git", mock.AsyncMock()) as create_index,
            mock.patch.object(background_tasks, "utc_now", return_value=datetime(2026, 1, 4, tzinfo=timezone.utc)),
        ):
            await service._check_and_trigger_git_reindex()

        create_index.assert_not_awaited()

    async def test_due_scheduled_git_index_with_partial_webhook_credentials_still_creates_job(self) -> None:
        service = background_tasks.BackgroundTaskService()
        metadata = _git_metadata(
            "partial-webhook",
            webhook_id="wh_partial",
            webhook_secret=None,
        )
        fake_db = SimpleNamespace(
            indexmetadata=SimpleNamespace(find_many=mock.AsyncMock(return_value=[metadata])),
            indexjob=SimpleNamespace(find_first=mock.AsyncMock(return_value=None)),
        )

        with (
            mock.patch.object(background_tasks.repository, "_get_db", mock.AsyncMock(return_value=fake_db)),
            mock.patch.object(background_tasks.indexer, "create_index_from_git", mock.AsyncMock()) as create_index,
            mock.patch.object(background_tasks, "utc_now", return_value=datetime(2026, 1, 4, tzinfo=timezone.utc)),
        ):
            await service._check_and_trigger_git_reindex()

        create_index.assert_awaited_once()
