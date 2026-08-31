import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from ragtime.indexer import routes


class GitIndexConfigCadenceTests(unittest.IsolatedAsyncioTestCase):
    def _metadata(self, **overrides: object) -> SimpleNamespace:
        base = {
            "sourceType": "git",
            "configSnapshot": {
                "reindex_interval_hours": 24,
                "reindex_start_minute": 180,
                "reindex_timezone": "America/Denver",
            },
            "gitBranch": "main",
            "webhookId": "webhook-1",
            "webhookSecret": "secret-1",
            "documentCount": 1,
            "chunkCount": 1,
        }
        base.update(overrides)
        return SimpleNamespace(**base)

    async def test_positive_interval_conflicts_with_complete_webhook_credentials(self) -> None:
        repo = mock.AsyncMock()
        repo.get_index_metadata = mock.AsyncMock(return_value=self._metadata())
        repo.update_index_config = mock.AsyncMock(return_value=True)

        with mock.patch.object(routes, "repository", repo):
            with self.assertRaises(HTTPException) as exc_info:
                await routes.update_index_config(
                    "ragtime",
                    routes.UpdateIndexConfigRequest(reindex_interval_hours=24),
                    _user=mock.sentinel.user,
                )

        self.assertEqual(exc_info.exception.status_code, 409)
        self.assertEqual(exc_info.exception.detail, "Disable webhook delivery before enabling scheduled re-indexing.")
        repo.update_index_config.assert_not_awaited()

    async def test_zero_interval_remains_allowed_with_complete_webhook_credentials(self) -> None:
        repo = mock.AsyncMock()
        repo.get_index_metadata = mock.AsyncMock(return_value=self._metadata())
        repo.update_index_config = mock.AsyncMock(return_value=True)

        with mock.patch.object(routes, "repository", repo):
            response = await routes.update_index_config(
                "ragtime",
                routes.UpdateIndexConfigRequest(reindex_interval_hours=0),
                _user=mock.sentinel.user,
            )

        self.assertEqual(response["config_snapshot"]["reindex_interval_hours"], 0)
        repo.update_index_config.assert_awaited_once()

    async def test_unrelated_config_update_remains_allowed_with_complete_webhook_credentials(self) -> None:
        repo = mock.AsyncMock()
        repo.get_index_metadata = mock.AsyncMock(return_value=self._metadata())
        repo.update_index_config = mock.AsyncMock(return_value=True)

        with mock.patch.object(routes, "repository", repo):
            response = await routes.update_index_config(
                "ragtime",
                routes.UpdateIndexConfigRequest(chunk_size=1200),
                _user=mock.sentinel.user,
            )

        self.assertEqual(response["config_snapshot"]["chunk_size"], 1200)
        repo.update_index_config.assert_awaited_once()
