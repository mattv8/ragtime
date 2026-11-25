import unittest
from types import SimpleNamespace
from unittest import mock

from prisma import Json

from ragtime.indexer.repository import IndexerRepository
from ragtime.indexer.task_policy import make_execution_policy


class ChatTaskCreationPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def _create_task(self, **kwargs):
        repo = IndexerRepository()
        fake_db = SimpleNamespace(
            chattask=SimpleNamespace(create=mock.AsyncMock(side_effect=lambda data: SimpleNamespace(id=data["id"]))),
            conversation=SimpleNamespace(update=mock.AsyncMock()),
        )
        model = object()
        with (
            mock.patch.object(repo, "_get_db", new=mock.AsyncMock(return_value=fake_db)),
            mock.patch.object(repo, "_prisma_task_to_model", return_value=model) as convert,
        ):
            result = await repo.create_chat_task("conversation-1", "Hello", **kwargs)

        fake_db.chattask.create.assert_awaited_once()
        data = fake_db.chattask.create.await_args.kwargs["data"]
        self.assertEqual(data["conversation"], {"connect": {"id": "conversation-1"}})
        self.assertEqual(data["userMessage"], "Hello")
        fake_db.conversation.update.assert_awaited_once_with(where={"id": "conversation-1"}, data={"activeTaskId": data["id"]})
        convert.assert_called_once()
        self.assertIs(result, model)
        return data

    async def test_omitted_policy_is_not_persisted(self):
        data = await self._create_task()
        self.assertNotIn("executionPolicy", data)

    async def test_none_policy_is_not_persisted(self):
        data = await self._create_task(execution_policy=None)
        self.assertNotIn("executionPolicy", data)

    async def test_policy_is_persisted_as_json(self):
        policy = make_execution_policy("general", source="workspace_agent")
        data = await self._create_task(execution_policy=policy)
        self.assertIsInstance(data["executionPolicy"], Json)
        self.assertEqual(data["executionPolicy"].data, policy)
