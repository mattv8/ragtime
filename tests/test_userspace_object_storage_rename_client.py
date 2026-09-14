import unittest
from unittest import mock

from ragtime.userspace.object_storage import client


class ObjectStorageRenameClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_rename_uses_only_the_lock_fenced_control_endpoint(self):
        response = {
            "workspace_id": "workspace",
            "bucket_name": "default",
            "key": "renamed.txt",
            "size_bytes": 12,
            "content_type": "text/plain",
        }
        with mock.patch.object(client.control, "request", new=mock.AsyncMock(return_value=response)) as request:
            result = await client.rename_object("workspace", "default", "source.txt", "renamed.txt")
        request.assert_awaited_once_with("POST", "/v1/workspaces/workspace/buckets/default/rename", {"key": "source.txt", "new_key": "renamed.txt"})
        self.assertEqual(response, result)

    async def test_rename_propagates_control_collision_without_sdk_delete(self):
        collision = client.HTTPException(status_code=409, detail="Object storage request was rejected")
        with mock.patch.object(client.control, "request", new=mock.AsyncMock(side_effect=collision)):
            with self.assertRaises(client.HTTPException) as raised:
                await client.rename_object("workspace", "default", "source", "destination")
        self.assertEqual(409, raised.exception.status_code)
