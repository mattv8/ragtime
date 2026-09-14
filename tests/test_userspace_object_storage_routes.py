from __future__ import annotations

import io
import unittest
from types import SimpleNamespace
from typing import BinaryIO
from unittest import mock

from fastapi import Request, UploadFile
from starlette.datastructures import Headers

from ragtime.userspace import routes
from ragtime.userspace.models import UploadUserSpaceObjectStorageObjectResponse


class ObjectStorageRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_stages_file_and_delegates_to_s3_client(self) -> None:
        upload = UploadFile(filename="report.csv", file=io.BytesIO(b"a,b\n1,2\n"), headers=Headers({"content-type": "text/csv"}))
        user = SimpleNamespace(id="user-1")
        service = SimpleNamespace(
            enforce_workspace_role=mock.AsyncMock(),
            _normalize_object_storage_prefix=lambda value: value.strip("/"),
        )
        received: dict[str, object] = {}

        async def upload_file(
            _workspace: str, _user: str, _bucket: str, key: str, fileobj: BinaryIO, **_kwargs: object
        ) -> UploadUserSpaceObjectStorageObjectResponse:
            content = fileobj.read()
            received["key"] = key
            received["content"] = content
            return UploadUserSpaceObjectStorageObjectResponse(workspace_id=_workspace, bucket_name=_bucket, key=key, size_bytes=len(content))

        service.upload_workspace_object_storage_file = mock.AsyncMock(side_effect=upload_file)

        with (
            mock.patch.object(routes, "userspace_service", service),
            mock.patch.object(routes, "get_app_settings", new=mock.AsyncMock(return_value={})),
        ):
            result = await routes.upload_workspace_object_storage_object("workspace-1", "uploads", upload, "data", user)

        self.assertEqual(received, {"key": "data/report.csv", "content": b"a,b\n1,2\n"})
        self.assertEqual(result.size_bytes, 8)
        service.enforce_workspace_role.assert_awaited_once_with("workspace-1", "user-1", "owner")

    async def test_download_forwards_range_and_conditionals_to_streaming_client(self) -> None:
        request = Request({"type": "http", "method": "GET", "path": "/", "headers": [(b"range", b"bytes=0-9"), (b"if-none-match", b"etag")]})
        user = SimpleNamespace(id="user-1")
        service = SimpleNamespace(enforce_workspace_role=mock.AsyncMock())
        response = object()
        download = mock.AsyncMock(return_value=response)
        service.download_workspace_object_storage_object = download

        with (
            mock.patch.object(routes, "userspace_service", service),
        ):
            result = await routes.download_workspace_object_storage_object("workspace-1", "uploads", "data/report.csv", request, user)

        self.assertIs(result, response)
        assert download.await_args is not None
        self.assertEqual(download.await_args.kwargs["headers"]["range"], "bytes=0-9")
        self.assertEqual(download.await_args.kwargs["headers"]["if-none-match"], "etag")
