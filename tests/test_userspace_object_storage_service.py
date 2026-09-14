import io
import unittest
from unittest import mock

from fastapi import HTTPException

from ragtime.userspace.object_storage import client
from ragtime.userspace.service import UserSpaceService


class _S3:
    def __init__(self) -> None:
        self.upload_fileobj = mock.Mock()
        self.head_object = mock.Mock(return_value={"ContentLength": 3, "ContentType": "text/plain"})
        self.list_objects_v2 = mock.Mock(return_value={"Contents": [], "IsTruncated": True, "NextContinuationToken": "next"})
        self.get_object = mock.Mock()


class ObjectStorageClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_list_forwards_continuation_token(self) -> None:
        s3 = _S3()
        with mock.patch.object(client, "_workspace_client", new=mock.AsyncMock(return_value=(s3, {"buckets": [{"name": "assets"}]}))):
            result = await client.list_objects("workspace", "assets", "public/", "cursor", 25)
        self.assertEqual("next", result["NextContinuationToken"])
        self.assertEqual("cursor", s3.list_objects_v2.call_args.kwargs["ContinuationToken"])
        self.assertEqual(25, s3.list_objects_v2.call_args.kwargs["MaxKeys"])

    async def test_upload_uses_file_object_and_returns_route_dto_fields(self) -> None:
        s3 = _S3()
        with mock.patch.object(client, "_workspace_client", new=mock.AsyncMock(return_value=(s3, {"buckets": [{"name": "assets"}]}))):
            result = await client.upload_file("workspace", "assets", "public/a.txt", io.BytesIO(b"abc"), "text/plain")
        self.assertEqual({"workspace_id": "workspace", "bucket_name": "assets", "key": "public/a.txt", "size_bytes": 3, "content_type": "text/plain"}, result)
        self.assertEqual("assets", s3.upload_fileobj.call_args.args[1])

    async def test_unknown_bucket_is_rejected_before_s3_call(self) -> None:
        s3 = _S3()
        with mock.patch.object(client, "_workspace_client", new=mock.AsyncMock(return_value=(s3, {"buckets": []}))):
            with self.assertRaisesRegex(Exception, "bucket"):
                await client.delete_object("workspace", "assets", "a.txt")
        s3.head_object.assert_not_called()

    async def test_not_modified_is_empty_304_response(self) -> None:
        from botocore.exceptions import ClientError  # type: ignore[import-untyped]

        s3 = _S3()
        s3.get_object = mock.Mock(
            side_effect=ClientError({"Error": {"Code": "304"}, "ResponseMetadata": {"HTTPStatusCode": 304, "HTTPHeaders": {"etag": "etag"}}}, "GetObject")
        )
        with mock.patch.object(client, "_workspace_client", new=mock.AsyncMock(return_value=(s3, {"buckets": [{"name": "assets"}]}))):
            response = await client.download_response("workspace", "assets", "a.txt", {"if-none-match": "etag"})
        self.assertEqual(304, response.status_code)
        self.assertEqual("etag", response.headers["etag"])


class ObjectStorageTransitionTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_gateway_workspace_imports_legacy_once(self) -> None:
        service = object.__new__(UserSpaceService)
        legacy = {"access_key_id": "legacy", "secret_access_key": "secret", "buckets": [{"name": "assets"}]}
        with (
            mock.patch(
                "ragtime.userspace.service.object_storage_control.get_workspace",
                new=mock.AsyncMock(side_effect=HTTPException(status_code=404, detail="missing")),
            ),
            mock.patch(
                "ragtime.userspace.service.object_storage_control.ensure_workspace", new=mock.AsyncMock(return_value={"buckets": [{"name": "assets"}]})
            ) as ensure,
            mock.patch("ragtime.userspace.service.object_storage_control.import_legacy", new=mock.AsyncMock()) as import_legacy,
            mock.patch.object(service, "_legacy_object_storage_payload", return_value=legacy),
        ):
            result = await service._ensure_managed_object_storage("workspace")
        self.assertEqual([{"name": "assets"}], result["buckets"])
        ensure.assert_awaited_once_with("workspace", legacy)
        import_legacy.assert_awaited_once_with("workspace")

    async def test_existing_gateway_workspace_does_not_import_legacy(self) -> None:
        service = object.__new__(UserSpaceService)
        config = {"buckets": [{"name": "assets"}]}
        with (
            mock.patch("ragtime.userspace.service.object_storage_control.get_workspace", new=mock.AsyncMock(return_value=config)),
            mock.patch("ragtime.userspace.service.object_storage_control.import_legacy", new=mock.AsyncMock()) as import_legacy,
        ):
            self.assertEqual(config, await service._ensure_managed_object_storage("workspace"))
        import_legacy.assert_not_awaited()

    async def test_failed_existing_legacy_import_is_retried_before_use(self) -> None:
        service = object.__new__(UserSpaceService)
        failed = {"state": "importing", "legacy_import_state": "failed", "buckets": [{"name": "assets"}]}
        complete = {**failed, "state": "ready", "legacy_import_state": "completed"}
        with (
            mock.patch("ragtime.userspace.service.object_storage_control.get_workspace", new=mock.AsyncMock(side_effect=[failed, complete])),
            mock.patch("ragtime.userspace.service.object_storage_control.import_legacy", new=mock.AsyncMock()) as retry,
        ):
            self.assertEqual(complete, await service._ensure_managed_object_storage("workspace"))
        retry.assert_awaited_once_with("workspace")

    async def test_failed_import_never_silently_unfences_or_returns_partial_data(self) -> None:
        service = object.__new__(UserSpaceService)
        with (
            mock.patch("ragtime.userspace.service.object_storage_control.get_workspace", new=mock.AsyncMock(return_value={"legacy_import_state": "failed"})),
            mock.patch(
                "ragtime.userspace.service.object_storage_control.import_legacy",
                new=mock.AsyncMock(side_effect=HTTPException(status_code=409, detail="legacy conflict")),
            ),
        ):
            with self.assertRaises(HTTPException) as failure:
                await service._ensure_managed_object_storage("workspace")
        self.assertEqual(failure.exception.status_code, 409)
