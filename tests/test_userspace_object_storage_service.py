import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
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
    async def asyncSetUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.settings = mock.patch("ragtime.userspace.service.settings.index_data_path", self.root)
        self.settings.start()
        self.service = UserSpaceService()
        self.workspace_id = "workspace"
        self.buckets = self.service._workspace_dir(self.workspace_id) / "s3" / "buckets" / "assets"
        self.buckets.mkdir(parents=True)
        self.service._workspace_object_storage_config_path(self.workspace_id).parent.mkdir(parents=True, exist_ok=True)
        self.service._workspace_object_storage_config_path(self.workspace_id).write_text(
            json.dumps({"access_key_id": "legacy", "buckets": [{"name": "assets"}]}), encoding="utf-8"
        )

    async def asyncTearDown(self) -> None:
        await self.service.shutdown_legacy_object_storage_reconciliation()
        self.settings.stop()
        self.tempdir.cleanup()

    async def test_bound_completed_gateway_job_resumes_gc_after_pending_poll(self) -> None:
        source = self.buckets / "a.txt"
        source.write_bytes(b"legacy")
        pending_config = {"state": "importing", "legacy_import_state": "pending"}
        complete_config = {"state": "ready", "legacy_import_state": "completed"}
        pending = {"state": "pending"}
        with (
            mock.patch("ragtime.userspace.service.object_storage_control.get_workspace", new=mock.AsyncMock(side_effect=[pending_config, complete_config])),
            mock.patch("ragtime.userspace.service.object_storage_control.ensure_workspace", new=mock.AsyncMock()),
            mock.patch("ragtime.userspace.service.object_storage_control.submit_legacy_import", new=mock.AsyncMock()) as submit,
            mock.patch(
                "ragtime.userspace.service.object_storage_control.get_legacy_import", new=mock.AsyncMock(side_effect=[pending, lambda workspace_id: None])
            ) as status,
            mock.patch("ragtime.userspace.service.object_storage_control.acknowledge_legacy_gc", new=mock.AsyncMock()) as acknowledge,
            mock.patch.object(self.service, "_legacy_object_storage_runtime_active", new=mock.AsyncMock(return_value=False)),
        ):
            first = await self.service._process_legacy_object_storage(self.workspace_id)
            receipt = self.service._legacy_object_storage_migrator._load_receipts(self.workspace_id)[0]
            status.side_effect = [
                {
                    "state": "completed",
                    "generation": receipt["generation"],
                    "manifest_sha256": receipt["manifest_sha256"],
                    "verified_files": ["buckets/assets/a.txt"],
                }
            ]
            second = await self.service._process_legacy_object_storage(self.workspace_id)
        self.assertFalse(first)
        self.assertTrue(second)
        self.assertFalse(source.exists())
        self.assertGreaterEqual(submit.await_count, 1)
        acknowledge.assert_awaited_once()

    async def test_pending_workspace_does_not_block_second_loop_tick_and_releases_before_staging_it(self) -> None:
        second = "second"
        second_source = self.service._workspace_dir(second) / "s3" / "buckets" / "assets" / "b.txt"
        second_source.parent.mkdir(parents=True)
        second_source.write_bytes(b"second")
        second_config = self.service._workspace_object_storage_config_path(second)
        second_config.parent.mkdir(parents=True, exist_ok=True)
        second_config.write_text(json.dumps({"buckets": [{"name": "assets"}]}), encoding="utf-8")
        (self.buckets / "a.txt").write_bytes(b"first")
        states = {self.workspace_id: "pending", second: "pending"}

        async def workspace_config(workspace_id: str) -> dict[str, str]:
            return {"state": "ready" if states[workspace_id] == "completed" else "importing", "legacy_import_state": states[workspace_id]}

        async def import_status(workspace_id: str) -> dict[str, object]:
            if states[workspace_id] != "completed":
                return {"state": "pending"}
            receipt = self.service._legacy_object_storage_migrator._load_receipts(workspace_id)[0]
            return {
                "state": "completed",
                "generation": receipt["generation"],
                "manifest_sha256": receipt["manifest_sha256"],
                "verified_files": [item["path"] for item in receipt["manifest"]["files"]],
            }

        with (
            mock.patch("ragtime.userspace.service.object_storage_control.get_workspace", new=mock.AsyncMock(side_effect=workspace_config)),
            mock.patch("ragtime.userspace.service.object_storage_control.submit_legacy_import", new=mock.AsyncMock()),
            mock.patch("ragtime.userspace.service.object_storage_control.get_legacy_import", new=mock.AsyncMock(side_effect=import_status)),
            mock.patch("ragtime.userspace.service.object_storage_control.acknowledge_legacy_gc", new=mock.AsyncMock()),
            mock.patch.object(self.service, "_legacy_object_storage_runtime_active", new=mock.AsyncMock(return_value=False)),
        ):
            self.assertFalse(await self.service._process_legacy_object_storage(self.workspace_id))
            self.assertFalse(await asyncio.wait_for(self.service._process_legacy_object_storage(second), timeout=0.1))
            self.assertFalse(self.service._legacy_object_storage_migrator._load_receipts(second))
            states[self.workspace_id] = "completed"
            self.assertTrue(await self.service._process_legacy_object_storage(self.workspace_id))
            self.assertFalse(await self.service._process_legacy_object_storage(second))
        self.assertTrue(self.service._legacy_object_storage_migrator._load_receipts(second))

    async def test_shutdown_drains_lazy_worker_without_startup_loop(self) -> None:
        async def wait_for_shutdown() -> None:
            await asyncio.Event().wait()

        worker = asyncio.create_task(wait_for_shutdown())
        self.service._legacy_object_storage_workspace_tasks[self.workspace_id] = worker
        await self.service.shutdown_legacy_object_storage_reconciliation()
        self.assertTrue(worker.done())

    async def test_orphan_legacy_directory_is_reportable_without_deletion(self) -> None:
        orphan = self.service._workspace_dir("orphan") / "s3" / "buckets" / "assets" / "keep.txt"
        orphan.parent.mkdir(parents=True)
        orphan.write_bytes(b"keep")
        self.assertEqual({"orphan"}, self.service._legacy_object_storage_orphan_ids({self.workspace_id}))
        self.assertTrue(orphan.exists())

    async def test_gateway_pending_fences_access_without_local_legacy_files(self) -> None:
        (self.service._workspace_object_storage_config_path(self.workspace_id)).unlink()
        with mock.patch(
            "ragtime.userspace.service.object_storage_control.get_workspace",
            new=mock.AsyncMock(return_value={"state": "importing", "legacy_import_state": "pending"}),
        ):
            with self.assertRaises(HTTPException) as failure:
                await self.service._ensure_managed_object_storage(self.workspace_id)
        self.assertEqual(503, failure.exception.status_code)
