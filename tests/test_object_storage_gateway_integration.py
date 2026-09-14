"""Explicitly opted-in tests; all writes target disposable gateway/upstream fixtures."""

import hashlib
import hmac
import os
import time
import unittest
import uuid
from typing import Any

import httpx


@unittest.skipUnless(os.environ.get("OBJECT_STORAGE_GATEWAY_INTEGRATION") == "1", "requires disposable gateway and upstream")
class GatewayIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        import boto3  # type: ignore[import-untyped]
        from botocore.config import Config  # type: ignore[import-untyped]

        self.boto3 = boto3
        self.sdk_config = Config(signature_version="s3v4", s3={"addressing_style": "path"}, retries={"max_attempts": 1})
        self.endpoint = os.environ["OBJECT_STORAGE_S3_CONTRACT_ENDPOINT"]
        token = hmac.new(os.environ["OBJECT_STORAGE_TEST_KEY"].encode(), b"ragtime-object-storage-control-v1", hashlib.sha256).hexdigest()
        self.control = httpx.Client(base_url=os.environ["OBJECT_STORAGE_TEST_CONTROL"], headers={"Authorization": f"Bearer {token}"}, timeout=30)
        self.addCleanup(self.control.close)

    def request(self, method, path, payload=None):
        response = self.control.request(method, path, json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def workspace(self):
        self.request("PUT", "/v1/settings", {"mode": "local"})
        workspace_id = f"integration-{uuid.uuid4().hex}"
        config = self.request("POST", f"/v1/workspaces/{workspace_id}/ensure", {"default_bucket_name": "uploads"})
        sdk = self.boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            region_name="us-east-1",
            config=self.sdk_config,
            aws_access_key_id=config["access_key_id"],
            aws_secret_access_key=config["secret_access_key"],
        )
        self.addCleanup(sdk.close)
        return workspace_id, sdk

    def test_verified_migration_preserves_bytes_metadata_identity_and_source(self):
        workspace_id, sdk = self.workspace()
        content = b"verified migration body" * 1024
        sdk.put_object(Bucket="uploads", Key="asset.txt", Body=content, ContentType="text/plain", Metadata={"owner": "fixture"}, CacheControl="max-age=30")
        local_config = self.request("GET", f"/v1/workspaces/{workspace_id}")
        settings = self.request(
            "PUT",
            "/v1/settings",
            {
                "mode": "external",
                "endpoint": os.environ["OBJECT_STORAGE_TEST_UPSTREAM"],
                "region": "us-east-1",
                "bucket": "migration-fixture",
                "access_key_id": "fixture-upstream",
                "secret_access_key": "fixture-upstream-secret",
                "create_bucket": True,
            },
        )
        self.assertTrue(settings["secret_key_configured"])
        self.assertNotIn("secret_access_key", settings)
        jobs = self.request("POST", "/v1/migrations", {"workspace_ids": [workspace_id]})["jobs"]
        self.assertEqual(len(jobs), 1)
        deadline = time.monotonic() + 30
        while True:
            job = next(item for item in self.request("GET", "/v1/migrations")["jobs"] if item["id"] == jobs[0]["id"])
            if job["state"] in {"failed", "completed"}:
                break
            self.assertLess(time.monotonic(), deadline, "migration timed out")
            time.sleep(0.05)
        self.assertEqual(job["state"], "completed", job)
        current = self.request("GET", f"/v1/workspaces/{workspace_id}")
        self.assertEqual(current["access_key_id"], local_config["access_key_id"])
        self.assertEqual(current["buckets"][0]["id"], local_config["buckets"][0]["id"])
        self.assertEqual(current["buckets"][0]["backend_id"], settings["default_backend_id"])
        response = sdk.get_object(Bucket="uploads", Key="asset.txt")
        try:
            self.assertEqual(response["Body"].read(), content)
        finally:
            response["Body"].close()
        self.assertEqual(response["Metadata"], {"owner": "fixture"})
        self.assertEqual(response["ContentType"], "text/plain")
        self.assertEqual(response["CacheControl"], "max-age=30")

    def test_managed_rename_and_backup_fence(self) -> None:
        workspace_id, sdk = self.workspace()
        sdk.put_object(Bucket="uploads", Key="before", Body=b"original", Metadata={"tag": "preserved"})
        result = self.request("POST", f"/v1/workspaces/{workspace_id}/buckets/uploads/rename", {"key": "before", "new_key": "after"})
        self.assertEqual(result["size_bytes"], 8)
        self.assertEqual(sdk.head_object(Bucket="uploads", Key="after")["Metadata"], {"tag": "preserved"})
        from botocore.exceptions import ClientError  # type: ignore[import-untyped]

        with self.assertRaises(ClientError) as missing:
            sdk.head_object(Bucket="uploads", Key="before")
        self.assertEqual(missing.exception.response["ResponseMetadata"]["HTTPStatusCode"], 404)
        sdk.put_object(Bucket="uploads", Key="occupied", Body=b"other")
        conflict = self.control.post(f"/v1/workspaces/{workspace_id}/buckets/uploads/rename", json={"key": "after", "new_key": "occupied"})
        self.assertEqual(conflict.status_code, 409)
        self.assertEqual(sdk.head_object(Bucket="uploads", Key="after")["ContentLength"], 8)
        lease = self.request("POST", "/v1/backup/prepare", {})
        try:
            self.assertTrue(lease["consistent"])
            self.assertEqual(sdk.head_object(Bucket="uploads", Key="after")["ContentLength"], 8)
        finally:
            self.request("POST", "/v1/backup/release", {"lease_id": lease["lease_id"]})
        sdk.put_object(Bucket="uploads", Key="after-release", Body=b"ok")
