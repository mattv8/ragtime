"""Opt-in conformance checks for a disposable Ragtime object-storage gateway.

This suite is deliberately inert unless a caller supplies an isolated endpoint and
credentials. It never discovers or writes to the default Compose gateway.
"""

import os
import subprocess
import unittest
from typing import Any, ClassVar
from urllib.error import HTTPError
from urllib.request import urlopen

_RUN = os.environ.get("OBJECT_STORAGE_S3_CONTRACT_RUN") == "1"
_ENDPOINT = os.environ.get("OBJECT_STORAGE_S3_CONTRACT_ENDPOINT", "")
_ACCESS_KEY = os.environ.get("OBJECT_STORAGE_S3_CONTRACT_ACCESS_KEY", "")
_SECRET_KEY = os.environ.get("OBJECT_STORAGE_S3_CONTRACT_SECRET_KEY", "")
_SECOND_ACCESS_KEY = os.environ.get("OBJECT_STORAGE_S3_CONTRACT_SECOND_ACCESS_KEY", "")
_SECOND_SECRET_KEY = os.environ.get("OBJECT_STORAGE_S3_CONTRACT_SECOND_SECRET_KEY", "")


@unittest.skipUnless(
    _RUN and _ENDPOINT and _ACCESS_KEY and _SECRET_KEY and _SECOND_ACCESS_KEY and _SECOND_SECRET_KEY,
    "set OBJECT_STORAGE_S3_CONTRACT_RUN=1 and disposable endpoint/tenant credentials to run S3 conformance checks",
)
class ObjectStorageS3ContractTests(unittest.TestCase):
    s3: ClassVar[Any]
    other_s3: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import boto3  # type: ignore[import-untyped]
            from botocore.config import Config  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise unittest.SkipTest("boto3 must be installed for the S3 contract suite") from exc
        config = Config(s3={"addressing_style": "path"}, retries={"max_attempts": 2, "mode": "standard"})
        cls.s3 = boto3.client(
            "s3", endpoint_url=_ENDPOINT, region_name="us-east-1", aws_access_key_id=_ACCESS_KEY, aws_secret_access_key=_SECRET_KEY, config=config
        )
        cls.other_s3 = boto3.client(
            "s3", endpoint_url=_ENDPOINT, region_name="us-east-1", aws_access_key_id=_SECOND_ACCESS_KEY, aws_secret_access_key=_SECOND_SECRET_KEY, config=config
        )

    def test_signed_sdk_operations_and_tenant_isolation(self) -> None:
        bucket = self.s3.list_buckets()["Buckets"][0]["Name"]
        key = "contract/metadata.txt"
        self.s3.put_object(Bucket=bucket, Key=key, Body=b"contract", ContentType="text/plain", Metadata={"suite": "contract"})
        head = self.s3.head_object(Bucket=bucket, Key=key)
        self.assertEqual(head["ContentType"], "text/plain")
        self.assertEqual(head["Metadata"], {"suite": "contract"})
        self.s3.copy_object(Bucket=bucket, Key="contract/copied.txt", CopySource={"Bucket": bucket, "Key": key})
        listing = self.s3.list_objects_v2(Bucket=bucket, Prefix="contract/", MaxKeys=1)
        self.assertTrue(listing["IsTruncated"])
        self.assertTrue(listing.get("NextContinuationToken"))
        with self.assertRaises(Exception):
            self.other_s3.get_object(Bucket=bucket, Key=key)

    def test_unsigned_requests_are_denied(self) -> None:
        with self.assertRaises(HTTPError) as raised:
            urlopen(f"{_ENDPOINT.rstrip('/')}/", timeout=5)  # nosec B310: explicit disposable test endpoint
        self.assertIn(raised.exception.code, {400, 401, 403})

    def test_wrong_credentials_are_denied(self) -> None:
        import boto3  # type: ignore[import-untyped]
        from botocore.config import Config  # type: ignore[import-untyped]

        invalid = boto3.client(
            "s3",
            endpoint_url=_ENDPOINT,
            region_name="us-east-1",
            aws_access_key_id="invalid-contract-key",
            aws_secret_access_key="invalid-contract-secret",
            config=Config(s3={"addressing_style": "path"}),
        )
        with self.assertRaises(Exception):
            invalid.list_buckets()

    def test_multipart_upload_is_bound_to_the_tenant(self) -> None:
        bucket = self.s3.list_buckets()["Buckets"][0]["Name"]
        upload = self.s3.create_multipart_upload(Bucket=bucket, Key="contract/multipart.bin")
        upload_id = upload["UploadId"]
        part = self.s3.upload_part(Bucket=bucket, Key="contract/multipart.bin", UploadId=upload_id, PartNumber=1, Body=b"x" * (5 * 1024 * 1024))
        self.s3.complete_multipart_upload(
            Bucket=bucket,
            Key="contract/multipart.bin",
            UploadId=upload_id,
            MultipartUpload={"Parts": [{"PartNumber": 1, "ETag": part["ETag"]}]},
        )
        self.assertEqual(self.s3.get_object(Bucket=bucket, Key="contract/multipart.bin")["Body"].read(1), b"x")

    @unittest.skipUnless(
        os.environ.get("OBJECT_STORAGE_S3_CONTRACT_BENCHMARK") == "1", "set OBJECT_STORAGE_S3_CONTRACT_BENCHMARK=1 for disposable 10k-key pagination check"
    )
    def test_ten_thousand_key_prefix_returns_a_bounded_page(self) -> None:
        bucket = self.s3.list_buckets()["Buckets"][0]["Name"]
        for number in range(10_000):
            self.s3.put_object(Bucket=bucket, Key=f"contract/benchmark/{number:05d}", Body=b"")
        page = self.s3.list_objects_v2(Bucket=bucket, Prefix="contract/benchmark/", MaxKeys=100)
        self.assertLessEqual(len(page.get("Contents", [])), 100)
        self.assertTrue(page["IsTruncated"])

    @unittest.skipUnless(
        os.environ.get("OBJECT_STORAGE_S3_CONTRACT_NODE_COMMAND"), "set OBJECT_STORAGE_S3_CONTRACT_NODE_COMMAND to an isolated AWS SDK v3 fixture command"
    )
    def test_node_aws_sdk_v3_fixture(self) -> None:
        result = subprocess.run(
            os.environ["OBJECT_STORAGE_S3_CONTRACT_NODE_COMMAND"],
            shell=True,
            env=os.environ | {"OBJECT_STORAGE_S3_ENDPOINT": _ENDPOINT},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
