import unittest

from ragtime.rag.prompts import build_userspace_object_storage_prompt_fragment


class UserSpaceObjectStoragePromptTests(unittest.TestCase):
    def test_sdk_guidance_keeps_credentials_backend_only(self) -> None:
        prompt = build_userspace_object_storage_prompt_fragment(
            object_storage_enabled=True,
            buckets=[{"name": "assets", "public_root": "public", "private_root": "private"}],
        )

        self.assertIn("@aws-sdk/client-s3", prompt)
        self.assertIn("boto3.client", prompt)
        self.assertIn("forcePathStyle: true", prompt)
        self.assertIn("Browser code must call the app's authenticated routes", prompt)
        self.assertIn("never an S3 endpoint with credentials", prompt)
