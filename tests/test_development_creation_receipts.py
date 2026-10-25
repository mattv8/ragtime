from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import TypedDict
from unittest import mock

from ragtime.userspace.development_creation_receipts import (
    MAX_CREATION_RECEIPT_BYTES,
    publish_completion_receipt,
    read_completion_receipt,
    receipt_path,
)


class ReceiptIdentity(TypedDict):
    workspace_id: str
    request_id: str
    user_id: str


class DevelopmentCreationReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.workspace_dir = Path(self.temp_dir.name)
        self.expected: ReceiptIdentity = {
            "workspace_id": "workspace-1",
            "request_id": "request-1",
            "user_id": "user-1",
        }

    def test_published_receipt_is_readable_only_for_its_exact_identity(self) -> None:
        publish_completion_receipt(self.workspace_dir, **self.expected)
        self.assertIsNotNone(read_completion_receipt(self.workspace_dir, **self.expected))
        self.assertIsNone(read_completion_receipt(self.workspace_dir, workspace_id="other", request_id="request-1", user_id="user-1"))

    def test_symlink_and_oversized_receipts_are_not_evidence(self) -> None:
        target = self.workspace_dir / "target"
        target.write_text("{}")
        os.symlink(target, receipt_path(self.workspace_dir))
        self.assertIsNone(read_completion_receipt(self.workspace_dir, **self.expected))
        receipt_path(self.workspace_dir).unlink()
        receipt_path(self.workspace_dir).write_bytes(b"x" * (MAX_CREATION_RECEIPT_BYTES + 1))
        self.assertIsNone(read_completion_receipt(self.workspace_dir, **self.expected))

    def test_fifo_receipt_is_rejected_without_blocking(self) -> None:
        fifo = receipt_path(self.workspace_dir)
        os.mkfifo(fifo)
        self.assertIsNone(read_completion_receipt(self.workspace_dir, **self.expected))

    def test_directory_fsync_failure_after_rename_leaves_receipt_as_evidence(self) -> None:
        original_fsync = os.fsync
        calls = 0

        def fail_directory_fsync(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("directory fsync failed")
            original_fsync(descriptor)

        with mock.patch("ragtime.userspace.development_creation_receipts.os.fsync", side_effect=fail_directory_fsync):
            with self.assertRaises(OSError):
                publish_completion_receipt(self.workspace_dir, **self.expected)
        self.assertIsNotNone(read_completion_receipt(self.workspace_dir, **self.expected))
