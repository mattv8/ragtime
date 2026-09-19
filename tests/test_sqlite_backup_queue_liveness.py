from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from uuid import uuid4

from ragtime.userspace import sqlite_backup_queue as queue


class _RecoveryStore:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.interrupted: list[str] = []

    async def stale_running(self, **kwargs):
        return [{"id": self.job_id, "owner_token": "dead-owner"}]

    async def interrupt(self, job_id: str, owner_token: str, **kwargs):
        self.interrupted.append(job_id)
        return True


class SqliteBackupQueueLivenessTests(unittest.IsolatedAsyncioTestCase):
    async def test_recovery_does_not_interrupt_when_another_process_holds_job_lock(self) -> None:
        job_id = str(uuid4())
        store = _RecoveryStore(job_id)
        service = queue.SqliteBackupQueueService(store)
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(queue.settings, "index_data_path", temp):
            fd = queue._try_job_lock(job_id)
            assert fd is not None
            try:
                self.assertEqual([], await service.recover_stale())
            finally:
                queue._release_job_lock(fd)
            self.assertEqual([job_id], await service.recover_stale())
        self.assertEqual([job_id], store.interrupted)

    async def test_child_inherits_context_job_fd(self) -> None:
        job_id = str(uuid4())
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(queue.settings, "index_data_path", temp):
            fd = queue._try_job_lock(job_id)
            assert fd is not None
            try:
                # A real subprocess observes the inherited descriptor; this is
                # the fence that survives a controller process crash.
                from ragtime.userspace.sqlite_capture_admission import inherit_capture_fds, run_admitted_subprocess

                with inherit_capture_fds((fd,)):
                    result = run_admitted_subprocess([sys.executable, "-c", "import os,sys; os.fstat(int(sys.argv[1]))", str(fd)], check=True)
                self.assertEqual(0, result.returncode)
            finally:
                queue._release_job_lock(fd)
