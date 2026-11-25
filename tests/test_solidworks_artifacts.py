import asyncio
import json
import os
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from ragtime.solidworks import artifacts


class ArtifactTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root_patch = patch.object(artifacts, "ARTIFACT_DIR", self.root / "store")
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    async def store(self, name='a\r\n"/\\b.step'):
        source = self.root / "source"
        source.write_bytes(b"123456")
        return await artifacts.store_artifact(
            source,
            file_name=name,
            kind="model",
            format="step",
            sha256="abc",
            owner_user_id="owner",
            tool_config_id="tool",
            surface="chat",
            mcp_route=None,
        )

    async def test_store_wire_expiry_cleanup(self):
        wire = await self.store()
        self.assertEqual(wire["file_name"], "ab.step")
        self.assertNotIn(str(self.root), json.dumps(wire))
        meta, path = await artifacts.load_artifact(wire["id"])
        self.assertEqual(path.read_bytes(), b"123456")
        self.assertFalse(meta["expired"])
        meta["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        (path.parent / "meta.json").write_text(json.dumps(meta))
        meta, _ = await artifacts.load_artifact(wire["id"])
        self.assertTrue(meta["expired"])
        with self.assertRaises(artifacts.ArtifactExpired):
            await artifacts.resolve_artifact_content(wire["id"], SimpleNamespace(id="owner", role="user"))
        self.assertEqual(await artifacts.cleanup_expired_artifacts(), 1)
        with self.assertRaises(artifacts.ArtifactNotFound):
            await artifacts.load_artifact(wire["id"])

    async def test_invalid_id_and_size(self):
        with self.assertRaises(artifacts.ArtifactNotFound):
            await artifacts.load_artifact("../other")
        with patch.object(artifacts, "MAX_ARTIFACT_BYTES", 5):
            with self.assertRaises(artifacts.ArtifactTooLarge):
                await self.store()
        self.assertFalse((self.root / "source").exists())

    async def test_sync_loader(self):
        wire = await self.store()
        meta, path = artifacts.load_artifact_sync(wire["id"])
        self.assertFalse(meta["expired"])
        self.assertEqual(path.read_bytes(), b"123456")
        meta["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        (path.parent / "meta.json").write_text(json.dumps(meta))
        self.assertTrue(artifacts.load_artifact_sync(wire["id"])[0]["expired"])
        with self.assertRaises(artifacts.ArtifactNotFound):
            artifacts.load_artifact_sync("../other")

    async def test_cleanup_stale_staging_and_symlinks(self):
        artifacts.ARTIFACT_DIR.mkdir()
        stale = artifacts.ARTIFACT_DIR / ("." + "a" * 32)
        fresh = artifacts.ARTIFACT_DIR / ("." + "b" * 32)
        stale.mkdir()
        fresh.mkdir()
        (stale / "content").write_bytes(b"abandoned")
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).timestamp()
        os.utime(stale, (old, old))
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "content").write_bytes(b"keep")
        (stale / "link").symlink_to(outside, target_is_directory=True)
        os.utime(stale, (old, old))
        (artifacts.ARTIFACT_DIR / ("." + "c" * 32)).symlink_to(outside, target_is_directory=True)
        self.assertEqual(await artifacts.cleanup_expired_artifacts(), 1)
        self.assertFalse(stale.exists())
        self.assertTrue(fresh.exists())
        self.assertEqual((outside / "content").read_bytes(), b"keep")

    async def test_cleanup_during_publication(self):
        ready = threading.Event()
        proceed = threading.Event()
        original_replace = artifacts.os.replace

        def pause_publication(source, destination):
            if Path(source).name.startswith("."):
                ready.set()
                if not proceed.wait(10):
                    raise TimeoutError("publication was not resumed")
            return original_replace(source, destination)

        with patch.object(artifacts.os, "replace", side_effect=pause_publication):
            task = asyncio.create_task(self.store())
            try:
                self.assertTrue(await asyncio.to_thread(ready.wait, 10))
                self.assertEqual(await artifacts.cleanup_expired_artifacts(), 0)
            finally:
                proceed.set()
            wire = await task
        _, path = await artifacts.load_artifact(wire["id"])
        self.assertEqual(path.read_bytes(), b"123456")

    async def test_access(self):
        wire = await self.store()
        meta, _ = await artifacts.load_artifact(wire["id"])
        await artifacts.authorize_artifact(meta, SimpleNamespace(id="owner", role="user"))
        await artifacts.authorize_artifact(meta, SimpleNamespace(id="other", role="admin"))
        user = SimpleNamespace(id="other", role="user")
        with (
            patch.object(artifacts.repository, "get_tool_config", AsyncMock(return_value=object())),
            patch.object(artifacts, "filter_tool_ids_by_access", AsyncMock(return_value=["tool"])),
        ):
            await artifacts.authorize_artifact(meta, user)
        with patch.object(artifacts.repository, "get_tool_config", AsyncMock(return_value=None)):
            with self.assertRaises(artifacts.ArtifactForbidden):
                await artifacts.authorize_artifact(meta, user)
        with (
            patch.object(artifacts.repository, "get_tool_config", AsyncMock(return_value=object())),
            patch.object(artifacts, "filter_tool_ids_by_access", AsyncMock(return_value=[])),
        ):
            with self.assertRaises(artifacts.ArtifactForbidden):
                await artifacts.authorize_artifact(meta, user)
