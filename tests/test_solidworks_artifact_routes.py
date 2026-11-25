import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI

from ragtime.core.security import get_current_user
from ragtime.solidworks import artifact_routes, artifacts


class ArtifactRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        root_patch = patch.object(artifacts, "ARTIFACT_DIR", root / "store")
        root_patch.start()
        self.addCleanup(root_patch.stop)
        source = root / "source"
        source.write_bytes(b"0123456789")
        self.wire = await artifacts.store_artifact(
            source,
            file_name='bad\r\n".step',
            kind="model",
            format="step",
            sha256="abc",
            owner_user_id="owner",
            tool_config_id="tool",
            surface="chat",
            mcp_route=None,
        )
        self.app = FastAPI()
        self.app.include_router(artifact_routes.router)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)
        self.url = f"{artifacts.ARTIFACT_URL_PREFIX}/{self.wire['id']}"
        self.tool_patch = patch.object(artifact_routes.repository, "get_tool_config", AsyncMock(return_value=None))
        self.tool_patch.start()
        self.addCleanup(self.tool_patch.stop)

    async def test_auth_metadata_content_range_and_expiry(self):
        self.assertEqual((await self.client.get(self.url)).status_code, 401)
        self.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="owner", role="user")
        response = await self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["expired"])
        response = await self.client.get(self.url + "/content", headers={"Range": "bytes=0-2"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.content, b"012")
        self.assertNotIn("\r", response.headers["content-disposition"])
        self.assertEqual((await self.client.get(artifacts.ARTIFACT_URL_PREFIX + "/invalid")).status_code, 404)
        meta, path = await artifacts.load_artifact(self.wire["id"])
        meta["expires_at"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        (path.parent / "meta.json").write_text(json.dumps(meta))
        self.assertTrue((await self.client.get(self.url)).json()["expired"])
        self.assertEqual((await self.client.get(self.url + "/content")).status_code, 410)
        self.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="other", role="user")
        self.assertEqual((await self.client.get(self.url)).status_code, 403)

    async def test_main_app_routes_non_admin_before_indexer(self):
        from ragtime.main import app

        app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="owner", role="user")
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                self.assertEqual((await client.get(self.url)).status_code, 200)
        finally:
            app.dependency_overrides.pop(get_current_user, None)
