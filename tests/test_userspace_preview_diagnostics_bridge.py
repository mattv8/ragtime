from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi import HTTPException

from ragtime.indexer.tool_presentation import HIDDEN_TOOL_VISIBILITY, normalize_tool_presentation
from ragtime.userspace.models import UserSpacePreviewDiagnosticsRequest
from ragtime.userspace.preview_host import preview_diagnostics, preview_host_app
from ragtime.userspace.service import _RUNTIME_BRIDGE_VERSION, _build_bridge_content


class UserSpacePreviewDiagnosticsBridgeTests(unittest.IsolatedAsyncioTestCase):
    def test_preview_host_exposes_diagnostics_endpoint_before_proxy_catchall(self) -> None:
        paths = [getattr(route, "path", "") for route in preview_host_app.routes]
        self.assertIn("/__ragtime/diagnostics", paths)
        self.assertLess(paths.index("/__ragtime/diagnostics"), paths.index("/{path:path}"))

    def test_runtime_bridge_batches_get_post_network_diagnostics(self) -> None:
        content = _build_bridge_content()
        self.assertGreaterEqual(_RUNTIME_BRIDGE_VERSION, 18)
        self.assertIn("/__ragtime/diagnostics", content)
        self.assertIn("queueNetworkDiagnostic", content)
        self.assertIn("normalizeDiagnosticTarget", content)
        self.assertIn("method === 'GET' || method === 'POST'", content)
        self.assertIn("/__ragtime/execute-component", content)

    def test_userspace_diagnostics_tool_is_hidden_from_frontend_presentation(self) -> None:
        self.assertEqual(
            normalize_tool_presentation("userspace_diagnostics"),
            {"visibility": HIDDEN_TOOL_VISIBILITY},
        )

    async def test_preview_diagnostics_rejects_shared_preview_sessions(self) -> None:
        payload = UserSpacePreviewDiagnosticsRequest(events=[])
        with mock.patch(
            "ragtime.userspace.preview_host._verify_preview_session_cookie",
            mock.AsyncMock(return_value={"workspace_id": "workspace-1", "preview_mode": "shared_public_host"}),
        ):
            with self.assertRaises(HTTPException) as raised:
                await preview_diagnostics(SimpleNamespace(), payload)  # type: ignore[arg-type]

        self.assertEqual(raised.exception.status_code, 403)
