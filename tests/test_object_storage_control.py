import hashlib
import hmac
import unittest
from unittest import mock

from fastapi import HTTPException
from fastapi.routing import APIRoute

from ragtime.userspace.object_storage import admin_routes, control


class ObjectStorageControlTests(unittest.IsolatedAsyncioTestCase):
    def test_token_uses_managed_key_and_strips_whitespace(self):
        with mock.patch.object(control.settings, "encryption_key", " secret\n"):
            self.assertEqual(
                control._control_token(),
                hmac.new(b"secret", b"ragtime-object-storage-control-v1", hashlib.sha256).hexdigest(),
            )

    async def test_transport_failures_are_sanitized(self):
        client = mock.AsyncMock()
        client.request.side_effect = control.httpx.ConnectError("provider password=not-for-output")
        with mock.patch.object(control, "_get_client", return_value=client):
            with self.assertRaises(HTTPException) as raised:
                await control.request("GET", "/v1/settings")
        self.assertEqual(raised.exception.status_code, 503)
        self.assertNotIn("password", raised.exception.detail)

    async def test_admin_settings_strip_secrets(self):
        value = {
            "mode": "external",
            "endpoint": "http://example.invalid",
            "region": "us-east-1",
            "bucket": "private-root",
            "access_key_configured": True,
            "secret_key_configured": True,
            "access_key_id": "do-not-return",
            "secret_access_key": "do-not-return",
        }
        with mock.patch.object(control, "request", new=mock.AsyncMock(return_value=value)):
            result = await admin_routes.get_settings(None)
        self.assertNotIn("access_key_id", result)
        self.assertNotIn("secret_access_key", result)
        self.assertTrue(result["secret_key_configured"])

    def test_every_admin_endpoint_requires_admin(self):
        for route in admin_routes.router.routes:
            assert isinstance(route, APIRoute)
            dependencies = route.dependant.dependencies
            self.assertTrue(any(item.call is admin_routes.require_admin for item in dependencies), route.path)
