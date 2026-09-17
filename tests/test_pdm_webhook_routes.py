import unittest
from unittest import mock

from fastapi import HTTPException
from starlette.requests import Request

from ragtime.pdm_automation.routes import receive_pdm_webhook


def _request(body=b"{}", authorization="Bearer secret"):
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": "POST", "path": "/webhooks/pdm/id", "headers": [(b"authorization", authorization.encode())]}, receive)


class PdmWebhookRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_credentials_are_rejected_before_body_parse(self):
        target = {"webhook_secret": "enc::secret", "enabled": True, "webhook_paused": False, "tool_config_id": "tool-1"}
        with mock.patch("ragtime.pdm_automation.routes.pdm_automation_repository.resolve_webhook", mock.AsyncMock(return_value=target)):
            with self.assertRaises(HTTPException) as raised:
                await receive_pdm_webhook("id", _request(b"not json", "Bearer wrong"))
        self.assertEqual(raised.exception.status_code, 401)
