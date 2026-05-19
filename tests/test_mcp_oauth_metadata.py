import base64
import json
import unittest
from typing import Optional
from unittest import mock

from ragtime.mcp import oauth


def _basic(client_id: str, client_secret: str) -> str:
    raw = f"{client_id}:{client_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _scope(
    *,
    method: str = "GET",
    path: str = "/mcp/cowork",
    headers: Optional[dict[bytes, bytes]] = None,
) -> dict:
    return {
        "type": "http",
        "method": method,
        "scheme": "https",
        "path": path,
        "headers": list((headers or {b"host": b"ragtime.example"}).items()),
    }


async def _empty_receive() -> dict:
    return {"type": "http.request", "body": b"", "more_body": False}


def _form_receive(body: bytes):
    sent = False

    async def receive() -> dict:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


async def _capture_response(call) -> tuple[int, dict[str, str], dict]:
    messages: list[dict] = []

    async def send(message: dict) -> None:
        messages.append(message)

    await call(send)
    status = messages[0]["status"]
    headers = {
        key.decode("latin1"): value.decode("latin1")
        for key, value in messages[0].get("headers", [])
    }
    body = json.loads(messages[-1].get("body", b"{}"))
    return status, headers, body


class McpOAuthMetadataTests(unittest.IsolatedAsyncioTestCase):
    def test_client_credentials_protected_resource_uses_route_issuer(self) -> None:
        metadata = oauth.build_protected_resource_metadata(
            "https://ragtime.example", "cowork"
        )

        self.assertEqual(metadata["resource"], "https://ragtime.example/mcp/cowork")
        self.assertEqual(
            metadata["authorization_servers"], ["https://ragtime.example/mcp/cowork"]
        )
        self.assertNotIn(".well-known", metadata["authorization_servers"][0])

    def test_oauth2_custom_route_uses_app_level_authorization_metadata(self) -> None:
        metadata = oauth.build_interactive_protected_resource_metadata(
            "https://ragtime.example", "workspace"
        )

        self.assertEqual(metadata["resource"], "https://ragtime.example/mcp/workspace")
        self.assertEqual(metadata["authorization_servers"], ["https://ragtime.example"])

    async def test_metadata_handler_selects_route_scoped_client_credentials(self) -> None:
        with mock.patch.object(
            oauth,
            "_get_route_client_credentials",
            new=mock.AsyncMock(return_value=("client-a", "secret-a")),
        ):
            status, _headers, body = await _capture_response(
                lambda send: oauth.handle_protected_resource_metadata(
                    _scope(), _empty_receive, send, "cowork"
                )
            )

        self.assertEqual(status, 200)
        self.assertEqual(body["resource"], "https://ragtime.example/mcp/cowork")
        self.assertEqual(body["authorization_servers"], ["https://ragtime.example/mcp/cowork"])

    async def test_metadata_handler_falls_back_for_oauth2_custom_route(self) -> None:
        with mock.patch.object(
            oauth,
            "_get_route_client_credentials",
            new=mock.AsyncMock(return_value=None),
        ):
            status, _headers, body = await _capture_response(
                lambda send: oauth.handle_protected_resource_metadata(
                    _scope(), _empty_receive, send, "workspace"
                )
            )

        self.assertEqual(status, 200)
        self.assertEqual(body["resource"], "https://ragtime.example/mcp/workspace")
        self.assertEqual(body["authorization_servers"], ["https://ragtime.example"])

    async def test_token_errors_are_rfc6749_top_level_json(self) -> None:
        status, headers, body = await _capture_response(
            lambda send: oauth.handle_token_request(
                _scope(method="POST"),
                _form_receive(b"grant_type=password"),
                send,
                "cowork",
            )
        )

        self.assertEqual(status, 400)
        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(body["error"], "unsupported_grant_type")
        self.assertIn("client_credentials", body["error_description"])
        self.assertNotIn("detail", body)

    async def test_token_endpoint_accepts_basic_and_binds_bearer_to_route(self) -> None:
        headers = {
            b"host": b"ragtime.example",
            b"authorization": _basic("client-a", "secret-a").encode("ascii"),
        }
        with mock.patch.object(
            oauth,
            "_get_route_client_credentials",
            new=mock.AsyncMock(return_value=("client-a", "secret-a")),
        ):
            status, _headers, body = await _capture_response(
                lambda send: oauth.handle_token_request(
                    _scope(method="POST", headers=headers),
                    _form_receive(b"grant_type=client_credentials"),
                    send,
                    "cowork",
                )
            )

        self.assertEqual(status, 200)
        self.assertEqual(body["token_type"], "Bearer")
        self.assertEqual(body["scope"], "mcp:cowork")

        matching_scope = _scope(
            headers={
                b"host": b"ragtime.example",
                b"authorization": f"Bearer {body['access_token']}".encode("ascii"),
            }
        )
        self.assertTrue(oauth.validate_client_credentials_bearer(matching_scope, "cowork"))
        self.assertFalse(oauth.validate_client_credentials_bearer(matching_scope, "other"))
        self.assertEqual(matching_scope["_mcp_client_id"], "client-a")


if __name__ == "__main__":
    unittest.main()