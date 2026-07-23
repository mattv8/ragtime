import unittest
from typing import Optional
from unittest import mock

from ragtime.mcp import oauth
from tests.asgi_test_utils import basic_auth_header, capture_response, form_receive


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


class McpOAuthMetadataTests(unittest.IsolatedAsyncioTestCase):
    def test_client_credentials_protected_resource_uses_route_issuer(self) -> None:
        metadata = oauth.build_protected_resource_metadata("https://ragtime.example", "cowork")

        self.assertEqual(metadata["resource"], "https://ragtime.example/mcp/cowork")
        self.assertEqual(metadata["authorization_servers"], ["https://ragtime.example/mcp/cowork"])
        self.assertNotIn(".well-known", metadata["authorization_servers"][0])

    def test_oauth2_custom_route_uses_app_level_authorization_metadata(self) -> None:
        metadata = oauth.build_interactive_protected_resource_metadata("https://ragtime.example", "workspace")

        self.assertEqual(metadata["resource"], "https://ragtime.example/mcp/workspace")
        self.assertEqual(metadata["authorization_servers"], ["https://ragtime.example"])

    async def test_metadata_handler_selects_route_scoped_client_credentials(self) -> None:
        with mock.patch.object(
            oauth,
            "_get_route_client_credentials",
            new=mock.AsyncMock(return_value=("client-a", "secret-a")),
        ):
            status, _headers, body = await capture_response(lambda send: oauth.handle_protected_resource_metadata(_scope(), _empty_receive, send, "cowork"))

        self.assertEqual(status, 200)
        self.assertEqual(body["resource"], "https://ragtime.example/mcp/cowork")
        self.assertEqual(body["authorization_servers"], ["https://ragtime.example/mcp/cowork"])

    async def test_metadata_handler_falls_back_for_oauth2_custom_route(self) -> None:
        with mock.patch.object(
            oauth,
            "_get_route_client_credentials",
            new=mock.AsyncMock(return_value=None),
        ):
            status, _headers, body = await capture_response(lambda send: oauth.handle_protected_resource_metadata(_scope(), _empty_receive, send, "workspace"))

        self.assertEqual(status, 200)
        self.assertEqual(body["resource"], "https://ragtime.example/mcp/workspace")
        self.assertEqual(body["authorization_servers"], ["https://ragtime.example"])

    async def test_token_errors_are_rfc6749_top_level_json(self) -> None:
        status, headers, body = await capture_response(
            lambda send: oauth.handle_token_request(
                _scope(method="POST"),
                form_receive(b"grant_type=password"),
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
            b"authorization": basic_auth_header("client-a", "secret-a").encode("ascii"),
        }
        with mock.patch.object(
            oauth,
            "_get_route_client_credentials",
            new=mock.AsyncMock(return_value=("client-a", "secret-a")),
        ):
            status, _headers, body = await capture_response(
                lambda send: oauth.handle_token_request(
                    _scope(method="POST", headers=headers),
                    form_receive(b"grant_type=client_credentials"),
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
