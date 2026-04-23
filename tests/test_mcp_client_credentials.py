"""
Smoke test for the OAuth 2.0 client_credentials auth method on MCP routes.

Runs against a live Ragtime dev stack (docker compose up) at
``RAGTIME_BASE_URL`` (default ``http://localhost:8000``). Requires local admin
credentials supplied via ``LOCAL_ADMIN_USER`` / ``LOCAL_ADMIN_PASSWORD``
environment variables (matching the dev .env).

Covers:
- Provisioning a custom MCP route with ``auth_method=client_credentials``.
- Rejection of anonymous requests (401).
- Direct HTTP Basic auth on MCP initialize (accepted).
- Token endpoint exchange returning a JWT bearer token.
- Bearer-token auth on MCP initialize (accepted).
- Bearer token bound to the route via ``aud`` (rejected on a different route).
- Invalid client credentials rejected (401 at token endpoint).
- Authorization Server discovery metadata is served.

Usage (from the repo root):

    docker exec ragtime-dev python -m pytest -xvs \
        tests/test_mcp_client_credentials.py

or directly:

    docker exec ragtime-dev python tests/test_mcp_client_credentials.py
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import sys
import uuid
from typing import Any

import httpx

BASE_URL = os.environ.get("RAGTIME_BASE_URL", "http://localhost:8000")
ADMIN_USER = os.environ.get("LOCAL_ADMIN_USER")
ADMIN_PASSWORD = os.environ.get("LOCAL_ADMIN_PASSWORD")

INIT_BODY = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "mcp-smoke-test", "version": "0.0.1"},
    },
}


def _basic(client_id: str, client_secret: str) -> str:
    raw = f"{client_id}:{client_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _admin_login(client: httpx.Client) -> None:
    _require(
        bool(ADMIN_USER and ADMIN_PASSWORD),
        "LOCAL_ADMIN_USER / LOCAL_ADMIN_PASSWORD must be set in the environment",
    )
    resp = client.post(
        f"{BASE_URL}/auth/login",
        json={"username": ADMIN_USER, "password": ADMIN_PASSWORD},
    )
    _require(
        resp.status_code == 200, f"Admin login failed: {resp.status_code} {resp.text}"
    )


def _create_route(
    client: httpx.Client, *, route_path: str, client_id: str, client_secret: str
) -> str:
    payload = {
        "name": f"Smoke Test {route_path}",
        "route_path": route_path,
        "description": "Automated client_credentials smoke test route",
        "require_auth": True,
        "auth_method": "client_credentials",
        "auth_client_id": client_id,
        "auth_password": client_secret,
        "include_knowledge_search": False,
        "include_git_history": False,
        "selected_document_indexes": [],
        "selected_filesystem_indexes": [],
        "selected_schema_indexes": [],
        "tool_config_ids": [],
    }
    resp = client.post(f"{BASE_URL}/mcp-routes", json=payload)
    _require(
        resp.status_code == 200,
        f"Route create failed: {resp.status_code} {resp.text}",
    )
    data = resp.json()
    return data["id"]


def _delete_route(client: httpx.Client, route_id: str) -> None:
    try:
        client.delete(f"{BASE_URL}/mcp-routes/{route_id}")
    except Exception:
        pass


def _mcp_init(
    client: httpx.Client, route_path: str, *, auth: str | None
) -> httpx.Response:
    headers = {
        "content-type": "application/json",
        "accept": "application/json, text/event-stream",
    }
    if auth:
        headers["authorization"] = auth
    return client.post(
        f"{BASE_URL}/mcp/{route_path}",
        content=json.dumps(INIT_BODY),
        headers=headers,
    )


def _token(
    client: httpx.Client,
    route_path: str,
    *,
    client_id: str,
    client_secret: str,
    auth_style: str = "basic",
) -> httpx.Response:
    url = f"{BASE_URL}/mcp/{route_path}/oauth/token"
    if auth_style == "basic":
        return client.post(
            url,
            data={"grant_type": "client_credentials"},
            headers={"authorization": _basic(client_id, client_secret)},
        )
    return client.post(
        url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
    )


def run() -> int:
    """Execute the smoke test. Returns a process exit code."""

    route_a = f"smoke_cc_{uuid.uuid4().hex[:8]}"
    route_b = f"smoke_cc_{uuid.uuid4().hex[:8]}"
    client_id_a = f"cid-{secrets.token_hex(4)}"
    client_secret_a = secrets.token_urlsafe(24)
    client_id_b = f"cid-{secrets.token_hex(4)}"
    client_secret_b = secrets.token_urlsafe(24)

    results: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, note: str = "") -> None:
        results.append((name, ok, note))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}{' - ' + note if note else ''}")

    with httpx.Client(timeout=30.0) as client:
        _admin_login(client)
        route_a_id = _create_route(
            client,
            route_path=route_a,
            client_id=client_id_a,
            client_secret=client_secret_a,
        )
        route_b_id = _create_route(
            client,
            route_path=route_b,
            client_id=client_id_b,
            client_secret=client_secret_b,
        )
        try:
            # 1. Anonymous initialize -> 401
            r = _mcp_init(client, route_a, auth=None)
            record(
                "anonymous request rejected",
                r.status_code == 401,
                f"status={r.status_code}",
            )

            # 2. Basic auth initialize -> 200
            r = _mcp_init(client, route_a, auth=_basic(client_id_a, client_secret_a))
            record(
                "basic auth accepted on MCP init",
                r.status_code in (200, 202),
                f"status={r.status_code}",
            )

            # 3. Wrong basic auth -> 401
            r = _mcp_init(client, route_a, auth=_basic(client_id_a, "not-the-secret"))
            record(
                "wrong client_secret rejected",
                r.status_code == 401,
                f"status={r.status_code}",
            )

            # 4. Token endpoint returns JWT
            r = _token(
                client,
                route_a,
                client_id=client_id_a,
                client_secret=client_secret_a,
            )
            record(
                "token endpoint issues JWT (basic)",
                r.status_code == 200 and "access_token" in r.json(),
                f"status={r.status_code} body={r.text[:120]}",
            )
            token_payload = r.json() if r.status_code == 200 else {}
            access_token = token_payload.get("access_token", "")
            _require(
                bool(access_token),
                "token endpoint did not return access_token - aborting",
            )

            # 5. Token endpoint via form body
            r_form = _token(
                client,
                route_a,
                client_id=client_id_a,
                client_secret=client_secret_a,
                auth_style="form",
            )
            record(
                "token endpoint issues JWT (form body)",
                r_form.status_code == 200 and "access_token" in r_form.json(),
                f"status={r_form.status_code}",
            )

            # 6. Bearer token initialize -> 200
            r = _mcp_init(client, route_a, auth=f"Bearer {access_token}")
            record(
                "bearer token accepted on MCP init",
                r.status_code in (200, 202),
                f"status={r.status_code}",
            )

            # 7. Bearer token rejected on a different route (aud binding)
            r = _mcp_init(client, route_b, auth=f"Bearer {access_token}")
            record(
                "bearer token rejected on other route (aud binding)",
                r.status_code == 401,
                f"status={r.status_code}",
            )

            # 8. Invalid credentials at token endpoint -> 401 invalid_client
            r = _token(
                client,
                route_a,
                client_id=client_id_a,
                client_secret="not-the-secret",
            )
            body: dict[str, Any] = {}
            try:
                body = r.json()
            except Exception:
                pass
            record(
                "token endpoint rejects bad credentials",
                r.status_code == 401 and body.get("error") == "invalid_client",
                f"status={r.status_code} body={body}",
            )

            # 9. AS metadata discovery
            r = client.get(
                f"{BASE_URL}/mcp/{route_a}/.well-known/oauth-authorization-server"
            )
            md: dict[str, Any] = {}
            try:
                md = r.json()
            except Exception:
                pass
            record(
                "authorization-server metadata reachable",
                r.status_code == 200
                and "client_credentials" in (md.get("grant_types_supported") or [])
                and route_a in (md.get("token_endpoint") or ""),
                f"status={r.status_code} token_endpoint={md.get('token_endpoint')}",
            )
        finally:
            _delete_route(client, route_a_id)
            _delete_route(client, route_b_id)

    failed = [r for r in results if not r[1]]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(run())
