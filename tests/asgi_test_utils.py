"""Shared helpers for testing ASGI handlers without spinning up a server."""

from __future__ import annotations

import base64
import json
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.types import Message, Receive, Send

AppCall = Callable[[Send], Awaitable[None]]


def basic_auth_header(client_id: str, client_secret: str) -> str:
    """Return a Basic Authorization header string for the given credentials."""
    raw = f"{client_id}:{client_secret}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def form_receive(body: bytes) -> Receive:
    """Build an ASGI receive callable that yields a single request body."""
    sent = False

    async def receive() -> Message:
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


async def capture_response(call: AppCall) -> tuple[int, dict[str, str], dict[str, Any]]:
    """Capture an ASGI app call and return (status, headers, parsed JSON body)."""
    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    await call(send)
    status = messages[0]["status"]
    headers = {key.decode("latin1"): value.decode("latin1") for key, value in messages[0].get("headers", [])}
    body = json.loads(messages[-1].get("body", b"{}"))
    return status, headers, body
