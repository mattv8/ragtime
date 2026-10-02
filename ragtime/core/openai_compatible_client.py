"""Safe Chat Completions client options for arbitrary compatible endpoints."""

from __future__ import annotations

from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

_KEYLESS_SENTINEL = "openai-compatible-keyless"
_ENV_HEADER_NAMES = {"authorization", "openai-organization", "openai-project"}


def _strip_environment_credentials(request: httpx.Request, *, keyless: bool) -> None:
    for name in _ENV_HEADER_NAMES:
        if keyless or name != "authorization":
            request.headers.pop(name, None)


def _request_hook(*, keyless: bool, asynchronous: bool = False):
    def strip_environment_credentials(request: httpx.Request) -> None:
        _strip_environment_credentials(request, keyless=keyless)

    async def async_strip_environment_credentials(request: httpx.Request) -> None:
        _strip_environment_credentials(request, keyless=keyless)

    return async_strip_environment_credentials if asynchronous else strip_environment_credentials


def compatible_http_clients(api_key: str) -> tuple[httpx.Client, httpx.AsyncClient]:
    """Create redirect-safe clients which isolate SDK environment credentials."""
    options: dict[str, Any] = {"follow_redirects": False, "trust_env": True}
    return (
        httpx.Client(**options, event_hooks={"request": [_request_hook(keyless=not api_key)]}),
        httpx.AsyncClient(**options, event_hooks={"request": [_request_hook(keyless=not api_key, asynchronous=True)]}),
    )


def compatible_chat_options(api_key: str) -> dict[str, Any]:
    """Return SDK options that work without a key and never inherit SDK env state."""
    http_client, http_async_client = compatible_http_clients(api_key)
    return {
        "api_key": SecretStr(api_key or _KEYLESS_SENTINEL),
        "organization": "",
        "openai_proxy": None,
        "default_headers": {},
        "http_client": http_client,
        "http_async_client": http_async_client,
    }


def compatible_embedding_options(api_key: str) -> dict[str, Any]:
    """Return credential-isolated HTTP clients for compatible embeddings."""
    return compatible_chat_options(api_key)


class CompatibleChatOpenAI(ChatOpenAI):
    """Keep Chat Completions' documented ``max_tokens`` wire field."""

    def _get_request_payload(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = super()._get_request_payload(*args, **kwargs)
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload
