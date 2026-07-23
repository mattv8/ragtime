from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlencode, urlsplit

import httpx

from ragtime.config.settings import settings
from ragtime.core.logging import get_logger
from ragtime.http_api.models import (
    DEFAULT_HTTP_API_METHOD_POLICIES,
    HTTP_API_SECRET_FIELDS,
    HttpApiAuthMode,
    HttpApiConnectionConfig,
    HttpApiExecutionResult,
    HttpApiMethodPolicy,
    HttpApiRequest,
    HttpApiValidationResult,
)
from ragtime.http_api.security import (
    HttpApiSecurityError,
    PinnedAsyncTransport,
    build_pinned_base_url,
    build_trusted_configured_headers,
    normalize_relative_path,
    sanitize_outbound_headers,
    validate_base_url,
    validate_resolved_addresses,
    validate_trusted_header_name,
)

_REQUEST_BODY_LIMIT = 256 * 1024
_RESPONSE_BODY_LIMIT = 1024 * 1024
_OPENAPI_RESPONSE_LIMIT = 2 * 1024 * 1024
_DEFAULT_TOKEN_TTL_SECONDS = 300
_TOKEN_CACHE_MAX_SIZE = 128
_SELECTOR_FORBIDDEN_TOKENS = ("*", "[", "]", "..", "$", "?", "(", ")")

HostnameResolver = Callable[[str], list[str] | Awaitable[list[str]]]


@dataclass(frozen=True)
class _ResolvedTarget:
    original_host: str
    pinned_host: str
    port: int | None
    scheme: str
    base_url: str


def _hash_config(config: HttpApiConnectionConfig) -> str:
    payload = json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sanitize_json_value(value: Any) -> Any:
    secret_names = {field.lower() for field in HTTP_API_SECRET_FIELDS} | {"authorization"}
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in secret_names:
                continue
            sanitized[str(key)] = _sanitize_json_value(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]
    return value


def _validate_selector(selector: str) -> list[str]:
    if not selector:
        return []
    if any(token in selector for token in _SELECTOR_FORBIDDEN_TOKENS):
        raise ValueError("Response selector contains unsupported syntax")
    parts = selector.split(".")
    if any(part == "" for part in parts):
        raise ValueError("Response selector contains empty segments")
    for part in parts:
        if part.isdigit():
            continue
        if not all(ch.isalnum() or ch in {"_", "-"} for ch in part):
            raise ValueError("Response selector contains unsupported syntax")
    return parts


def _select_json_value(value: Any, selector: str) -> Any:
    selected = value
    for segment in _validate_selector(selector):
        if isinstance(selected, list):
            if not segment.isdigit():
                raise ValueError("Response selector segment must be a list index")
            selected = selected[int(segment)]
        elif isinstance(selected, dict):
            if segment not in selected:
                raise ValueError(f"Response selector segment not found: {segment}")
            selected = selected[segment]
        else:
            raise ValueError("Response selector reached a scalar before completion")
    return selected


def _normalize_rows_and_columns(value: Any, max_results: int) -> tuple[list[dict[str, Any]], list[str], int | None]:
    if isinstance(value, list):
        rows = [item for item in value if isinstance(item, dict)][:max_results]
        columns = sorted({key for row in rows for key in row.keys()}) if rows else []
        return rows, columns, len(rows) if rows or value == [] else None
    if isinstance(value, dict):
        rows = [dict(value)][:max_results]
        return rows, list(value.keys()), len(rows)
    return [], [], None


def _join_base_and_path(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    suffix = path if path.startswith("/") else f"/{path}"
    return f"{base}{suffix}"


async def resolve_http_api_hostname(hostname: str) -> list[str]:
    loop = asyncio.get_running_loop()
    results = await loop.getaddrinfo(
        hostname,
        None,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    addresses: list[str] = []
    seen: set[str] = set()
    for family, socktype, proto, canonname, sockaddr in results:
        del family, socktype, proto, canonname
        address = str(sockaddr[0])
        if address not in seen:
            seen.add(address)
            addresses.append(address)
    return addresses


class HttpApiBroker:
    def __init__(
        self,
        *,
        resolver: HostnameResolver | None = None,
        base_transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] | None = None,
        client_factory: type[httpx.AsyncClient] = httpx.AsyncClient,
        logger=None,
    ) -> None:
        self._resolver: HostnameResolver = resolver or (lambda _host: [])
        self._base_transport = base_transport
        self._clock = clock or time.time
        self._client_factory = client_factory
        self._logger = logger or get_logger(__name__)
        self._token_cache: dict[tuple[str, str], tuple[str, float]] = {}
        self._token_locks: dict[tuple[str, str], asyncio.Lock] = {}

    async def validate_configuration(self, config: HttpApiConnectionConfig, perform_login: bool = False) -> HttpApiValidationResult:
        self._preflight_config(config)
        target = await self._resolve_target(config.base_url)
        if not perform_login:
            async with self._build_client(target, timeout_seconds=30):
                pass
            return HttpApiValidationResult(success=True, message="Configuration is valid - no live request was sent.")
        if config.auth_mode not in {HttpApiAuthMode.LOGIN_EXCHANGE, HttpApiAuthMode.TOKEN_EXCHANGE}:
            raise ValueError("Live authentication validation requires login_exchange or token_exchange mode")
        async with self._build_client(target, timeout_seconds=30) as client:
            token = await self._get_login_token("__validation__", config, target=target, client=client, force_refresh=True)
        if not token:
            raise ValueError("Login exchange did not return a token")
        return HttpApiValidationResult(success=True, message=self._validation_success_message(config))

    async def execute(self, tool_id, config, request, *, allow_write, timeout_seconds, max_results) -> HttpApiExecutionResult:
        policy = (config.method_policies or DEFAULT_HTTP_API_METHOD_POLICIES).get(request.method, HttpApiMethodPolicy.DISABLED)
        if policy == HttpApiMethodPolicy.DISABLED:
            raise ValueError(f"HTTP method {request.method} is disabled")
        if policy == HttpApiMethodPolicy.WRITE and not allow_write:
            raise PermissionError(f"HTTP method {request.method} requires write access")

        self._preflight_config(config)
        target = await self._resolve_target(config.base_url)
        async with self._build_client(target, timeout_seconds=timeout_seconds) as client:
            response = await self._send_request(tool_id, config, request, target=target, client=client, timeout_seconds=timeout_seconds, refresh_on_401=True)
        output = _sanitize_json_value(response.get("parsed", response.get("raw")))
        rows, columns, row_count = _normalize_rows_and_columns(output, max_results)
        return HttpApiExecutionResult(status=response["status"], output=output, rows=rows, columns=columns, row_count=row_count)

    async def fetch_openapi_document(self, spec_url: str, timeout_seconds: int = 30) -> str:
        target = await self._resolve_target(spec_url)
        async with self._build_client(target, timeout_seconds=timeout_seconds) as client:
            raw = await self._request_bytes(
                client, "GET", target, normalize_relative_path(urlsplit(spec_url).path or "/"), {}, None, None, response_limit=_OPENAPI_RESPONSE_LIMIT
            )
        return raw.decode("utf-8", errors="replace")

    async def _resolve_target(self, base_url: str) -> _ResolvedTarget:
        parsed = validate_base_url(base_url, debug_mode=settings.debug_mode)
        maybe_addresses = self._resolver(parsed.host)
        if isinstance(maybe_addresses, list):
            addresses = maybe_addresses
        else:
            addresses = await maybe_addresses
        validated = validate_resolved_addresses(addresses, debug_mode=settings.debug_mode)
        return _ResolvedTarget(
            original_host=parsed.host,
            pinned_host=str(validated[0]),
            port=parsed.port,
            scheme=parsed.scheme,
            base_url=str(parsed),
        )

    def _build_client(self, target: _ResolvedTarget, *, timeout_seconds: int) -> httpx.AsyncClient:
        transport = self._base_transport or httpx.AsyncHTTPTransport(retries=0)
        pinned_transport = PinnedAsyncTransport(
            transport,
            pinned_host=target.pinned_host,
            original_host=target.original_host,
            original_port=target.port,
        )
        return self._client_factory(transport=pinned_transport, follow_redirects=False, trust_env=False, timeout=timeout_seconds)

    async def _request_bytes(
        self,
        client: httpx.AsyncClient,
        method: str,
        target: _ResolvedTarget,
        path: str,
        headers: dict[str, str],
        json_body: Any,
        form_body: dict[str, Any] | None,
        *,
        response_limit: int,
    ) -> bytes:
        url = _join_base_and_path(build_pinned_base_url(target.base_url, target.pinned_host), path)
        request_obj = client.build_request(method, url, headers=headers, json=json_body, data=form_body)
        response = await client.send(request_obj, stream=True)
        raw = bytearray()
        try:
            async for chunk in response.aiter_bytes():
                raw.extend(chunk)
                if len(raw) > response_limit:
                    raise ValueError("HTTP API response exceeds limit")
        finally:
            await response.aclose()
        if response.status_code >= 0:
            response.extensions["raw_bytes"] = bytes(raw)
        return bytes(raw)

    async def _send_request(
        self,
        tool_id: str,
        config: HttpApiConnectionConfig,
        request: HttpApiRequest,
        *,
        target: _ResolvedTarget,
        client: httpx.AsyncClient,
        timeout_seconds: int,
        refresh_on_401: bool,
        trusted_headers: dict[str, str] | None = None,
        trusted_query: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del timeout_seconds
        path = normalize_relative_path(request.path)
        headers = sanitize_outbound_headers(request.headers, config.approved_request_headers)
        query = dict(request.query)
        headers.update(trusted_headers or {})
        query.update(trusted_query or {})
        configured_request_headers = self._build_configured_request_headers(config)
        validated_token_header_name = validate_trusted_header_name(config.token_header_name)
        if config.auth_mode == HttpApiAuthMode.TOKEN_EXCHANGE:
            self._check_token_header_conflict(config, configured_request_headers)
        headers.update(configured_request_headers)
        json_body = request.json_body
        form_body = request.form_body
        self._inject_static_auth(config, headers, query)
        if config.auth_mode in {HttpApiAuthMode.LOGIN_EXCHANGE, HttpApiAuthMode.TOKEN_EXCHANGE}:
            token = await self._get_login_token(tool_id, config, target=target, client=client)
            prefix = f"{config.token_prefix} " if config.token_prefix else ""
            headers[validated_token_header_name] = f"{prefix}{token}".strip()

        body_bytes = b""
        if json_body is not None:
            body_bytes = json.dumps(json_body, separators=(",", ":")).encode("utf-8")
        elif form_body is not None:
            body_bytes = urlencode(form_body, doseq=True).encode("utf-8")
        if len(body_bytes) > _REQUEST_BODY_LIMIT:
            raise ValueError("HTTP API request body exceeds limit")

        request_path = path
        if query:
            request_path = f"{request_path}{'&' if '?' in request_path else '?'}{urlencode(query, doseq=True)}"

        start = self._clock()
        request_obj = client.build_request(
            request.method,
            _join_base_and_path(build_pinned_base_url(target.base_url, target.pinned_host), request_path),
            headers=headers,
            json=json_body,
            data=form_body,
        )
        response = await client.send(request_obj, stream=True)
        raw = bytearray()
        try:
            async for chunk in response.aiter_bytes():
                raw.extend(chunk)
                if len(raw) > _RESPONSE_BODY_LIMIT:
                    raise ValueError("HTTP API response exceeds limit")
        finally:
            await response.aclose()

        if response.status_code == 401 and refresh_on_401 and config.auth_mode in {HttpApiAuthMode.LOGIN_EXCHANGE, HttpApiAuthMode.TOKEN_EXCHANGE}:
            self._clear_cached_token(tool_id, config)
            return await self._send_request(tool_id, config, request, target=target, client=client, timeout_seconds=0, refresh_on_401=False)

        raw_bytes = bytes(raw)
        parsed: Any
        try:
            parsed = json.loads(raw_bytes.decode("utf-8"))
        except Exception:
            parsed = raw_bytes.decode("utf-8", errors="replace")
        selector = request.response_selector or config.default_response_selector
        if selector:
            parsed = _select_json_value(parsed, selector)
        duration_ms = int((self._clock() - start) * 1000)
        self._logger.info(
            "HTTP API request completed tool_id=%s method=%s path=%s status=%s duration_ms=%s response_bytes=%s",
            tool_id,
            request.method,
            path.split("?", 1)[0],
            response.status_code,
            duration_ms,
            len(raw_bytes),
        )
        return {"status": response.status_code, "parsed": parsed, "raw": raw_bytes.decode("utf-8", errors="replace")}

    def _inject_static_auth(self, config: HttpApiConnectionConfig, headers: dict[str, str], query: dict[str, Any]) -> None:
        if config.auth_mode == HttpApiAuthMode.API_KEY and config.api_key_name and config.send_api_key_to_requests:
            self._apply_api_key(config, headers, query)
        elif config.auth_mode == HttpApiAuthMode.BASIC:
            headers["Authorization"] = httpx.BasicAuth(config.basic_username, config.basic_password)._auth_header
        elif config.auth_mode == HttpApiAuthMode.BEARER:
            headers["Authorization"] = f"Bearer {config.bearer_token}"

    def _apply_api_key(self, config: HttpApiConnectionConfig, headers: dict[str, str], query: dict[str, Any]) -> None:
        value = f"{config.api_key_prefix} {config.api_key}".strip() if config.api_key_prefix else config.api_key
        if config.api_key_location == "query":
            query[config.api_key_name] = value
        else:
            headers[config.api_key_name] = value

    def _build_configured_request_headers(self, config: HttpApiConnectionConfig) -> dict[str, str]:
        if config.auth_mode not in {HttpApiAuthMode.HEADERS, HttpApiAuthMode.TOKEN_EXCHANGE}:
            return {}
        return build_trusted_configured_headers(config.request_headers)

    def _check_token_header_conflict(self, config: HttpApiConnectionConfig, configured_headers: dict[str, str]) -> None:
        if config.auth_mode != HttpApiAuthMode.TOKEN_EXCHANGE:
            return
        token_header_name = config.token_header_name.strip().lower()
        conflicting = next((name for name in configured_headers if name.strip().lower() == token_header_name), None)
        if conflicting is not None:
            raise HttpApiSecurityError(f"Configured request_headers conflict with token_header_name '{config.token_header_name}'")

    def _preflight_config(self, config: HttpApiConnectionConfig) -> None:
        if config.auth_mode == HttpApiAuthMode.HEADERS:
            build_trusted_configured_headers(config.request_headers)
            return
        if config.auth_mode not in {HttpApiAuthMode.LOGIN_EXCHANGE, HttpApiAuthMode.TOKEN_EXCHANGE}:
            return
        validate_trusted_header_name(config.token_header_name)
        if config.auth_mode == HttpApiAuthMode.TOKEN_EXCHANGE:
            configured_headers = build_trusted_configured_headers(config.request_headers)
            build_trusted_configured_headers(config.token_request_headers)
            self._check_token_header_conflict(config, configured_headers)

    def _validation_success_message(self, config: HttpApiConnectionConfig) -> str:
        if config.auth_mode == HttpApiAuthMode.TOKEN_EXCHANGE:
            return "Token exchange succeeded - token received."
        return "Login exchange succeeded - token received."

    async def _get_login_token(
        self,
        tool_id: str,
        config: HttpApiConnectionConfig,
        *,
        target: _ResolvedTarget,
        client: httpx.AsyncClient,
        force_refresh: bool = False,
    ) -> str:
        cache_key = (tool_id, _hash_config(config))
        now = self._clock()
        if not force_refresh:
            cached = self._token_cache.get(cache_key)
            if cached and cached[1] > now:
                return cached[0]

        lock = self._token_locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            now = self._clock()
            if not force_refresh:
                cached = self._token_cache.get(cache_key)
                if cached and cached[1] > now:
                    return cached[0]

            headers: dict[str, str] = {}
            query: dict[str, Any] = {}
            trusted_headers: dict[str, str] = {}
            trusted_query: dict[str, Any] = {}
            if config.send_api_key_to_login and config.api_key and config.api_key_name:
                self._apply_api_key(config, trusted_headers, trusted_query)
            extra_headers = build_trusted_configured_headers(config.token_request_headers) if config.auth_mode == HttpApiAuthMode.TOKEN_EXCHANGE else {}
            trusted_headers.update(extra_headers)
            if config.auth_mode == HttpApiAuthMode.TOKEN_EXCHANGE:
                payload = {field.name: field.value for field in config.token_request_fields}
            else:
                payload = {
                    config.login_username_field: config.login_username,
                    config.login_password_field: config.login_password,
                }
            request = HttpApiRequest(
                method=config.login_method,
                path=config.login_path,
                headers=headers,
                query=query,
                json_body=payload if config.login_body_format == "json" else None,
                form_body=payload if config.login_body_format == "form" else None,
            )
            response = await self._send_request(
                tool_id or "__login__",
                HttpApiConnectionConfig(**{**config.model_dump(), "auth_mode": "none"}),
                request,
                target=target,
                client=client,
                timeout_seconds=0,
                refresh_on_401=False,
                trusted_headers=trusted_headers,
                trusted_query=trusted_query,
            )
            parsed = response["parsed"]
            token = _select_json_value(parsed, config.token_response_path)
            expires_in = _DEFAULT_TOKEN_TTL_SECONDS
            if config.token_expires_in_path:
                try:
                    expires_in = int(_select_json_value(parsed, config.token_expires_in_path))
                except Exception:
                    expires_in = _DEFAULT_TOKEN_TTL_SECONDS
            if len(self._token_cache) >= _TOKEN_CACHE_MAX_SIZE and cache_key not in self._token_cache:
                self._token_cache.pop(next(iter(self._token_cache)), None)
            self._token_cache[cache_key] = (str(token), now + max(expires_in, 1))
            return str(token)

    def _clear_cached_token(self, tool_id: str, config: HttpApiConnectionConfig) -> None:
        self._token_cache.pop((tool_id, _hash_config(config)), None)


http_api_broker = HttpApiBroker(resolver=resolve_http_api_hostname)
