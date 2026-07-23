from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import socket
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlencode, urlsplit

import httpx

from ragtime.config.settings import settings
from ragtime.core.logging import get_logger
from ragtime.http_api.body import validate_configured_content_type
from ragtime.http_api.models import (
    DEFAULT_HTTP_API_METHOD_POLICIES,
    HTTP_API_SECRET_FIELDS,
    HttpApiAuthMode,
    HttpApiBodyFormat,
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
OAuthCredentialUpdater = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True)
class _ResolvedTarget:
    original_host: str
    pinned_host: str
    port: int | None
    scheme: str
    base_url: str


@dataclass(frozen=True)
class _ResolvedEndpoint:
    target: _ResolvedTarget
    path: str
    reuse_resource_client: bool


class HttpApiConfigurationError(ValueError):
    """Safe, value-free configuration error suitable for API responses."""


@dataclass(frozen=True)
class _PreparedBody:
    json_body: Any = None
    form_body: dict[str, Any] | None = None
    multipart_body: list[tuple[str, tuple[None, str]]] | None = None


def _hash_config(config: HttpApiConnectionConfig) -> str:
    payload = json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sanitize_json_value(value: Any) -> Any:
    secret_names = {field.lower() for field in HTTP_API_SECRET_FIELDS} | {
        "access_token",
        "authorization",
        "client_secret",
        "refresh_token",
    }
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


def _effective_port(url: httpx.URL) -> int | None:
    return url.port or {"http": 80, "https": 443}.get(url.scheme)


def _origin_url(url: httpx.URL) -> str:
    default_port = {"http": 80, "https": 443}.get(url.scheme)
    port = None if url.port == default_port else url.port
    return str(httpx.URL(scheme=url.scheme, host=url.host, port=port))


def _basic_auth_header(username: str, password: str) -> str:
    encoded = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {encoded}"


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
        self._oauth_refresh_locks: dict[str, asyncio.Lock] = {}

    async def validate_configuration(self, config: HttpApiConnectionConfig, perform_login: bool = False) -> HttpApiValidationResult:
        self._preflight_config(config)
        try:
            target = await self._resolve_target(config.base_url)
        except (ValueError, HttpApiSecurityError, httpx.InvalidURL) as exc:
            raise HttpApiSecurityError("Resource target validation failed") from exc
        if config.auth_mode == HttpApiAuthMode.OAUTH2:
            await self._validate_oauth_endpoints(config)
            if not perform_login:
                async with self._build_client(target, timeout_seconds=30):
                    pass
                return HttpApiValidationResult(success=True, message="Configuration is valid - no live request was sent.")
            if not config.oauth_access_token or self._oauth_expiring(config.oauth_token_expires_at):
                raise ValueError("OAuth validation requires a usable stored access token")
            return HttpApiValidationResult(success=True, message="OAuth configuration validated - stored access token is available.")
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

    async def validate_connectivity(self, config: HttpApiConnectionConfig) -> HttpApiValidationResult:
        self._preflight_config(config)
        try:
            target = await self._resolve_target(config.base_url)
        except (ValueError, HttpApiSecurityError, httpx.InvalidURL) as exc:
            raise HttpApiSecurityError("Resource target validation failed") from exc
        try:
            async with self._build_client(target, timeout_seconds=30) as client:
                request = client.build_request("HEAD", build_pinned_base_url(target.base_url, target.pinned_host))
                response = await client.send(request, stream=True)
                try:
                    status = response.status_code
                finally:
                    await response.aclose()
        except (httpx.HTTPError, OSError) as exc:
            return HttpApiValidationResult(success=False, message=f"Connectivity failed: {type(exc).__name__}")
        return HttpApiValidationResult(
            success=True,
            message=f"Connectivity succeeded with HTTP {status}. Authentication and request body acceptance were not tested.",
            details={"status": status, "auth_tested": False},
        )

    async def execute(
        self, tool_id, config, request, *, allow_write, timeout_seconds, max_results, oauth_credential_updater: OAuthCredentialUpdater | None = None
    ) -> HttpApiExecutionResult:
        policy = (config.method_policies or DEFAULT_HTTP_API_METHOD_POLICIES).get(request.method, HttpApiMethodPolicy.DISABLED)
        if policy == HttpApiMethodPolicy.DISABLED:
            raise ValueError(f"HTTP method {request.method} is disabled")
        if policy == HttpApiMethodPolicy.WRITE and not allow_write:
            raise PermissionError(f"HTTP method {request.method} requires write access")

        self._preflight_config(config)
        try:
            target = await self._resolve_target(config.base_url)
        except (ValueError, HttpApiSecurityError, httpx.InvalidURL) as exc:
            raise HttpApiSecurityError("Resource target validation failed") from exc
        async with self._build_client(target, timeout_seconds=timeout_seconds) as client:
            response = await self._send_request(
                tool_id,
                config,
                request,
                target=target,
                client=client,
                timeout_seconds=timeout_seconds,
                refresh_on_401=True,
                oauth_credential_updater=oauth_credential_updater,
            )
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

    async def _resolve_token_endpoint(self, config: HttpApiConnectionConfig, resource_target: _ResolvedTarget) -> _ResolvedEndpoint:
        raw_endpoint = (config.token_url or config.login_path).strip()
        if not raw_endpoint:
            raise HttpApiConfigurationError("OAuth endpoint is required")
        parsed = urlsplit(raw_endpoint)
        if not parsed.scheme and not parsed.netloc:
            return _ResolvedEndpoint(resource_target, normalize_relative_path(raw_endpoint), True)
        try:
            validated = validate_base_url(raw_endpoint, debug_mode=settings.debug_mode)
        except (ValueError, HttpApiSecurityError, httpx.InvalidURL) as exc:
            raise HttpApiSecurityError("Token target validation failed") from exc
        same_origin = (
            validated.scheme == resource_target.scheme
            and validated.host == resource_target.original_host
            and _effective_port(validated) == (resource_target.port or {"http": 80, "https": 443}.get(resource_target.scheme))
        )
        if same_origin:
            target = _ResolvedTarget(
                validated.host,
                resource_target.pinned_host,
                resource_target.port,
                resource_target.scheme,
                _origin_url(validated),
            )
        else:
            try:
                target = await self._resolve_target(_origin_url(validated))
            except (ValueError, HttpApiSecurityError, httpx.InvalidURL) as exc:
                raise HttpApiSecurityError("Token target validation failed") from exc
        return _ResolvedEndpoint(target, normalize_relative_path(validated.path or "/"), same_origin)

    async def validate_oauth_url(self, url: str) -> None:
        parsed = validate_base_url(url, debug_mode=settings.debug_mode)
        await self._resolve_target(_origin_url(parsed))

    async def _validate_oauth_endpoints(self, config: HttpApiConnectionConfig) -> None:
        endpoints = {
            config.oauth_issuer_url,
            config.oauth_authorization_url,
            config.oauth_device_authorization_url,
            config.oauth_token_url,
        }
        for endpoint in endpoints:
            if endpoint:
                await self.validate_oauth_url(endpoint)

    async def oauth_request_json(
        self,
        url: str,
        *,
        method: str = "GET",
        form: dict[str, Any] | None = None,
        json_body: Any = None,
        client_auth: Any = None,
        client_id: str = "",
        client_secret: str = "",
    ) -> tuple[int, dict[str, Any]]:
        parsed = validate_base_url(url, debug_mode=settings.debug_mode)
        target = await self._resolve_target(_origin_url(parsed))
        headers: dict[str, str] = {"Accept": "application/json"}
        payload = dict(form or {})
        if client_auth == "client_secret_post" and client_secret:
            payload["client_secret"] = client_secret
            payload.setdefault("client_id", client_id)
        elif getattr(client_auth, "value", client_auth) == "client_secret_basic" and client_secret:
            headers["Authorization"] = _basic_auth_header(client_id, client_secret)
        async with self._build_client(target, timeout_seconds=30) as client:
            request = client.build_request(
                method,
                _join_base_and_path(build_pinned_base_url(target.base_url, target.pinned_host), normalize_relative_path(parsed.path or "/")),
                headers=headers,
                json=json_body,
                data=payload or None,
            )
            response = await client.send(request, stream=True)
            raw = bytearray()
            try:
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw) > _RESPONSE_BODY_LIMIT:
                        raise ValueError("OAuth provider response exceeds limit")
            finally:
                await response.aclose()
        try:
            parsed_body = json.loads(bytes(raw).decode("utf-8"))
        except Exception:
            parsed_body = {}
        return response.status_code, parsed_body if isinstance(parsed_body, dict) else {}

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
        oauth_credential_updater: OAuthCredentialUpdater | None = None,
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
        if config.auth_mode in {HttpApiAuthMode.TOKEN_EXCHANGE, HttpApiAuthMode.OAUTH2}:
            self._check_token_header_conflict(config, configured_request_headers)
        headers.update(configured_request_headers)
        prepared = self._prepare_resource_body(config, request)
        self._inject_static_auth(config, headers, query)
        if config.auth_mode == HttpApiAuthMode.OAUTH2:
            token = await self._get_oauth_token(tool_id, config, target=target, updater=oauth_credential_updater)
            prefix = f"{config.token_prefix} " if config.token_prefix else ""
            headers[validated_token_header_name] = f"{prefix}{token}".strip()
        if config.auth_mode in {HttpApiAuthMode.LOGIN_EXCHANGE, HttpApiAuthMode.TOKEN_EXCHANGE}:
            token = await self._get_login_token(tool_id, config, target=target, client=client)
            prefix = f"{config.token_prefix} " if config.token_prefix else ""
            headers[validated_token_header_name] = f"{prefix}{token}".strip()

        request_path = path
        if query:
            request_path = f"{request_path}{'&' if '?' in request_path else '?'}{urlencode(query, doseq=True)}"

        start = self._clock()
        request_obj = client.build_request(
            request.method,
            _join_base_and_path(build_pinned_base_url(target.base_url, target.pinned_host), request_path),
            headers=headers,
            json=prepared.json_body,
            data=prepared.form_body,
            files=prepared.multipart_body,
        )
        encoded_body = await request_obj.aread()
        if len(encoded_body) > _REQUEST_BODY_LIMIT:
            raise HttpApiConfigurationError("HTTP API request body exceeds the configured size limit")
        response = await client.send(request_obj, stream=True)
        raw = bytearray()
        try:
            async for chunk in response.aiter_bytes():
                raw.extend(chunk)
                if len(raw) > _RESPONSE_BODY_LIMIT:
                    raise ValueError("HTTP API response exceeds limit")
        finally:
            await response.aclose()

        if (
            response.status_code == 401
            and refresh_on_401
            and config.auth_mode in {HttpApiAuthMode.LOGIN_EXCHANGE, HttpApiAuthMode.TOKEN_EXCHANGE, HttpApiAuthMode.OAUTH2}
        ):
            if config.auth_mode == HttpApiAuthMode.OAUTH2 and not config.oauth_refresh_token:
                raise HttpApiConfigurationError("OAuth connection requires reconnecting")
            if config.auth_mode == HttpApiAuthMode.OAUTH2:
                config.oauth_access_token = ""
            self._clear_cached_token(tool_id, config)
            return await self._send_request(
                tool_id,
                config,
                request,
                target=target,
                client=client,
                timeout_seconds=0,
                refresh_on_401=False,
                oauth_credential_updater=oauth_credential_updater,
            )

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

    def _prepare_resource_body(self, config: HttpApiConnectionConfig, request: HttpApiRequest) -> _PreparedBody:
        configured = {field.name: field.value for field in config.request_body_fields}
        if not configured or config.auth_mode not in {HttpApiAuthMode.HEADERS, HttpApiAuthMode.TOKEN_EXCHANGE, HttpApiAuthMode.OAUTH2}:
            return _PreparedBody(json_body=request.json_body, form_body=request.form_body)
        if config.request_body_format == HttpApiBodyFormat.JSON:
            if request.form_body is not None:
                raise HttpApiConfigurationError("Configured request body format does not match the operation body format")
            operation = {} if request.json_body is None else request.json_body
            if not isinstance(operation, dict):
                raise HttpApiConfigurationError("Configured JSON fields require an operation JSON object")
            return _PreparedBody(json_body={**operation, **configured})
        if config.request_body_format == HttpApiBodyFormat.FORM:
            if request.json_body is not None:
                raise HttpApiConfigurationError("Configured request body format does not match the operation body format")
            return _PreparedBody(form_body={**(request.form_body or {}), **configured})
        if request.json_body is not None or request.form_body is not None:
            raise HttpApiConfigurationError("Configured request body format does not match the operation body format")
        return _PreparedBody(multipart_body=[(name, (None, value)) for name, value in configured.items()])

    def _inject_static_auth(self, config: HttpApiConnectionConfig, headers: dict[str, str], query: dict[str, Any]) -> None:
        if config.auth_mode == HttpApiAuthMode.API_KEY and config.api_key_name and config.send_api_key_to_requests:
            self._apply_api_key(config, headers, query)
        elif config.auth_mode == HttpApiAuthMode.BASIC:
            headers["Authorization"] = _basic_auth_header(config.basic_username, config.basic_password)
        elif config.auth_mode == HttpApiAuthMode.BEARER:
            headers["Authorization"] = f"Bearer {config.bearer_token}"

    def _apply_api_key(self, config: HttpApiConnectionConfig, headers: dict[str, str], query: dict[str, Any]) -> None:
        value = f"{config.api_key_prefix} {config.api_key}".strip() if config.api_key_prefix else config.api_key
        if config.api_key_location == "query":
            query[config.api_key_name] = value
        else:
            headers[config.api_key_name] = value

    def _build_configured_request_headers(self, config: HttpApiConnectionConfig) -> dict[str, str]:
        if config.auth_mode not in {HttpApiAuthMode.HEADERS, HttpApiAuthMode.TOKEN_EXCHANGE, HttpApiAuthMode.OAUTH2}:
            return {}
        return build_trusted_configured_headers(config.request_headers)

    def _check_token_header_conflict(self, config: HttpApiConnectionConfig, configured_headers: dict[str, str]) -> None:
        if config.auth_mode not in {HttpApiAuthMode.TOKEN_EXCHANGE, HttpApiAuthMode.OAUTH2}:
            return
        token_header_name = config.token_header_name.strip().lower()
        conflicting = next((name for name in configured_headers if name.strip().lower() == token_header_name), None)
        if conflicting is not None:
            raise HttpApiSecurityError(f"Configured request_headers conflict with token_header_name '{config.token_header_name}'")

    def _preflight_config(self, config: HttpApiConnectionConfig) -> None:
        if config.auth_mode == HttpApiAuthMode.HEADERS:
            build_trusted_configured_headers(config.request_headers)
            validate_configured_content_type(
                config.request_headers, config.request_body_format, has_body=bool(config.request_body_fields), section="request body"
            )
            return
        if config.auth_mode == HttpApiAuthMode.OAUTH2:
            validate_trusted_header_name(config.token_header_name)
            configured_headers = build_trusted_configured_headers(config.request_headers)
            validate_configured_content_type(
                config.request_headers, config.request_body_format, has_body=bool(config.request_body_fields), section="request body"
            )
            self._check_token_header_conflict(config, configured_headers)
            return
        if config.auth_mode not in {HttpApiAuthMode.LOGIN_EXCHANGE, HttpApiAuthMode.TOKEN_EXCHANGE}:
            return
        validate_trusted_header_name(config.token_header_name)
        if config.auth_mode == HttpApiAuthMode.TOKEN_EXCHANGE:
            configured_headers = build_trusted_configured_headers(config.request_headers)
            build_trusted_configured_headers(config.token_request_headers)
            validate_configured_content_type(
                config.request_headers, config.request_body_format, has_body=bool(config.request_body_fields), section="request body"
            )
            validate_configured_content_type(
                config.token_request_headers, config.login_body_format, has_body=bool(config.token_request_fields), section="token request body"
            )
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
            endpoint = await self._resolve_token_endpoint(config, target)
            request = HttpApiRequest(
                method=config.login_method,
                path=endpoint.path,
                headers=headers,
                query=query,
                json_body=payload if config.login_body_format == "json" else None,
                form_body=payload if config.login_body_format == "form" else None,
            )
            auth_config = HttpApiConnectionConfig(
                **{
                    **config.model_dump(),
                    "auth_mode": HttpApiAuthMode.NONE,
                    "default_response_selector": "",
                }
            )
            if endpoint.reuse_resource_client:
                response = await self._send_request(
                    tool_id or "__login__",
                    auth_config,
                    request,
                    target=endpoint.target,
                    client=client,
                    timeout_seconds=0,
                    refresh_on_401=False,
                    trusted_headers=trusted_headers,
                    trusted_query=trusted_query,
                )
            else:
                async with self._build_client(endpoint.target, timeout_seconds=30) as token_client:
                    response = await self._send_request(
                        tool_id or "__login__",
                        auth_config,
                        request,
                        target=endpoint.target,
                        client=token_client,
                        timeout_seconds=0,
                        refresh_on_401=False,
                        trusted_headers=trusted_headers,
                        trusted_query=trusted_query,
                    )
            status = int(response["status"])
            if status < 200 or status >= 300:
                raise HttpApiConfigurationError(f"Token exchange failed with HTTP {status}")
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

    async def _get_oauth_token(self, tool_id: str, config: HttpApiConnectionConfig, *, target: _ResolvedTarget, updater: OAuthCredentialUpdater | None) -> str:
        del target
        if config.oauth_access_token and not self._oauth_expiring(config.oauth_token_expires_at):
            return config.oauth_access_token
        if not config.oauth_refresh_token:
            raise HttpApiConfigurationError("OAuth connection requires reconnecting")
        lock = self._oauth_refresh_locks.setdefault(str(tool_id), asyncio.Lock())
        async with lock:
            if config.oauth_access_token and not self._oauth_expiring(config.oauth_token_expires_at):
                return config.oauth_access_token
            used_refresh_token = config.oauth_refresh_token
            status, payload = await self.oauth_request_json(
                config.oauth_token_url,
                method="POST",
                form={
                    "grant_type": "refresh_token",
                    "refresh_token": used_refresh_token,
                    "client_id": config.oauth_client_id,
                    "scope": " ".join(config.oauth_scopes),
                },
                client_auth=config.oauth_client_auth_method,
                client_id=config.oauth_client_id,
                client_secret=config.oauth_client_secret,
            )
            if status != 200:
                raise HttpApiConfigurationError("OAuth refresh failed; reconnect required")
            from ragtime.http_api.oauth import _credential_update

            update = _credential_update(payload)
            proposed = update.as_config()
            proposed["oauth_refresh_token_used"] = used_refresh_token
            if updater is not None:
                await updater(proposed)
            config.oauth_access_token = update.access_token
            if update.refresh_token:
                config.oauth_refresh_token = update.refresh_token
            config.oauth_token_type = update.token_type
            config.oauth_token_expires_at = update.expires_at
            return config.oauth_access_token

    def _oauth_expiring(self, value: str) -> bool:
        if not value:
            return True
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            expiry = parsed.timestamp()
        except (TypeError, ValueError):
            return True
        return expiry <= time.time() + 60


http_api_broker = HttpApiBroker(resolver=resolve_http_api_hostname)
