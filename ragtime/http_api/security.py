from __future__ import annotations

import ipaddress
from collections.abc import Iterable, Sequence
from urllib.parse import unquote, urlsplit

import httpx

from ragtime.core.security import validate_external_url
from ragtime.http_api.header_policy import (
    HTTP_API_BLOCKED_HEADER_NAMES,
    validate_header_name,
    validate_header_value,
)
from ragtime.http_api.header_policy import (
    is_blocked_header_name as _is_blocked_header_name,
)


class HttpApiSecurityError(ValueError):
    pass


IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def is_public_ip_address(address: IpAddress) -> bool:
    return not (address.is_private or address.is_loopback or address.is_link_local or address.is_multicast or address.is_reserved or address.is_unspecified)


def validate_resolved_addresses(addresses: Iterable[str], *, debug_mode: bool) -> list[IpAddress]:
    resolved: list[IpAddress] = []
    for raw in addresses:
        address = ipaddress.ip_address(raw)
        if not debug_mode and not is_public_ip_address(address):
            raise HttpApiSecurityError("HTTP API target resolves to a non-public address")
        resolved.append(address)
    if not resolved:
        raise HttpApiSecurityError("HTTP API target did not resolve to any addresses")
    return resolved


def build_pinned_base_url(base_url: str, pinned_host: str) -> str:
    return str(httpx.URL(base_url).copy_with(host=pinned_host))


def validate_base_url(base_url: str, *, debug_mode: bool) -> httpx.URL:
    ok, error = validate_external_url(base_url, allow_private_networks=debug_mode)
    if not ok:
        raise HttpApiSecurityError(error)
    parsed = httpx.URL(base_url)
    if parsed.userinfo or parsed.fragment:
        raise HttpApiSecurityError("HTTP API base URL contains unsupported URL components")
    if parsed.query:
        raise HttpApiSecurityError("HTTP API base URL must not include query parameters")
    if not debug_mode and parsed.scheme != "https":
        raise HttpApiSecurityError("HTTP API base URL must use HTTPS outside debug mode")
    return parsed


def normalize_relative_path(path: str) -> str:
    candidate = (path or "").strip()
    if not candidate:
        return "/"
    parsed = urlsplit(candidate)
    if parsed.scheme or parsed.netloc:
        raise HttpApiSecurityError("HTTP API request path must be relative")
    if candidate.startswith("//"):
        raise HttpApiSecurityError("HTTP API request path must not replace the origin")
    if "#" in candidate or parsed.fragment:
        raise HttpApiSecurityError("HTTP API request path must not include a fragment")
    raw_path = parsed.path or ""
    decoded_path = unquote(raw_path)
    if "\\" in candidate or "\\" in decoded_path:
        raise HttpApiSecurityError("HTTP API request path must not contain backslashes")
    if "@" in decoded_path:
        raise HttpApiSecurityError("HTTP API request path must not contain userinfo-like segments")
    if raw_path.lower().startswith("/%2f") or decoded_path.startswith("//") or decoded_path.startswith("///"):
        raise HttpApiSecurityError("HTTP API request path must not replace the origin")
    segments = [segment for segment in decoded_path.split("/") if segment]
    if any(segment == ".." for segment in segments):
        raise HttpApiSecurityError("HTTP API request path must not contain traversal")
    normalized_path = raw_path if raw_path.startswith("/") else f"/{raw_path}"
    return normalized_path + (f"?{parsed.query}" if parsed.query else "")


def is_blocked_header_name(name: str) -> bool:
    try:
        return _is_blocked_header_name(name)
    except Exception:
        return True


def validate_trusted_header_name(name: object) -> str:
    try:
        return validate_header_name(name, allow_configured_auth_headers=True)
    except ValueError as exc:
        raise HttpApiSecurityError(str(exc)) from exc


def sanitize_outbound_headers(headers: dict[str, str] | None, approved_headers: list[str]) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    approved = {header.lower() for header in approved_headers}
    for name, value in (headers or {}).items():
        try:
            validated_name = validate_header_name(name, allow_configured_auth_headers=False)
            validated_value = validate_header_value(value, allow_blank=True)
        except ValueError as exc:
            raise HttpApiSecurityError(str(exc)) from exc
        lowered = validated_name.lower()
        if lowered not in approved:
            raise HttpApiSecurityError(f"Header '{validated_name}' is not approved")
        sanitized[validated_name] = validated_value
    return sanitized


def build_trusted_configured_headers(headers: Sequence[object] | None) -> dict[str, str]:
    trusted: dict[str, str] = {}
    seen: set[str] = set()
    for row in headers or []:
        if isinstance(row, dict):
            if "name" not in row:
                raise HttpApiSecurityError("Configured header name is required")
            if "value" not in row:
                raise HttpApiSecurityError("Configured header value is required")
            name = row.get("name")
            value = row.get("value")
        else:
            if not hasattr(row, "name"):
                raise HttpApiSecurityError("Configured header name is required")
            if not hasattr(row, "value"):
                raise HttpApiSecurityError("Configured header value is required")
            name = getattr(row, "name")
            value = getattr(row, "value")
        try:
            validated_name = validate_header_name(name, allow_configured_auth_headers=True)
        except ValueError as exc:
            raise HttpApiSecurityError(str(exc)) from exc
        if not isinstance(value, str) or value.strip() == "":
            raise HttpApiSecurityError(f"Header '{validated_name}' must have a non-blank value")
        try:
            validated_value = validate_header_value(value, allow_blank=True)
        except ValueError as exc:
            raise HttpApiSecurityError(str(exc)) from exc
        lowered = validated_name.lower()
        if lowered in seen:
            raise HttpApiSecurityError(f"Duplicate header '{validated_name}' is not allowed")
        seen.add(lowered)
        trusted[validated_name] = validated_value
    return trusted


class PinnedAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, transport: httpx.AsyncBaseTransport, *, pinned_host: str, original_host: str, original_port: int | None) -> None:
        self._transport = transport
        self._pinned_host = pinned_host
        self._original_host = original_host
        self._original_port = original_port

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request.url = request.url.copy_with(host=self._pinned_host)
        default_port = 443 if request.url.scheme == "https" else 80
        include_port = self._original_port is not None and self._original_port != default_port
        host_header = self._original_host if not include_port else f"{self._original_host}:{self._original_port}"
        request.headers["host"] = host_header
        request.extensions["sni_hostname"] = self._original_host
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self._transport.aclose()
