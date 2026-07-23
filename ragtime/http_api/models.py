from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

from ragtime.http_api.body import validate_configured_content_type
from ragtime.http_api.header_policy import validate_header_name, validate_header_value

HTTP_API_SECRET_FIELDS = (
    "api_key",
    "basic_password",
    "bearer_token",
    "login_password",
    "oauth_client_secret",
    "oauth_access_token",
    "oauth_refresh_token",
)

HTTP_API_STANDARD_METHODS = (
    "GET",
    "HEAD",
    "OPTIONS",
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
)


def _validate_http_header_name(name: str) -> str:
    return validate_header_name(name, allow_configured_auth_headers=True)


def _validate_http_header_value(value: str) -> str:
    return validate_header_value(value, allow_blank=True)


def _merge_header_rows(existing: Any, incoming: Any) -> Any:
    if not isinstance(incoming, list):
        return incoming
    existing_by_name = {}
    for row in existing or []:
        if isinstance(row, dict):
            existing_by_name[str(row.get("name", "")).strip().lower()] = row

    merged: list[dict[str, Any]] = []
    for row in incoming:
        if not isinstance(row, dict):
            merged.append(row)
            continue
        merged_row = dict(row)
        existing_row = existing_by_name.get(str(row.get("name", "")).strip().lower())
        if existing_row and merged_row.get("value") in (None, ""):
            merged_row["value"] = existing_row.get("value", "")
        merged.append(merged_row)
    return merged


def _merge_secret_field_rows(existing: Any, incoming: Any) -> Any:
    if not isinstance(incoming, list):
        return incoming
    existing_by_name = {}
    for row in existing or []:
        if isinstance(row, dict):
            existing_by_name[str(row.get("name", "")).strip()] = row

    merged: list[dict[str, Any]] = []
    for row in incoming:
        if not isinstance(row, dict):
            merged.append(row)
            continue
        merged_row = dict(row)
        existing_row = existing_by_name.get(str(row.get("name", "")).strip())
        if existing_row and merged_row.get("value") in (None, ""):
            merged_row["value"] = existing_row.get("value", "")
        merged.append(merged_row)
    return merged


def redact_http_api_connection_config(
    config: "HttpApiConnectionConfig | dict[str, Any] | None",
    secret_paths: set[str] | None = None,
) -> dict[str, Any]:
    redacted: dict[str, Any]
    if isinstance(config, HttpApiConnectionConfig):
        redacted = config.model_dump(mode="json")
    elif config is None:
        redacted = {}
    else:
        redacted = HttpApiConnectionConfig(**config).model_dump(mode="json")
    for field in HTTP_API_SECRET_FIELDS:
        redacted.pop(field, None)
    if secret_paths is not None:
        source = config.model_dump(mode="python") if isinstance(config, HttpApiConnectionConfig) else dict(config or {})
        for field in ("token_request_fields", "request_body_fields"):
            source_rows = source.get(field)
            redacted_rows = redacted.get(field)
            if not isinstance(source_rows, list) or not isinstance(redacted_rows, list):
                continue
            source_by_name = {str(row.get("name", "")): row for row in source_rows if isinstance(row, dict)}
            for row in redacted_rows:
                name = str(row.get("name", ""))
                source_row = source_by_name.get(name)
                if source_row is None:
                    continue
                path = f"{field}.{name}"
                if source_row.get("secret", True) is False and path not in secret_paths:
                    row["secret"] = source_row.get("secret", True)
                    row["value"] = source_row.get("value", "")
    return redacted


class HttpApiAuthMode(str, Enum):
    NONE = "none"
    API_KEY = "api_key"
    BASIC = "basic"
    BEARER = "bearer"
    HEADERS = "headers"
    LOGIN_EXCHANGE = "login_exchange"
    TOKEN_EXCHANGE = "token_exchange"
    OAUTH2 = "oauth2"


class HttpApiOAuthFlow(str, Enum):
    DEVICE_CODE = "device_code"
    AUTHORIZATION_CODE_PKCE = "authorization_code_pkce"


class HttpApiOAuthClientAuthMethod(str, Enum):
    NONE = "none"
    CLIENT_SECRET_POST = "client_secret_post"
    CLIENT_SECRET_BASIC = "client_secret_basic"


class HttpApiApiKeyLocation(str, Enum):
    HEADER = "header"
    QUERY = "query"


class HttpApiLoginBodyFormat(str, Enum):
    JSON = "json"
    FORM = "form"


class HttpApiBodyFormat(str, Enum):
    JSON = "json"
    FORM = "form"
    MULTIPART = "multipart"


class HttpApiMethod(str, Enum):
    GET = "GET"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class HttpApiMethodPolicy(str, Enum):
    DISABLED = "disabled"
    READ = "read"
    WRITE = "write"


DEFAULT_HTTP_API_METHOD_POLICIES: dict[str, HttpApiMethodPolicy] = {
    "GET": HttpApiMethodPolicy.READ,
    "HEAD": HttpApiMethodPolicy.READ,
    "OPTIONS": HttpApiMethodPolicy.READ,
    "POST": HttpApiMethodPolicy.DISABLED,
    "PUT": HttpApiMethodPolicy.DISABLED,
    "PATCH": HttpApiMethodPolicy.DISABLED,
    "DELETE": HttpApiMethodPolicy.DISABLED,
}


def sanitize_persisted_http_api_connection_config(config: dict[str, Any] | None) -> dict[str, Any]:
    persisted = dict(config or {})
    persisted.pop("raw_openapi_document", None)
    normalized = HttpApiConnectionConfig(**persisted)
    sanitized = normalized.model_dump()
    # Live configs normalize legacy markers, but persisted configs must retain
    # explicit false markers so exports can distinguish public fields.
    for field in ("token_request_fields", "request_body_fields"):
        original_rows = persisted.get(field)
        sanitized_rows = sanitized.get(field)
        if not isinstance(original_rows, list) or not isinstance(sanitized_rows, list):
            continue
        original_by_name = {str(row.get("name", "")): row for row in original_rows if isinstance(row, dict)}
        for row in sanitized_rows:
            original = original_by_name.get(str(row.get("name", "")))
            if isinstance(original, dict) and isinstance(original.get("secret"), bool):
                row["secret"] = original["secret"]
    return sanitized


def merge_http_api_secret_updates(existing: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(existing or {})
    updates = dict(incoming or {})

    for field in HTTP_API_SECRET_FIELDS:
        if field not in updates:
            continue
        value = updates[field]
        updates[field] = "" if value in (None, "") else value

    if "request_headers" in updates:
        updates["request_headers"] = _merge_header_rows(merged.get("request_headers"), updates.get("request_headers"))
    if "token_request_headers" in updates:
        updates["token_request_headers"] = _merge_header_rows(merged.get("token_request_headers"), updates.get("token_request_headers"))
    if "token_request_fields" in updates:
        updates["token_request_fields"] = _merge_secret_field_rows(merged.get("token_request_fields"), updates.get("token_request_fields"))
    if "request_body_fields" in updates:
        updates["request_body_fields"] = _merge_secret_field_rows(merged.get("request_body_fields"), updates.get("request_body_fields"))

    merged.update(updates)
    sanitized = sanitize_persisted_http_api_connection_config(merged)
    # Keep the legacy update-helper contract: markers supplied to this helper
    # are normalized for live configuration handling. Persistence callers that
    # need the explicit marker retain it on the stored payload.
    for field in ("token_request_fields", "request_body_fields"):
        for row in sanitized.get(field, []):
            if isinstance(row, dict) and "secret" in row:
                row["secret"] = True
    return sanitized


class OpenApiCatalogOperation(BaseModel):
    operation_id: str = Field(default="", description="Stable OpenAPI operation identifier")
    method: HttpApiMethod = Field(description="HTTP method")
    path: str = Field(description="Relative API path")
    summary: str = Field(default="", description="Short operation summary")
    description: str = Field(default="", description="Compact operation description")
    tags: list[str] = Field(default_factory=list, description="Operation tags")


class OpenApiCatalog(BaseModel):
    title: str = Field(default="", description="OpenAPI document title")
    version: str = Field(default="", description="OpenAPI document version")
    operations: list[OpenApiCatalogOperation] = Field(default_factory=list, description="Bounded normalized operation catalog")


class HttpApiConfiguredHeader(BaseModel):
    name: str = Field(description="Header name")
    value: str = Field(default="", description="Header value")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _validate_http_header_name(value)

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: str) -> str:
        return _validate_http_header_value(value)


class HttpApiTokenField(BaseModel):
    name: str = Field(description="Token request field name")
    value: str = Field(default="", description="Token request field value")
    secret: bool = Field(default=True, description="Whether the field value is secret")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        candidate = str(value or "").strip()
        if not candidate:
            raise ValueError("Token request field name is required")
        if "\r" in candidate or "\n" in candidate:
            raise ValueError("Token request field name must not contain CR or LF characters")
        return candidate

    @field_validator("value")
    @classmethod
    def _validate_value(cls, value: str) -> str:
        candidate = str(value or "")
        if "\r" in candidate or "\n" in candidate:
            raise ValueError("Token request field value must not contain CR or LF characters")
        return candidate

    @field_validator("secret", mode="before")
    @classmethod
    def _normalize_secret_marker(cls, _secret: Any) -> bool:
        return True


class HttpApiConnectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(default="", description="Manual API base URL")
    documentation_url: str = Field(default="", description="Optional HTTP(S) API documentation URL")
    auth_mode: HttpApiAuthMode = Field(
        default=HttpApiAuthMode.NONE,
        description="Authentication mode",
    )
    api_key: str = Field(default="", description="Static API key secret")
    api_key_location: HttpApiApiKeyLocation = Field(default=HttpApiApiKeyLocation.HEADER, description="API key location")
    api_key_name: str = Field(default="", description="Header or query parameter name used for API-key auth")
    api_key_prefix: str = Field(default="", description="Optional API-key header prefix")
    basic_username: str = Field(default="", description="Basic auth username")
    basic_password: str = Field(default="", description="Basic auth password")
    bearer_token: str = Field(default="", description="Static bearer token")
    oauth_flow: HttpApiOAuthFlow = Field(default=HttpApiOAuthFlow.DEVICE_CODE, description="OAuth 2.0 interactive grant")
    oauth_issuer_url: str = Field(default="", description="OAuth issuer URL")
    oauth_authorization_url: str = Field(default="", description="OAuth authorization endpoint")
    oauth_device_authorization_url: str = Field(default="", description="OAuth device authorization endpoint")
    oauth_token_url: str = Field(default="", description="OAuth token endpoint")
    oauth_client_id: str = Field(default="", description="OAuth client identifier")
    oauth_client_secret: str = Field(default="", description="OAuth client secret")
    oauth_client_auth_method: HttpApiOAuthClientAuthMethod = Field(
        default=HttpApiOAuthClientAuthMethod.NONE, description="OAuth token endpoint client authentication"
    )
    oauth_scopes: list[str] = Field(default_factory=list, description="OAuth scopes")
    oauth_access_token: str = Field(default="", description="OAuth access token")
    oauth_refresh_token: str = Field(default="", description="OAuth refresh token")
    oauth_token_type: str = Field(default="Bearer", description="OAuth token type")
    oauth_token_expires_at: str = Field(default="", description="OAuth access token expiry in ISO-8601 UTC")
    oauth_session_id: str = Field(default="", exclude=True, description="Transient OAuth connection session")
    login_path: str = Field(default="", description="Relative login exchange path")
    token_url: str = Field(default="", description="Relative or absolute token exchange endpoint")
    login_method: HttpApiMethod = Field(default=HttpApiMethod.POST, description="Login exchange HTTP method")
    login_body_format: HttpApiLoginBodyFormat = Field(default=HttpApiLoginBodyFormat.JSON, description="Login exchange body format")
    login_username: str = Field(default="", description="Login exchange username")
    login_password: str = Field(default="", description="Login exchange password")
    login_username_field: str = Field(default="username", description="Login payload username field name")
    login_password_field: str = Field(default="password", description="Login payload password field name")
    request_headers: list[HttpApiConfiguredHeader] = Field(default_factory=list, description="Additional request headers applied to API calls")
    request_body_format: HttpApiBodyFormat = Field(
        default=HttpApiBodyFormat.JSON,
        description="Encoding for trusted configured resource request body fields",
    )
    request_body_fields: list[HttpApiTokenField] = Field(
        default_factory=list,
        description="Trusted fields merged into every resource request body",
    )
    token_request_headers: list[HttpApiConfiguredHeader] = Field(default_factory=list, description="Additional headers applied to token exchange requests")
    token_request_fields: list[HttpApiTokenField] = Field(default_factory=list, description="Additional fields applied to token exchange requests")
    send_api_key_to_login: bool = Field(default=False, description="Whether login exchange should include the configured API key")
    send_api_key_to_requests: bool = Field(default=False, description="Whether requests should include the configured API key")
    token_response_path: str = Field(default="access_token", description="Dot path to the token returned by the login response")
    token_expires_in_path: str = Field(default="", description="Optional dot path to token expiry seconds")
    token_header_name: str = Field(default="Authorization", description="Header name used for login-exchange tokens")
    token_prefix: str = Field(default="Bearer", description="Prefix used for login-exchange tokens")
    openapi_source_url: str = Field(default="", description="Optional OpenAPI URL")
    openapi_source_name: str = Field(default="", description="Optional uploaded OpenAPI source filename")
    openapi_source_hash: str = Field(default="", description="Optional OpenAPI source hash")
    openapi_catalog: Optional[OpenApiCatalog] = Field(default=None, description="Bounded normalized OpenAPI catalog")
    raw_openapi_document: str = Field(default="", exclude=True, description="Transient uploaded OpenAPI JSON or YAML document")
    method_policies: dict[str, HttpApiMethodPolicy] = Field(
        default_factory=lambda: dict(DEFAULT_HTTP_API_METHOD_POLICIES),
        description="Per-method read/write/disabled policy map",
    )
    approved_request_headers: list[str] = Field(default_factory=list, description="Approved request headers that may be set per request")
    default_response_selector: str = Field(default="", description="Default constrained response selector")

    @field_validator("documentation_url")
    @classmethod
    def _validate_documentation_url(cls, value: str) -> str:
        candidate = str(value or "").strip()
        if not candidate:
            return ""
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in candidate):
            raise ValueError("documentation_url must not contain ASCII control characters")
        try:
            parsed = urlsplit(candidate)
        except ValueError as exc:
            raise ValueError("documentation_url must be an absolute HTTP(S) URL") from exc
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc or not parsed.hostname:
            raise ValueError("documentation_url must be an absolute HTTP(S) URL")
        return candidate

    @field_validator("method_policies", mode="before")
    @classmethod
    def _validate_method_policies(cls, value: Any) -> dict[str, HttpApiMethodPolicy]:
        if value is None:
            return dict(DEFAULT_HTTP_API_METHOD_POLICIES)
        if not isinstance(value, dict):
            raise TypeError("method_policies must be a mapping")

        normalized: dict[str, HttpApiMethodPolicy] = dict(DEFAULT_HTTP_API_METHOD_POLICIES)
        for raw_key, raw_policy in value.items():
            key = str(raw_key).upper()
            if key not in HTTP_API_STANDARD_METHODS:
                raise ValueError(f"Unsupported HTTP method policy key: {raw_key!r}")
            try:
                normalized[key] = HttpApiMethodPolicy(str(getattr(raw_policy, "value", raw_policy)))
            except ValueError as exc:
                raise ValueError(f"Unsupported HTTP method policy value for {key}: {raw_policy!r}") from exc
        return normalized

    @field_validator("request_headers", "token_request_headers")
    @classmethod
    def _validate_unique_header_names(cls, value: list[HttpApiConfiguredHeader]) -> list[HttpApiConfiguredHeader]:
        seen: set[str] = set()
        for row in value:
            lowered = row.name.lower()
            if lowered in seen:
                raise ValueError(f"Duplicate header name: {row.name}")
            seen.add(lowered)
        return value

    @field_validator("token_request_fields", "request_body_fields")
    @classmethod
    def _validate_unique_secret_field_names(cls, value: list[HttpApiTokenField], info: Any) -> list[HttpApiTokenField]:
        seen: set[str] = set()
        label = "request body" if info.field_name == "request_body_fields" else "token request"
        for row in value:
            if row.name in seen:
                raise ValueError(f"Duplicate {label} field name: {row.name}")
            seen.add(row.name)
        return value

    @model_validator(mode="after")
    def _validate_configured_body_content_types(self) -> "HttpApiConnectionConfig":
        validate_configured_content_type(
            self.request_headers,
            self.request_body_format,
            has_body=bool(self.request_body_fields),
            section="request body",
        )
        validate_configured_content_type(
            self.token_request_headers,
            self.login_body_format,
            has_body=bool(self.token_request_fields),
            section="token request body",
        )
        return self

    @field_validator("oauth_scopes", mode="before")
    @classmethod
    def _normalize_oauth_scopes(cls, value: Any) -> list[str]:
        if value is None:
            return []
        values = value.split() if isinstance(value, str) else value
        if not isinstance(values, (list, tuple, set)):
            raise ValueError("oauth_scopes must be a list of scope names")
        return sorted({str(scope).strip() for scope in values if str(scope).strip()})

    @model_validator(mode="after")
    def _validate_oauth_config(self) -> "HttpApiConnectionConfig":
        if self.auth_mode != HttpApiAuthMode.OAUTH2:
            return self
        if not self.oauth_client_id.strip() or not self.oauth_token_url.strip():
            raise ValueError("OAuth client ID and token endpoint are required")
        if self.oauth_client_auth_method != HttpApiOAuthClientAuthMethod.NONE and not self.oauth_client_secret:
            raise ValueError("OAuth client secret is required for the selected client authentication method")
        if self.oauth_flow == HttpApiOAuthFlow.DEVICE_CODE and not self.oauth_device_authorization_url.strip():
            raise ValueError("OAuth device authorization endpoint is required")
        if self.oauth_flow == HttpApiOAuthFlow.AUTHORIZATION_CODE_PKCE and not self.oauth_authorization_url.strip():
            raise ValueError("OAuth authorization endpoint is required")
        return self

    @field_validator("token_header_name")
    @classmethod
    def _validate_token_header_name(cls, value: str) -> str:
        return validate_header_name(value, allow_configured_auth_headers=True)

    @field_validator("token_prefix")
    @classmethod
    def _validate_token_prefix(cls, value: str) -> str:
        candidate = str(value or "")
        if "\r" in candidate or "\n" in candidate:
            raise ValueError("token_prefix must not contain CR or LF characters")
        return candidate

    @field_serializer(*HTTP_API_SECRET_FIELDS, when_used="json")
    def _serialize_secret_field(self, _value: str) -> str:
        return ""

    @field_serializer("request_headers", when_used="json")
    def _serialize_request_headers(self, value: list[HttpApiConfiguredHeader]) -> list[dict[str, str]]:
        return [{"name": row.name, "value": ""} for row in value]

    @field_serializer("token_request_headers", when_used="json")
    def _serialize_token_request_headers(self, value: list[HttpApiConfiguredHeader]) -> list[dict[str, str]]:
        return [{"name": row.name, "value": ""} for row in value]

    @field_serializer("token_request_fields", when_used="json")
    def _serialize_token_request_fields(self, value: list[HttpApiTokenField]) -> list[dict[str, Any]]:
        return [
            {
                "name": row.name,
                "value": "",
                "secret": True,
            }
            for row in value
        ]

    @field_serializer("request_body_fields", when_used="json")
    def _serialize_request_body_fields(self, value: list[HttpApiTokenField]) -> list[dict[str, Any]]:
        return [
            {
                "name": row.name,
                "value": "",
                "secret": True,
            }
            for row in value
        ]


class HttpApiRequest(BaseModel):
    method: HttpApiMethod = Field(description="Configured HTTP method")
    path: str = Field(description="Relative request path")
    query: dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    headers: dict[str, str] = Field(default_factory=dict, description="Approved request headers")
    json_body: Optional[Any] = Field(default=None, description="Optional JSON body; configured server-side fields merge in and win duplicate keys")
    form_body: Optional[dict[str, Any]] = Field(default=None, description="Optional form body; configured server-side fields merge in and win duplicate keys")
    response_selector: str = Field(default="", description="Optional constrained response selector")


class HttpApiExecutionResult(BaseModel):
    status: Optional[int] = Field(default=None, description="HTTP response status code")
    output: Any = Field(description="Raw JSON-compatible output")
    rows: list[dict[str, Any]] = Field(default_factory=list, description="Optional normalized rows")
    columns: list[str] = Field(default_factory=list, description="Optional normalized columns")
    row_count: Optional[int] = Field(default=None, description="Optional normalized row count")
    error: Optional[str] = Field(default=None, description="Execution error message")
    error_kind: Optional[str] = Field(default=None, description="Execution error category")
    response_bytes: Optional[int] = Field(default=None, description="Raw response size in bytes")


class HttpApiValidationResult(BaseModel):
    success: bool = Field(description="Whether test connection validation succeeded")
    message: str = Field(description="Human-readable validation result message")
    details: Optional[dict[str, Any]] = Field(default=None, description="Optional validation details")
