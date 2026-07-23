from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ragtime.core.encryption import ENCRYPTED_PREFIX, attempt_decrypt, decrypt_secret, encrypt_secret

_NESTED_SECRET_ROW_FIELDS: tuple[str, ...] = (
    "request_headers",
    "token_request_headers",
    "token_request_fields",
)


def _copy_rows(config: dict[str, Any], field: str) -> list[dict[str, Any]]:
    rows = config.get(field)
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _is_secret_bearing_row(field: str, row: dict[str, Any]) -> bool:
    if field == "token_request_fields":
        return bool(row.get("secret"))
    return True


def _iter_nested_secret_rows(config: dict[str, Any] | None) -> Iterable[tuple[str, dict[str, Any]]]:
    payload = dict(config or {})
    for field in _NESTED_SECRET_ROW_FIELDS:
        for row in _copy_rows(payload, field):
            if _is_secret_bearing_row(field, row):
                yield field, row


def _secret_row_path(field: str, row: dict[str, Any]) -> str:
    return f"{field}.{row.get('name', '')}"


def _secret_row_value(row: dict[str, Any]) -> str:
    return str(row.get("value", ""))


def _is_encrypted_secret_value(value: str) -> bool:
    return bool(value) and value.startswith(ENCRYPTED_PREFIX)


def encrypt_http_api_nested_secrets(config: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(config or {})

    for field in _NESTED_SECRET_ROW_FIELDS:
        rows = _copy_rows(result, field)
        for row in rows:
            if _is_secret_bearing_row(field, row) and row.get("value"):
                row["value"] = encrypt_secret(str(row["value"]))
        if field in result:
            result[field] = rows

    return result


def decrypt_http_api_nested_secrets(config: dict[str, Any] | None) -> dict[str, Any]:
    result = dict(config or {})

    for field in _NESTED_SECRET_ROW_FIELDS:
        rows = _copy_rows(result, field)
        for row in rows:
            if _is_secret_bearing_row(field, row):
                row["value"] = decrypt_secret(_secret_row_value(row))
        if field in result:
            result[field] = rows

    return result


def configured_http_api_secret_paths(config: dict[str, Any] | None) -> list[str]:
    return [_secret_row_path(field, row) for field, row in _iter_nested_secret_rows(config) if row.get("value")]


def undecryptable_http_api_secret_paths(config: dict[str, Any] | None) -> list[str]:
    return [_secret_row_path(field, row) for field, row in _iter_nested_secret_rows(config) if (value := _secret_row_value(row)) and not attempt_decrypt(value)]


def iter_http_api_encrypted_secret_values(config: dict[str, Any] | None) -> list[str]:
    return [value for _field, row in _iter_nested_secret_rows(config) if _is_encrypted_secret_value(value := _secret_row_value(row))]


def clear_undecryptable_http_api_nested_secrets(config: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    payload = dict(config or {})
    cleared_paths: list[str] = []

    for field in _NESTED_SECRET_ROW_FIELDS:
        if field not in payload:
            continue
        cleaned_rows: list[dict[str, Any]] = []
        for row in _copy_rows(payload, field):
            value = _secret_row_value(row)
            if _is_secret_bearing_row(field, row) and _is_encrypted_secret_value(value) and not attempt_decrypt(value):
                cleared_paths.append(_secret_row_path(field, row))
                continue
            cleaned_rows.append(row)
        payload[field] = cleaned_rows

    return payload, cleared_paths
