from __future__ import annotations

import hashlib
import json
from typing import Any

import yaml
from yaml.events import AliasEvent

from ragtime.http_api.models import HttpApiMethod, OpenApiCatalog, OpenApiCatalogOperation

_OPENAPI_MAX_SOURCE_BYTES = 2 * 1024 * 1024
_OPENAPI_MAX_OPERATIONS = 500
_OPENAPI_MAX_DEPTH = 40


def _load_document(document: str) -> dict[str, Any]:
    payload = (document or "").encode("utf-8")
    if len(payload) > _OPENAPI_MAX_SOURCE_BYTES:
        raise ValueError("OpenAPI source exceeds size limit")
    stripped = (document or "").strip()
    if not stripped:
        return {}
    if stripped[:1] not in "[{":
        for event in yaml.parse(stripped):
            if isinstance(event, AliasEvent):
                raise ValueError("OpenAPI YAML aliases are not supported")
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        loaded = yaml.safe_load(stripped)
    if not isinstance(loaded, dict):
        raise ValueError("OpenAPI source must decode to an object")
    return loaded


def _scan_for_external_refs(value: Any, depth: int = 0) -> None:
    if depth > _OPENAPI_MAX_DEPTH:
        raise ValueError("OpenAPI document exceeds nesting limit")
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "$ref" and isinstance(item, str) and not item.startswith("#"):
                raise ValueError("OpenAPI external references are not supported")
            _scan_for_external_refs(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _scan_for_external_refs(item, depth + 1)


def normalize_openapi_source(
    *, spec_url: str | None = None, document: str | None = None, document_name: str | None = None
) -> tuple[OpenApiCatalog, dict[str, str]]:
    spec = _load_document(document or "")
    _scan_for_external_refs(spec)
    info = spec.get("info") or {}
    operations: list[OpenApiCatalogOperation] = []
    for path, methods in (spec.get("paths") or {}).items():
        if not isinstance(methods, dict):
            continue
        for method_name, operation in methods.items():
            upper = str(method_name).upper()
            if upper not in HttpApiMethod._value2member_map_:
                continue
            if not isinstance(operation, dict):
                continue
            if len(operations) >= _OPENAPI_MAX_OPERATIONS:
                raise ValueError("OpenAPI document exceeds operation limit")
            operations.append(
                OpenApiCatalogOperation(
                    operation_id=str(operation.get("operationId") or ""),
                    method=HttpApiMethod(upper),
                    path=str(path),
                    summary=str(operation.get("summary") or ""),
                    description=str(operation.get("description") or ""),
                    tags=[str(tag) for tag in (operation.get("tags") or [])[:10]],
                )
            )
    source_hash = hashlib.sha256((document or "").encode("utf-8")).hexdigest() if document else ""
    return (
        OpenApiCatalog(
            title=str(info.get("title") or ""),
            version=str(info.get("version") or ""),
            operations=operations,
        ),
        {
            "openapi_source_url": spec_url or "",
            "openapi_source_name": document_name or "",
            "openapi_source_hash": source_hash,
        },
    )


def search_openapi_catalog(catalog: OpenApiCatalog, query: str, limit: int = 10) -> list[OpenApiCatalogOperation]:
    terms = [term for term in query.lower().split() if term]
    if not terms:
        return catalog.operations[:limit]
    scored: list[tuple[int, OpenApiCatalogOperation]] = []
    for operation in catalog.operations:
        haystack = " ".join(
            [
                operation.operation_id,
                operation.method,
                operation.path,
                operation.summary,
                operation.description,
                " ".join(operation.tags),
            ]
        ).lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            scored.append((score, operation))
    scored.sort(key=lambda item: (-item[0], item[1].path, item[1].method))
    return [operation for _score, operation in scored[:limit]]
