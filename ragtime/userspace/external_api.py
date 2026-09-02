from __future__ import annotations

import hashlib
import hmac
import json
import logging
import re
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from ragtime.core.database import get_db
from ragtime.core.datetimes import utc_now
from ragtime.userspace.service import userspace_service

if TYPE_CHECKING:
    from prisma.types import WorkspaceApiRequestLogWhereInput

logger = logging.getLogger(__name__)

_MANIFEST_RELATIVE_PATH = Path(".ragtime") / "external-api.json"
_TOKEN_PREFIX = "rtws"
_TOKEN_SELECTOR_HEX_LENGTH = 32
_TOKEN_SECRET_BYTES = 32
_REQUEST_LOG_RETENTION_DAYS = 90
_ALLOWED_METHODS = {"GET", "HEAD"}
_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_REQUEST_LOG_CLEANUP_LAST_RUN_ON: date | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceServicePrincipal:
    credential_id: str
    credential_label: str
    workspace_id: str
    endpoint_id: str
    endpoint_key: str
    endpoint_label: str
    method: str
    path_template: str


@dataclass(frozen=True, slots=True)
class ExternalApiManifestCandidate:
    key: str
    label: str
    description: str
    method: str
    path: str
    valid: bool
    errors: tuple[str, ...] = ()
    definition_hash: str = ""


@dataclass(frozen=True, slots=True)
class ExternalApiManifest:
    version: int | None
    valid: bool
    errors: list[str] = field(default_factory=list)
    candidates: list[ExternalApiManifestCandidate] = field(default_factory=list)


def _workspace_files_root(workspace_id: str) -> Path:
    return userspace_service._workspace_files_dir(workspace_id)


def _manifest_path(files_root: Path) -> Path:
    return files_root / _MANIFEST_RELATIVE_PATH


def _canonical_definition_hash(*, version: int, key: str, method: str, path: str) -> str:
    payload = json.dumps(
        {"version": version, "key": key, "method": method, "path": path},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonicalize_external_api_path(
    raw_path: str,
    *,
    raw_path_bytes: bytes | str | None = None,
    allow_root: bool,
    reject_reserved: bool,
) -> str:
    path = str(raw_path or "").strip()
    if not path:
        raise ValueError("path is required")
    if raw_path_bytes is not None:
        encoded_path = raw_path_bytes.encode("utf-8", errors="ignore") if isinstance(raw_path_bytes, str) else bytes(raw_path_bytes)
        lowered_encoded_path = encoded_path.lower()
        if b"%2f" in lowered_encoded_path or b"%5c" in lowered_encoded_path or b"%2e" in lowered_encoded_path:
            raise ValueError("path must not contain encoded separators or dot segments")
    if any(ord(char) > 127 for char in path):
        raise ValueError("path must be ASCII")
    if "\\" in path:
        raise ValueError("path must not contain backslashes")
    if "?" in path or "#" in path:
        raise ValueError("path must not contain query strings or fragments")
    lowered = path.lower()
    if "%2f" in lowered or "%5c" in lowered or "%2e" in lowered:
        raise ValueError("path must not contain encoded separators or dot segments")
    if not path.startswith("/"):
        raise ValueError("path must be absolute")
    if path != "/" and path.endswith("/"):
        raise ValueError("path must not have a trailing slash")
    if path.startswith("//") or "//" in path:
        raise ValueError("path must not contain duplicate slashes")
    segments = path.split("/")
    if any(segment in {".", ".."} for segment in segments):
        raise ValueError("path must not contain dot segments")
    if path == "/" and not allow_root:
        raise ValueError("path is reserved")
    if reject_reserved and (lowered == "/__ragtime" or lowered.startswith("/__ragtime/")):
        raise ValueError("path is reserved")
    if reject_reserved and (lowered == "/auth" or lowered.startswith("/auth/")):
        raise ValueError("path is reserved")
    return path


def _normalize_manifest_path(raw_path: str) -> str:
    return _canonicalize_external_api_path(raw_path, allow_root=False, reject_reserved=True)


def parse_external_api_manifest(files_root: Path) -> ExternalApiManifest:
    manifest_path = _manifest_path(files_root)
    if not manifest_path.exists():
        return ExternalApiManifest(version=1, valid=True, errors=[], candidates=[])

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return ExternalApiManifest(version=None, valid=False, errors=[f"invalid manifest JSON: {exc}"], candidates=[])

    if not isinstance(payload, dict):
        return ExternalApiManifest(version=None, valid=False, errors=["manifest must be a JSON object"], candidates=[])

    version = payload.get("version")
    top_level_errors: list[str] = []
    if version != 1:
        top_level_errors.append("version must be exactly 1")

    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, list):
        return ExternalApiManifest(
            version=version if isinstance(version, int) else None, valid=False, errors=top_level_errors + ["endpoints must be an array"], candidates=[]
        )

    candidates: list[ExternalApiManifestCandidate] = []
    key_counts: dict[str, int] = {}
    route_counts: dict[tuple[str, str], int] = {}
    aggregate_errors = list(top_level_errors)
    parsed_rows: list[dict[str, Any]] = []
    for index, item in enumerate(endpoints):
        errors: list[str] = []
        if not isinstance(item, dict):
            aggregate_errors.append(f"endpoint[{index}] must be an object")
            continue
        key = str(item.get("key") or "")
        label = str(item.get("label") or "")
        description = str(item.get("description") or "")
        method = str(item.get("method") or "").upper()
        path = str(item.get("path") or "")
        if not _KEY_RE.fullmatch(key):
            errors.append(f"endpoint[{index}] key is invalid")
        if not (1 <= len(label) <= 100):
            errors.append(f"endpoint[{index}] label must be 1-100 characters")
        if len(description) > 500:
            errors.append(f"endpoint[{index}] description must be 0-500 characters")
        if method not in _ALLOWED_METHODS:
            errors.append(f"endpoint[{index}] method must be GET or HEAD")
        try:
            normalized_path = _normalize_manifest_path(path)
        except ValueError as exc:
            errors.append(f"endpoint[{index}] path is not canonical: {exc}")
            normalized_path = path
        key_counts[key] = key_counts.get(key, 0) + 1
        route_counts[(method, normalized_path)] = route_counts.get((method, normalized_path), 0) + 1
        parsed_rows.append(
            {
                "key": key,
                "label": label,
                "description": description,
                "method": method,
                "path": normalized_path,
                "errors": errors,
            }
        )

    for row in parsed_rows:
        row_errors = list(row["errors"])
        if key_counts.get(row["key"], 0) > 1:
            duplicate_error = f"duplicate endpoint key: {row['key']}"
            if duplicate_error not in aggregate_errors:
                aggregate_errors.append(duplicate_error)
            row_errors.append(duplicate_error)
        if route_counts.get((row["method"], row["path"]), 0) > 1:
            duplicate_error = f"duplicate endpoint route: {row['method']} {row['path']}"
            if duplicate_error not in aggregate_errors:
                aggregate_errors.append(duplicate_error)
            row_errors.append(duplicate_error)
        definition_hash = ""
        if not row_errors and version == 1:
            definition_hash = _canonical_definition_hash(
                version=1,
                key=row["key"],
                method=row["method"],
                path=row["path"],
            )
        candidate = ExternalApiManifestCandidate(
            key=row["key"],
            label=row["label"],
            description=row["description"],
            method=row["method"],
            path=row["path"],
            valid=not row_errors and version == 1,
            errors=tuple(row_errors),
            definition_hash=definition_hash,
        )
        candidates.append(candidate)
        aggregate_errors.extend(error for error in row_errors if error not in aggregate_errors)

    return ExternalApiManifest(
        version=version if isinstance(version, int) else None,
        valid=not aggregate_errors,
        errors=aggregate_errors,
        candidates=candidates,
    )


def _parse_service_token(token: str) -> tuple[str, str]:
    parts = str(token or "").split("_", 2)
    if len(parts) != 3 or parts[0] != _TOKEN_PREFIX:
        raise HTTPException(status_code=401, detail="Invalid service credential")
    selector = parts[1]
    secret = parts[2]
    if len(selector) != _TOKEN_SELECTOR_HEX_LENGTH or any(char not in "0123456789abcdef" for char in selector) or not secret:
        raise HTTPException(status_code=401, detail="Invalid service credential")
    return selector, secret


def _hash_service_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _build_service_token() -> tuple[str, str, str]:
    selector = secrets.token_hex(_TOKEN_SELECTOR_HEX_LENGTH // 2)
    token = f"{_TOKEN_PREFIX}_{selector}_{secrets.token_urlsafe(_TOKEN_SECRET_BYTES)}"
    return token, selector, _hash_service_token(token)


def _display_token_prefix(selector: str) -> str:
    return f"{_TOKEN_PREFIX}_{selector}_"


def _manifest_candidate_map(manifest: ExternalApiManifest) -> dict[str, ExternalApiManifestCandidate]:
    return {candidate.key: candidate for candidate in manifest.candidates if candidate.valid}


def _endpoint_response_item(row: Any, *, stale: bool) -> dict[str, Any]:
    approved_at = getattr(row, "approvedAt", None)
    return {
        "id": row.id,
        "key": row.key,
        "label": row.label,
        "description": row.description,
        "method": row.method,
        "path": row.path,
        "enabled": bool(row.enabled),
        "stale": stale,
        "definition_hash": row.definitionHash,
        "approved_at": approved_at.isoformat() if isinstance(approved_at, datetime) else None,
    }


def _credential_endpoint_keys(row: Any) -> list[str]:
    grants = list(getattr(row, "endpointGrants", []) or [])
    keys = [grant.endpoint.key for grant in grants if getattr(grant, "endpoint", None) is not None]
    return sorted({str(key) for key in keys})


def _credential_response_item(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "label": row.label,
        "token_prefix": _display_token_prefix(row.tokenPrefix),
        "enabled": bool(row.enabled),
        "expires_at": row.expiresAt.isoformat() if getattr(row, "expiresAt", None) else None,
        "last_used_at": row.lastUsedAt.isoformat() if getattr(row, "lastUsedAt", None) else None,
        "request_count": int(getattr(row, "requestCount", 0) or 0),
        "revoked_at": row.revokedAt.isoformat() if getattr(row, "revokedAt", None) else None,
        "endpoint_keys": _credential_endpoint_keys(row),
    }


async def authenticate_workspace_service_request(*, workspace_id: str, method: str, path: str, bearer_token: str) -> WorkspaceServicePrincipal:
    selector, _ = _parse_service_token(bearer_token)
    normalized_path = _normalize_manifest_path(path)
    db = await get_db()
    credential = await db.workspaceservicecredential.find_first(
        where={"workspaceId": workspace_id, "tokenPrefix": selector},
        include={"endpointGrants": {"include": {"endpoint": True}}},
    )
    if credential is None:
        raise HTTPException(status_code=401, detail="Invalid service credential")
    if not hmac.compare_digest(str(getattr(credential, "tokenHash", "")), _hash_service_token(bearer_token)):
        raise HTTPException(status_code=401, detail="Invalid service credential")
    now = utc_now()
    if not bool(getattr(credential, "enabled", False)) or getattr(credential, "revokedAt", None) is not None:
        raise HTTPException(status_code=401, detail="Invalid service credential")
    expires_at = getattr(credential, "expiresAt", None)
    if expires_at is not None and expires_at <= now:
        raise HTTPException(status_code=401, detail="Invalid service credential")

    manifest = parse_external_api_manifest(_workspace_files_root(workspace_id))
    manifest_candidates = _manifest_candidate_map(manifest)
    normalized_method = str(method or "").upper()
    for grant in list(getattr(credential, "endpointGrants", []) or []):
        endpoint = getattr(grant, "endpoint", None)
        if endpoint is None or not bool(getattr(endpoint, "enabled", False)):
            continue
        if str(endpoint.method).upper() != normalized_method or str(endpoint.path) != normalized_path:
            continue
        candidate = manifest_candidates.get(str(endpoint.key))
        if candidate is None or candidate.definition_hash != getattr(endpoint, "definitionHash", ""):
            raise HTTPException(status_code=403, detail="Published endpoint is stale or unapproved")
        return WorkspaceServicePrincipal(
            credential_id=str(credential.id),
            credential_label=str(credential.label),
            workspace_id=str(workspace_id),
            endpoint_id=str(endpoint.id),
            endpoint_key=str(endpoint.key),
            endpoint_label=str(endpoint.label),
            method=str(endpoint.method),
            path_template=str(endpoint.path),
        )

    raise HTTPException(status_code=403, detail="Service credential does not grant this endpoint")


async def _cleanup_old_request_logs_if_due(db: Any, *, now: datetime) -> None:
    global _REQUEST_LOG_CLEANUP_LAST_RUN_ON
    if _REQUEST_LOG_CLEANUP_LAST_RUN_ON == now.date():
        return
    try:
        await db.workspaceapirequestlog.delete_many(where={"createdAt": {"lt": now - timedelta(days=_REQUEST_LOG_RETENTION_DAYS)}})
        _REQUEST_LOG_CLEANUP_LAST_RUN_ON = now.date()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to clean up workspace external API request logs")


async def record_workspace_api_request(principal: WorkspaceServicePrincipal, *, status_code: int, duration_ms: int, client_fingerprint: str | None) -> None:
    db = await get_db()
    now = utc_now()
    await _cleanup_old_request_logs_if_due(db, now=now)
    await db.workspaceservicecredential.update(
        where={"id": principal.credential_id},
        data={"lastUsedAt": now, "requestCount": {"increment": 1}},
    )
    await db.workspaceapirequestlog.create(
        data={
            "id": str(uuid.uuid4()),
            "workspaceId": principal.workspace_id,
            "credentialId": principal.credential_id,
            "credentialLabel": principal.credential_label,
            "endpointKey": principal.endpoint_key,
            "endpointLabel": principal.endpoint_label,
            "method": principal.method,
            "pathTemplate": principal.path_template,
            "statusCode": int(status_code),
            "durationMs": int(duration_ms),
            "clientFingerprint": client_fingerprint,
            "createdAt": now,
        }
    )


async def record_workspace_api_denied_request(
    *,
    workspace_id: str,
    method: str,
    path: str,
    status_code: int,
    reason: str,
    token_selector: str | None,
    client_fingerprint: str | None,
) -> None:
    # Denied-request rows intentionally omit auth failure internals so the log
    # preserves only stable request accounting fields, not investigative hints
    # or public token selectors that are outside the fixed response contract.
    _ = reason, token_selector
    db = await get_db()
    now = utc_now()
    await _cleanup_old_request_logs_if_due(db, now=now)
    safe_path = str(path or "").split("?", 1)[0].split("#", 1)[0] or "/"
    try:
        safe_path = _normalize_manifest_path(safe_path)
    except Exception:  # noqa: BLE001
        safe_path = safe_path if safe_path.startswith("/") else "/"
    await db.workspaceapirequestlog.create(
        data={
            "id": str(uuid.uuid4()),
            "workspaceId": workspace_id,
            "credentialId": None,
            "credentialLabel": "",
            "endpointKey": "",
            "endpointLabel": "",
            "method": str(method or "").upper(),
            "pathTemplate": safe_path,
            "statusCode": int(status_code),
            "durationMs": 0,
            "clientFingerprint": client_fingerprint,
            "createdAt": now,
        }
    )


async def _record_management_audit(workspace_id: str, user_id: str, event_type: str, payload: dict[str, Any]) -> None:
    try:
        recorder = getattr(userspace_service, "_record_runtime_audit_event", None)
        if recorder is None:
            return
        await recorder(workspace_id, user_id, event_type, payload)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to record workspace external API audit event")


async def get_external_api_manifest_payload(*, workspace_id: str, preview_origin: str) -> dict[str, Any]:
    manifest = parse_external_api_manifest(_workspace_files_root(workspace_id))
    return {
        "preview_origin": preview_origin,
        "version": manifest.version,
        "valid": manifest.valid,
        "errors": list(manifest.errors),
        "candidates": [
            {
                "key": candidate.key,
                "label": candidate.label,
                "description": candidate.description,
                "method": candidate.method,
                "path": candidate.path,
                "valid": candidate.valid,
                "errors": list(candidate.errors),
            }
            for candidate in manifest.candidates
        ],
    }


async def list_workspace_published_endpoints_payload(*, workspace_id: str, preview_origin: str) -> dict[str, Any]:
    db = await get_db()
    rows = await db.workspacepublishedendpoint.find_many(where={"workspaceId": workspace_id}, order={"createdAt": "asc"})
    manifest = parse_external_api_manifest(_workspace_files_root(workspace_id))
    manifest_candidates = _manifest_candidate_map(manifest)
    return {
        "preview_origin": preview_origin,
        "items": [
            _endpoint_response_item(
                row,
                stale=(manifest_candidates.get(str(row.key)) is None or manifest_candidates[str(row.key)].definition_hash != row.definitionHash),
            )
            for row in rows
        ],
    }


async def publish_workspace_endpoint(*, workspace_id: str, key: str, user_id: str) -> dict[str, Any]:
    db = await get_db()
    manifest = parse_external_api_manifest(_workspace_files_root(workspace_id))
    if not manifest.valid:
        raise HTTPException(status_code=400, detail="External API manifest is invalid")
    candidate = next((item for item in manifest.candidates if item.key == key and item.valid), None)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Manifest endpoint not found")
    now = utc_now()
    existing = await db.workspacepublishedendpoint.find_first(where={"workspaceId": workspace_id, "key": key})
    if existing is None:
        row = await db.workspacepublishedendpoint.create(
            data={
                "id": str(uuid.uuid4()),
                "workspaceId": workspace_id,
                "key": candidate.key,
                "label": candidate.label,
                "description": candidate.description,
                "method": candidate.method,
                "path": candidate.path,
                "definitionHash": candidate.definition_hash,
                "enabled": True,
                "approvedByUserId": user_id,
                "approvedAt": now,
                "createdAt": now,
                "updatedAt": now,
            }
        )
    else:
        updated_row = await db.workspacepublishedendpoint.update(
            where={"id": existing.id},
            data={
                "label": candidate.label,
                "description": candidate.description,
                "method": candidate.method,
                "path": candidate.path,
                "definitionHash": candidate.definition_hash,
                "enabled": True,
                "approvedByUser": {"connect": {"id": user_id}},
                "approvedAt": now,
                "updatedAt": now,
            },
        )
        if updated_row is None:
            raise HTTPException(status_code=404, detail="Published endpoint not found")
        row = updated_row
    await _record_management_audit(workspace_id, user_id, "external_api.endpoint_published", {"endpoint_key": key})
    return _endpoint_response_item(row, stale=False)


async def unpublish_workspace_endpoint(*, workspace_id: str, endpoint_id: str, user_id: str) -> dict[str, Any]:
    db = await get_db()
    row = await db.workspacepublishedendpoint.find_unique(where={"id": endpoint_id})
    if row is None or str(row.workspaceId) != workspace_id:
        raise HTTPException(status_code=404, detail="Published endpoint not found")
    await db.workspacepublishedendpoint.delete(where={"id": endpoint_id})
    await _record_management_audit(workspace_id, user_id, "external_api.endpoint_unpublished", {"endpoint_id": endpoint_id, "endpoint_key": row.key})
    return _endpoint_response_item(row, stale=False)


async def list_workspace_service_credentials_payload(*, workspace_id: str) -> dict[str, Any]:
    db = await get_db()
    rows = await db.workspaceservicecredential.find_many(
        where={"workspaceId": workspace_id},
        include={"endpointGrants": {"include": {"endpoint": True}}},
        order={"createdAt": "asc"},
    )
    return {"items": [_credential_response_item(row) for row in rows]}


async def create_workspace_service_credential(
    *, workspace_id: str, user_id: str, label: str, endpoint_keys: list[str], expires_at: datetime | None
) -> dict[str, Any]:
    cleaned_label = str(label or "").strip()
    if not (1 <= len(cleaned_label) <= 100):
        raise HTTPException(status_code=400, detail="label must be 1-100 characters")
    unique_endpoint_keys = sorted({str(key).strip() for key in endpoint_keys if str(key).strip()})
    if not unique_endpoint_keys:
        raise HTTPException(status_code=400, detail="At least one endpoint key is required")
    if expires_at is not None and expires_at <= utc_now():
        raise HTTPException(status_code=400, detail="expires_at must be in the future")
    db = await get_db()
    endpoint_rows = await db.workspacepublishedendpoint.find_many(where={"workspaceId": workspace_id}, order={"createdAt": "asc"})
    manifest_candidates = _manifest_candidate_map(parse_external_api_manifest(_workspace_files_root(workspace_id)))
    by_key = {str(row.key): row for row in endpoint_rows if bool(getattr(row, "enabled", False))}
    selected = [by_key[key] for key in unique_endpoint_keys if key in by_key]
    if len(selected) != len(unique_endpoint_keys):
        raise HTTPException(status_code=400, detail="All endpoint keys must reference published endpoints")
    for endpoint in selected:
        candidate = manifest_candidates.get(str(endpoint.key))
        if candidate is None or candidate.definition_hash != getattr(endpoint, "definitionHash", ""):
            raise HTTPException(status_code=400, detail="Published endpoints must be reapproved before granting access")
    token, selector, token_hash = _build_service_token()
    now = utc_now()
    row = await db.workspaceservicecredential.create(
        data={
            "id": str(uuid.uuid4()),
            "workspaceId": workspace_id,
            "label": cleaned_label,
            "tokenPrefix": selector,
            "tokenHash": token_hash,
            "enabled": True,
            "expiresAt": expires_at,
            "createdByUserId": user_id,
            "createdAt": now,
            "updatedAt": now,
        }
    )
    grants = []
    for endpoint in selected:
        await db.workspaceservicecredentialendpoint.create(data={"credentialId": row.id, "endpointId": endpoint.id})
        grants.append(SimpleNamespace(endpoint=endpoint))
    await _record_management_audit(workspace_id, user_id, "external_api.credential_created", {"credential_id": row.id, "endpoint_keys": unique_endpoint_keys})
    payload = _credential_response_item(
        SimpleNamespace(
            id=row.id,
            label=row.label,
            tokenPrefix=row.tokenPrefix,
            enabled=row.enabled,
            expiresAt=row.expiresAt,
            lastUsedAt=row.lastUsedAt,
            requestCount=row.requestCount,
            revokedAt=row.revokedAt,
            endpointGrants=grants,
        )
    )
    payload["token"] = token
    return payload


async def rotate_workspace_service_credential(*, workspace_id: str, credential_id: str, user_id: str) -> dict[str, Any]:
    db = await get_db()
    row = await db.workspaceservicecredential.find_unique(
        where={"id": credential_id},
        include={"endpointGrants": {"include": {"endpoint": True}}},
    )
    if row is None or str(row.workspaceId) != workspace_id:
        raise HTTPException(status_code=404, detail="Service credential not found")
    if getattr(row, "revokedAt", None) is not None or not bool(getattr(row, "enabled", False)):
        raise HTTPException(status_code=400, detail="Revoked credentials cannot be rotated")
    token, selector, token_hash = _build_service_token()
    updated = await db.workspaceservicecredential.update(
        where={"id": credential_id},
        data={"tokenPrefix": selector, "tokenHash": token_hash, "updatedAt": utc_now()},
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Service credential not found")
    await _record_management_audit(workspace_id, user_id, "external_api.credential_rotated", {"credential_id": credential_id})
    payload = _credential_response_item(
        SimpleNamespace(
            id=updated.id,
            label=row.label,
            tokenPrefix=updated.tokenPrefix,
            enabled=updated.enabled,
            expiresAt=row.expiresAt,
            lastUsedAt=getattr(row, "lastUsedAt", None),
            requestCount=getattr(row, "requestCount", 0),
            revokedAt=getattr(row, "revokedAt", None),
            endpointGrants=list(getattr(row, "endpointGrants", []) or []),
        )
    )
    payload["token"] = token
    return payload


async def revoke_workspace_service_credential(*, workspace_id: str, credential_id: str, user_id: str) -> dict[str, Any]:
    db = await get_db()
    row = await db.workspaceservicecredential.find_unique(
        where={"id": credential_id},
        include={"endpointGrants": {"include": {"endpoint": True}}},
    )
    if row is None or str(row.workspaceId) != workspace_id:
        raise HTTPException(status_code=404, detail="Service credential not found")
    revoked_at = utc_now()
    updated = await db.workspaceservicecredential.update(
        where={"id": credential_id},
        data={"enabled": False, "revokedAt": revoked_at, "updatedAt": revoked_at},
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Service credential not found")
    await _record_management_audit(workspace_id, user_id, "external_api.credential_revoked", {"credential_id": credential_id})
    return _credential_response_item(
        SimpleNamespace(
            id=updated.id,
            label=row.label,
            tokenPrefix=updated.tokenPrefix,
            enabled=updated.enabled,
            expiresAt=row.expiresAt,
            lastUsedAt=getattr(row, "lastUsedAt", None),
            requestCount=getattr(row, "requestCount", 0),
            revokedAt=updated.revokedAt,
            endpointGrants=list(getattr(row, "endpointGrants", []) or []),
        )
    )


async def list_workspace_api_requests_payload(*, workspace_id: str, cursor: str | None, limit: int) -> dict[str, Any]:
    db = await get_db()
    where: WorkspaceApiRequestLogWhereInput = {"workspaceId": workspace_id}
    if cursor:
        cursor_row = await db.workspaceapirequestlog.find_unique(where={"id": cursor})
        if cursor_row is not None and str(getattr(cursor_row, "workspaceId", "")) == workspace_id:
            cursor_created_at = getattr(cursor_row, "createdAt", None)
            cursor_id = str(getattr(cursor_row, "id", ""))
            if isinstance(cursor_created_at, datetime):
                where = {
                    "workspaceId": workspace_id,
                    "OR": [
                        {"createdAt": {"lt": cursor_created_at}},
                        {"AND": [{"createdAt": cursor_created_at}, {"id": {"lt": cursor_id}}]},
                    ],
                }
    rows = await db.workspaceapirequestlog.find_many(
        where=where,
        take=limit + 1,
        order=[{"createdAt": "desc"}, {"id": "desc"}],
    )
    page = list(rows[:limit])
    next_cursor = str(getattr(page[-1], "id", "")) if len(rows) > limit and page else None
    return {
        "cursor": next_cursor,
        "limit": limit,
        "items": [
            {
                "id": row.id,
                "credential_id": getattr(row, "credentialId", None),
                "credential_label": getattr(row, "credentialLabel", ""),
                "endpoint_key": getattr(row, "endpointKey", ""),
                "endpoint_label": getattr(row, "endpointLabel", ""),
                "method": row.method,
                "path_template": row.pathTemplate,
                "status_code": row.statusCode,
                "duration_ms": row.durationMs,
                "created_at": row.createdAt.isoformat(),
            }
            for row in page
        ],
    }
