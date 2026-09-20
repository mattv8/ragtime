"""Public, transport-neutral content-protection enforcement service."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Any, Iterator
from uuid import UUID, uuid4

from langchain_core.messages import BaseMessage
from prisma import Json

from ragtime.content_protection.models import PUBLIC_SCOPE, ContentProtectionConfig, ContentProtectionError, Profile, ProtectionContext, default_profiles
from ragtime.content_protection.policy import resolve_required
from ragtime.content_protection.provider import classify, security_classification_context
from ragtime.content_protection.store import load_config_record, resolve_identities, save_config_record, validate_references
from ragtime.core import database

_MAX_BYTES = 1024 * 1024
_CACHE_TTL = 60.0
_TURN_BUDGET = 30.0
_queue = asyncio.Semaphore(16)
_cache_secret = secrets.token_bytes(32)


@dataclass
class _TurnState:
    started: float = field(default_factory=monotonic)
    spent: float = 0.0
    terminal: ContentProtectionError | None = None


@dataclass(frozen=True)
class _ResolvedPolicy:
    required: bool
    provenance: str
    verified: set[str]
    groups: dict[str, set[str]]
    expiries: dict[str, str | None]
    profile_sets: list[list[dict[str, object]]]


_context: ContextVar[ProtectionContext | None] = ContextVar("content_protection_context", default=None)
_turn: ContextVar[_TurnState | None] = ContextVar("content_protection_turn", default=None)
_decision_cache: dict[str, tuple[float, dict[str, str]]] = {}


@contextmanager
def protection_context(context: ProtectionContext) -> Iterator[None]:
    """Bind context while retaining the same mutable budget/terminal state when nested."""
    token = _context.set(context)
    state = _turn.get()
    state_token = None if state is not None else _turn.set(_TurnState())
    try:
        yield
    finally:
        _context.reset(token)
        if state_token is not None:
            _turn.reset(state_token)


def current_context() -> ProtectionContext | None:
    return _context.get()


def canonical_serialize(value: Any) -> bytes:
    """Serialize known inspectable transport values; never stringify opaque values."""

    def validate(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if item != item or item in (float("inf"), float("-inf")):
                raise TypeError("non-finite number")
            return item
        if isinstance(item, (bytes, bytearray)):
            raise TypeError("binary")
        if isinstance(item, (datetime, UUID)):
            return item.isoformat() if isinstance(item, datetime) else str(item)
        normalized = _normalize_known_transport_model(item)
        if normalized is not None:
            return validate(normalized)
        if isinstance(item, (list, tuple)):
            return [validate(part) for part in item]
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise TypeError("non-string key")
            return {key: validate(part) for key, part in item.items()}
        raise TypeError("opaque value")

    return json.dumps(validate(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def normalize_transport_value(value: Any) -> Any:
    """Return the exact JSON value sent to the classifier for supported transports.

    This is deliberately a small allowlist.  In particular it does not use
    ``str()``, ``dict()``, or a generic Pydantic fallback, because doing so can
    turn an uninspectable attachment or arbitrary object into apparent text.
    """

    def normalize(item: Any) -> Any:
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if item != item or item in (float("inf"), float("-inf")):
                raise TypeError("non-finite number")
            return item
        if isinstance(item, (bytes, bytearray)):
            raise TypeError("binary")
        if isinstance(item, (datetime, UUID)):
            return item.isoformat() if isinstance(item, datetime) else str(item)
        known = _normalize_known_transport_model(item)
        if known is not None:
            return normalize(known)
        if isinstance(item, (list, tuple)):
            return [normalize(part) for part in item]
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise TypeError("non-string key")
            content_type = item.get("type")
            if content_type in {"image", "image_url", "input_image", "audio", "input_audio", "video", "file", "document", "resource", "binary"}:
                raise TypeError("unsupported modality")
            return {key: normalize(part) for key, part in item.items()}
        raise TypeError("opaque value")

    return normalize(value)


def _normalize_known_transport_model(value: Any) -> Any | None:
    """Losslessly dump only models used by supported Ragtime transports."""
    if isinstance(value, BaseMessage):
        return value.model_dump(mode="json")
    module = type(value).__module__
    # MCP TextContent/CallToolResult and Ragtime API/indexer response models
    # are Pydantic transport contracts.  Their JSON dump preserves text,
    # ordering, tool arguments, metadata, datetimes, and UUIDs.
    if module.startswith(("mcp.types", "ragtime.api.", "ragtime.indexer.", "ragtime.models")) and hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return None


def _digest(value: Any) -> str:
    return hmac.new(_cache_secret, canonical_serialize(value), hashlib.sha256).hexdigest()


async def _resolve(config: ContentProtectionConfig, context: ProtectionContext, *, tool_id: str | None = None) -> _ResolvedPolicy:
    effective = ProtectionContext(
        user_id=context.user_id,
        audience_user_ids=context.audience_user_ids,
        surface=context.surface,
        mcp_route=context.mcp_route,
        tool_id=tool_id or context.tool_id,
        resource_id=context.resource_id,
        public=context.public,
        baseline=context.baseline,
    )
    identities = {identity for identity in (effective.user_id, *effective.audience_user_ids) if identity}
    verified, groups, expiries = await resolve_identities(identities)
    required, provenance = resolve_required(config, effective, groups, verified)
    profiles = {profile.id: profile for profile in config.profiles}
    mapped = {mapping.group_id: mapping.profile_id for mapping in config.group_profiles}
    profile_sets: list[list[dict[str, object]]] = []
    # Each identity/audience is an independent audience set: union inside, intersection across.
    audience = tuple(dict.fromkeys((effective.user_id, *effective.audience_user_ids)))
    if effective.baseline == "public" or effective.public:
        # This is intentionally not an editable profile mapping.
        profile_sets.append([Profile(id="public_baseline", name="Public", level=0, scope=PUBLIC_SCOPE).model_dump(mode="json")])
    elif effective.baseline in {"anonymous", "service"}:
        profile_sets.append([profiles.get("standard", default_profiles()[0]).model_dump(mode="json")])
    for identity in audience:
        if not identity or identity not in verified:
            continue
        ids = {"standard"} | {mapped[group] for group in groups.get(identity, set()) if group in mapped}
        profile_sets.append([profiles[profile_id].model_dump(mode="json") for profile_id in sorted(ids) if profile_id in profiles])
    if not profile_sets:
        profile_sets.append([profiles.get("standard", default_profiles()[0]).model_dump(mode="json")])
    return _ResolvedPolicy(required, provenance, verified, groups, expiries, profile_sets)


async def classification_required(context: ProtectionContext | None = None) -> bool:
    try:
        config = await load_config()
    except ContentProtectionError:
        raise
    except Exception:
        # A missing authoritative policy must never be interpreted as master-off.
        raise ContentProtectionError("classifier_unavailable", str(uuid4())) from None
    if not config.enabled:
        return False
    resolved = context or current_context() or ProtectionContext(baseline="anonymous")
    try:
        return (await _resolve(config, resolved)).required
    except ContentProtectionError:
        raise
    except Exception:
        raise ContentProtectionError("classifier_unavailable", str(uuid4())) from None


async def load_config() -> ContentProtectionConfig:
    return await load_config_record()


async def save_config(config: ContentProtectionConfig | dict[str, object], expected_revision: int, actor_id: str | None) -> ContentProtectionConfig:
    candidate = ContentProtectionConfig.model_validate(config)
    await validate_references(candidate)
    current = await load_config()
    model_changed = candidate.classifier_model != current.classifier_model
    enabling = not current.enabled and candidate.enabled
    if candidate.enabled and (enabling or model_changed):
        await probe_readiness(candidate)
    return await save_config_record(candidate, expected_revision, actor_id)


async def preview_policy(context: ProtectionContext, config: ContentProtectionConfig | None = None) -> dict[str, object]:
    active = config or await load_config()
    resolved = await _resolve(active, context)
    return {"required": resolved.required, "provenance": resolved.provenance, "profiles": resolved.profile_sets}


async def test_sample(config: ContentProtectionConfig | dict[str, object], sample: object, profile_ids: list[str]) -> dict[str, object]:
    active = ContentProtectionConfig.model_validate(config)
    try:
        normalized_sample = normalize_transport_value(sample)
    except (TypeError, ValueError):
        raise ContentProtectionError("content_unclassifiable", str(uuid4())) from None
    if len(canonical_serialize(normalized_sample)) > _MAX_BYTES:
        raise ContentProtectionError("content_unclassifiable", str(uuid4()))
    selected = [profile.model_dump(mode="json") for profile in active.profiles if profile.id in set(profile_ids)]
    if profile_ids and len(selected) != len(set(profile_ids)):
        raise ValueError("unknown_profile_reference")
    started = monotonic()
    verdict = await _classify(
        active,
        {
            "direction": "sample",
            "candidate": normalized_sample,
            "audience_constraints": [selected or [default_profiles()[0].model_dump(mode="json")]],
            "policy_revision": active.revision,
        },
        include_reason=True,
    )
    return {
        "code": verdict.get("reason_code", "permitted"),
        "verdict": verdict.get("verdict"),
        "reason": verdict.get("reason"),
        "latency": monotonic() - started,
    }


async def probe_readiness(config: ContentProtectionConfig | dict[str, object]) -> dict[str, object]:
    active = ContentProtectionConfig.model_validate(config)
    if not active.classifier_model:
        raise ContentProtectionError("classifier_unavailable", str(uuid4()))
    allow = await _classify(
        active,
        {
            "direction": "probe",
            "candidate": "ordinary operational information",
            "audience_constraints": [[default_profiles()[0].model_dump(mode="json")]],
            "policy_revision": active.revision,
        },
    )
    deny = await _classify(
        active,
        {
            "direction": "probe",
            "candidate": "restricted company financial records",
            "audience_constraints": [[default_profiles()[0].model_dump(mode="json")]],
            "policy_revision": active.revision,
        },
    )
    if allow.get("verdict") != "allow" or deny.get("verdict") != "deny":
        raise ContentProtectionError("classifier_invalid_response", str(uuid4()))
    return {"code": "ready", "verdict": "allow"}


async def list_decisions(limit: int = 50) -> list[dict[str, object]]:
    rows = await (await database.get_db()).contentprotectiondecision.find_many(take=max(1, min(limit, 50)), order={"createdAt": "desc"})
    return [dict(getattr(row, "metadata", {}), request_id=row.requestId, created_at=row.createdAt.isoformat()) for row in rows]


async def _audit(request_id: str, metadata: dict[str, object]) -> None:
    """Persist body-free metadata and opportunistically retain thirty days only."""
    try:
        db = await database.get_db()
        await db.contentprotectiondecision.create(data={"requestId": request_id, "metadata": Json(metadata)})
        await db.contentprotectiondecision.delete_many(where={"createdAt": {"lt": datetime.now(UTC) - timedelta(days=30)}})
    except Exception:
        # Audit failure must not leak a candidate or turn a safe denial into a 500.
        return


async def _provider_settings_identity() -> str:
    """Bind reuse to the configured provider credentials/base URL without logging either."""
    settings = await (await database.get_db()).appsettings.find_unique(where={"id": "default"})
    if settings is None:
        return "missing"
    updated_at = getattr(settings, "updatedAt", None)
    data = settings.model_dump(mode="json") if hasattr(settings, "model_dump") else {"updated_at": updated_at.isoformat() if updated_at is not None else None}
    return _digest(data)


def _store_allow(key: str, verdict: dict[str, str]) -> None:
    now = monotonic()
    for cache_key, (created, _value) in list(_decision_cache.items()):
        if now - created > _CACHE_TTL:
            _decision_cache.pop(cache_key, None)
    if len(_decision_cache) >= 1024:
        oldest = min(_decision_cache, key=lambda cache_key: _decision_cache[cache_key][0])
        _decision_cache.pop(oldest, None)
    _decision_cache[key] = (now, verdict)


async def _classify_with_limits(config: ContentProtectionConfig, envelope: dict[str, object], state: _TurnState) -> dict[str, str]:
    if state.terminal is not None:
        raise state.terminal
    if state.spent >= _TURN_BUDGET:
        raise ContentProtectionError("classifier_unavailable", str(uuid4()))
    queued = monotonic()
    try:
        await asyncio.wait_for(_queue.acquire(), timeout=1)
    except TimeoutError:
        raise ContentProtectionError("classifier_unavailable", str(uuid4())) from None
    state.spent += monotonic() - queued
    try:
        if state.spent >= _TURN_BUDGET:
            raise ContentProtectionError("classifier_unavailable", str(uuid4()))
        started = monotonic()
        verdict = await asyncio.wait_for(_classify(config, envelope), timeout=min(5.0, _TURN_BUDGET - state.spent))
        state.spent += monotonic() - started
        return verdict
    finally:
        _queue.release()


async def _classify(config: ContentProtectionConfig, envelope: dict[str, object], *, include_reason: bool = False) -> dict[str, str]:
    """Use the provider only through the private security-classification path."""
    with security_classification_context():
        return await classify(config, envelope, include_reason=include_reason)


async def authorize_content(
    candidate: Any,
    *,
    direction: str,
    context: ProtectionContext | None = None,
    tool_id: str | None = None,
    operation: str | None = None,
    supporting_context: Any = None,
    _rechecked: bool = False,
) -> None:
    resolved_context = context or current_context() or ProtectionContext(baseline="anonymous")
    state = _turn.get() or _TurnState()
    if state.terminal is not None:
        raise state.terminal
    request_id = str(uuid4())
    try:
        config = await load_config()
    except ContentProtectionError:
        raise
    except Exception:
        raise ContentProtectionError("classifier_unavailable", request_id) from None
    # Master-off is a complete production bypass. Existing authentication and
    # authorization gates still run in their owning adapters; this service does
    # not need (or attempt) an identity lookup in that state.
    if not config.enabled:
        return
    # Reject opaque/oversized values before any identity/provider work when the
    # global policy already covers this boundary. Group-only coverage still
    # needs the authoritative membership read below.
    preliminary_required = resolve_required(
        config,
        ProtectionContext(
            user_id=resolved_context.user_id,
            audience_user_ids=resolved_context.audience_user_ids,
            surface=resolved_context.surface,
            mcp_route=resolved_context.mcp_route,
            tool_id=tool_id or resolved_context.tool_id,
            resource_id=resolved_context.resource_id,
            public=resolved_context.public,
            baseline=resolved_context.baseline,
        ),
    )[0]
    if preliminary_required:
        try:
            preliminary = canonical_serialize(normalize_transport_value(candidate))
        except (TypeError, ValueError):
            error = ContentProtectionError("content_unclassifiable", request_id)
            state.terminal = error
            raise error
        if len(preliminary) > _MAX_BYTES:
            error = ContentProtectionError("content_unclassifiable", request_id)
            state.terminal = error
            raise error
    try:
        policy = await _resolve(config, resolved_context, tool_id=tool_id)
    except ContentProtectionError:
        raise
    except Exception:
        error = ContentProtectionError("classifier_unavailable", request_id)
        state.terminal = error
        raise error from None
    if not policy.required:
        return
    try:
        normalized_candidate = normalize_transport_value(candidate)
        normalized_supporting_context = normalize_transport_value(supporting_context) if supporting_context is not None else None
        serialized = canonical_serialize(normalized_candidate)
        supporting_serialized = canonical_serialize(normalized_supporting_context)
    except (TypeError, ValueError):
        error = ContentProtectionError("content_unclassifiable", request_id)
        state.terminal = error
        raise error
    if len(serialized) + len(supporting_serialized) > _MAX_BYTES:
        error = ContentProtectionError("content_unclassifiable", request_id)
        state.terminal = error
        raise error
    try:
        provider_settings = await _provider_settings_identity()
    except Exception:
        error = ContentProtectionError("classifier_unavailable", request_id)
        state.terminal = error
        raise error from None
    fingerprint = {
        "candidate": _digest(normalized_candidate),
        "supporting_context": _digest(normalized_supporting_context),
        "revision": config.revision,
        "model": config.classifier_model,
        "enabled": config.enabled,
        "coverage": config.coverage_mode,
        "direction": direction,
        "surface": resolved_context.surface,
        "tool": tool_id or resolved_context.tool_id,
        "operation": operation,
        "resource": resolved_context.resource_id,
        "baseline": resolved_context.baseline,
        "public": resolved_context.public,
        "audiences": policy.profile_sets,
        "memberships": {key: sorted(value) for key, value in policy.groups.items()},
        "expiries": policy.expiries,
        "provenance": policy.provenance,
        "provider_settings": provider_settings,
    }
    key = _digest(fingerprint)
    cached = _decision_cache.get(key)
    cached_verdict = cached[1] if cached is not None else None
    cache_hit = cached is not None and monotonic() - cached[0] <= _CACHE_TTL
    verdict: dict[str, str] | None = cached_verdict if cache_hit else None
    if verdict is None:
        envelope = {
            "direction": direction,
            "candidate": normalized_candidate,
            "supporting_context": normalized_supporting_context,
            "surface": resolved_context.surface,
            "tool_id": tool_id or resolved_context.tool_id,
            "operation": operation,
            "resource_id": resolved_context.resource_id,
            "policy_revision": config.revision,
            "audience_constraints": policy.profile_sets,
        }
        try:
            verdict = await _classify_with_limits(config, envelope, state)
        except ContentProtectionError as protection_error:
            state.terminal = protection_error
            await _audit(
                request_id,
                {
                    "surface": resolved_context.surface,
                    "direction": direction,
                    "code": protection_error.code,
                    "policy_revision": config.revision,
                    "provenance": policy.provenance,
                },
            )
            raise
        except Exception:
            error = ContentProtectionError("classifier_unavailable", request_id)
            state.terminal = error
            await _audit(
                request_id,
                {
                    "surface": resolved_context.surface,
                    "direction": direction,
                    "code": error.code,
                    "policy_revision": config.revision,
                    "provenance": policy.provenance,
                },
            )
            raise error from None
        if verdict.get("verdict") == "allow":
            _store_allow(key, verdict)
    if verdict.get("verdict") != "allow":
        error = ContentProtectionError("content_denied" if verdict.get("verdict") == "deny" else "classifier_invalid_response", request_id)
        state.terminal = error
        await _audit(
            request_id,
            {
                "surface": resolved_context.surface,
                "direction": direction,
                "code": error.code,
                "policy_revision": config.revision,
                "provenance": policy.provenance,
                "cache_hit": cache_hit,
            },
        )
        raise error
    # A fresh authoritative read catches changed memberships/config before release.
    try:
        latest = await load_config()
        if not latest.enabled:
            # A deliberate master-off edit is authoritative at the release
            # boundary and ends classification rather than using stale allow.
            return
        latest_policy = await _resolve(latest, resolved_context, tool_id=tool_id)
    except ContentProtectionError:
        raise
    except Exception:
        error = ContentProtectionError("classifier_unavailable", request_id)
        state.terminal = error
        raise error from None
    if latest.revision != config.revision or latest_policy != policy:
        # One re-evaluation is permitted; another changed policy is a fixed error.
        if not _rechecked:
            _decision_cache.pop(key, None)
            await authorize_content(
                candidate,
                direction=direction,
                context=resolved_context,
                tool_id=tool_id,
                operation=operation,
                supporting_context=supporting_context,
                _rechecked=True,
            )
            return
        error = ContentProtectionError("policy_changed", request_id)
        state.terminal = error
        raise error
    await _audit(
        request_id,
        {
            "surface": resolved_context.surface,
            "direction": direction,
            "code": "permitted",
            "policy_revision": config.revision,
            "provenance": policy.provenance,
            "cache_hit": cache_hit,
        },
    )


async def reject_unsupported_if_required(context: ProtectionContext | None = None) -> None:
    if await classification_required(context):
        raise ContentProtectionError("content_unclassifiable", str(uuid4()))
