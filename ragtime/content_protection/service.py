"""Public, transport-neutral content-protection enforcement service."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import weakref
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
from ragtime.core.logging import get_logger
from ragtime.core.performance import timed_operation

_MAX_BYTES = 1024 * 1024
_CACHE_TTL = 60.0
_TURN_BUDGET = 30.0
_cache_secret = secrets.token_bytes(32)
logger = get_logger(__name__)
_last_audit_retention_sweep = 0.0


@dataclass
class _TurnRecord:
    started: float = field(default_factory=monotonic)
    spent: float = 0.0
    recovery_consumed: bool = False
    boundary_count: int = 0
    reserved: float = 0.0


@dataclass
class _TurnState:
    record: _TurnRecord = field(default_factory=_TurnRecord)
    terminal: ContentProtectionError | None = None


@dataclass
class _InFlightVerdict:
    task: asyncio.Task[tuple[dict[str, str], float, float]]
    waiters: int = 0


class _LoopFlights:
    def __init__(self) -> None:
        self.entries: dict[str, _InFlightVerdict] = {}
        self.queue = asyncio.Semaphore(16)


class _SharedFailure(Exception):
    def __init__(self, code: str) -> None:
        self.code = code


_inflight_by_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _LoopFlights] = weakref.WeakKeyDictionary()


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
_timing: ContextVar[dict[str, float] | None] = ContextVar("content_protection_timing", default=None)


@contextmanager
def protection_context(context: ProtectionContext) -> Iterator[None]:
    """Bind context while retaining the same mutable turn record when nested."""
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


def terminal_error() -> ContentProtectionError | None:
    """Return the sticky terminal error for this attempt, if any."""
    state = _turn.get()
    return state.terminal if state is not None else None


def ensure_active_attempt() -> None:
    """Prevent stale copied contexts from executing or releasing work."""
    error = terminal_error()
    if error is not None:
        raise error


def _ensure_attempt(state: _TurnState) -> None:
    current = _turn.get()
    if state.terminal is not None:
        raise state.terminal
    if current is not None and current is not state:
        # A successor was installed in this caller; old captured work may not
        # release into the successor attempt.
        raise ContentProtectionError("classifier_unavailable", str(uuid4()))


def _set_terminal(state: _TurnState, error: ContentProtectionError) -> None:
    """First terminal error wins for all copied contexts of an attempt."""
    _ensure_attempt(state)
    state.terminal = error


def _record_timing(stage: str, elapsed: float) -> None:
    timing = _timing.get()
    if timing is not None:
        timing[stage] += elapsed


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
    started = monotonic()
    try:
        db = await database.get_db()
        await db.contentprotectiondecision.create(data={"requestId": request_id, "metadata": Json(metadata)})
        global _last_audit_retention_sweep
        now = monotonic()
        if now - _last_audit_retention_sweep >= 3600:
            _last_audit_retention_sweep = now
            await db.contentprotectiondecision.delete_many(where={"createdAt": {"lt": datetime.now(UTC) - timedelta(days=30)}})
    except Exception:
        # Audit failure must not leak a candidate or turn a safe denial into a 500.
        return
    finally:
        timing = _timing.get()
        if timing is not None:
            timing["audit"] += monotonic() - started


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
    # Reasons are user-facing refusal prose, never reusable allow metadata.
    _decision_cache[key] = (now, {"verdict": verdict["verdict"], "reason_code": verdict["reason_code"]})


async def _physical_classify(config: ContentProtectionConfig, envelope: dict[str, object], flights: _LoopFlights) -> tuple[dict[str, str], float, float]:
    """The shareable provider operation; it never owns a caller's error ID."""
    queued = monotonic()
    try:
        await asyncio.wait_for(flights.queue.acquire(), timeout=1)
    except TimeoutError:
        raise _SharedFailure("classifier_unavailable") from None
    queue_duration = monotonic() - queued
    try:
        started = monotonic()
        try:
            verdict = await asyncio.wait_for(_classify(config, envelope, include_reason=True), timeout=5.0)
        except ContentProtectionError as error:
            raise _SharedFailure(error.code) from None
        except Exception:
            raise _SharedFailure("classifier_unavailable") from None
        return verdict, queue_duration, monotonic() - started
    finally:
        flights.queue.release()


@timed_operation("content_protection.classifier_wait")
async def _singleflight_classify(
    config: ContentProtectionConfig, envelope: dict[str, object], state: _TurnState, key: str, request_id: str
) -> tuple[dict[str, str], bool, float, float, float]:
    """Share only an identical provider verdict; each caller retains its own release checks."""
    loop = asyncio.get_running_loop()
    flights = _inflight_by_loop.setdefault(loop, _LoopFlights())
    entry = flights.entries.get(key)
    joined = entry is not None
    if entry is None:
        if len(flights.entries) >= 256:
            # Overflow remains bounded but is not coalesced.
            entry = _InFlightVerdict(task=loop.create_task(_physical_classify(config, envelope, flights)))
        else:
            entry = _InFlightVerdict(task=loop.create_task(_physical_classify(config, envelope, flights)))
            flights.entries[key] = entry
    entry.waiters += 1
    started = monotonic()
    reservation = 0.0
    try:
        _ensure_attempt(state)
        available = _TURN_BUDGET - state.record.spent - state.record.reserved
        if available <= 0:
            raise ContentProtectionError("classifier_unavailable", request_id)
        # Reserve the maximum physical queue/provider lifetime before allowing
        # another same-turn waiter to start.  This prevents concurrent callers
        # from silently borrowing past the cumulative turn budget.
        reservation = min(6.0, available)
        state.record.reserved += reservation
        try:
            verdict, queue_duration, provider_duration = await asyncio.wait_for(asyncio.shield(entry.task), timeout=reservation)
        except _SharedFailure as failure:
            raise ContentProtectionError(failure.code, request_id) from None
        _ensure_attempt(state)
        return verdict, joined, monotonic() - started, queue_duration, provider_duration
    finally:
        elapsed = monotonic() - started
        if reservation:
            state.record.reserved = max(0.0, state.record.reserved - reservation)
            state.record.spent += elapsed
        entry.waiters -= 1
        if entry.waiters == 0:
            if not entry.task.done():
                entry.task.cancel()
            try:
                await asyncio.shield(entry.task)
            except (asyncio.CancelledError, _SharedFailure):
                pass
            if flights.entries.get(key) is entry:
                flights.entries.pop(key, None)


def begin_recovery(error: ContentProtectionError) -> _TurnState:
    """Atomically consume this hosted turn's one recovery allowance.

    The old attempt remains terminal for copied ContextVars; only this caller is
    advanced to the successor attempt.
    """
    state = _turn.get()
    if state is None or state.terminal is not error or not error.recovery_eligible or state.record.recovery_consumed:
        raise error
    state.record.recovery_consumed = True
    successor = _TurnState(record=state.record)
    _turn.set(successor)
    return successor


@contextmanager
def recovery_attempt(error: ContentProtectionError) -> Iterator[_TurnState]:
    """Install one successor attempt, restoring the old terminal on failure.

    A successful block deliberately keeps its successor bound so outer release
    and persistence checks see the approved recovery attempt.
    """
    state = _turn.get()
    if state is None or state.terminal is not error or not error.recovery_eligible or state.record.recovery_consumed:
        raise error
    state.record.recovery_consumed = True
    successor = _TurnState(record=state.record)
    token = _turn.set(successor)
    try:
        yield successor
    except BaseException:
        _turn.reset(token)
        raise


@timed_operation("content_protection.classifier")
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
    """Authorize content and emit one payload-free finalized timing record."""
    started = monotonic()
    timing = {"initial": 0.0, "queue": 0.0, "provider": 0.0, "release": 0.0, "audit": 0.0}
    token = _timing.set(timing)
    allowed_directions = {"inbound", "stored_readback", "assistant_response", "tool_result", "proposed_operation", "outbound"}
    safe_direction = direction if direction in allowed_directions else "other"
    outcome = "permitted"
    try:
        await _authorize_content(
            candidate,
            direction=direction,
            context=context,
            tool_id=tool_id,
            operation=operation,
            supporting_context=supporting_context,
            _rechecked=_rechecked,
        )
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except ContentProtectionError as error:
        outcome = "denied" if error.code == "content_denied" else "failed"
        raise
    except Exception:
        outcome = "failed"
        raise
    finally:
        state = _turn.get()
        total_elapsed = monotonic() - started
        # Failures before policy resolution still spent initial-stage time;
        # retain it rather than emitting a misleading zero.
        if timing["initial"] == 0.0:
            timing["initial"] = max(0.0, total_elapsed - timing["queue"] - timing["provider"] - timing["release"] - timing["audit"])
        logger.info(
            "content_protection_timing",
            extra={
                "content_protection": {
                    "direction": safe_direction,
                    "outcome": outcome,
                    "initial_ms": round(timing["initial"] * 1000, 3),
                    "queue_ms": round(timing["queue"] * 1000, 3),
                    "provider_ms": round(timing["provider"] * 1000, 3),
                    "release_recheck_ms": round(timing["release"] * 1000, 3),
                    "audit_ms": round(timing["audit"] * 1000, 3),
                    "total_ms": round(total_elapsed * 1000, 3),
                    "boundary_count": state.record.boundary_count if state is not None else 1,
                }
            },
        )
        _timing.reset(token)


async def _authorize_content(
    candidate: Any,
    *,
    direction: str,
    context: ProtectionContext | None = None,
    tool_id: str | None = None,
    operation: str | None = None,
    supporting_context: Any = None,
    _rechecked: bool = False,
) -> None:
    total_started = monotonic()
    initial_started = total_started
    initial_duration = 0.0
    queue_duration = 0.0
    provider_duration = 0.0
    release_duration = 0.0
    audit_duration = 0.0
    outcome = "error"
    resolved_context = context or current_context() or ProtectionContext(baseline="anonymous")
    bound_state = _turn.get()
    state = bound_state or _TurnState()
    state.record.boundary_count += 1
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
        _ensure_attempt(state)
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
            _set_terminal(state, error)
            raise error
        if len(preliminary) > _MAX_BYTES:
            error = ContentProtectionError("content_unclassifiable", request_id)
            _set_terminal(state, error)
            raise error
    try:
        policy = await _resolve(config, resolved_context, tool_id=tool_id)
    except ContentProtectionError:
        raise
    except Exception:
        error = ContentProtectionError("classifier_unavailable", request_id)
        _set_terminal(state, error)
        raise error from None
    initial_duration = monotonic() - initial_started
    _record_timing("initial", initial_duration)
    _ensure_attempt(state)
    if not policy.required:
        return
    try:
        normalized_candidate = normalize_transport_value(candidate)
        normalized_supporting_context = normalize_transport_value(supporting_context) if supporting_context is not None else None
        serialized = canonical_serialize(normalized_candidate)
        supporting_serialized = canonical_serialize(normalized_supporting_context)
    except (TypeError, ValueError):
        error = ContentProtectionError("content_unclassifiable", request_id)
        _set_terminal(state, error)
        raise error
    if len(serialized) + len(supporting_serialized) > _MAX_BYTES:
        error = ContentProtectionError("content_unclassifiable", request_id)
        _set_terminal(state, error)
        raise error
    try:
        provider_settings = await _provider_settings_identity()
    except Exception:
        error = ContentProtectionError("classifier_unavailable", request_id)
        _set_terminal(state, error)
        raise error from None
    _ensure_attempt(state)
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
    coalesced = False
    wait_duration = 0.0
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
            verdict, coalesced, wait_duration, queue_duration, provider_duration = await _singleflight_classify(config, envelope, state, key, request_id)
            _record_timing("queue", queue_duration)
            _record_timing("provider", provider_duration)
        except ContentProtectionError as protection_error:
            _set_terminal(state, protection_error)
            audit_started = monotonic()
            await _audit(
                request_id,
                {
                    "surface": resolved_context.surface,
                    "direction": direction,
                    "code": protection_error.code,
                    "policy_revision": config.revision,
                    "provenance": policy.provenance,
                    "coalesced": coalesced,
                    "classifier_wait_ms": round(wait_duration * 1000, 3),
                },
            )
            audit_duration += monotonic() - audit_started
            raise
        except Exception:
            error = ContentProtectionError("classifier_unavailable", request_id)
            _set_terminal(state, error)
            audit_started = monotonic()
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
            audit_duration += monotonic() - audit_started
            raise error from None
        _ensure_attempt(state)
        if verdict.get("verdict") == "allow":
            _ensure_attempt(state)
            _store_allow(key, verdict)
    if verdict.get("verdict") != "allow":
        eligible = (
            verdict.get("verdict") == "deny"
            and ((direction == "assistant_response" and operation is None) or direction == "tool_result")
            and not state.record.recovery_consumed
        )
        error = ContentProtectionError(
            "content_denied" if verdict.get("verdict") == "deny" else "classifier_invalid_response",
            request_id,
            reason=verdict.get("reason"),
            reason_code=verdict.get("reason_code"),
            recovery_eligible=eligible,
            execution_status="completed_response_withheld" if eligible and direction == "tool_result" else None,
            attempts_remaining=1 if eligible and bound_state is not None else None,
        )
        _set_terminal(state, error)
        audit_started = monotonic()
        await _audit(
            request_id,
            {
                "surface": resolved_context.surface,
                "direction": direction,
                "code": error.code,
                "policy_revision": config.revision,
                "provenance": policy.provenance,
                "cache_hit": cache_hit,
                "coalesced": coalesced if not cache_hit else False,
                "classifier_wait_ms": round(wait_duration * 1000, 3) if not cache_hit else 0.0,
            },
        )
        audit_duration += monotonic() - audit_started
        raise error
    # A fresh authoritative read catches changed memberships/config before release.
    release_started = monotonic()
    try:
        latest = await load_config()
        if not latest.enabled:
            # A deliberate master-off edit is authoritative at the release
            # boundary and ends classification rather than using stale allow.
            _ensure_attempt(state)
            outcome = "permitted"
            return
        latest_policy = await _resolve(latest, resolved_context, tool_id=tool_id)
    except ContentProtectionError:
        _record_timing("release", monotonic() - release_started)
        raise
    except Exception:
        _record_timing("release", monotonic() - release_started)
        error = ContentProtectionError("classifier_unavailable", request_id)
        _set_terminal(state, error)
        await _audit(
            request_id,
            {
                "surface": resolved_context.surface,
                "direction": direction,
                "code": error.code,
                "policy_revision": config.revision,
                "provenance": policy.provenance,
                "stage": "release_recheck",
            },
        )
        raise error from None
    release_duration = monotonic() - release_started
    _record_timing("release", release_duration)
    _ensure_attempt(state)
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
        _set_terminal(state, error)
        raise error
    _ensure_attempt(state)
    audit_started = monotonic()
    await _audit(
        request_id,
        {
            "surface": resolved_context.surface,
            "direction": direction,
            "code": "permitted",
            "policy_revision": config.revision,
            "provenance": policy.provenance,
            "cache_hit": cache_hit,
            "coalesced": coalesced if not cache_hit else False,
            "classifier_wait_ms": round(wait_duration * 1000, 3) if not cache_hit else 0.0,
        },
    )
    audit_duration += monotonic() - audit_started
    outcome = "permitted"


async def reject_unsupported_if_required(context: ProtectionContext | None = None) -> None:
    if await classification_required(context):
        raise ContentProtectionError("content_unclassifiable", str(uuid4()))
