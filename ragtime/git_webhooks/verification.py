from __future__ import annotations

import hashlib
import hmac
from typing import Any, Mapping

from ragtime.git_webhooks.models import GitPushEvent, GitWebhookTarget


class WebhookAuthenticationError(Exception):
    pass


def verify_webhook_request(
    target: GitWebhookTarget,
    headers: Mapping[str, str],
    body: bytes,
    query_token: str | None,
) -> None:
    normalized = _normalize_headers(headers)
    provider = (target.provider or "generic").strip().lower()
    if provider == "github":
        if _verify_prefixed_sha256_signature(target.secret, body, normalized.get("x-hub-signature-256"), prefix="sha256="):
            return
        raise WebhookAuthenticationError("Invalid webhook credentials")
    if provider == "gitlab":
        if _verify_token(target.secret, normalized.get("x-gitlab-token")):
            return
        raise WebhookAuthenticationError("Invalid webhook credentials")
    if _verify_any_generic_credential(target.secret, body, normalized, query_token):
        return
    raise WebhookAuthenticationError("Invalid webhook credentials")


def parse_git_events(provider: str, headers: Mapping[str, str], payload: Any) -> list[GitPushEvent]:
    normalized = _normalize_headers(headers)
    normalized_provider = (provider or "generic").strip().lower()
    event_name = _event_name(normalized_provider, normalized)
    delivery_id = _delivery_id(normalized_provider, normalized)
    if event_name and not _is_push_event(normalized_provider, event_name):
        return [_ignored_event(event_name=event_name, delivery_id=delivery_id, message="Non-push webhook events are ignored.")]
    if not isinstance(payload, Mapping):
        return [_ignored_event(event_name=event_name, delivery_id=delivery_id, message="Push branch could not be determined.")]

    ref_events = _parse_ref_payload(payload, event_name=event_name, delivery_id=delivery_id)
    if ref_events is not None:
        return ref_events

    bitbucket_events = _parse_bitbucket_changes(payload, event_name=event_name, delivery_id=delivery_id)
    if bitbucket_events:
        return bitbucket_events

    azure_events = _parse_azure_ref_updates(payload, event_name=event_name, delivery_id=delivery_id)
    if azure_events:
        return azure_events

    return [_ignored_event(event_name=event_name, delivery_id=delivery_id, message="Push branch could not be determined.")]


def _normalize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {str(key).lower(): str(value).strip() for key, value in headers.items()}


def _verify_any_generic_credential(
    secret: str,
    body: bytes,
    headers: Mapping[str, str],
    query_token: str | None,
) -> bool:
    return any(
        (
            _verify_prefixed_sha256_signature(secret, body, headers.get("x-hub-signature-256"), prefix="sha256="),
            _verify_token(secret, headers.get("x-gitlab-token")),
            _verify_raw_sha256_signature(secret, body, headers.get("x-gitea-signature")),
            _verify_raw_sha256_signature(secret, body, headers.get("x-gogs-signature")),
            _verify_token(secret, headers.get("x-ragtime-webhook-token")),
            _verify_token(secret, _bearer_token(headers.get("authorization"))),
            _verify_token(secret, query_token),
        )
    )


def _verify_prefixed_sha256_signature(secret: str, body: bytes, signature: str | None, *, prefix: str) -> bool:
    if not signature:
        return False
    value = signature.strip()
    if not value.startswith(prefix):
        return False
    hex_value = value[len(prefix) :]
    if not _is_sha256_hex(hex_value):
        return False
    expected = prefix + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, value)


def _verify_raw_sha256_signature(secret: str, body: bytes, signature: str | None) -> bool:
    if not signature:
        return False
    value = signature.strip()
    if not _is_sha256_hex(value):
        return False
    normalized_value = value.lower()
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, normalized_value)


def _verify_token(secret: str, provided: str | None) -> bool:
    if provided is None:
        return False
    return hmac.compare_digest(secret, provided.strip())


def _bearer_token(value: str | None) -> str | None:
    if value is None:
        return None
    scheme, _, token = value.strip().partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _event_name(provider: str, headers: Mapping[str, str]) -> str:
    if provider == "github":
        return headers.get("x-github-event", "")
    if provider == "gitlab":
        return headers.get("x-gitlab-event", "")
    return headers.get("x-event-key") or headers.get("x-event-type") or headers.get("x-github-event") or headers.get("x-gitlab-event") or ""


def _delivery_id(provider: str, headers: Mapping[str, str]) -> str | None:
    if provider == "github":
        return headers.get("x-github-delivery") or None
    if provider == "gitlab":
        return headers.get("x-gitlab-event-uuid") or headers.get("x-request-id") or None
    return headers.get("x-request-uuid") or headers.get("x-request-id") or headers.get("x-github-delivery") or headers.get("x-gitlab-event-uuid") or None


def _is_push_event(provider: str, event_name: str) -> bool:
    normalized = event_name.strip().lower()
    if provider == "github":
        return normalized == "push"
    if provider == "gitlab":
        return normalized == "push hook"
    return normalized in {"push", "repo:push", "git.push", "push hook"}


def _parse_ref_payload(payload: Mapping[str, Any], *, event_name: str, delivery_id: str | None) -> list[GitPushEvent] | None:
    ref = payload.get("ref")
    if not isinstance(ref, str) or not ref:
        return None
    branch = _branch_from_ref(ref)
    if branch is None:
        return [_ignored_event(event_name=event_name, delivery_id=delivery_id, message="Push branch could not be determined.")]
    head_commit = _string_value(payload.get("after"))
    if _is_deleted_head(head_commit):
        return [_ignored_event(event_name=event_name, delivery_id=delivery_id, branch=branch, message="Deleted branches are ignored.")]
    return [_push_event(event_name=event_name, delivery_id=delivery_id, branch=branch, head_commit=head_commit)]


def _parse_bitbucket_changes(payload: Mapping[str, Any], *, event_name: str, delivery_id: str | None) -> list[GitPushEvent]:
    push = payload.get("push")
    if not isinstance(push, Mapping):
        return []
    changes = push.get("changes")
    if not isinstance(changes, list):
        return []
    events: list[GitPushEvent] = []
    for change in changes:
        if not isinstance(change, Mapping):
            continue
        new_value = change.get("new")
        old_value = change.get("old")
        if new_value is None:
            branch = _branch_name_from_change(old_value)
            if branch is not None:
                events.append(_ignored_event(event_name=event_name, delivery_id=delivery_id, branch=branch, message="Deleted branches are ignored."))
            continue
        if not isinstance(new_value, Mapping):
            continue
        if str(new_value.get("type") or "") != "branch":
            continue
        branch = _string_value(new_value.get("name"))
        if not branch:
            continue
        head_commit = _string_value((new_value.get("target") or {}).get("hash")) if isinstance(new_value.get("target"), Mapping) else None
        events.append(_push_event(event_name=event_name, delivery_id=delivery_id, branch=branch, head_commit=head_commit))
    return events


def _parse_azure_ref_updates(payload: Mapping[str, Any], *, event_name: str, delivery_id: str | None) -> list[GitPushEvent]:
    resource = payload.get("resource")
    if not isinstance(resource, Mapping):
        return []
    updates = resource.get("refUpdates")
    if not isinstance(updates, list):
        return []
    events: list[GitPushEvent] = []
    for update in updates:
        if not isinstance(update, Mapping):
            continue
        ref_name = _string_value(update.get("name"))
        if not ref_name:
            continue
        branch = _branch_from_ref(ref_name)
        if branch is None:
            continue
        new_object_id = _string_value(update.get("newObjectId"))
        if _is_deleted_head(new_object_id):
            events.append(_ignored_event(event_name=event_name, delivery_id=delivery_id, branch=branch, message="Deleted branches are ignored."))
            continue
        events.append(_push_event(event_name=event_name, delivery_id=delivery_id, branch=branch, head_commit=new_object_id))
    return events


def _branch_from_ref(ref: str) -> str | None:
    prefix = "refs/heads/"
    if ref.startswith(prefix):
        branch = ref[len(prefix) :]
        return branch or None
    return None


def _is_deleted_head(head_commit: str | None) -> bool:
    if not head_commit:
        return False
    stripped = head_commit.strip().lower()
    return bool(stripped) and set(stripped) == {"0"}


def _branch_name_from_change(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    if str(value.get("type") or "") != "branch":
        return None
    return _string_value(value.get("name"))


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _push_event(*, event_name: str, delivery_id: str | None, branch: str, head_commit: str | None) -> GitPushEvent:
    return GitPushEvent(
        kind="push",
        message=None,
        provider_delivery_id=delivery_id,
        event_name=event_name or "push",
        branch=branch,
        head_commit=head_commit,
    )


def _ignored_event(
    *,
    event_name: str,
    delivery_id: str | None,
    message: str,
    branch: str | None = None,
) -> GitPushEvent:
    return GitPushEvent(
        kind="ignored",
        message=message,
        provider_delivery_id=delivery_id,
        event_name=event_name or "push",
        branch=branch,
        head_commit=None,
    )
