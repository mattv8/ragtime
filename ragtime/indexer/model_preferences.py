from dataclasses import dataclass
from typing import Protocol, cast

from ragtime.core.database import get_db
from ragtime.core.model_providers import normalize_provider_name
from ragtime.indexer.models import AppSettings
from ragtime.indexer.repository import _resolve_default_conversation_model


@dataclass(frozen=True)
class ModelAvailabilitySnapshot:
    available_model_ids: frozenset[str]
    authoritative_providers: frozenset[str]


class _UserTable(Protocol):
    async def find_unique(self, *, where: dict[str, str]) -> object | None: ...

    async def update(self, *, where: dict[str, str], data: dict[str, str | None]) -> object | None: ...

    async def update_many(self, *, where: dict[str, str], data: dict[str, str | None]) -> object: ...


class _WorkspaceUserPreferenceTable(Protocol):
    async def find_first(self, *, where: dict[str, str]) -> object | None: ...

    async def delete_many(self, *, where: dict[str, str]) -> object: ...

    async def upsert(
        self,
        *,
        where: dict[str, dict[str, str]],
        create: dict[str, str],
        update: dict[str, str],
    ) -> object: ...


class _ModelPreferenceDb(Protocol):
    user: _UserTable
    workspaceuserpreference: _WorkspaceUserPreferenceTable


async def _get_model_preference_db() -> _ModelPreferenceDb:
    return cast(_ModelPreferenceDb, await get_db())


def normalize_default_model(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    if "::" not in normalized:
        return normalized
    provider, _, model_id = normalized.partition("::")
    normalized_provider = normalize_provider_name(provider)
    normalized_model_id = model_id.strip()
    if not normalized_provider or not normalized_model_id:
        return normalized
    return f"{normalized_provider}::{normalized_model_id}"


def _split_scoped_model_identifier(identifier: str | None) -> tuple[str | None, str]:
    normalized = normalize_default_model(identifier)
    if normalized is None or "::" not in normalized:
        return None, normalized or ""
    provider, _, model_id = normalized.partition("::")
    if not provider or not model_id:
        return None, normalized
    return provider, model_id


def _is_stale_candidate(candidate: str | None, availability: ModelAvailabilitySnapshot | None) -> bool:
    normalized_candidate = normalize_default_model(candidate)
    if normalized_candidate is None or availability is None:
        return False
    provider, _ = _split_scoped_model_identifier(normalized_candidate)
    if provider is None or provider not in availability.authoritative_providers:
        return False
    return normalized_candidate not in availability.available_model_ids


async def get_user_default_model(user_id: str) -> str | None:
    db = await _get_model_preference_db()
    user = await db.user.find_unique(where={"id": user_id})
    return normalize_default_model(getattr(user, "defaultChatModel", None))


async def set_user_default_model(user_id: str, model: str | None) -> str | None:
    db = await _get_model_preference_db()
    normalized_model = normalize_default_model(model)
    user = await db.user.update(where={"id": user_id}, data={"defaultChatModel": normalized_model})
    return normalize_default_model(getattr(user, "defaultChatModel", normalized_model))


async def _get_workspace_user_preference_record(user_id: str, workspace_id: str):
    db = await _get_model_preference_db()
    return await db.workspaceuserpreference.find_first(where={"workspaceId": workspace_id, "userId": user_id})


async def get_workspace_user_default_model(user_id: str, workspace_id: str) -> str | None:
    record = await _get_workspace_user_preference_record(user_id, workspace_id)
    return normalize_default_model(getattr(record, "defaultChatModel", None))


async def set_workspace_user_default_model(user_id: str, workspace_id: str, model: str | None) -> str | None:
    db = await _get_model_preference_db()
    normalized_model = normalize_default_model(model)

    if normalized_model is None:
        await db.workspaceuserpreference.delete_many(where={"workspaceId": workspace_id, "userId": user_id})
        return None

    record = await db.workspaceuserpreference.upsert(
        where={"workspaceId_userId": {"workspaceId": workspace_id, "userId": user_id}},
        create={
            "workspaceId": workspace_id,
            "userId": user_id,
            "defaultChatModel": normalized_model,
        },
        update={"defaultChatModel": normalized_model},
    )
    return normalize_default_model(getattr(record, "defaultChatModel", normalized_model))


async def resolve_new_conversation_model(
    app_settings: AppSettings | None,
    *,
    user_id: str,
    workspace_id: str | None = None,
    explicit_model: str | None = None,
    availability: ModelAvailabilitySnapshot | None = None,
) -> str:
    if explicit_model is not None:
        return explicit_model

    workspace_model = None
    if workspace_id is not None:
        workspace_model = await get_workspace_user_default_model(user_id, workspace_id)
        if workspace_model is not None and _is_stale_candidate(workspace_model, availability):
            await clear_matching_personal_defaults(user_id, workspace_id, workspace_model)
        elif workspace_model is not None:
            return workspace_model

    user_model = await get_user_default_model(user_id)
    if user_model is not None and _is_stale_candidate(user_model, availability):
        await clear_matching_personal_defaults(user_id, None, user_model)
    elif user_model is not None:
        return user_model

    return _resolve_default_conversation_model(app_settings)


async def clear_matching_personal_defaults(user_id: str, workspace_id: str | None, model: str) -> None:
    db = await _get_model_preference_db()
    normalized_model = normalize_default_model(model)
    if normalized_model is None:
        return

    if workspace_id is not None:
        await db.workspaceuserpreference.delete_many(where={"workspaceId": workspace_id, "userId": user_id, "defaultChatModel": normalized_model})

    await db.user.update_many(
        where={"id": user_id, "defaultChatModel": normalized_model},
        data={"defaultChatModel": None},
    )
