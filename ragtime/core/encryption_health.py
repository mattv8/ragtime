from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any, Final

from prisma import Prisma

from ragtime.core.database import get_db
from ragtime.core.encryption import (
    CONNECTION_CONFIG_PASSWORD_FIELDS,
    ENCRYPTED_PREFIX,
    attempt_decrypt,
    reset_key_mismatch_state,
)

_APP_SETTINGS_SECRET_FIELDS: Final[tuple[str, ...]] = (
    "openaiApiKey",
    "anthropicApiKey",
    "openrouterApiKey",
    "githubModelsApiToken",
    "githubCopilotAccessToken",
    "githubCopilotRefreshToken",
    "githubCopilotOauthRefreshToken",
    "openaiCodexAccessToken",
    "openaiCodexRefreshToken",
    "postgresPassword",
    "lmstudioApiKey",
    "omlxApiKey",
    "mcpDefaultRoutePassword",
)


def _encrypted_value(value: Any) -> str | None:
    match value:
        case str() as text if text.startswith(ENCRYPTED_PREFIX):
            return text
        case _:
            return None


def _iter_row_secret_values(rows: Iterable[Any], field_names: tuple[str, ...]) -> Iterable[str]:
    for row in rows:
        for field_name in field_names:
            encrypted_value = _encrypted_value(getattr(row, field_name, None))
            if encrypted_value is not None:
                yield encrypted_value


def _iter_connection_config_secret_values(rows: Iterable[Any], field_name: str = "connectionConfig") -> Iterable[str]:
    for row in rows:
        match getattr(row, field_name, None):
            case dict() as connection_config:
                for password_field in CONNECTION_CONFIG_PASSWORD_FIELDS:
                    encrypted_value = _encrypted_value(connection_config.get(password_field))
                    if encrypted_value is not None:
                        yield encrypted_value
            case _:
                continue


async def _find_many_with_encrypted_field(model: Any, field_name: str) -> list[Any]:
    return await model.find_many(
        where={field_name: {"startsWith": ENCRYPTED_PREFIX}},
        select={field_name: True},
    )


async def _iter_encrypted_secret_values(db: Prisma) -> AsyncIterator[str]:
    app_settings_rows = await db.appsettings.find_many(
        select={field_name: True for field_name in _APP_SETTINGS_SECRET_FIELDS},
    )
    for encrypted_value in _iter_row_secret_values(app_settings_rows, _APP_SETTINGS_SECRET_FIELDS):
        yield encrypted_value

    tool_config_rows = await db.toolconfig.find_many(select={"connectionConfig": True})
    for encrypted_value in _iter_connection_config_secret_values(tool_config_rows):
        yield encrypted_value

    mcp_route_rows = await _find_many_with_encrypted_field(db.mcprouteconfig, "authPassword")
    for encrypted_value in _iter_row_secret_values(mcp_route_rows, ("authPassword",)):
        yield encrypted_value

    index_job_rows = await _find_many_with_encrypted_field(db.indexjob, "gitToken")
    for encrypted_value in _iter_row_secret_values(index_job_rows, ("gitToken",)):
        yield encrypted_value

    index_metadata_rows = await _find_many_with_encrypted_field(db.indexmetadata, "gitToken")
    for encrypted_value in _iter_row_secret_values(index_metadata_rows, ("gitToken",)):
        yield encrypted_value

    ldap_config_rows = await _find_many_with_encrypted_field(db.ldapconfig, "bindPassword")
    for encrypted_value in _iter_row_secret_values(ldap_config_rows, ("bindPassword",)):
        yield encrypted_value

    workspace_rows = await _find_many_with_encrypted_field(db.workspace, "scmToken")
    for encrypted_value in _iter_row_secret_values(workspace_rows, ("scmToken",)):
        yield encrypted_value

    conversation_share_rows = await _find_many_with_encrypted_field(db.conversationshare, "sharePassword")
    for encrypted_value in _iter_row_secret_values(conversation_share_rows, ("sharePassword",)):
        yield encrypted_value

    workspace_share_rows = await _find_many_with_encrypted_field(db.workspaceshare, "sharePassword")
    for encrypted_value in _iter_row_secret_values(workspace_share_rows, ("sharePassword",)):
        yield encrypted_value

    workspace_env_rows = await _find_many_with_encrypted_field(db.workspaceenvironmentvariable, "value")
    for encrypted_value in _iter_row_secret_values(workspace_env_rows, ("value",)):
        yield encrypted_value

    global_env_rows = await _find_many_with_encrypted_field(db.globalenvironmentvariable, "value")
    for encrypted_value in _iter_row_secret_values(global_env_rows, ("value",)):
        yield encrypted_value

    userspace_mount_source_rows = await db.userspacemountsource.find_many(select={"connectionConfig": True})
    for encrypted_value in _iter_connection_config_secret_values(userspace_mount_source_rows):
        yield encrypted_value

    user_mount_source_rows = await db.useruserspacemountsource.find_many(select={"connectionConfig": True})
    for encrypted_value in _iter_connection_config_secret_values(user_mount_source_rows):
        yield encrypted_value

    user_cloud_oauth_rows = await db.usercloudoauthaccount.find_many(
        where={
            "OR": [
                {"accessToken": {"startsWith": ENCRYPTED_PREFIX}},
                {"refreshToken": {"startsWith": ENCRYPTED_PREFIX}},
            ]
        },
        select={"accessToken": True, "refreshToken": True},
    )
    for encrypted_value in _iter_row_secret_values(user_cloud_oauth_rows, ("accessToken", "refreshToken")):
        yield encrypted_value


async def recheck_encryption_key_health() -> bool:
    db = await get_db()
    async for encrypted_value in _iter_encrypted_secret_values(db):
        if not attempt_decrypt(encrypted_value):
            return False

    reset_key_mismatch_state()
    return True
