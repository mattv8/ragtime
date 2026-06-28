from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Final, cast

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
        row_values = getattr(row, "__dict__", None)
        for field_name in field_names:
            value = row_values.get(field_name) if isinstance(row_values, dict) else getattr(row, field_name, None)
            encrypted_value = _encrypted_value(value)
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
    return await model.find_many(where={field_name: {"startsWith": ENCRYPTED_PREFIX}})


async def _load_encrypted_secret_values(db: Prisma) -> list[str]:
    encrypted_values: list[str] = []

    app_settings_rows = await db.appsettings.find_many(where={"id": "default"})  # type: ignore[call-arg]
    encrypted_values.extend(_iter_row_secret_values(app_settings_rows, _APP_SETTINGS_SECRET_FIELDS))

    tool_config_rows = await db.toolconfig.find_many()  # type: ignore[call-arg]
    encrypted_values.extend(_iter_connection_config_secret_values(tool_config_rows))

    mcp_route_rows = await _find_many_with_encrypted_field(db.mcprouteconfig, "authPassword")
    encrypted_values.extend(_iter_row_secret_values(mcp_route_rows, ("authPassword",)))

    index_job_rows = await _find_many_with_encrypted_field(db.indexjob, "gitToken")
    encrypted_values.extend(_iter_row_secret_values(index_job_rows, ("gitToken",)))

    index_metadata_rows = await _find_many_with_encrypted_field(db.indexmetadata, "gitToken")
    encrypted_values.extend(_iter_row_secret_values(index_metadata_rows, ("gitToken",)))

    ldap_config_rows = await _find_many_with_encrypted_field(db.ldapconfig, "bindPassword")
    encrypted_values.extend(_iter_row_secret_values(ldap_config_rows, ("bindPassword",)))

    workspace_rows = await _find_many_with_encrypted_field(db.workspace, "scmToken")
    encrypted_values.extend(_iter_row_secret_values(workspace_rows, ("scmToken",)))

    conversation_share_rows = await _find_many_with_encrypted_field(db.conversationshare, "sharePassword")
    encrypted_values.extend(_iter_row_secret_values(conversation_share_rows, ("sharePassword",)))

    workspace_share_rows = await _find_many_with_encrypted_field(db.workspaceshare, "sharePassword")
    encrypted_values.extend(_iter_row_secret_values(workspace_share_rows, ("sharePassword",)))

    workspace_env_rows = await _find_many_with_encrypted_field(db.workspaceenvironmentvariable, "value")
    encrypted_values.extend(_iter_row_secret_values(workspace_env_rows, ("value",)))

    global_env_rows = await _find_many_with_encrypted_field(db.globalenvironmentvariable, "value")
    encrypted_values.extend(_iter_row_secret_values(global_env_rows, ("value",)))

    userspace_mount_source_rows = await db.userspacemountsource.find_many()  # type: ignore[call-arg]
    encrypted_values.extend(_iter_connection_config_secret_values(userspace_mount_source_rows))

    user_mount_source_rows = await db.useruserspacemountsource.find_many()  # type: ignore[call-arg]
    encrypted_values.extend(_iter_connection_config_secret_values(user_mount_source_rows))

    user_cloud_oauth_where = cast(
        Any,
        {
            "OR": [
                {"accessToken": {"startsWith": ENCRYPTED_PREFIX}},
                {"refreshToken": {"startsWith": ENCRYPTED_PREFIX}},
            ]
        },
    )
    user_cloud_oauth_rows = await db.usercloudoauthaccount.find_many(where=user_cloud_oauth_where)  # type: ignore[call-arg, arg-type]
    encrypted_values.extend(_iter_row_secret_values(user_cloud_oauth_rows, ("accessToken", "refreshToken")))

    return encrypted_values


async def recheck_encryption_key_health() -> bool:
    db = await get_db()
    for encrypted_value in await _load_encrypted_secret_values(db):
        if not attempt_decrypt(encrypted_value):
            return False

    reset_key_mismatch_state()
    return True
