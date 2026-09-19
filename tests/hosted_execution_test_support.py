from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from ragtime.core import hosted_execution_policy


def enabled_hosted_execution_policy(*user_ids: str):
    """Patch the policy DB with explicit globally and per-user enabled records."""
    users = [SimpleNamespace(id=user_id, hostedChatEnabled=True) for user_id in user_ids]
    db = SimpleNamespace(
        appsettings=SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(hostedChatEnabled=True))),
        user=SimpleNamespace(find_many=mock.AsyncMock(return_value=users)),
    )
    return mock.patch.object(hosted_execution_policy, "get_db", new=mock.AsyncMock(return_value=db))
