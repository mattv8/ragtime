from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

from ragtime.core import generation_policy
from ragtime.core.generation_policy import GenerationSurface


@contextmanager
def enabled_generation_policy(*user_ids: str, surface: GenerationSurface = "chat"):
    """Bind an enabled trusted surface backed by explicit enabled policy rows."""
    users = [SimpleNamespace(id=user_id, chatEnabled=True, userspaceGenerationEnabled=True) for user_id in user_ids]
    db = SimpleNamespace(
        appsettings=SimpleNamespace(find_unique=mock.AsyncMock(return_value=SimpleNamespace(chatEnabled=True, userspaceGenerationEnabled=True))),
        user=SimpleNamespace(find_many=mock.AsyncMock(return_value=users)),
    )
    with (
        mock.patch.object(generation_policy, "get_db", new=mock.AsyncMock(return_value=db)),
        generation_policy.generation_context(surface, *user_ids),
    ):
        yield
