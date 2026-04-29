"""Helpers for normalizing user identity for public and prompt-facing use."""

from __future__ import annotations

_LOCAL_USERNAME_PREFIX = "local:"


def normalize_user_identity(
    username: str | None,
    display_name: str | None = None,
    *,
    lowercase_username: bool = False,
) -> tuple[str, str | None]:
    """Normalize stored user identity for user-facing rendering.

    Local users are stored with a ``local:`` username prefix to avoid
    collisions with LDAP users. User-facing surfaces should not expose that
    storage detail, so this helper strips the prefix, trims whitespace, and
    optionally lowercases the public username for slug/share-path use.
    """

    normalized_username = str(username or "").strip()
    if (
        normalized_username[: len(_LOCAL_USERNAME_PREFIX)].lower()
        == _LOCAL_USERNAME_PREFIX
    ):
        normalized_username = normalized_username[len(_LOCAL_USERNAME_PREFIX) :].strip()

    if lowercase_username:
        normalized_username = normalized_username.lower()

    normalized_display_name = str(display_name or "").strip() or None
    return normalized_username, normalized_display_name
