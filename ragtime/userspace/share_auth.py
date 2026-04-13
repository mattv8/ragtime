from __future__ import annotations

import hashlib
from collections.abc import Mapping

from fastapi import Response

SHARE_AUTH_HEADER = "x-userspace-share-auth"


def share_auth_cookie_name_for_slug(owner_username: str, share_slug: str) -> str:
    digest = hashlib.sha256(
        f"{owner_username}:{share_slug}".encode("utf-8")
    ).hexdigest()[:16]
    return f"userspace_share_auth_{digest}"


def share_auth_cookie_name_for_token(share_token: str) -> str:
    digest = hashlib.sha256((share_token or "").encode("utf-8")).hexdigest()[:16]
    return f"userspace_share_auth_tok_{digest}"


def share_auth_token_from_request(
    headers: Mapping[str, str] | None,
    cookies: Mapping[str, str] | None,
    *,
    share_token: str | None = None,
    owner_username: str | None = None,
    share_slug: str | None = None,
) -> str | None:
    header_token = str((headers or {}).get(SHARE_AUTH_HEADER, "") or "").strip()
    if header_token:
        return header_token

    if share_token is not None:
        cookie_name = share_auth_cookie_name_for_token(share_token)
    elif owner_username is not None and share_slug is not None:
        cookie_name = share_auth_cookie_name_for_slug(owner_username, share_slug)
    else:
        return None

    cookie_token = str((cookies or {}).get(cookie_name, "") or "").strip()
    return cookie_token or None


def set_share_auth_cookie(
    response: Response,
    token: str,
    *,
    max_age: int,
    secure: bool,
    share_token: str | None = None,
    owner_username: str | None = None,
    share_slug: str | None = None,
) -> None:
    if share_token is not None:
        cookie_name = share_auth_cookie_name_for_token(share_token)
    elif owner_username is not None and share_slug is not None:
        cookie_name = share_auth_cookie_name_for_slug(owner_username, share_slug)
    else:
        raise ValueError("share route identity is required")

    response.set_cookie(
        key=cookie_name,
        value=token,
        max_age=max_age,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
    )


def clear_share_auth_cookie(
    response: Response,
    *,
    share_token: str | None = None,
    owner_username: str | None = None,
    share_slug: str | None = None,
) -> None:
    if share_token is not None:
        cookie_name = share_auth_cookie_name_for_token(share_token)
    elif owner_username is not None and share_slug is not None:
        cookie_name = share_auth_cookie_name_for_slug(owner_username, share_slug)
    else:
        raise ValueError("share route identity is required")

    response.delete_cookie(key=cookie_name, path="/")
