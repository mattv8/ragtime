"""Small HTTP-client primitives shared by control-plane clients."""

from __future__ import annotations

from http.cookiejar import CookieJar
from typing import Any


class RejectResponseCookies(CookieJar):
    """A cookie jar that deliberately never persists or replays cookies.

    A pooled ``httpx`` client must not retain an upstream application's session
    cookie and attach it to a later request for another user or workspace.
    Explicit ``Cookie`` headers are unaffected because they are part of the
    individual request, not this jar.
    """

    def extract_cookies(self, response: Any, request: Any) -> None:
        """Discard every response cookie before CookieJar can store it."""

    def set_cookie(self, cookie: Any) -> None:
        """Keep direct CookieJar writes from making a cookie replayable."""

    def add_cookie_header(self, request: Any) -> None:
        """Never add a jar-owned Cookie header to an outgoing request."""
