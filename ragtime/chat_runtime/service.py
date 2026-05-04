"""Chat-only diagnostics runtime client (terminal + Playwright web browse).

Leases dedicated runtime-manager sessions keyed by conversation id, using a
synthetic ``chat-diag-<conversation_id>`` workspace namespace so the existing
worker sandbox is reused without writing any Workspace/UserSpaceRuntimeSession
DB rows.

Exposes two capabilities to the chat tool layer:

- ``exec_command``: runs a read-only diagnostic shell command in the leased
  sandbox. The actual command-policy validation lives in
  ``ragtime.core.security.validate_chat_diagnostic_command`` and must be
  applied by callers BEFORE invoking this client.
- ``browse_url``: drive Playwright against an arbitrary external URL via the
    runtime manager's ``/external-browse`` endpoint. URL safety is enforced via
    ``validate_external_url`` in callers.
- ``search_web``: query the configured web-search backend. The default preset
    uses the internal SearXNG service unless ``TAVILY_API_KEY`` is configured.
    Playwright is only used for direct page browse.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from fastapi import HTTPException

from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_BROWSE_TIMEOUT_MAX_SECONDS
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_COMMAND_TIMEOUT_MAX_SECONDS
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_ENABLED
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_SEARCH_MAX_RESULTS
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_SEARXNG_BASE_URL
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_SESSION_IDLE_TTL_SECONDS
from ragtime.config import settings
from ragtime.core.logging import get_logger
from ragtime.core.runtime_manager_client import runtime_manager_enabled
from ragtime.core.runtime_manager_client import runtime_manager_request

logger = get_logger(__name__)


_CHAT_DIAG_PROVIDER_USER_ID = "chat-diagnostics"
_CHAT_DIAG_WORKSPACE_PREFIX = "chat-diag-"
_CHAT_DIAG_CLEANUP_INTERVAL_SECONDS = 60
_TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"


@dataclass
class _ChatDiagSession:
    conversation_id: str
    workspace_id: str
    provider_session_id: str
    last_used_at: float
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ChatRuntimeService:
    def __init__(self) -> None:
        self._sessions: dict[str, _ChatDiagSession] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None

    # -- Configuration helpers ------------------------------------------------

    def _enabled(self) -> bool:
        if not CHAT_DIAGNOSTICS_ENABLED:
            return False
        return runtime_manager_enabled()

    def is_enabled(self) -> bool:
        return self._enabled()

    def _idle_ttl_seconds(self) -> int:
        return max(
            60,
            int(CHAT_DIAGNOSTICS_SESSION_IDLE_TTL_SECONDS),
        )

    def _command_timeout_max(self) -> int:
        return max(
            1,
            min(
                120,
                int(CHAT_DIAGNOSTICS_COMMAND_TIMEOUT_MAX_SECONDS),
            ),
        )

    def _browse_timeout_max(self) -> int:
        return max(
            5,
            min(
                60,
                int(CHAT_DIAGNOSTICS_BROWSE_TIMEOUT_MAX_SECONDS),
            ),
        )

    def _max_search_results(self) -> int:
        return max(
            1,
            min(
                25,
                int(CHAT_DIAGNOSTICS_SEARCH_MAX_RESULTS),
            ),
        )

    @staticmethod
    def _tavily_api_key() -> str:
        return str(getattr(settings, "tavily_api_key", "")).strip()

    @staticmethod
    def _compact_snippet(value: Any) -> str:
        return " ".join(str(value or "").split())[:280]

    @classmethod
    def _normalize_search_result(
        cls,
        item: dict[str, Any],
        *,
        provider: str,
    ) -> dict[str, Any] | None:
        result_url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        if not result_url or not title:
            return None

        result_payload: dict[str, Any] = {
            "title": title[:200],
            "url": result_url,
            "snippet": cls._compact_snippet(item.get("content") or item.get("snippet")),
        }
        score = item.get("score")
        if isinstance(score, (int, float)):
            result_payload["score"] = float(score)

        favicon = str(item.get("favicon") or "").strip()
        if favicon:
            result_payload["favicon"] = favicon

        engine = item.get("engine")
        engines = item.get("engines")
        if isinstance(engine, str) and engine.strip():
            result_payload["engine"] = engine.strip()
        elif isinstance(engines, list):
            clean_engines = [
                str(value).strip() for value in engines if str(value).strip()
            ]
            if clean_engines:
                result_payload["engine"] = ", ".join(clean_engines[:3])

        result_payload["source_provider"] = provider
        return result_payload

    async def _search_web_searxng(
        self,
        *,
        query: str,
        max_results: int,
    ) -> dict[str, Any]:
        base_url = str(CHAT_DIAGNOSTICS_SEARXNG_BASE_URL or "").strip().rstrip("/")
        if not base_url:
            raise HTTPException(
                status_code=503,
                detail="Chat search provider 'searxng' is missing a base URL preset",
            )

        endpoint = f"{base_url}/search"
        timeout = httpx.Timeout(max(10.0, float(self._browse_timeout_max()) + 5.0))
        params = {
            "q": query,
            "format": "json",
            "categories": "general",
            "language": "en",
        }
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
            ) as client:
                response = await client.get(
                    endpoint,
                    params=params,
                    headers={"Accept": "application/json"},
                )
        except Exception as exc:
            exc_type = exc.__class__.__name__
            exc_message = str(exc).strip()
            detail = f"SearXNG search request failed ({exc_type})"
            if exc_message:
                detail = f"{detail}: {exc_message}"
            raise HTTPException(status_code=502, detail=detail) from exc

        if response.status_code >= 400:
            body_preview = response.text[:256]
            raise HTTPException(
                status_code=502,
                detail=(
                    f"SearXNG search request failed ({response.status_code}): "
                    f"{body_preview}"
                ),
            )

        try:
            data = response.json()
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="SearXNG search returned invalid JSON",
            ) from exc

        results: list[dict[str, Any]] = []
        for item in data.get("results") or []:
            if len(results) >= max_results or not isinstance(item, dict):
                break
            normalized = self._normalize_search_result(item, provider="searxng")
            if normalized is not None:
                results.append(normalized)

        answers = data.get("answers") or []
        answer = ""
        if isinstance(answers, list) and answers:
            answer = self._compact_snippet(answers[0])

        return {
            "ok": True,
            "blocked": False,
            "query": query,
            "provider": "searxng",
            "results": results,
            "result_count": len(results),
            "engine_url": str(response.url),
            "answer": answer,
            "response_time": data.get("search_time"),
            "request_id": "",
            "suggestions": data.get("suggestions") or [],
            "unresponsive_engines": data.get("unresponsive_engines") or [],
        }

    async def _search_web_tavily(
        self,
        *,
        query: str,
        max_results: int,
    ) -> dict[str, Any]:
        api_key = self._tavily_api_key()
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Chat search provider 'tavily' requires TAVILY_API_KEY to be set"
                ),
            )

        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "include_answer": True,
            "include_favicon": True,
        }
        timeout = httpx.Timeout(max(10.0, float(self._browse_timeout_max()) + 5.0))
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
            ) as client:
                response = await client.post(
                    _TAVILY_SEARCH_ENDPOINT,
                    json=payload,
                    headers=headers,
                )
        except Exception as exc:
            exc_type = exc.__class__.__name__
            exc_message = str(exc).strip()
            detail = f"Tavily search request failed ({exc_type})"
            if exc_message:
                detail = f"{detail}: {exc_message}"
            raise HTTPException(status_code=502, detail=detail) from exc

        if response.status_code >= 400:
            body_preview = response.text[:256]
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Tavily search request failed ({response.status_code}): "
                    f"{body_preview}"
                ),
            )

        try:
            data = response.json()
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="Tavily search returned invalid JSON",
            ) from exc

        results: list[dict[str, Any]] = []
        for item in data.get("results") or []:
            if len(results) >= max_results or not isinstance(item, dict):
                break
            normalized = self._normalize_search_result(item, provider="tavily")
            if normalized is not None:
                results.append(normalized)

        return {
            "ok": True,
            "blocked": False,
            "query": query,
            "provider": "tavily",
            "results": results,
            "result_count": len(results),
            "engine_url": _TAVILY_SEARCH_ENDPOINT,
            "answer": str(data.get("answer") or "").strip(),
            "response_time": data.get("response_time"),
            "request_id": str(data.get("request_id") or "").strip(),
        }

    # -- Manager HTTP plumbing ------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        timeout_override_seconds: float | None = None,
    ) -> dict[str, Any]:
        if not self._enabled():
            raise HTTPException(
                status_code=503,
                detail="Chat diagnostics runtime is disabled or misconfigured",
            )
        return await runtime_manager_request(
            method,
            path,
            json_payload=json_payload,
            timeout_override_seconds=timeout_override_seconds,
            unavailable_detail_prefix="Chat diagnostics runtime manager unavailable",
            request_failed_detail_prefix="Chat diagnostics runtime manager request failed",
        )

    # -- Session lifecycle ----------------------------------------------------

    @staticmethod
    def _sanitize_conversation_id(conversation_id: str) -> str:
        cleaned = (conversation_id or "").strip()
        if not cleaned:
            cleaned = "anonymous"
        # Workspace IDs propagate to filesystem paths in the worker; restrict
        # to a conservative character set.
        return re.sub(r"[^A-Za-z0-9_\-]", "-", cleaned)[:40] or "anonymous"

    def _workspace_id_for(self, conversation_id: str) -> str:
        return f"{_CHAT_DIAG_WORKSPACE_PREFIX}{self._sanitize_conversation_id(conversation_id)}"

    async def _ensure_cleanup_task(self) -> None:
        if self._cleanup_task is None or self._cleanup_task.done():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._cleanup_task = loop.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(_CHAT_DIAG_CLEANUP_INTERVAL_SECONDS)
                await self._reap_idle_sessions()
        except asyncio.CancelledError:
            return
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("chat_runtime cleanup loop failed: %s", exc)

    async def _reap_idle_sessions(self) -> None:
        ttl = self._idle_ttl_seconds()
        now = time.monotonic()
        expired: list[_ChatDiagSession] = []
        async with self._lock:
            for conv_id, sess in list(self._sessions.items()):
                if now - sess.last_used_at > ttl:
                    expired.append(sess)
                    self._sessions.pop(conv_id, None)
        for sess in expired:
            try:
                await self._request(
                    "POST",
                    f"/sessions/{sess.provider_session_id}/stop",
                )
                logger.info(
                    "chat_runtime released idle session conv=%s provider=%s",
                    sess.conversation_id,
                    sess.provider_session_id,
                )
            except Exception as exc:
                logger.warning(
                    "chat_runtime failed to stop idle session conv=%s: %s",
                    sess.conversation_id,
                    exc,
                )

    async def _ensure_session(self, conversation_id: str) -> _ChatDiagSession:
        await self._ensure_cleanup_task()
        async with self._lock:
            existing = self._sessions.get(conversation_id)
            if existing:
                existing.last_used_at = time.monotonic()
                return existing

        # Lease a fresh provider session against a synthetic workspace id.
        workspace_id = self._workspace_id_for(conversation_id)
        payload = {
            "workspace_id": workspace_id,
            "leased_by_user_id": _CHAT_DIAG_PROVIDER_USER_ID,
            "workspace_env": {},
            "workspace_env_visibility": {},
            "workspace_mounts": [],
        }
        response = await self._request("POST", "/sessions/start", json_payload=payload)
        provider_session_id = str(response.get("provider_session_id") or "").strip()
        if not provider_session_id:
            raise HTTPException(
                status_code=502,
                detail="Chat diagnostics runtime manager did not return a session id",
            )

        async with self._lock:
            existing = self._sessions.get(conversation_id)
            if existing:
                # A concurrent ensure won the race; release the duplicate lease
                # in the background.
                duplicate_provider = provider_session_id
                existing.last_used_at = time.monotonic()

                async def _release_duplicate() -> None:
                    try:
                        await self._request(
                            "POST",
                            f"/sessions/{duplicate_provider}/stop",
                        )
                    except Exception:
                        pass

                asyncio.create_task(_release_duplicate())
                return existing

            session = _ChatDiagSession(
                conversation_id=conversation_id,
                workspace_id=workspace_id,
                provider_session_id=provider_session_id,
                last_used_at=time.monotonic(),
            )
            self._sessions[conversation_id] = session
            return session

    async def release_conversation(self, conversation_id: str) -> None:
        """Best-effort release of the diagnostic session for a conversation."""
        async with self._lock:
            session = self._sessions.pop(conversation_id, None)
        if not session:
            return
        try:
            await self._request(
                "POST",
                f"/sessions/{session.provider_session_id}/stop",
            )
        except Exception as exc:
            logger.warning(
                "chat_runtime release_conversation conv=%s failed: %s",
                conversation_id,
                exc,
            )

    # -- Public capabilities --------------------------------------------------

    async def exec_command(
        self,
        *,
        conversation_id: str,
        command: str,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        """Execute a (caller-validated) read-only command in the chat sandbox."""
        bounded_timeout = max(1, min(self._command_timeout_max(), int(timeout_seconds)))
        session = await self._ensure_session(conversation_id)
        async with session.lock:
            session.last_used_at = time.monotonic()
            response = await self._request(
                "POST",
                f"/sessions/{session.provider_session_id}/exec",
                json_payload={
                    "command": command,
                    "timeout_seconds": bounded_timeout,
                },
                timeout_override_seconds=bounded_timeout + 15.0,
            )
        return response

    async def browse_url(
        self,
        *,
        conversation_id: str,
        url: str,
        timeout_seconds: int | None = None,
        wait_after_load_ms: int = 1500,
        extract_links: bool = True,
        max_text_chars: int = 4000,
        max_links: int = 20,
        user_agent: str = "",
    ) -> dict[str, Any]:
        """Drive Playwright against an external URL via runtime manager."""
        bounded_timeout_s = max(
            5,
            min(
                self._browse_timeout_max(),
                int(timeout_seconds or self._browse_timeout_max()),
            ),
        )
        timeout_ms = bounded_timeout_s * 1000
        payload: dict[str, Any] = {
            "url": url,
            "timeout_ms": timeout_ms,
            "wait_after_load_ms": max(0, min(15000, int(wait_after_load_ms))),
            "extract_links": bool(extract_links),
            "max_text_chars": max(200, min(20000, int(max_text_chars))),
            "max_links": max(0, min(100, int(max_links))),
            "user_agent": str(user_agent or ""),
        }
        session = await self._ensure_session(conversation_id)
        async with session.lock:
            session.last_used_at = time.monotonic()
            return await self._request(
                "POST",
                f"/sessions/{session.provider_session_id}/external-browse",
                json_payload=payload,
                timeout_override_seconds=bounded_timeout_s + 15.0,
            )

    async def search_web(
        self,
        *,
        conversation_id: str,
        query: str,
        max_results: int | None = None,
    ) -> dict[str, Any]:
        """Run a web search via the configured chat diagnostics provider.

        Returns a structured payload with up to ``max_results`` ranked
        hits (title/url/snippet). Snippets are extracted from the configured
        provider response.
        """
        del conversation_id

        cleaned_query = (query or "").strip()
        if not cleaned_query:
            raise HTTPException(status_code=400, detail="Search query is empty")
        if len(cleaned_query) > 512:
            raise HTTPException(
                status_code=400, detail="Search query exceeds 512 characters"
            )
        cap = self._max_search_results()
        if max_results is not None:
            cap = max(1, min(cap, int(max_results)))

        if self._tavily_api_key():
            return await self._search_web_tavily(
                query=cleaned_query,
                max_results=cap,
            )
        return await self._search_web_searxng(
            query=cleaned_query,
            max_results=cap,
        )


chat_runtime_service = ChatRuntimeService()
