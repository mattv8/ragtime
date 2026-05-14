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
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException

from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_BLOCK_PRIVATE_NETWORKS
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_BROWSE_TIMEOUT_MAX_SECONDS
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_COMMAND_TIMEOUT_MAX_SECONDS
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_ENABLED
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_PDF_READ_DEFAULT_TEXT_CHARS
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_PDF_READ_MAX_TEXT_CHARS
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_SEARCH_MAX_RESULTS
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_SEARCH_PDF_MAX_BYTES
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_SEARCH_PDF_MAX_RESULTS
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_SEARXNG_BASE_URL
from ragtime.chat_runtime.presets import CHAT_DIAGNOSTICS_SESSION_IDLE_TTL_SECONDS
from ragtime.chat_runtime.presets import CHAT_WEB_READ_PDF_TOOL_ID
from ragtime.config import settings
from ragtime.core.logging import get_logger
from ragtime.core.runtime_manager_client import runtime_manager_enabled
from ragtime.core.runtime_manager_client import runtime_manager_request
from ragtime.core.security import validate_chat_diagnostic_command
from ragtime.core.security import validate_external_url

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

    def validate_command(self, command: str) -> tuple[bool, str]:
        return validate_chat_diagnostic_command(command)

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

    def _max_search_pdf_results(self) -> int:
        return max(
            0,
            min(
                self._max_search_results(),
                int(CHAT_DIAGNOSTICS_SEARCH_PDF_MAX_RESULTS),
            ),
        )

    def _max_search_pdf_bytes(self) -> int:
        return max(256 * 1024, int(CHAT_DIAGNOSTICS_SEARCH_PDF_MAX_BYTES))

    def _default_pdf_read_text_chars(self) -> int:
        return max(500, int(CHAT_DIAGNOSTICS_PDF_READ_DEFAULT_TEXT_CHARS))

    def _max_pdf_read_text_chars(self) -> int:
        return max(
            self._default_pdf_read_text_chars(),
            int(CHAT_DIAGNOSTICS_PDF_READ_MAX_TEXT_CHARS),
        )

    def _bound_pdf_read_text_chars(self, max_chars: int | None) -> int:
        if max_chars is None:
            return self._default_pdf_read_text_chars()
        return max(1, min(self._max_pdf_read_text_chars(), int(max_chars)))

    @staticmethod
    def _tavily_api_key() -> str:
        return str(getattr(settings, "tavily_api_key", "")).strip()

    @staticmethod
    def _compact_snippet(value: Any) -> str:
        return " ".join(str(value or "").split())[:280]

    @staticmethod
    def _is_runtime_session_not_found_error(exc: Exception) -> bool:
        if not isinstance(exc, HTTPException):
            return False
        detail = str(getattr(exc, "detail", "") or "")
        return "Runtime session not found" in detail and "(404)" in detail

    @staticmethod
    def _is_pdf_url(value: str) -> bool:
        try:
            parsed = urlsplit(value or "")
        except ValueError:
            return False
        path = parsed.path.lower()
        return path.endswith(".pdf") or "/pdf/" in path or path.endswith("/pdf")

    @classmethod
    def _is_likely_pdf_result(cls, result: dict[str, Any]) -> bool:
        url = str(result.get("url") or "")
        if cls._is_pdf_url(url):
            return True
        title = str(result.get("title") or "").lower()
        snippet = str(result.get("snippet") or result.get("content") or "").lower()
        return "[pdf]" in title or " pdf " in f" {title} " or "[pdf]" in snippet

    async def _attach_pdf_metadata(
        self,
        results: list[dict[str, Any]],
        *,
        include_pdf_metadata: bool,
    ) -> int:
        if not include_pdf_metadata:
            return 0
        max_pdf_results = self._max_search_pdf_results()
        if max_pdf_results <= 0:
            return 0

        candidates: list[tuple[int, dict[str, Any]]] = []
        for idx, result in enumerate(results):
            if len(candidates) >= max_pdf_results:
                break
            if self._is_likely_pdf_result(result):
                candidates.append((idx, result))
        if not candidates:
            return 0

        for idx, _ in candidates:
            results[idx]["pdf"] = {
                "status": "available",
                "read_tool": CHAT_WEB_READ_PDF_TOOL_ID,
                "max_bytes": self._max_search_pdf_bytes(),
            }
        return len(candidates)

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
        include_pdf_metadata: bool,
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

        pdf_result_count = await self._attach_pdf_metadata(
            results,
            include_pdf_metadata=include_pdf_metadata,
        )

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
            "pdf_result_count": pdf_result_count,
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
        include_pdf_metadata: bool,
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

        pdf_result_count = await self._attach_pdf_metadata(
            results,
            include_pdf_metadata=include_pdf_metadata,
        )

        return {
            "ok": True,
            "blocked": False,
            "query": query,
            "provider": "tavily",
            "results": results,
            "result_count": len(results),
            "pdf_result_count": pdf_result_count,
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

    async def _forget_session_if_current(
        self,
        conversation_id: str,
        provider_session_id: str,
    ) -> None:
        async with self._lock:
            existing = self._sessions.get(conversation_id)
            if existing and existing.provider_session_id == provider_session_id:
                self._sessions.pop(conversation_id, None)

    async def _request_with_session_retry(
        self,
        *,
        conversation_id: str,
        operation_name: str,
        path_template: str,
        json_payload: dict[str, Any],
        timeout_override_seconds: float | None = None,
    ) -> dict[str, Any]:
        for attempt in range(2):
            session = await self._ensure_session(conversation_id)
            async with session.lock:
                session.last_used_at = time.monotonic()
                try:
                    return await self._request(
                        "POST",
                        path_template.format(
                            provider_session_id=session.provider_session_id
                        ),
                        json_payload=json_payload,
                        timeout_override_seconds=timeout_override_seconds,
                    )
                except Exception as exc:
                    if attempt == 0 and self._is_runtime_session_not_found_error(exc):
                        await self._forget_session_if_current(
                            conversation_id,
                            session.provider_session_id,
                        )
                        logger.info(
                            "chat_runtime evicted stale %s session conv=%s provider=%s",
                            operation_name,
                            conversation_id,
                            session.provider_session_id,
                        )
                        continue
                    raise
        raise HTTPException(
            status_code=502,
            detail=f"Chat diagnostics {operation_name} retry failed",
        )

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
        return await self._request_with_session_retry(
            conversation_id=conversation_id,
            operation_name="exec",
            path_template="/sessions/{provider_session_id}/exec",
            json_payload={
                "command": command,
                "timeout_seconds": bounded_timeout,
            },
            timeout_override_seconds=bounded_timeout + 15.0,
        )

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
        return await self._request_with_session_retry(
            conversation_id=conversation_id,
            operation_name="browse",
            path_template="/sessions/{provider_session_id}/external-browse",
            json_payload=payload,
            timeout_override_seconds=bounded_timeout_s + 15.0,
        )

    async def search_web(
        self,
        *,
        conversation_id: str,
        query: str,
        max_results: int | None = None,
        include_pdf_metadata: bool = True,
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
                include_pdf_metadata=include_pdf_metadata,
            )
        return await self._search_web_searxng(
            query=cleaned_query,
            max_results=cap,
            include_pdf_metadata=include_pdf_metadata,
        )

    async def read_pdf_url(
        self,
        *,
        conversation_id: str,
        url: str,
        start_char: int = 0,
        max_chars: int | None = None,
        query: str = "",
        max_matches: int = 5,
    ) -> dict[str, Any]:
        """Fetch a PDF URL and return a targeted text range or query snippets."""
        cleaned_url = (url or "").strip()
        if not cleaned_url:
            raise HTTPException(status_code=400, detail="PDF URL is empty")
        ok, error_message = validate_external_url(
            cleaned_url,
            allow_private_networks=not CHAT_DIAGNOSTICS_BLOCK_PRIVATE_NETWORKS,
        )
        if not ok:
            return {"status": "skipped", "ok": False, "error": error_message}

        bounded_max_chars = self._bound_pdf_read_text_chars(max_chars)
        payload = {
            "url": cleaned_url,
            "start_char": max(0, int(start_char)),
            "max_chars": bounded_max_chars,
            "query": " ".join(str(query or "").split()),
            "max_matches": max(1, min(20, int(max_matches))),
            "max_bytes": self._max_search_pdf_bytes(),
        }
        return await self._request_with_session_retry(
            conversation_id=conversation_id,
            operation_name="pdf-read",
            path_template="/sessions/{provider_session_id}/pdf-read",
            json_payload=payload,
            timeout_override_seconds=max(30.0, float(self._browse_timeout_max()) + 20.0),
        )


chat_runtime_service = ChatRuntimeService()
