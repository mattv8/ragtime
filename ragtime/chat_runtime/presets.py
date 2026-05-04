"""Opinionated chat diagnostics presets.

These values are intentionally code-owned rather than environment-driven.
They define the default operational envelope for chat-only diagnostics.
"""

from __future__ import annotations

CHAT_DIAGNOSTICS_ENABLED = True
CHAT_DIAGNOSTIC_COMMAND_TOOL_ID = "run_chat_diagnostic_command"
CHAT_WEB_SEARCH_TOOL_ID = "web_search"
CHAT_WEB_BROWSE_TOOL_ID = "web_browse"
CHAT_LEGACY_WEB_BROWSE_SEARCH_TOOL_ID = "chat_web_browse_search"
CHAT_DIAGNOSTIC_BUILTIN_TOOL_IDS = frozenset(
    {
        CHAT_DIAGNOSTIC_COMMAND_TOOL_ID,
        CHAT_WEB_SEARCH_TOOL_ID,
        CHAT_WEB_BROWSE_TOOL_ID,
    }
)
CHAT_LEGACY_BUILTIN_TOOL_ID_ALIASES = {
    CHAT_LEGACY_WEB_BROWSE_SEARCH_TOOL_ID: (
        CHAT_WEB_SEARCH_TOOL_ID,
        CHAT_WEB_BROWSE_TOOL_ID,
    )
}
CHAT_DIAGNOSTICS_COMMAND_TIMEOUT_MAX_SECONDS = 30
CHAT_DIAGNOSTICS_BROWSE_TIMEOUT_MAX_SECONDS = 25
CHAT_DIAGNOSTICS_SESSION_IDLE_TTL_SECONDS = 1800
# Default open for local diagnostics; set True to enforce public-only targets.
CHAT_DIAGNOSTICS_BLOCK_PRIVATE_NETWORKS = False
CHAT_DIAGNOSTICS_SEARCH_MAX_RESULTS = 5
CHAT_DIAGNOSTICS_SEARXNG_BASE_URL = "http://searxng:8080"
