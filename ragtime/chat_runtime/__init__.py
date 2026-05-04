"""Chat-only diagnostics runtime (read-only terminal + Playwright browse).

This package owns a control-plane client for the runtime manager dedicated to
chat agents. It deliberately does NOT use ``UserSpaceRuntimeSession`` rows or
any Workspace DB record; instead it uses a synthetic per-conversation
``workspace_id`` namespace so the runtime worker's existing sandbox primitives
(``_resolve_workspace_root`` + ``spawn_sandboxed``) can be reused without
coupling to userspace persistence.
"""

from ragtime.chat_runtime.service import chat_runtime_service

__all__ = ["chat_runtime_service"]
