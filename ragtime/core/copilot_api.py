"""Shared GitHub Copilot API request metadata helpers."""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

COPILOT_DEFAULT_BASE_URL = "https://api.githubcopilot.com"
COPILOT_API_VERSION = "2025-05-01"
COPILOT_EDITOR_VERSION = "vscode/1.99.0"
COPILOT_PLUGIN_VERSION = "copilot-chat/0.26.3"
COPILOT_INTEGRATION_ID = "vscode-chat"
COPILOT_USER_AGENT = "GitHubCopilotChat/0.26.3"
COPILOT_DEFAULT_INTENT = "conversation-agent"
COPILOT_DEFAULT_INITIATOR = "user"


def build_copilot_headers(
    *,
    access_token: Optional[str] = None,
    interaction_id: Optional[str] = None,
    intent: str = COPILOT_DEFAULT_INTENT,
    accept: str = "application/json",
    include_content_type: bool = False,
) -> dict[str, str]:
    """Build Copilot headers that match the shipped VS Code client shape."""
    headers = {
        "Accept": accept,
        "Openai-Intent": intent,
        "X-Initiator": COPILOT_DEFAULT_INITIATOR,
        "X-GitHub-Api-Version": COPILOT_API_VERSION,
        "Copilot-Integration-Id": COPILOT_INTEGRATION_ID,
        "X-Interaction-Id": interaction_id or str(uuid4()),
        "User-Agent": COPILOT_USER_AGENT,
        "Editor-Version": COPILOT_EDITOR_VERSION,
        "Editor-Plugin-Version": COPILOT_PLUGIN_VERSION,
    }
    if include_content_type:
        headers["Content-Type"] = "application/json"
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers