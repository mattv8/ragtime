"""Structured build briefs submitted by external planning agents."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated

from pydantic import BaseModel, Field

BriefItem = Annotated[str, Field(min_length=1, max_length=2000)]


class BuildBriefInput(BaseModel):
    """Structured build request from an external planning agent."""

    idempotency_key: str = Field(
        min_length=8,
        max_length=128,
        description="Client-generated unique key. Retries with the same key return the original task instead of starting a duplicate build.",
    )
    title: str = Field(min_length=1, max_length=120, description="Short task title; becomes the conversation title.")
    objective: str = Field(min_length=1, max_length=4000, description="What the builder should accomplish.")
    requirements: list[BriefItem] = Field(min_length=1, max_length=100, description="Concrete, verifiable requirements.")
    acceptance_criteria: list[BriefItem] = Field(min_length=1, max_length=100, description="How to judge the work is done.")
    constraints: list[BriefItem] = Field(default_factory=list, max_length=100, description="Hard constraints (tech choices, style, scope).")
    preserve_paths: list[BriefItem] = Field(
        default_factory=list,
        max_length=200,
        description="Existing workspace files that must not be rewritten wholesale. Each must exist in the workspace.",
    )
    data_component_ids: list[BriefItem] = Field(
        default_factory=list,
        max_length=100,
        description="Workspace-selected tool component IDs the app must wire live data from.",
    )
    non_goals: list[BriefItem] = Field(default_factory=list, max_length=100, description="Explicitly out of scope.")
    context_revision: str | None = Field(
        default=None,
        description="Informational: the context_revision from /context when this brief was drafted.",
    )


def compute_brief_payload_hash(brief: BuildBriefInput) -> str:
    """Stable content hash used for idempotency conflict detection."""
    payload = brief.model_dump(exclude={"idempotency_key"})
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _section(header: str, items: list[str]) -> str:
    lines = "\n".join(f"- {str(item).strip()}" for item in items if str(item).strip())
    if not lines:
        return ""
    return f"\n## {header}\n\n{lines}\n"


def render_build_brief(brief: BuildBriefInput, workspace_name: str) -> str:
    """Render the brief into the deterministic user message sent to the builder."""
    parts = [
        "# External Agent Build Brief",
        "",
        f'Workspace: "{workspace_name}"',
        f"Title: {brief.title.strip()}",
    ]
    if brief.context_revision:
        parts.append(f"Planning context revision: {brief.context_revision}")
    parts += ["", "## Objective", "", brief.objective.strip(), ""]
    body = "\n".join(parts)
    body += _section("Requirements", brief.requirements)
    body += _section("Acceptance Criteria", brief.acceptance_criteria)
    body += _section("Constraints", brief.constraints)
    body += _section("Existing Files To Preserve", brief.preserve_paths)
    body += _section("Live Data Components (component_id)", brief.data_component_ids)
    body += _section("Non-Goals", brief.non_goals)
    body += (
        "\n## Execution Notes\n\n"
        "- This brief was submitted by an external planning agent on the user's behalf.\n"
        "- Inspect the existing workspace first and extend its current architecture.\n"
        "- Work autonomously: make reasonable assumptions instead of ending your turn "
        "with a question. If an assumption is significant, state it in your final summary.\n"
    )
    return body
