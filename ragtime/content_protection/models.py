"""Validated, transport-independent content-protection data models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

STANDARD_SCOPE = (
    "Ordinary operational/business discussion and nonsensitive returned records; "
    "general educational discussion about finance or HR is allowed. Company-specific "
    "nonpublic finance, personal employee/customer records, and secrets are excluded."
)
PUBLIC_SCOPE = "Public, nonsensitive information only. Nonpublic operational, financial, personnel, strategic information, and secrets are excluded."


def default_profiles() -> list["Profile"]:
    """Return fresh, editable policy defaults for an unconfigured installation."""
    return [
        Profile(id="standard", name="Standard", level=0, scope=STANDARD_SCOPE),
        Profile(
            id="finance",
            name="Finance",
            level=1,
            scope="Standard plus nonpublic company financial records, prices, costs, margins, forecasts, and financial analysis. Individually identifiable compensation and HR records remain excluded.",
        ),
        Profile(
            id="people",
            name="People",
            level=1,
            scope="Standard plus personnel and individual compensation information. Does not grant unrelated company finance or strategic records.",
        ),
        Profile(
            id="trusted_business",
            name="Trusted business",
            level=2,
            scope="Nonpublic company financial records, personnel/individual compensation information, and restricted strategic/business information. Does not override credentials, platform authorization, or workspace isolation.",
        ),
    ]


CoverageMode = Literal["all_supported_traffic", "selected_scopes"]
RequirementMode = Literal["require", "inherit"]
OverrideMode = Literal["inherit", "always_classify", "never_classify"]


@dataclass(frozen=True)
class ProtectionContext:
    user_id: str | None = None
    audience_user_ids: tuple[str | None, ...] = ()
    surface: str = "chat"
    mcp_route: str | None = None
    tool_id: str | None = None
    resource_id: str | None = None
    public: bool = False
    baseline: Literal["user", "anonymous", "service", "public"] = "user"


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    level: int = Field(ge=0, le=2)
    scope: str = Field(min_length=1, max_length=4000)


class GroupProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)


class Requirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope_kind: Literal["group", "tool", "mcp_route", "surface"]
    scope_key: str = Field(min_length=1, max_length=256)
    mode: RequirementMode


class UserOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1)
    mode: OverrideMode


class ContentProtectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: int = Field(default=0, ge=0)
    enabled: bool = False
    classifier_model: str | None = None
    coverage_mode: CoverageMode = "all_supported_traffic"
    profiles: list[Profile] = Field(default_factory=default_profiles)
    group_profiles: list[GroupProfile] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    user_overrides: list[UserOverride] = Field(default_factory=list)

    def model_post_init(self, __context: object) -> None:
        ids = [profile.id for profile in self.profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("profile ids must be unique")
        groups = [item.group_id for item in self.group_profiles]
        if len(groups) != len(set(groups)):
            raise ValueError("group profile mappings must be unique")
        overrides = [item.user_id for item in self.user_overrides]
        if len(overrides) != len(set(overrides)):
            raise ValueError("user overrides must be unique")


class ContentProtectionError(Exception):
    """Fixed, safe error data for transports to serialize."""

    def __init__(self, code: str, request_id: str) -> None:
        self.code = code
        self.request_id = request_id
        super().__init__(code)

    def public_detail(self) -> dict[str, str]:
        messages = {
            "content_denied": "This content is not available under your access profile.",
            "operation_completed_response_withheld": "The operation completed but its response is not available under your access profile.",
            "classifier_unavailable": "Content protection is currently unavailable.",
            "classifier_invalid_response": "Content protection is currently unavailable.",
            "content_unclassifiable": "This content cannot be inspected safely.",
            "policy_changed": "Content protection policy changed; retry the request.",
        }
        return {"code": self.code, "message": messages.get(self.code, "Content protection request failed."), "request_id": self.request_id}
