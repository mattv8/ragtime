"""Bounded wire models for remote-client enrollment."""

import base64
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ragtime.indexer.models import ToolType


class PairingCreateRequest(BaseModel):
    tool_type: ToolType
    tool_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("tool_type")
    @classmethod
    def validate_shell_type(cls, value: ToolType) -> ToolType:
        if value not in {ToolType.SSH_SHELL, ToolType.WINDOWS_SHELL}:
            raise ValueError("tool_type must be ssh_shell or windows_shell")
        return value


class RemoteFacts(BaseModel):
    os: Literal["linux", "macos", "windows"]
    username: str = Field(min_length=1, max_length=256)
    hostname: str = Field(min_length=1, max_length=255)
    addresses: list[str] = Field(default_factory=list, max_length=8)
    sshd_port: int = Field(default=22, ge=1, le=65535)
    host_keys: list[str] = Field(default_factory=list, max_length=6)
    shared_admin_keyfile: bool = False
    errors: list[str] = Field(default_factory=list, max_length=8)
    python_available: bool = False

    @field_validator("addresses")
    @classmethod
    def bounded_addresses(cls, value: list[str]) -> list[str]:
        import ipaddress

        if any(len(item) > 64 for item in value):
            raise ValueError("address is too long")
        try:
            for item in value:
                ipaddress.ip_address(item)
        except ValueError as exc:
            raise ValueError("addresses must be numeric IP addresses") from exc
        return value

    @field_validator("host_keys")
    @classmethod
    def bounded_host_keys(cls, value: list[str]) -> list[str]:
        for item in value:
            parts = item.split()
            if len(item) > 8192 or len(parts) < 2 or not parts[0].startswith(("ssh-", "ecdsa-")):
                raise ValueError("host_keys must be bounded OpenSSH public keys")
            try:
                base64.b64decode(parts[1], validate=True)
            except ValueError as exc:
                raise ValueError("host_keys must contain base64 key data") from exc
        return value

    @field_validator("errors")
    @classmethod
    def bounded_errors(cls, value: list[str]) -> list[str]:
        if any(len(item) > 512 for item in value):
            raise ValueError("error is too long")
        return value


class EnrollmentCompleteRequest(BaseModel):
    nonce: str = Field(min_length=16, max_length=256)
    facts: RemoteFacts
    mode: Literal["auto", "direct", "tunnel"]


class OfflineCompleteRequest(BaseModel):
    blob: str = Field(min_length=1, max_length=32_768)


class PairingResponse(BaseModel):
    id: str
    tool_id: str
    status: str
    expires_at: datetime
    facts: dict | None = None
    transport: str | None = None
    error: str | None = None
    connection_config: dict = Field(default_factory=dict)


class PairingCreatedResponse(PairingResponse):
    public_key: str
    commands: dict[str, str]
