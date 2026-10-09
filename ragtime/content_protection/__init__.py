"""Transport-neutral content protection contracts."""

from ragtime.content_protection.models import ContentProtectionError, ProtectionContext
from ragtime.content_protection.service import (
    access_guidance,
    authorize_content,
    classification_required,
    current_context,
    protection_context,
    reject_unsupported_if_required,
)

__all__ = [
    "ContentProtectionError",
    "ProtectionContext",
    "authorize_content",
    "access_guidance",
    "classification_required",
    "current_context",
    "protection_context",
    "reject_unsupported_if_required",
]
