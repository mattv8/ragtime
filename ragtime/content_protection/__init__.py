"""Transport-neutral content protection contracts."""

from ragtime.content_protection.models import ContentProtectionError, ProtectionContext
from ragtime.content_protection.service import (
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
    "classification_required",
    "current_context",
    "protection_context",
    "reject_unsupported_if_required",
]
