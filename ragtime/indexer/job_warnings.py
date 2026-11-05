"""Bounded, safe summaries of source-level indexing warnings."""

from __future__ import annotations

from hashlib import sha256
from pathlib import PurePosixPath
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, Field, model_validator

WarningCategory = Literal["timeout", "output_exhausted", "incomplete_response", "invalid_output"]
MAX_WARNING_SAMPLES = 100
MAX_WARNING_SOURCE_LENGTH = 512


def warning_source(source: str) -> str:
    """Encode unsafe filename characters without exposing an absolute location."""
    if PurePosixPath(source).is_absolute() or ".." in PurePosixPath(source).parts:
        raise ValueError("Warning source must be relative")
    encoded = quote(source, safe="/-._~")
    if len(encoded) > MAX_WARNING_SOURCE_LENGTH:
        suffix = "~" + sha256(encoded.encode()).hexdigest()[:16]
        return encoded[: MAX_WARNING_SOURCE_LENGTH - len(suffix)] + suffix
    return encoded


class IndexingWarning(BaseModel):
    """A safe category for one omitted relative source."""

    source: str = Field(min_length=1, max_length=MAX_WARNING_SOURCE_LENGTH, strict=True)
    category: WarningCategory

    @model_validator(mode="after")
    def validate_relative_source(self) -> "IndexingWarning":
        parts = self.source.split("/")
        if (
            self.source != self.source.strip()
            or self.source.startswith("/")
            or "\\" in self.source
            or ":" in self.source
            or ".." in parts
            or any(ord(character) < 32 for character in self.source)
        ):
            raise ValueError("source must be a relative path")
        return self


class IndexingWarnings(BaseModel):
    """Bounded warning samples with a total for all omitted sources."""

    count: int = Field(default=0, ge=0)
    items: list[IndexingWarning] = Field(default_factory=list, max_length=MAX_WARNING_SAMPLES)

    @model_validator(mode="after")
    def validate_count(self) -> "IndexingWarnings":
        if self.count < len(self.items):
            raise ValueError("count cannot be less than the retained warning sample count")
        return self

    def record(self, source: str, category: WarningCategory) -> None:
        """Record one source warning, updating an existing retained sample."""
        warning = IndexingWarning(source=source, category=category)
        for index, item in enumerate(self.items):
            if item.source == warning.source:
                self.items[index] = warning
                return
        self.count += 1
        if len(self.items) < MAX_WARNING_SAMPLES:
            self.items.append(warning)


def warnings_from_persisted(value: Any) -> IndexingWarnings:
    """Hydrate stored warnings, accepting only the legacy null or missing shape."""
    if value is None:
        return IndexingWarnings()
    data = getattr(value, "data", value)
    return IndexingWarnings.model_validate(data)
