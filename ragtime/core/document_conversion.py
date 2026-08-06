"""Shared AnyDoc conversion adapter."""

import os
import threading
from dataclasses import dataclass
from enum import Enum


class DocumentConversionFailure(str, Enum):
    UNSUPPORTED = "unsupported"
    MALFORMED = "malformed"
    ENCRYPTED = "encrypted"
    RESOURCE_LIMIT = "resource_limit"
    MISSING_PART = "missing_part"
    DEPENDENCY = "dependency"
    UNEXPECTED = "unexpected"


@dataclass(frozen=True)
class DocumentConversionResult:
    text: str
    failure: DocumentConversionFailure | None = None
    detail: str | None = None


_MAX_CONCURRENT_CONVERSIONS = max(1, min(8, os.cpu_count() or 1))
_CONVERSION_SEMAPHORE = threading.Semaphore(_MAX_CONCURRENT_CONVERSIONS)


def convert_document_bytes(content: bytes, suffix: str) -> DocumentConversionResult:
    try:
        import anydoc
    except ImportError as exc:
        return DocumentConversionResult(
            text="",
            failure=DocumentConversionFailure.DEPENDENCY,
            detail=str(exc),
        )

    try:
        detected_format = anydoc.format_from_bytes(content)
        normalized_format = detected_format or anydoc.format_from_extension(suffix)
        if not normalized_format:
            return DocumentConversionResult(
                text="",
                failure=DocumentConversionFailure.UNSUPPORTED,
                detail=f"Unsupported document format for {suffix or 'content'}",
            )

        with _CONVERSION_SEMAPHORE:
            markdown = anydoc.to_markdown_bytes(content, normalized_format)
    except ImportError as exc:
        return DocumentConversionResult(
            text="",
            failure=DocumentConversionFailure.DEPENDENCY,
            detail=str(exc),
        )
    except anydoc.UnsupportedError as exc:
        return DocumentConversionResult(text="", failure=DocumentConversionFailure.UNSUPPORTED, detail=str(exc))
    except anydoc.MalformedError as exc:
        return DocumentConversionResult(text="", failure=DocumentConversionFailure.MALFORMED, detail=str(exc))
    except anydoc.EncryptedError as exc:
        return DocumentConversionResult(text="", failure=DocumentConversionFailure.ENCRYPTED, detail=str(exc))
    except anydoc.ResourceLimitError as exc:
        return DocumentConversionResult(text="", failure=DocumentConversionFailure.RESOURCE_LIMIT, detail=str(exc))
    except anydoc.MissingPartError as exc:
        return DocumentConversionResult(text="", failure=DocumentConversionFailure.MISSING_PART, detail=str(exc))
    except Exception as exc:
        return DocumentConversionResult(text="", failure=DocumentConversionFailure.UNEXPECTED, detail=str(exc))

    if isinstance(markdown, bytes):
        text = markdown.decode("utf-8", errors="replace")
    else:
        text = str(markdown)
    return DocumentConversionResult(text=text)
