"""
Document Content Parser

Extracts text content from various document formats:
- AnyDoc-supported documents (PDF, Office, OpenDocument, RTF, EPUB, CSV)
- Email (.eml, .msg)
- HTML (.html, .htm) - with tag stripping
- Images with OCR (.png, .jpg, .jpeg, .tiff, .bmp, .gif, .webp)
- Plain text (.txt, .md, .rst, .json, .xml)
- Code files (.py, .js, .ts, etc.)

OCR Modes:
- disabled: Skip image files
- tesseract: Traditional OCR (fast, basic text extraction)
- vision: Semantic OCR with multimodal vision models (slower, better understanding)
"""

import asyncio
import email
import io
import subprocess
import time
from email.policy import default
from pathlib import Path
from typing import Any, Literal, Optional

from ragtime.core.document_conversion import convert_document_bytes
from ragtime.core.file_constants import ANYDOC_DOCUMENT_EXTENSIONS, DOCUMENT_EXTENSIONS, OCR_EXTENSIONS, RAW_CAMERA_EXTENSIONS
from ragtime.core.logging import get_logger
from ragtime.core.vision_models import (
    VisionOcrResult,
    extract_text_with_vision,
    extract_text_with_vision_structured,
)

logger = get_logger(__name__)

# Type alias for OCR mode
OcrModeType = Literal["disabled", "tesseract", "vision"]


def extract_text_from_file(
    file_path: Path,
    content: Optional[bytes] = None,
    enable_ocr: bool = False,
    ocr_mode: OcrModeType = "disabled",
    ocr_provider: Optional[str] = None,
    ocr_vision_model: Optional[str] = None,
    vision_base_url: Optional[str] = None,
    vision_api_key: Optional[str] = None,
) -> str:
    """
    Extract text content from a file based on its extension.

    For async OCR with vision models, use extract_text_from_file_async instead.

    Args:
        file_path: Path to the file
        content: Optional pre-loaded file content (bytes)
        enable_ocr: Legacy flag - if True and ocr_mode is 'disabled', uses tesseract
        ocr_mode: OCR mode ('disabled', 'tesseract', or 'vision')
        ocr_provider: Provider for semantic vision OCR
        ocr_vision_model: Vision model for OCR
        vision_base_url: Provider base URL for semantic vision OCR
        vision_api_key: Optional provider API key

    Returns:
        Extracted text content as string
    """
    # Handle legacy enable_ocr flag
    effective_ocr_mode = ocr_mode
    if enable_ocr and ocr_mode == "disabled":
        effective_ocr_mode = "tesseract"

    suffix = file_path.suffix.lower()

    # Load content if not provided
    if content is None:
        try:
            content = file_path.read_bytes()
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            return ""

    # Route to appropriate parser
    try:
        if suffix in ANYDOC_DOCUMENT_EXTENSIONS:
            return _extract_anydoc(content, suffix, file_path)
        elif suffix == ".eml":
            return _extract_eml(content)
        elif suffix == ".msg":
            return _extract_msg(content)
        elif suffix in {".html", ".htm"}:
            return _extract_html(content)
        elif suffix in OCR_EXTENSIONS and effective_ocr_mode == "tesseract":
            return _extract_image_ocr(content)
        elif suffix in OCR_EXTENSIONS and effective_ocr_mode == "vision":
            # Vision OCR requires async - log warning and fall back to tesseract
            logger.warning(f"Vision OCR requires async. Use extract_text_from_file_async() for {file_path.name}. Falling back to tesseract.")
            return _extract_image_ocr(content)
        elif suffix in OCR_EXTENSIONS:
            # OCR disabled, skip image files
            logger.debug(f"Skipping image {file_path.name} - OCR disabled")
            return ""
        else:
            # Plain text files
            return _extract_text(content)
    except Exception as e:
        logger.warning(f"Failed to extract text from {file_path}: {e}")
        return ""


async def extract_image_structured_async(
    file_path: Path,
    content: Optional[bytes] = None,
    ocr_provider: Optional[str] = None,
    ocr_vision_model: Optional[str] = None,
    vision_base_url: Optional[str] = None,
    vision_api_key: Optional[str] = None,
):
    """
    Extract structured OCR data from an image for semantic chunking.

    Returns VisionOcrResult with semantic segments that can be chunked
    intelligently (keeping classification metadata together, etc.).

    Args:
        file_path: Path to the image file
        content: Optional pre-loaded file content
        ocr_provider: Provider for semantic vision OCR
        ocr_vision_model: Vision model name
        vision_base_url: Provider base URL
        vision_api_key: Optional provider API key

    Returns:
        VisionOcrResult with structured semantic data, or None if extraction fails
    """
    suffix = file_path.suffix.lower()

    if suffix not in OCR_EXTENSIONS:
        return None

    effective_provider = ocr_provider or "ollama"
    effective_base_url = vision_base_url

    if not ocr_vision_model or not effective_base_url:
        logger.warning("Structured OCR requires vision model and base URL")
        return None

    # Load content if not provided
    if content is None:
        try:
            content = file_path.read_bytes()
        except Exception as e:
            logger.warning(f"Failed to read image {file_path}: {e}")
            return None

    try:
        return await extract_text_with_vision_structured(
            image_content=content,
            base_url=effective_base_url,
            model=ocr_vision_model,
            provider=effective_provider,
            api_key=vision_api_key,
            source_format=suffix,
            include_classification=True,
        )
    except Exception as e:
        logger.warning(f"Structured OCR failed for {file_path}: {e}")
        return None


async def extract_text_from_file_async(
    file_path: Path,
    content: Optional[bytes] = None,
    enable_ocr: bool = False,
    ocr_mode: OcrModeType = "disabled",
    ocr_provider: Optional[str] = None,
    ocr_vision_model: Optional[str] = None,
    vision_base_url: Optional[str] = None,
    vision_api_key: Optional[str] = None,
) -> str:
    """
    Async version of extract_text_from_file with vision OCR support.

    Document parsing is run in a thread pool to avoid blocking the event loop,
    keeping the server responsive during indexing.

    Args:
        file_path: Path to the file
        content: Optional pre-loaded file content (bytes)
        enable_ocr: Legacy flag - if True and ocr_mode is 'disabled', uses tesseract
        ocr_mode: OCR mode ('disabled', 'tesseract', or 'vision')
        ocr_provider: Provider for semantic vision OCR
        ocr_vision_model: Vision model for OCR
        vision_base_url: Provider base URL for semantic vision OCR
        vision_api_key: Optional provider API key

    Returns:
        Extracted text content as string
    """
    # Handle legacy enable_ocr flag
    effective_ocr_mode = ocr_mode
    if enable_ocr and ocr_mode == "disabled":
        effective_ocr_mode = "tesseract"

    suffix = file_path.suffix.lower()

    # Load content if not provided - run file I/O in thread
    if content is None:
        try:
            content = await asyncio.to_thread(file_path.read_bytes)
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            return ""

        # Re-use the file-level binary detector to skip files that look
        # binary but escaped collection-time filtering (e.g. CAD formats
        # whose first 8KB is a text translation table). Without this guard,
        # the plain-text fallback below would hand binary garbage to
        # chonkie's tree-sitter language detector, which then spins forever.
        # DRY: this is the same bytes check used by has_binary_content in
        # ragtime.indexer.file_utils.
        if not suffix or (suffix not in DOCUMENT_EXTENSIONS and suffix not in OCR_EXTENSIONS):
            from ragtime.indexer.file_utils import _looks_binary

            if _looks_binary(content):
                logger.debug(f"Skipping binary file {file_path.name} (looks like non-text content)")
                return ""

    # Route to appropriate parser
    try:
        # Handle OCR cases for images
        if suffix in OCR_EXTENSIONS:
            if effective_ocr_mode == "vision":
                effective_provider = ocr_provider or "ollama"
                effective_base_url = vision_base_url
                if not ocr_vision_model or not effective_base_url:
                    logger.warning(f"Vision OCR requires model and base_url. Falling back to tesseract for {file_path.name}")
                    return await asyncio.to_thread(_extract_image_ocr, content)
                return await _extract_image_vision_ocr(
                    content,
                    effective_base_url,
                    ocr_vision_model,
                    provider=effective_provider,
                    api_key=vision_api_key,
                    source_format=suffix,
                )
            elif effective_ocr_mode == "tesseract":
                return await asyncio.to_thread(_extract_image_ocr, content)
            else:
                logger.debug(f"Skipping image {file_path.name} - OCR disabled")
                return ""

        # All other file types run in thread pool to avoid blocking event loop
        if suffix in ANYDOC_DOCUMENT_EXTENSIONS:
            return await asyncio.to_thread(_extract_anydoc, content, suffix, file_path)
        elif suffix == ".eml":
            return await asyncio.to_thread(_extract_eml, content)
        elif suffix == ".msg":
            return await asyncio.to_thread(_extract_msg, content)
        elif suffix in {".html", ".htm"}:
            return await asyncio.to_thread(_extract_html, content)
        else:
            # Plain text - offload to thread for large files to avoid
            # blocking the event loop during decode + whitespace cleanup.
            return await asyncio.to_thread(_extract_text, content)
    except Exception as e:
        logger.warning(f"Failed to extract text from {file_path}: {e}")
        return ""


def _extract_anydoc(content: bytes, suffix: str, file_path: Path) -> str:
    """Extract text from AnyDoc-supported formats."""
    result = convert_document_bytes(content, suffix)
    if result.failure is not None:
        detail = f": {result.detail}" if result.detail else ""
        logger.warning(f"AnyDoc extraction failed for {file_path.name} ({result.failure.value}){detail}")
        return ""
    return result.text


def _extract_text(content: bytes) -> str:
    """Extract text from plain text file."""
    # Try common encodings
    for encoding in ["utf-8", "latin-1", "cp1252", "ascii"]:
        try:
            return content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue

    # Last resort: decode with errors replaced
    return content.decode("utf-8", errors="replace")


def _extract_eml(content: bytes) -> str:
    """Extract text from email EML file."""
    try:
        msg = email.message_from_bytes(content, policy=default)
        text_parts = []

        # Add headers
        if msg.get("Subject"):
            text_parts.append(f"Subject: {msg.get('Subject')}")
        if msg.get("From"):
            text_parts.append(f"From: {msg.get('From')}")
        if msg.get("To"):
            text_parts.append(f"To: {msg.get('To')}")
        if msg.get("Date"):
            text_parts.append(f"Date: {msg.get('Date')}")

        text_parts.append("")  # Blank line after headers

        # Extract body
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        charset = part.get_content_charset() or "utf-8"
                        text_parts.append(payload.decode(charset, errors="replace"))
                elif content_type == "text/html":
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        # Strip HTML tags for plain text
                        try:
                            from bs4 import BeautifulSoup

                            charset = part.get_content_charset() or "utf-8"
                            soup = BeautifulSoup(payload.decode(charset, errors="replace"), "html.parser")
                            text_parts.append(soup.get_text(separator="\n", strip=True))
                        except ImportError:
                            # Fallback: just decode and append
                            charset = part.get_content_charset() or "utf-8"
                            text_parts.append(payload.decode(charset, errors="replace"))
        else:
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                charset = msg.get_content_charset() or "utf-8"
                text_parts.append(payload.decode(charset, errors="replace"))

        return "\n".join(text_parts)
    except Exception as e:
        logger.warning(f"EML extraction error: {e}")
        return ""


def _extract_msg(content: bytes) -> str:
    """Extract text from Outlook MSG file."""
    try:
        import extract_msg
    except ImportError:
        logger.warning("extract-msg not installed, cannot extract MSG content")
        return ""

    try:
        msg: Any = extract_msg.openMsg(io.BytesIO(content))
        text_parts = []

        # Add headers
        if msg.subject:
            text_parts.append(f"Subject: {msg.subject}")
        if msg.sender:
            text_parts.append(f"From: {msg.sender}")
        if msg.to:
            text_parts.append(f"To: {msg.to}")
        if msg.date:
            text_parts.append(f"Date: {msg.date}")

        text_parts.append("")  # Blank line after headers

        # Add body
        if msg.body:
            text_parts.append(msg.body)

        msg.close()
        return "\n".join(text_parts)
    except Exception as e:
        logger.warning(f"MSG extraction error: {e}")
        return ""


def _extract_html(content: bytes) -> str:
    """Extract text from HTML file, stripping tags."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Fallback to plain text extraction
        return _extract_text(content)

    try:
        # Decode content
        text = _extract_text(content)

        # Parse and extract text
        soup = BeautifulSoup(text, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()

        # Get text with newlines preserved
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        logger.warning(f"HTML extraction error: {e}")
        return _extract_text(content)


def _extract_image_ocr(content: bytes) -> str:
    """Extract text from image using OCR (Tesseract)."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.warning("pytesseract/Pillow not installed, cannot perform OCR. Install with: pip install pytesseract Pillow")
        return ""

    try:
        import time

        start_time = time.time()

        # Open image from bytes
        image: Image.Image = Image.open(io.BytesIO(content))
        original_size = image.size

        # Aggressively resize for fast OCR - 1200px is plenty for text extraction
        max_dimension = 1200
        if max(image.size) > max_dimension:
            ratio = max_dimension / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug(f"OCR: Resized image from {original_size} to {image.size}")
        else:
            logger.debug(f"OCR: Image size {image.size} within limit, no resize needed")

        # Perform OCR with 10s timeout
        import subprocess

        text = pytesseract.image_to_string(image, timeout=10)
        elapsed = time.time() - start_time
        logger.debug(f"OCR completed in {elapsed:.2f}s, extracted {len(text)} chars")

        return text.strip()
    except subprocess.TimeoutExpired:
        logger.warning("OCR timeout (10s) - skipping image")
        return ""
    except Exception as e:
        # Check if Tesseract is not installed
        if "tesseract is not installed" in str(e).lower():
            logger.warning("Tesseract OCR not installed on system. Install with: apt-get install tesseract-ocr")
        else:
            logger.warning(f"OCR extraction error: {e}")
        return ""


async def _extract_image_vision_ocr(
    content: bytes,
    vision_base_url: str,
    vision_model: str,
    provider: str = "ollama",
    api_key: str | None = None,
    source_format: str | None = None,
    timeout: float = 60.0,
) -> str:
    """
    Extract text from image using a vision model (semantic OCR).

    This performs semantic OCR - the model understands the image content
    and extracts text while preserving meaning and structure. Images are
    automatically resized and optimized. If no text is found, falls back
    to image classification for searchable descriptions.

    Concurrency is managed at the vision_models layer via centralized semaphore.

    Args:
        content: Raw image bytes
        vision_base_url: Provider base URL
        vision_model: Vision model name
        provider: OCR provider name
        api_key: Optional provider API key
        source_format: Source file extension (e.g., '.cr2' for raw formats)
        timeout: Request timeout in seconds

    Returns:
        Extracted text or image classification
    """
    start_time = time.time()

    try:
        text = await extract_text_with_vision(
            image_content=content,
            base_url=vision_base_url,
            model=vision_model,
            provider=provider,
            api_key=api_key,
            timeout=timeout,
            preprocess=True,  # Enable image resizing/optimization
            source_format=source_format,  # Pass format for raw handling
            include_classification=True,  # Include image classification for search
        )
        elapsed = time.time() - start_time
        logger.debug(f"Vision OCR ({provider}/{vision_model}) completed in {elapsed:.2f}s, extracted {len(text)} chars")
        return text
    except Exception as e:
        logger.warning(f"Vision OCR error with {vision_model}: {e}")
        # Fall back to tesseract for standard formats, skip for raw
        if source_format and source_format.lower() in RAW_CAMERA_EXTENSIONS:
            logger.info(f"Cannot fall back to Tesseract for raw format {source_format}")
            return ""
        logger.info("Falling back to Tesseract OCR")
        return _extract_image_ocr(content)


def is_supported_document(file_path: Path) -> bool:
    """Check if a file type is supported for text extraction."""
    return file_path.suffix.lower() in DOCUMENT_EXTENSIONS


def is_ocr_supported(file_path: Path) -> bool:
    """Check if a file type supports OCR extraction."""
    return file_path.suffix.lower() in OCR_EXTENSIONS
