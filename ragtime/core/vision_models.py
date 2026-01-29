"""
Vision model detection and OCR functionality for Ollama.

Provides functions to:
- Detect vision-capable models from an Ollama server via capabilities API
- Perform OCR using Ollama vision models
- Image preprocessing (resize, format conversion) for efficient processing
- Image classification for non-text images
- Manage vision model metadata

This module uses centralized capability detection from ollama.py.
Detection uses Ollama's /api/show 'capabilities' array which is the authoritative
source for model capabilities (e.g., ["completion", "vision", "tools"]).
"""

import base64
import io
import json
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from PIL import Image

from ragtime.core.file_constants import RAW_CAMERA_EXTENSIONS
from ragtime.core.logging import get_logger
from ragtime.core.ollama import (
    extract_capabilities,
    get_model_details,
    is_vision_model_by_capability,
    supports_structured_output,
)

logger = get_logger(__name__)

# Default max dimension for image resizing (pixels on longest edge)
DEFAULT_MAX_IMAGE_DIMENSION = 2048
# JPEG quality for resized images (balance between size and quality)
DEFAULT_JPEG_QUALITY = 85


# =============================================================================
# PROMPTS
# =============================================================================

# Prompt for classification-only (used by classify_image)
CLASSIFICATION_PROMPT = """Analyze this image and provide a structured description for search indexing. Return valid JSON.

Include these fields:
- image_type: photo, screenshot, diagram, chart, illustration, logo, etc.
- subject: main subject or focus
- description: brief factual description
- objects: list of identifiable objects, people, animals
- colors: list of dominant colors
- setting: location, environment, or context
- tags: list of search keywords

Be factual and objective. Include specific details useful for search."""

# JSON schema for structured classification output
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "image_type": {"type": "string"},
        "subject": {"type": "string"},
        "description": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "colors": {"type": "array", "items": {"type": "string"}},
        "setting": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["image_type", "subject", "description", "objects", "tags"],
}

# Prompt for OCR with classification (structured JSON output)
OCR_WITH_CLASSIFICATION_PROMPT = """Analyze this image. Extract all visible text AND describe the image for search indexing. Return valid JSON.

For extracted_text: list each text block with its type (Title, NarrativeText, ListItem, Table, Code, Handwritten, Uncategorized) and the exact text content.

For image analysis: describe the image type, subject, description, objects, setting, and search tags.

If there is no text in the image, return an empty extracted_text array."""

# JSON schema for OCR with classification
OCR_WITH_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "extracted_text": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "Title",
                            "NarrativeText",
                            "ListItem",
                            "Table",
                            "Code",
                            "Handwritten",
                            "Uncategorized",
                        ],
                    },
                    "content": {"type": "string"},
                },
                "required": ["type", "content"],
            },
        },
        "image_type": {"type": "string"},
        "subject": {"type": "string"},
        "description": {"type": "string"},
        "objects": {"type": "array", "items": {"type": "string"}},
        "setting": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["extracted_text", "image_type", "subject", "description", "tags"],
}

# Prompt for OCR only (no classification)
OCR_ONLY_PROMPT = """Extract all visible text from this image with semantic structure. Return valid JSON.

For each text block, identify its type:
- Title: titles, headings, section headers
- NarrativeText: paragraphs, prose, body text
- ListItem: bullet points, numbered items
- Table: tabular data
- Code: code snippets, commands
- Handwritten: handwritten text
- Uncategorized: other text

Preserve text exactly. Maintain reading order.
If there is no text, return an empty array."""

# JSON schema for OCR only
OCR_ONLY_SCHEMA = {
    "type": "object",
    "properties": {
        "extracted_text": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "Title",
                            "NarrativeText",
                            "ListItem",
                            "Table",
                            "Code",
                            "Header",
                            "Footer",
                            "FigureCaption",
                            "Formula",
                            "Handwritten",
                            "Annotation",
                            "Uncategorized",
                        ],
                    },
                    "content": {"type": "string"},
                },
                "required": ["type", "content"],
            },
        },
    },
    "required": ["extracted_text"],
}


@dataclass
class VisionModelInfo:
    """Information about an Ollama vision model."""

    name: str
    modified_at: Optional[str] = None
    size: Optional[int] = None
    family: Optional[str] = None
    parameter_size: Optional[str] = None
    capabilities: list[str] | None = None


def is_vision_capable(details: dict, model_name: str = "") -> bool:
    """
    Determine if a model supports vision/multimodal inputs.

    Uses centralized capability detection from ollama.py.
    Checks 'capabilities' array from /api/show response for "vision".

    Args:
        details: Full response from /api/show
        model_name: The model name (unused, kept for API compatibility)

    Returns:
        True if the model supports vision inputs
    """
    return is_vision_model_by_capability(details)


async def list_vision_models(
    base_url: str,
    timeout: float = 10.0,
) -> list[VisionModelInfo]:
    """
    List vision-capable models from an Ollama server.

    Queries Ollama's /api/show endpoint for each model and checks the
    'capabilities' array for "vision" capability.

    Args:
        base_url: Ollama server base URL (e.g., 'http://localhost:11434')
        timeout: Request timeout in seconds

    Returns:
        List of VisionModelInfo objects for models that support vision
    """
    models = []

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Get list of all models
            response = await client.get(f"{base_url}/api/tags")
            response.raise_for_status()
            data = response.json()

            for model in data.get("models", []):
                model_name = model.get("name", "")
                if not model_name:
                    continue

                # Get detailed info to check capabilities
                details = await get_model_details(model_name, base_url, client)

                if is_vision_capable(details, model_name):
                    detail_info = details.get("details", {})
                    capabilities = extract_capabilities(details)
                    models.append(
                        VisionModelInfo(
                            name=model_name,
                            modified_at=model.get("modified_at"),
                            size=model.get("size"),
                            family=detail_info.get("family"),
                            parameter_size=detail_info.get("parameter_size"),
                            capabilities=capabilities if capabilities else None,
                        )
                    )

    except httpx.ConnectError:
        logger.warning(f"Cannot connect to Ollama at {base_url}")
        raise
    except httpx.TimeoutException:
        logger.warning(f"Timeout connecting to Ollama at {base_url}")
        raise
    except Exception as e:
        logger.warning(f"Error listing Ollama vision models: {e}")
        raise

    return models


def _load_image(image_content: bytes, source_format: str | None) -> tuple[Any, str]:
    """
    Helper to load image from bytes, handling raw formats.

    Args:
        image_content: Raw image bytes
        source_format: Optional source format hint

    Returns:
        Tuple of (PIL.Image or None, status string)
    """
    is_raw = source_format and source_format.lower() in RAW_CAMERA_EXTENSIONS

    if not is_raw:
        return Image.open(io.BytesIO(image_content)), "standard"

    # Try to process raw format with rawpy
    try:
        import rawpy
    except ImportError:
        logger.warning(
            f"rawpy not installed, cannot process {source_format}. "
            "Install with: pip install rawpy"
        )
        return None, "unsupported_raw"

    try:
        with rawpy.imread(io.BytesIO(image_content)) as raw:
            # Use camera white balance and auto brightness
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=False,
                output_bps=8,
            )
        return Image.fromarray(rgb), "raw"
    except Exception as e:
        logger.warning(f"Failed to process raw format {source_format}: {e}")
        return None, "raw_error"


def _process_image_content(
    img: Any, max_dimension: int, quality: int
) -> tuple[bytes, str]:
    """
    Helper to resize and compress image.

    Args:
        img: PIL Image object
        max_dimension: Max pixel dimension
        quality: JPEG quality

    Returns:
        Tuple of (bytes, format)
    """
    # Convert to RGB if necessary (handles RGBA, palette, etc.)
    if img.mode in ("RGBA", "LA", "P"):
        # Create white background for transparency
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(
            img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None
        )
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Calculate resize dimensions
    width, height = img.size
    if width > max_dimension or height > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))

        # Use high-quality resampling
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

    # Convert to JPEG bytes
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue(), "jpeg"


def preprocess_image(
    image_content: bytes,
    max_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
    quality: int = DEFAULT_JPEG_QUALITY,
    source_format: str | None = None,
) -> tuple[bytes, str]:
    """
    Preprocess image for vision model processing.

    Resizes large images to reduce processing time and converts to JPEG
    for efficient transfer. Handles raw camera formats if rawpy is available.

    Args:
        image_content: Raw image bytes
        max_dimension: Maximum pixels on longest edge (default 2048)
        quality: JPEG compression quality 1-100 (default 85)
        source_format: Optional source format hint (e.g., '.cr2', '.nef')

    Returns:
        Tuple of (processed_image_bytes, format_used)
    """
    try:
        original_size = len(image_content)

        # 1. Load Image
        img, status = _load_image(image_content, source_format)

        if img is None:
            # Failed to load (raw error or unsupported)
            return image_content, status

        # 2. Process Image (Resize & Convert)
        processed_bytes, fmt = _process_image_content(img, max_dimension, quality)

        # Log reduction
        new_size = len(processed_bytes)
        if new_size < original_size:
            reduction = ((original_size - new_size) / original_size) * 100
            logger.debug(
                f"Image preprocessing: {original_size:,} -> {new_size:,} bytes "
                f"({reduction:.1f}% reduction)"
            )
        else:
            logger.debug(
                f"Image preprocessing: {original_size:,} -> {new_size:,} bytes"
            )

        return processed_bytes, fmt

    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}, using original")
        return image_content, "original"


async def classify_image(
    image_content: bytes,
    base_url: str,
    model: str,
    timeout: float = 60.0,
) -> str:
    """
    Classify and describe an image when no text is found.

    Generates a semantic description of the image content for retrieval purposes.
    Useful for photos, diagrams, and other non-text images.

    Args:
        image_content: Preprocessed image bytes
        base_url: Ollama server base URL
        model: Vision model name
        timeout: Request timeout in seconds

    Returns:
        Image classification/description text formatted for retrieval
    """
    image_b64 = base64.b64encode(image_content).decode("utf-8")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": CLASSIFICATION_PROMPT,
                    "images": [image_b64],
                    "stream": False,
                    "format": CLASSIFICATION_SCHEMA,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 1024,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            response_text = data.get("response", "").strip()

            # Parse structured JSON response
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse classification JSON, returning raw")
                return response_text

            # Format for retrieval indexing
            lines = []
            if result.get("image_type"):
                lines.append(f"[ImageType] {result['image_type']}")
            if result.get("subject"):
                lines.append(f"[Subject] {result['subject']}")
            if result.get("description"):
                lines.append(f"[Description] {result['description']}")
            if result.get("objects"):
                lines.append(f"[Objects] {', '.join(result['objects'])}")
            if result.get("colors"):
                lines.append(f"[Colors] {', '.join(result['colors'])}")
            if result.get("setting"):
                lines.append(f"[Setting] {result['setting']}")
            if result.get("tags"):
                lines.append(f"[Tags] {', '.join(result['tags'])}")

            return "\n".join(lines)

    except Exception as e:
        logger.warning(f"Image classification error with {model}: {e}")
        return ""


async def extract_text_with_vision(
    image_content: bytes,
    base_url: str,
    model: str,
    timeout: float = 60.0,
    prompt: Optional[str] = None,
    preprocess: bool = True,
    max_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
    source_format: str | None = None,
    include_classification: bool = True,
) -> str:
    """
    Extract text from an image using an Ollama vision model.

    This performs semantic OCR - the model understands the image content
    and extracts text while preserving meaning and structure. When
    include_classification is True, also appends image classification/description
    which enables semantic search for image content (e.g., finding photos of birds).

    Args:
        image_content: Raw image bytes (PNG, JPEG, raw formats, etc.)
        base_url: Ollama server base URL
        model: Vision model name (e.g., 'llava', 'granite3.2-vision:2b')
        timeout: Request timeout in seconds (vision models can be slow)
        prompt: Optional custom prompt (defaults to text extraction prompt)
        preprocess: Whether to resize/optimize the image (default True)
        max_dimension: Max pixels on longest edge when preprocessing
        source_format: Source file extension hint (e.g., '.cr2' for raw)
        include_classification: Include image classification for semantic search (default True)

    Returns:
        Extracted text plus image classification (when enabled)
    """
    # Preprocess image (resize, convert format)
    if preprocess:
        processed_content, format_used = preprocess_image(
            image_content,
            max_dimension=max_dimension,
            source_format=source_format,
        )
        if format_used == "unsupported_raw":
            return "[UnsupportedFormat] Raw camera format could not be processed"
    else:
        processed_content = image_content

    # Check if model supports structured output (tools capability)
    # If not, we'll use plain text prompts instead of JSON schema
    use_structured_output = False
    if prompt is None:
        try:
            model_details = await get_model_details(model, base_url)
            if model_details and supports_structured_output(model_details):
                use_structured_output = True
                logger.debug(f"Model {model} supports structured output")
            else:
                capabilities = (
                    extract_capabilities(model_details) if model_details else []
                )
                logger.debug(
                    f"Model {model} does not support structured output "
                    f"(capabilities: {capabilities}), using text prompts"
                )
        except Exception as e:
            logger.debug(f"Could not check model capabilities: {e}, using text prompts")

    # Select prompt and schema based on classification mode and model capabilities
    # Custom prompts bypass structured output (for backwards compatibility)
    schema: dict | None = None
    if prompt is None:
        if include_classification:
            prompt = OCR_WITH_CLASSIFICATION_PROMPT
            if use_structured_output:
                schema = OCR_WITH_CLASSIFICATION_SCHEMA
        else:
            prompt = OCR_ONLY_PROMPT
            if use_structured_output:
                schema = OCR_ONLY_SCHEMA

    image_b64 = base64.b64encode(processed_content).decode("utf-8")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            request_json = {
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 4096,
                },
            }
            if schema is not None:
                request_json["format"] = schema

            response = await client.post(
                f"{base_url}/api/generate",
                json=request_json,
            )
            response.raise_for_status()
            data = response.json()

            # Get response text - some models (like qwen3-vl with "thinking" capability)
            # put structured output in "thinking" field instead of "response"
            response_text = data.get("response", "").strip()
            thinking_text = data.get("thinking", "").strip()

            # Prefer response, but fall back to thinking if response is empty
            if not response_text and thinking_text:
                logger.debug(
                    "Using 'thinking' field for structured output (qwen-style)"
                )
                response_text = thinking_text

            # Custom prompts return raw text
            if schema is None:
                return response_text

            # Parse structured JSON response
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse OCR JSON: {e}. "
                    f"Raw response ({len(response_text)} chars): {response_text[:500]!r}"
                )
                return response_text

            # Format extracted text
            lines = []
            for item in result.get("extracted_text", []):
                text_type = item.get("type", "text")
                content = item.get("content", "")
                if content:
                    lines.append(f"[{text_type.title()}] {content}")

            # Append classification if present
            if include_classification:
                if result.get("image_type"):
                    lines.append(f"[ImageType] {result['image_type']}")
                if result.get("subject"):
                    lines.append(f"[Subject] {result['subject']}")
                if result.get("description"):
                    lines.append(f"[Description] {result['description']}")
                if result.get("objects"):
                    lines.append(f"[Objects] {', '.join(result['objects'])}")
                if result.get("colors"):
                    lines.append(f"[Colors] {', '.join(result['colors'])}")
                if result.get("setting"):
                    lines.append(f"[Setting] {result['setting']}")
                if result.get("tags"):
                    lines.append(f"[Tags] {', '.join(result['tags'])}")

            return "\n".join(lines)

    except httpx.ConnectError:
        logger.warning(f"Cannot connect to Ollama at {base_url} for vision OCR")
        raise
    except httpx.TimeoutException:
        logger.warning(f"Vision OCR timeout ({timeout}s) for model {model}")
        raise
    except Exception as e:
        logger.warning(f"Vision OCR error with {model}: {e}")
        raise


async def validate_vision_model(
    model: str,
    base_url: str,
) -> tuple[bool, str]:
    """
    Validate that a model exists and supports vision.

    Checks the 'capabilities' array from Ollama's /api/show endpoint
    for "vision" capability.

    Args:
        model: Model name to validate
        base_url: Ollama server base URL

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        details = await get_model_details(model, base_url)
        if not details:
            return False, f"Model '{model}' not found on Ollama server"

        capabilities = extract_capabilities(details)
        if is_vision_capable(details, model):
            return (
                True,
                f"Model '{model}' supports vision (capabilities: {capabilities})",
            )
        else:
            return (
                False,
                f"Model '{model}' does not support vision (capabilities: {capabilities})",
            )

    except Exception as e:
        return False, f"Error validating model: {e}"
