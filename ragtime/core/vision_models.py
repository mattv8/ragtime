"""
Vision model detection and OCR functionality for Ollama.

Provides functions to:
- Detect vision-capable models from an Ollama server via capabilities API
- Perform OCR using Ollama vision models
- Manage vision model metadata

This module uses centralized capability detection from ollama.py.
Detection uses Ollama's /api/show 'capabilities' array which is the authoritative
source for model capabilities (e.g., ["completion", "vision", "tools"]).
"""

import base64
from dataclasses import dataclass
from typing import Optional

import httpx

from ragtime.core.logging import get_logger
from ragtime.core.ollama import extract_capabilities, is_vision_model_by_capability

logger = get_logger(__name__)


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
    from ragtime.core.ollama import get_model_details

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


async def extract_text_with_vision(
    image_content: bytes,
    base_url: str,
    model: str,
    timeout: float = 60.0,
    prompt: Optional[str] = None,
) -> str:
    """
    Extract text from an image using an Ollama vision model.

    This performs semantic OCR - the model understands the image content
    and extracts text while preserving meaning and structure.

    Args:
        image_content: Raw image bytes (PNG, JPEG, etc.)
        base_url: Ollama server base URL
        model: Vision model name (e.g., 'llava', 'granite3.2-vision:2b')
        timeout: Request timeout in seconds (vision models can be slow)
        prompt: Optional custom prompt (defaults to text extraction prompt)

    Returns:
        Extracted text from the image
    """
    if prompt is None:
        prompt = """Extract all text from this image with semantic structure annotations.

For each distinct content block, output the text with a type label prefix. Use these labels:
- [Title] for titles, headings, section headers
- [NarrativeText] for paragraphs, prose, body text
- [ListItem] for bullet points, numbered items, list entries
- [Table] for tabular data (preserve structure with | separators)
- [FigureCaption] for image captions, figure labels
- [Header] for page headers, document headers
- [Footer] for page footers, page numbers
- [Code] for code snippets, technical commands (preserve formatting)
- [Formula] for mathematical equations, formulas
- [Handwritten] for handwritten text
- [Annotation] for marginalia, notes, comments
- [Uncategorized] for text that doesn't fit other categories

Rules:
1. Preserve original text exactly - no paraphrasing
2. Maintain reading order (top-to-bottom, left-to-right)
3. Keep tables structured with | column separators
4. Each content block on its own line(s) with its label prefix
5. For multi-line blocks, only put label on first line
6. No explanations or commentary - only extracted content with labels

Example output:
[Title] Quarterly Report Q3 2024
[NarrativeText] The third quarter showed significant growth across all sectors.
[ListItem] Revenue increased by 15%
[ListItem] Customer base expanded to 50,000
[Table] Product | Sales | Growth
[Table] Widget A | $1.2M | +20%
[Table] Widget B | $800K | +12%"""

    # Encode image as base64
    image_b64 = base64.b64encode(image_content).decode("utf-8")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for factual extraction
                        "num_predict": 4096,  # Allow longer responses for text-heavy images
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

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
    from ragtime.core.ollama import get_model_details

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
