"""
Centralized Ollama API client for model discovery and metadata.

Provides functions to:
- List all available models from an Ollama server
- Check model capabilities via /api/show (embedding, vision, tools, etc.)
- Filter for embedding-capable or vision-capable models
- Get model dimensions via /api/show

This module consolidates all Ollama API interactions to ensure consistent
behavior across the codebase.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import httpx

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Ollama capability types from /api/show response
OllamaCapability = Literal["embedding", "vision", "completion", "tools", "thinking"]


@dataclass
class OllamaModelInfo:
    """Information about an Ollama model."""

    name: str
    modified_at: Optional[str] = None
    size: Optional[int] = None
    dimensions: Optional[int] = None
    is_embedding_model: bool = False
    is_vision_model: bool = False
    family: Optional[str] = None
    capabilities: list[str] | None = None


# -----------------------------------------------------------------------------
# Capability Detection (centralized for all Ollama model types)
# -----------------------------------------------------------------------------


def extract_capabilities(details: dict) -> list[str]:
    """
    Extract capabilities list from Ollama /api/show response.

    The 'capabilities' array is the authoritative source for model capabilities.
    Possible values: ["embedding", "vision", "completion", "tools", "thinking"]

    Args:
        details: Full response from /api/show

    Returns:
        List of capability strings, empty list if none found
    """
    capabilities = details.get("capabilities")
    if isinstance(capabilities, list):
        return capabilities
    return []


def has_capability(details: dict, capability: OllamaCapability) -> bool:
    """
    Check if a model has a specific capability.

    Uses the 'capabilities' array from Ollama's /api/show response.
    This is the authoritative API-based detection method.

    Args:
        details: Full response from /api/show
        capability: The capability to check for ("embedding", "vision", etc.)

    Returns:
        True if the model has the specified capability
    """
    return capability in extract_capabilities(details)


def is_embedding_model_by_capability(details: dict) -> bool:
    """
    Check if a model is an embedding model via capabilities API.

    Args:
        details: Full response from /api/show

    Returns:
        True if the model has "embedding" capability
    """
    return has_capability(details, "embedding")


def is_vision_model_by_capability(details: dict) -> bool:
    """
    Check if a model is a vision/multimodal model via capabilities API.

    Args:
        details: Full response from /api/show

    Returns:
        True if the model has "vision" capability
    """
    return has_capability(details, "vision")


async def is_reachable(
    base_url: str, timeout: float = 10.0
) -> tuple[bool, Optional[str]]:
    """
    Check if an Ollama server is reachable.

    Args:
        base_url: Ollama server base URL (e.g., 'http://localhost:11434')
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_reachable, error_message).
        If reachable, returns (True, None).
        If not reachable, returns (False, error_description).
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                return True, None
            return (
                False,
                f"Ollama at {base_url} returned status {response.status_code}. Make sure Ollama is running.",
            )
    except httpx.ConnectError:
        return (
            False,
            f"Failed to connect to Ollama at {base_url}. Make sure Ollama is running and the URL is correct.",
        )
    except httpx.TimeoutException:
        return (
            False,
            f"Connection to Ollama at {base_url} timed out. The server may be overloaded or unreachable.",
        )
    except Exception as e:
        return False, f"Error connecting to Ollama: {e}"


async def get_model_details(
    model_name: str,
    base_url: str,
    client: Optional[httpx.AsyncClient] = None,
) -> dict:
    """
    Get detailed model info from Ollama's /api/show endpoint.

    Args:
        model_name: The model name (e.g., 'nomic-embed-text:latest')
        base_url: Ollama server base URL
        client: Optional existing httpx client to reuse

    Returns:
        Dict with model_info from Ollama, or empty dict on failure
    """
    try:
        if client:
            response = await client.post(
                f"{base_url}/api/show",
                json={"name": model_name},
                timeout=5.0,
            )
        else:
            async with httpx.AsyncClient(timeout=5.0) as new_client:
                response = await new_client.post(
                    f"{base_url}/api/show",
                    json={"name": model_name},
                )

        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        logger.debug(f"Failed to get details for {model_name}: {e}")
        return {}


def extract_embedding_dimension(model_info: dict) -> Optional[int]:
    """
    Extract embedding dimension from Ollama model_info.

    The dimension is stored as '<architecture>.embedding_length' in model_info.
    Different model architectures use different prefixes (bert, nomic-bert, etc).

    Args:
        model_info: The model_info dict from /api/show response

    Returns:
        Embedding dimension if found, None otherwise
    """
    for key, value in model_info.items():
        if key.endswith(".embedding_length") and isinstance(value, int):
            return value
    return None


def extract_model_family(details: dict) -> Optional[str]:
    """Extract model family from Ollama /api/show response."""
    detail_info = details.get("details", {})
    return detail_info.get("family")


def is_embedding_capable(details: dict, model_name: str = "") -> bool:
    """
    Determine if a model is designed for generating embeddings.

    This detects purpose-built embedding models (like nomic-embed-text, bge, etc.)
    NOT general LLMs that happen to have an embedding layer.

    Detection logic (in order):
    1. Capabilities API (authoritative - Ollama's /api/show 'capabilities' array)
    2. Model family is BERT-based (fallback for older Ollama versions)
    3. Model name contains 'embed' (common naming convention)

    Note: We do NOT check for embedding_length alone since all LLMs have
    internal embedding layers with this field.

    Args:
        details: Full response from /api/show
        model_name: The model name for pattern matching fallback

    Returns:
        True if the model is a purpose-built embedding model
    """
    # Primary: Use capabilities API (authoritative source)
    if is_embedding_model_by_capability(details):
        return True

    # Fallback: Check model family for embedding-specific architectures
    # (for older Ollama versions without capabilities array)
    family = extract_model_family(details)
    if family:
        family_lower = family.lower()
        # BERT-based architectures are specifically designed for embeddings
        embedding_families = ["bert", "nomic-bert", "xlm-roberta"]
        if any(ef in family_lower for ef in embedding_families):
            return True

    # Fallback: Check model name for 'embed' pattern
    name_to_check = model_name.lower() if model_name else ""
    if not name_to_check:
        # Fallback to modelfile content if available
        name_to_check = details.get("modelfile", "").lower()

    if "embed" in name_to_check:
        return True

    return False


async def list_models(
    base_url: str,
    embeddings_only: bool = False,
    include_dimensions: bool = True,
) -> list[OllamaModelInfo]:
    """
    List available models from an Ollama server.

    Args:
        base_url: Ollama server base URL (e.g., 'http://localhost:11434')
        embeddings_only: If True, only return models capable of embeddings
        include_dimensions: If True, query /api/show for each model to get dimensions

    Returns:
        List of OllamaModelInfo objects
    """
    models = []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get list of all models
            response = await client.get(f"{base_url}/api/tags")
            response.raise_for_status()
            data = response.json()

            for model in data.get("models", []):
                model_name = model.get("name", "")
                if not model_name:
                    continue

                dimensions = None
                is_embedding = False
                family = None

                # Get detailed info if needed
                if include_dimensions or embeddings_only:
                    details = await get_model_details(model_name, base_url, client)
                    model_info = details.get("model_info", {})

                    dimensions = extract_embedding_dimension(model_info)
                    family = extract_model_family(details)
                    is_embedding = is_embedding_capable(details, model_name)

                # Filter if embeddings_only is requested
                if embeddings_only and not is_embedding:
                    continue

                models.append(
                    OllamaModelInfo(
                        name=model_name,
                        modified_at=model.get("modified_at"),
                        size=model.get("size"),
                        dimensions=dimensions,
                        is_embedding_model=is_embedding,
                        family=family,
                    )
                )

    except httpx.ConnectError:
        logger.warning(f"Cannot connect to Ollama at {base_url}")
        raise
    except httpx.TimeoutException:
        logger.warning(f"Timeout connecting to Ollama at {base_url}")
        raise
    except Exception as e:
        logger.warning(f"Error listing Ollama models: {e}")
        raise

    return models


async def get_model_embedding_dimension(
    model_name: str,
    base_url: str,
) -> Optional[int]:
    """
    Get the embedding dimension for a specific Ollama model.

    This is the preferred way to get dimensions - queries Ollama directly
    for the authoritative value.

    Args:
        model_name: The model name (e.g., 'nomic-embed-text:latest')
        base_url: Ollama server base URL

    Returns:
        Embedding dimension if available, None otherwise
    """
    details = await get_model_details(model_name, base_url)
    model_info = details.get("model_info", {})
    return extract_embedding_dimension(model_info)


async def validate_ollama_connection(base_url: str) -> tuple[bool, str]:
    """
    Validate that an Ollama server is reachable.

    Args:
        base_url: Ollama server base URL

    Returns:
        Tuple of (success, message)
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                count = len(data.get("models", []))
                return True, f"Connected. Found {count} model(s)."
            return False, f"Server returned status {response.status_code}"
    except httpx.ConnectError:
        return False, f"Cannot connect to Ollama at {base_url}. Is Ollama running?"
    except httpx.TimeoutException:
        return False, f"Connection to {base_url} timed out."
    except Exception as e:
        return False, f"Connection failed: {str(e)}"
