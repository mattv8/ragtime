"""
Centralized Ollama API client for model discovery, metadata, and warmup.

Provides functions to:
- List all available models from an Ollama server
- Check model capabilities via /api/show (embedding, vision, tools, etc.)
- Filter for embedding-capable or vision-capable models
- Get model dimensions via /api/show
- Warm up LLM and embedding models onto GPU

This module consolidates all Ollama API interactions to ensure consistent
behavior across the codebase.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import httpx

from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default Ollama runtime parameters — single source of truth
# ---------------------------------------------------------------------------
# num_gpu=-1: offload ALL model layers to GPU (if available).
# Without this, Ollama may default to CPU-only on some systems.
NUM_GPU: int = -1

# keep_alive=900: keep the model loaded in memory for 15 minutes after last
# use.  This avoids expensive reloads between closely-spaced requests while
# still allowing GPU memory to be reclaimed during idle periods.
KEEP_ALIVE: int = 900

# Default Ollama base URL
DEFAULT_BASE_URL: str = "http://localhost:11434"

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


def supports_structured_output(details: dict) -> bool:
    """
    Check if a model supports structured output (JSON schema) via capabilities API.

    The "tools" capability indicates support for function calling and structured
    generation, which includes the `format` parameter for JSON schema output.

    Args:
        details: Full response from /api/show

    Returns:
        True if the model has "tools" capability (structured output support)
    """
    return has_capability(details, "tools")


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


def extract_context_length(model_info: dict) -> Optional[int]:
    """
    Extract context length from Ollama model_info.

    The length is stored as '<architecture>.context_length' in model_info.
    Works for all architectures (llama, qwen35, gemma, phi, etc.).

    Args:
        model_info: The model_info dict from /api/show response

    Returns:
        Context length if found, None otherwise
    """
    for key, value in model_info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


def extract_num_ctx_override(details: dict) -> Optional[int]:
    """
    Extract num_ctx override from the parameters field of /api/show.

    Users can set num_ctx in their Modelfile/model config to override
    the training context length (e.g. to save VRAM).

    Args:
        details: Full response from /api/show

    Returns:
        num_ctx value if explicitly set, None otherwise
    """
    params = details.get("parameters", "")
    if not params:
        return None
    for line in params.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].strip() == "num_ctx":
            try:
                return int(parts[-1])
            except (ValueError, TypeError):
                pass
    return None


def extract_effective_context_length(details: dict) -> Optional[int]:
    """
    Extract the effective context length from a full /api/show response.

    Prefers a user-set num_ctx override (from the parameters field) over
    the training context length from model_info.  This respects users who
    intentionally limit context to save VRAM.

    Args:
        details: Full response from /api/show

    Returns:
        Effective context length, or None if not determinable
    """
    num_ctx = extract_num_ctx_override(details)
    if num_ctx is not None:
        return num_ctx
    model_info = details.get("model_info", {})
    return extract_context_length(model_info)


async def get_model_context_length(
    model_name: str,
    base_url: str,
) -> Optional[int]:
    """
    Get the effective context length for an Ollama model.

    Queries /api/show and returns the effective context length, preferring
    a user-set num_ctx override over the training context length.

    Args:
        model_name: The Ollama model name
        base_url: Ollama server base URL

    Returns:
        Context length if found, None otherwise
    """
    details = await get_model_details(model_name, base_url)
    if not details:
        return None

    return extract_effective_context_length(details)


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


# ---------------------------------------------------------------------------
# Model Warmup — preload models onto GPU so first request is fast
# ---------------------------------------------------------------------------


async def _warn_if_cpu_loaded(
    client: httpx.AsyncClient, model: str, base_url: str
) -> Optional[bool]:
    """Return GPU residency status and warn when model is still CPU-resident.

    Returns:
        True when model has non-zero VRAM usage.
        False when model is loaded but reports ``size_vram == 0``.
        None when status could not be determined.
    """
    try:
        ps_resp = await client.get(f"{base_url.rstrip('/')}/api/ps")
        if ps_resp.status_code != 200:
            return None
        data = ps_resp.json()
        for loaded_model in data.get("models", []):
            if loaded_model.get("name") != model:
                continue
            size_vram = loaded_model.get("size_vram")
            if isinstance(size_vram, int) and size_vram == 0:
                logger.warning(
                    "Ollama model '%s' is loaded but reports size_vram=0. "
                    "This indicates CPU residency; verify GPU availability on the Ollama host.",
                    model,
                )
                return False
            if isinstance(size_vram, int) and size_vram > 0:
                return True
            return None
    except Exception:
        # Non-fatal diagnostics only.
        return None
    return None


async def _unload_model(client: httpx.AsyncClient, model: str, base_url: str) -> None:
    """Unload a model first so next warmup can apply new runtime options."""
    try:
        await client.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={"model": model, "messages": [], "keep_alive": 0, "stream": False},
        )
    except Exception:
        # Best-effort; warmup continues even if unload fails.
        return


async def warmup_model(model: str, base_url: str) -> bool:
    """Preload an Ollama LLM onto GPU memory via /api/generate.

    Sends an empty-prompt generate request with ``num_gpu=-1`` and
    ``keep_alive=-1`` so the model is loaded onto GPU and kept resident.

    Args:
        model: Ollama model name (e.g. ``qwen3.5:latest``).
        base_url: Ollama server base URL.

    Returns:
        ``True`` if the model was loaded successfully, ``False`` otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # If model was previously pinned on CPU, unload before warmup so
            # runtime options (num_gpu) are applied on fresh load.
            await _unload_model(client, model, base_url)
            resp = await client.post(
                f"{base_url.rstrip('/')}/api/generate",
                json={
                    "model": model,
                    "prompt": "",
                    "keep_alive": KEEP_ALIVE,
                    "stream": False,
                    "options": {"num_gpu": NUM_GPU},
                },
            )
            if resp.status_code == 200:
                gpu_resident = await _warn_if_cpu_loaded(client, model, base_url)
                if gpu_resident is True:
                    logger.info(f"Ollama model '{model}' is loaded and ready (GPU)")
                else:
                    logger.info(f"Ollama model '{model}' is loaded and ready")
                return True
            logger.warning(
                f"Ollama warmup for '{model}' returned status {resp.status_code}"
            )
    except Exception as e:
        logger.warning(f"Could not warm up Ollama model '{model}': {e}")
    return False


async def warmup_embedding_model(model: str, base_url: str) -> bool:
    """Preload an Ollama embedding model onto GPU memory via /api/embed.

    Embedding-only models (e.g. ``nomic-embed-text``) do not support
    ``/api/generate``, so we use ``/api/embed`` with a minimal input.

    Args:
        model: Ollama embedding model name.
        base_url: Ollama server base URL.

    Returns:
        ``True`` if the model was loaded successfully, ``False`` otherwise.
    """
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # Ensure fresh reload so GPU options are not ignored due to an
            # already-resident CPU instance.
            await _unload_model(client, model, base_url)
            resp = await client.post(
                f"{base_url.rstrip('/')}/api/embed",
                json={
                    "model": model,
                    "input": "warmup",
                    "keep_alive": KEEP_ALIVE,
                    "options": {"num_gpu": NUM_GPU},
                },
            )
            if resp.status_code == 200:
                gpu_resident = await _warn_if_cpu_loaded(client, model, base_url)
                if gpu_resident is True:
                    logger.info(
                        f"Ollama embedding model '{model}' is loaded and ready (GPU)"
                    )
                else:
                    logger.info(f"Ollama embedding model '{model}' is loaded and ready")
                return True
            logger.warning(
                f"Ollama embedding warmup for '{model}' returned status {resp.status_code}"
            )
    except Exception as e:
        logger.warning(f"Could not warm up Ollama embedding model '{model}': {e}")
    return False
