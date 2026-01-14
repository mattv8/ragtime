"""
Memory estimation utilities for FAISS indexes.

Provides functions to estimate RAM requirements based on:
- Number of chunks/vectors
- Embedding dimensions (from LiteLLM or tracked app settings)
- Historical data from previous loads
"""

from typing import Optional

from ragtime.core.embedding_models import EmbeddingModelInfo
from ragtime.core.logging import get_logger

logger = get_logger(__name__)

# Memory overhead factors observed in practice
# During pickle deserialization, Python temporarily holds both serialized
# and deserialized data, which can roughly double memory usage
PEAK_MEMORY_FACTOR = 1.8  # Peak is ~1.8x steady state during loading

# Docstore overhead: each chunk stores text content + metadata
# Approximate: avg_chunk_chars * 1.5 bytes (unicode + dict overhead)
DOCSTORE_OVERHEAD_FACTOR = 1.5

# Default average chunk size in characters (when not known)
DEFAULT_AVG_CHUNK_CHARS = 800

# Fallback dimensions for common Ollama models not in LiteLLM
# LiteLLM only tracks cloud provider models, so we need this for local models
OLLAMA_MODEL_DIMENSIONS = {
    # Nomic
    "nomic-embed-text": 768,
    # BGE models
    "bge-small": 384,
    "bge-base": 768,
    "bge-large": 1024,
    "bge-m3": 1024,
    # MXBai
    "mxbai-embed-large": 1024,
    # All-MiniLM
    "all-minilm": 384,
    # Snowflake
    "snowflake-arctic-embed": 1024,
    # E5 models
    "e5-small": 384,
    "e5-base": 768,
    "e5-large": 1024,
    # GTE models
    "gte-small": 384,
    "gte-base": 768,
    "gte-large": 1024,
    # Qwen models (large dimensions)
    "qwen3-embedding": 4096,
}


def get_embedding_dimension_from_litellm(
    model: str,
    embedding_models: dict[str, EmbeddingModelInfo],
) -> Optional[int]:
    """Look up embedding dimension from LiteLLM model data.

    Tries exact match first, then partial matching for versioned models.

    Args:
        model: Model name (e.g., "text-embedding-3-small", "nomic-embed-text")
        embedding_models: Dict of model_id -> EmbeddingModelInfo from get_embedding_models()

    Returns:
        Dimension if found in LiteLLM data, None otherwise
    """
    # Exact match
    if model in embedding_models:
        dim = embedding_models[model].output_vector_size
        if dim:
            return dim

    # Try lowercase match
    model_lower = model.lower()
    for model_id, info in embedding_models.items():
        if model_id.lower() == model_lower and info.output_vector_size:
            return info.output_vector_size

    # Try partial match for versioned models (e.g., "nomic-embed-text:latest")
    base_model = model.split(":")[0]
    if base_model in embedding_models:
        dim = embedding_models[base_model].output_vector_size
        if dim:
            return dim

    return None


def get_embedding_dimension(
    model: str,
    embedding_models: dict[str, EmbeddingModelInfo],
    tracked_dim: Optional[int] = None,
) -> Optional[int]:
    """Get the embedding dimension using fallback chain.

    Order of precedence:
    1. Tracked dimension (from app_settings, set by previous index creation)
    2. LiteLLM model data lookup (cloud providers)
    3. Ollama model lookup table (local models)
    4. None (unknown - will be determined at index creation)

    Args:
        model: Model name
        embedding_models: Dict from get_embedding_models()
        tracked_dim: Dimension tracked from previous runs (app_settings.embedding_dimension)

    Returns:
        Dimension if known, None otherwise
    """
    # 1. Tracked dimension from previous runs (most reliable)
    if tracked_dim is not None and tracked_dim > 0:
        return tracked_dim

    # 2. LiteLLM lookup (cloud provider models)
    litellm_dim = get_embedding_dimension_from_litellm(model, embedding_models)
    if litellm_dim:
        return litellm_dim

    # 3. Ollama model lookup table
    model_lower = model.lower()
    # Strip version tags like :latest
    base_model = model_lower.split(":")[0]

    if base_model in OLLAMA_MODEL_DIMENSIONS:
        return OLLAMA_MODEL_DIMENSIONS[base_model]

    # Try partial match for model variants
    for known_model, dim in OLLAMA_MODEL_DIMENSIONS.items():
        if base_model.startswith(known_model) or known_model in base_model:
            return dim

    # Unknown dimension
    logger.debug(
        f"Unknown embedding dimension for {model}, "
        "will be determined at index creation time"
    )
    return None


def estimate_index_memory(
    num_chunks: int,
    embedding_dim: int,
    avg_chunk_chars: int = DEFAULT_AVG_CHUNK_CHARS,
) -> dict:
    """Estimate memory requirements for a FAISS index.

    Args:
        num_chunks: Number of document chunks/vectors
        embedding_dim: Dimension of embedding vectors
        avg_chunk_chars: Average characters per chunk

    Returns:
        Dictionary with:
        - vector_memory_bytes: Memory for FAISS vectors
        - docstore_memory_bytes: Memory for document store
        - steady_memory_bytes: Total steady-state RAM
        - peak_memory_bytes: Peak RAM during loading
    """
    # Vector memory: num_vectors * dimensions * 4 bytes (float32)
    vector_memory = num_chunks * embedding_dim * 4

    # Docstore memory: chunks * avg_chars * overhead factor
    docstore_memory = int(num_chunks * avg_chunk_chars * DOCSTORE_OVERHEAD_FACTOR)

    # Steady state = vectors + docstore + ~10% overhead for Python structures
    steady_memory = int((vector_memory + docstore_memory) * 1.1)

    # Peak during loading (pickle deserialization doubles data temporarily)
    peak_memory = int(steady_memory * PEAK_MEMORY_FACTOR)

    return {
        "vector_memory_bytes": vector_memory,
        "docstore_memory_bytes": docstore_memory,
        "steady_memory_bytes": steady_memory,
        "peak_memory_bytes": peak_memory,
    }


def estimate_memory_at_dimensions(
    num_chunks: int,
    embedding_models: dict[str, EmbeddingModelInfo],
    avg_chunk_chars: int = DEFAULT_AVG_CHUNK_CHARS,
) -> list[dict]:
    """Generate memory estimates for common embedding dimensions.

    Useful for showing users a comparison table.

    Args:
        num_chunks: Number of document chunks
        embedding_models: Dict from get_embedding_models() for example lookups
        avg_chunk_chars: Average characters per chunk

    Returns:
        List of dicts with dimension and memory estimates
    """
    common_dimensions = [384, 768, 1024, 1536, 2048, 3072, 4096]
    results = []

    for dim in common_dimensions:
        est = estimate_index_memory(num_chunks, dim, avg_chunk_chars)
        results.append(
            {
                "dimension": dim,
                "steady_memory_mb": round(
                    est["steady_memory_bytes"] / (1024 * 1024), 1
                ),
                "peak_memory_mb": round(est["peak_memory_bytes"] / (1024 * 1024), 1),
                "examples": get_dimension_examples(dim, embedding_models),
            }
        )

    return results


def get_dimension_examples(
    dim: int,
    embedding_models: dict[str, EmbeddingModelInfo],
) -> list[str]:
    """Get example models for a given dimension from LiteLLM data."""
    examples = []
    for model_id, info in embedding_models.items():
        if info.output_vector_size == dim:
            examples.append(model_id)
            if len(examples) >= 3:  # Limit to 3 examples
                break
    return examples


def refine_peak_estimate_from_history(
    base_peak_bytes: int,
    historical_peak_bytes: Optional[int],
    historical_steady_bytes: Optional[int],
) -> int:
    """Refine peak memory estimate using historical data.

    If we have observed peak/steady ratios from previous loads,
    use that ratio instead of the default factor.

    Args:
        base_peak_bytes: Calculated peak using default factor
        historical_peak_bytes: Observed peak from previous load
        historical_steady_bytes: Observed steady from previous load

    Returns:
        Refined peak memory estimate
    """
    if (
        historical_peak_bytes
        and historical_steady_bytes
        and historical_steady_bytes > 0
    ):
        observed_ratio = historical_peak_bytes / historical_steady_bytes
        # Sanity check: ratio should be between 1.0 and 3.0
        if 1.0 <= observed_ratio <= 3.0:
            # Use observed ratio instead of default
            return int(base_peak_bytes * (observed_ratio / PEAK_MEMORY_FACTOR))

    return base_peak_bytes
