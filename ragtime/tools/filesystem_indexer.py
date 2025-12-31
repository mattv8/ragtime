"""
Filesystem Indexer Tool - Search indexed filesystem content via pgvector.

This tool provides semantic search over filesystem content that has been
indexed into PostgreSQL using pgvector embeddings. The actual indexing
is handled by the filesystem indexer service; this tool only performs searches.
"""

from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from ragtime.core.logging import get_logger
from ragtime.core.database import get_db

logger = get_logger(__name__)


class FilesystemSearchInput(BaseModel):
    """Input schema for filesystem search tool."""
    query: str = Field(
        description="Natural language search query to find relevant documents/files"
    )
    index_name: Optional[str] = Field(
        default=None,
        description="Optional: specific index name to search (searches all if not specified)"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results to return"
    )


async def search_filesystem_index(
    query: str,
    index_name: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """
    Search the filesystem index using semantic similarity.

    This function queries the pgvector embeddings table to find relevant
    document chunks based on the query.
    """
    from ragtime.core.app_settings import get_app_settings

    try:
        db = await get_db()
        app_settings = await get_app_settings()

        # Get the embedding for the query
        embedding = await _get_query_embedding(query, app_settings)
        if embedding is None:
            return "Error: Could not generate query embedding. Check embedding provider settings."

        # Build the search query
        # We use raw SQL because Prisma doesn't natively support pgvector operations
        results = await _search_embeddings(
            db=db,
            embedding=embedding,
            index_name=index_name,
            max_results=max_results,
        )

        if not results:
            scope = f"in index '{index_name}'" if index_name else "across all indexes"
            return f"No relevant documents found {scope} for query: {query}"

        # Format results
        output_parts = [f"Found {len(results)} relevant document(s):\n"]
        for i, result in enumerate(results, 1):
            output_parts.append(f"--- Result {i} ---")
            output_parts.append(f"File: {result['file_path']}")
            output_parts.append(f"Index: {result['index_name']}")
            output_parts.append(f"Similarity: {result['similarity']:.3f}")
            output_parts.append(f"Content:\n{result['content']}\n")

        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"Filesystem search error: {e}")
        return f"Error searching filesystem index: {str(e)}"


async def _get_query_embedding(query: str, app_settings: dict) -> Optional[list]:
    """Generate embedding for the search query using configured provider."""
    try:
        provider = app_settings.get("embedding_provider", "ollama")
        model = app_settings.get("embedding_model", "nomic-embed-text")
        dimensions = app_settings.get("embedding_dimensions")

        if provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            base_url = app_settings.get("ollama_base_url", "http://localhost:11434")
            embeddings = OllamaEmbeddings(model=model, base_url=base_url)
        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            api_key = app_settings.get("openai_api_key", "")
            if not api_key:
                logger.error("OpenAI API key not configured for embeddings")
                return None
            # Pass dimensions for text-embedding-3-* models (supports MRL)
            kwargs = {"model": model, "api_key": api_key}
            if dimensions and model.startswith("text-embedding-3"):
                kwargs["dimensions"] = dimensions
            embeddings = OpenAIEmbeddings(**kwargs)
        else:
            logger.error(f"Unknown embedding provider: {provider}")
            return None

        # Generate embedding
        result = await embeddings.aembed_query(query)
        return result

    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        return None


async def _search_embeddings(
    db,
    embedding: list,
    index_name: Optional[str],
    max_results: int,
) -> list:
    """Execute similarity search against pgvector embeddings table."""
    try:
        # Build embedding vector string for SQL
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        # Build WHERE clause
        where_clause = ""
        if index_name:
            where_clause = f"WHERE index_name = '{index_name}'"

        # Use cosine similarity (<=> operator in pgvector)
        # Note: <=> returns distance, so we compute 1 - distance for similarity
        query = f"""
            SELECT
                id,
                index_name,
                file_path,
                chunk_index,
                content,
                metadata,
                1 - (embedding <=> '{embedding_str}'::vector) as similarity
            FROM filesystem_embeddings
            {where_clause}
            ORDER BY embedding <=> '{embedding_str}'::vector
            LIMIT {max_results}
        """

        results = await db.query_raw(query)

        return [
            {
                "id": row["id"],
                "index_name": row["index_name"],
                "file_path": row["file_path"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "metadata": row["metadata"],
                "similarity": float(row["similarity"]),
            }
            for row in results
        ]

    except Exception as e:
        logger.error(f"Error searching embeddings: {e}")
        # Check if it's a pgvector not installed error
        if "vector" in str(e).lower() and "type" in str(e).lower():
            raise RuntimeError(
                "pgvector extension not installed. Run: CREATE EXTENSION IF NOT EXISTS vector;"
            ) from e
        raise


# The tool is created dynamically in components.py based on tool configs
# This is a template/base that can be used for creating tool instances
def create_filesystem_search_tool(
    tool_name: str,
    description: str,
    index_name: Optional[str] = None,
) -> StructuredTool:
    """
    Create a filesystem search tool instance.

    Args:
        tool_name: Name for the tool (used by LangChain)
        description: Description shown to the LLM
        index_name: Optional index name to restrict searches to
    """

    async def _search(query: str, max_results: int = 10) -> str:
        return await search_filesystem_index(
            query=query,
            index_name=index_name,
            max_results=max_results,
        )

    # Create a specific input schema for this instance
    class SearchInput(BaseModel):
        query: str = Field(
            description="Natural language search query to find relevant documents/files"
        )
        max_results: int = Field(
            default=10,
            ge=1,
            le=50,
            description="Maximum number of results to return"
        )

    return StructuredTool.from_function(
        coroutine=_search,
        name=tool_name,
        description=description,
        args_schema=SearchInput,
    )


# Default tool for registry auto-discovery (searches all indexes)
filesystem_indexer_tool = StructuredTool.from_function(
    coroutine=search_filesystem_index,
    name="filesystem_search",
    description="""Search through indexed filesystem content using semantic similarity.
Use this tool to find relevant documents, files, or text content that has been
indexed from configured filesystem sources (network shares, local paths, etc.).
The search uses vector embeddings for semantic matching, so natural language
queries work well.""",
    args_schema=FilesystemSearchInput,
)
