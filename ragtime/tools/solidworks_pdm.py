"""
SolidWorks PDM Search Tool - Semantic search over indexed PDM metadata.

This tool provides natural language search over PDM document metadata
that has been indexed into PostgreSQL using pgvector embeddings.

Example queries:
- "What material is used for part SW-13392-A?"
- "Find all drawings created by John Smith"
- "Show me parts with Carbon Steel material"
- "What's the BOM for assembly CU-CL9053?"
- "Find parts in the Canopies folder"
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ragtime.core.logging import get_logger

logger = get_logger(__name__)


class PdmSearchInput(BaseModel):
    """Input schema for PDM search tool."""

    query: str = Field(
        description=(
            "Natural language search query to find PDM documents. "
            "Examples: 'parts with aluminum material', "
            "'drawings for assembly CU-CL9053', 'part number 22-012-02015'"
        )
    )
    document_type: Optional[str] = Field(
        default=None,
        description=(
            "Optional filter by document type: SLDPRT (parts), "
            "SLDASM (assemblies), SLDDRW (drawings), or None for all types"
        ),
    )


async def execute_pdm_search(
    query: str,
    index_name: str,
    document_type: Optional[str] = None,
    max_results: int = 10,
) -> str:
    """
    Execute a semantic search against the PDM index.

    Args:
        query: Natural language search query
        index_name: The PDM index name to search
        document_type: Optional filter by document type
        max_results: Maximum results to return

    Returns:
        Formatted string with matching PDM documents
    """
    from ragtime.indexer.pdm_service import search_pdm_index

    logger.info(f"PDM Search: {query[:100]}...")
    if document_type:
        logger.debug(f"  Filter: document_type={document_type}")

    result = await search_pdm_index(
        query=query,
        index_name=index_name,
        document_type=document_type,
        max_results=max_results,
    )

    return result


def create_pdm_search_tool(
    name: str,
    index_name: str,
    description: str = "",
    max_results: int = 10,
) -> StructuredTool:
    """
    Create a configured PDM search tool for LangChain.

    Args:
        name: Tool name (e.g., 'search_ham_pdm')
        index_name: The PDM index name to search
        description: Tool description for the LLM
        max_results: Maximum results per search

    Returns:
        Configured StructuredTool for PDM search
    """
    if not description:
        description = (
            "Search SolidWorks PDM for parts, assemblies, and drawings. "
            "Query using natural language to find documents by part number, "
            "material, description, author, folder, or BOM relationships. "
            "Returns matching documents with their metadata."
        )

    async def search_pdm(query: str, document_type: Optional[str] = None) -> str:
        return await execute_pdm_search(
            query=query,
            index_name=index_name,
            document_type=document_type,
            max_results=max_results,
        )

    return StructuredTool.from_function(
        coroutine=search_pdm,
        name=name,
        description=description,
        args_schema=PdmSearchInput,
    )


# Export the tool factory
solidworks_pdm_tool = create_pdm_search_tool
