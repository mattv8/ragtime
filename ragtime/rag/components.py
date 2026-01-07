"""
RAG Components - FAISS Vector Store and LangChain Agent setup.
"""

import asyncio
import re
from pathlib import Path
from typing import Any, List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from ragtime.config import settings
from ragtime.core.app_settings import get_app_settings, get_tool_configs
from ragtime.core.logging import get_logger
from ragtime.tools import get_all_tools, get_enabled_tools
from ragtime.tools.odoo_shell import filter_odoo_output

logger = get_logger(__name__)

# Base system prompt template - tool and index descriptions will be appended
BASE_SYSTEM_PROMPT = """You are an intelligent AI assistant with access to indexed documentation and live system connections.
Your role is to help users understand their systems by combining knowledge from documentation with real-time data.

AVAILABLE TOOLS:

1. **search_knowledge** - Search indexed documentation (code, configs, technical docs)
   Returns relevant code snippets, schema definitions, and implementation details.
   Useful for understanding how systems work, what fields exist, and how data flows.

2. **Query tools** (database/system-specific) - Access live data in real-time
   Execute queries against actual databases and systems to get current data.
   Each tool connects to a specific system - read descriptions to understand which.

WHEN TO USE EACH TOOL:

**search_knowledge** is best for:
- Understanding schemas, models, or table structures before querying
- Finding implementation details, business logic, or validation rules
- Looking up field names, data types, or relationships
- Exploring unfamiliar systems or codebases

**Query tools** are best for:
- Retrieving actual data values ("How many orders?" "Show me user X")
- Checking current state or status of records
- Running reports or aggregations on live data
- Verifying what's actually in the database

**Combine both** for investigative questions:
- Use search_knowledge if you're unsure about the schema
- Use query tools when you know what you're looking for
- Iterate: search docs -> query data -> search more if needed

GUIDELINES:
- Choose the right tool based on what the question asks for
- If you know the schema already, go straight to querying
- If a query fails due to unknown columns/tables, search docs to find correct names
- You can call search_knowledge multiple times with different queries
- Always use LIMIT clauses in SQL queries

RESPONSE FORMAT:
- Lead with the answer, not implementation details
- Present data in tables or lists when appropriate
- Show queries/code only if explicitly requested
- Be concise but thorough"""


def build_index_system_prompt(index_metadata: List[dict]) -> str:
    """
    Build system prompt section describing available knowledge indexes.

    Args:
        index_metadata: List of index metadata dictionaries from database.

    Returns:
        System prompt section with index descriptions.
    """
    if not index_metadata:
        return """

NOTE: No knowledge indexes are currently loaded. You can only use live query tools.
To answer questions about code structure or implementation details, index the relevant codebase first.
"""

    # Filter to only enabled indexes
    enabled_indexes = [idx for idx in index_metadata if idx.get("enabled", True)]
    if not enabled_indexes:
        return """

NOTE: All knowledge indexes are disabled. Enable them in the Indexes tab to use documentation context.
"""

    index_sections = []
    for idx in enabled_indexes:
        name = idx.get("name", "Unnamed Index")
        description = idx.get("description", "")
        doc_count = idx.get("document_count", 0)
        chunk_count = idx.get("chunk_count", 0)
        source_type = idx.get("source_type", "unknown")

        section = (
            f"- **{name}** ({source_type}, {doc_count} files, {chunk_count} chunks)"
        )
        if description:
            section += f"\n  {description}"
        index_sections.append(section)

    return f"""

INDEXED KNOWLEDGE SOURCES:
{chr(10).join(index_sections)}

These indexes are AUTOMATICALLY searched for each query. The relevant documentation context is injected into your input.
Use this context to:
- Understand table/model schemas before writing queries
- Learn business logic and validation rules
- Find implementation details for debugging
- Discover relationships between different parts of the system

If the automatically retrieved context doesn't answer the question, you may need to query live data using the available tools.
"""


def build_tool_system_prompt(tool_configs: List[dict]) -> str:
    """
    Build system prompt section describing available tools.

    Args:
        tool_configs: List of tool configuration dictionaries.

    Returns:
        System prompt section with tool descriptions.
    """
    if not tool_configs:
        return """

NOTE: No query tools are configured. You can only answer from indexed documentation.
To query live data, configure tools in the Tools tab.
"""

    tool_sections = []
    for config in tool_configs:
        tool_type = config.get("tool_type", "unknown")
        name = config.get("name", "Unnamed Tool")
        description = config.get("description", "")

        # Build type-specific guidance
        if tool_type == "postgres":
            type_hint = "SQL database - execute SQL queries"
            query_hint = "Use standard SQL syntax with LIMIT clauses"
        elif tool_type == "odoo_shell":
            type_hint = "Odoo ORM - execute Python ORM code"
            query_hint = "Use env['model.name'].search() and browse() methods"
        elif tool_type == "ssh_shell":
            type_hint = "SSH shell - execute shell commands"
            query_hint = "Use standard shell commands"
        else:
            type_hint = "Query tool"
            query_hint = ""

        section = f"- **{name}** ({type_hint})"
        if description:
            section += f"\n  Target: {description}"
        if query_hint:
            section += f"\n  Usage: {query_hint}"
        tool_sections.append(section)

    return f"""

AVAILABLE QUERY TOOLS:
{chr(10).join(tool_sections)}

CRITICAL: Read each tool's "Target" description to understand what system it connects to.
Different tools connect to DIFFERENT systems - choose based on where the data lives.
If the user's question refers to a specific system, use the tool that connects to that system.
"""


class RAGComponents:
    """Container for RAG components initialized at startup."""

    def __init__(self):
        self.retrievers: dict[str, Any] = {}
        self.agent_executor: Optional[AgentExecutor] = None
        self.llm: Optional[Any] = None  # ChatOpenAI, ChatAnthropic, or ChatOllama
        self.is_ready: bool = False
        self._app_settings: Optional[dict] = None
        self._tool_configs: Optional[List[dict]] = None
        self._index_metadata: Optional[List[dict]] = None
        self._system_prompt: str = BASE_SYSTEM_PROMPT

    async def initialize(self):
        """Initialize all RAG components."""
        logger.info("Initializing RAG components...")

        # Load settings from database
        self._app_settings = await get_app_settings()
        self._tool_configs = await get_tool_configs()
        self._index_metadata = await self._load_index_metadata()

        # Build system prompt with tool and index descriptions
        tool_prompt_section = build_tool_system_prompt(self._tool_configs)
        index_prompt_section = build_index_system_prompt(self._index_metadata)
        self._system_prompt = (
            BASE_SYSTEM_PROMPT + index_prompt_section + tool_prompt_section
        )

        # Initialize LLM based on provider from database settings
        await self._init_llm()

        # Load embedding model based on database settings
        embedding_model = await self._get_embedding_model()

        # Load FAISS indexes from configured paths
        await self._load_faiss_indexes(embedding_model)

        # Create the agent with tools
        await self._create_agent()

        self.is_ready = True
        logger.info("RAG components initialized successfully")

    async def _init_llm(self):
        """Initialize LLM based on database settings."""
        assert self._app_settings is not None  # Set by initialize()
        provider = self._app_settings.get("llm_provider", "openai").lower()
        model = self._app_settings.get("llm_model", "gpt-4-turbo")

        if provider == "ollama":
            try:
                from langchain_ollama import \
                    ChatOllama  # type: ignore[import-untyped,import-not-found]

                base_url = self._app_settings.get(
                    "ollama_base_url", "http://localhost:11434"
                )
                self.llm = ChatOllama(
                    model=model,
                    base_url=base_url,
                    temperature=0,
                )
                logger.info(f"Using Ollama LLM: {model} at {base_url}")
                return
            except ImportError:
                logger.warning("langchain-ollama not installed, falling back to OpenAI")

        if provider == "anthropic":
            api_key = self._app_settings.get("anthropic_api_key", "")
            if api_key:
                try:
                    from langchain_anthropic import ChatAnthropic

                    self.llm = ChatAnthropic(  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
                        model=model, temperature=0, anthropic_api_key=api_key
                    )
                    logger.info(f"Using Anthropic LLM: {model}")
                    return
                except ImportError:
                    logger.warning(
                        "langchain-anthropic not installed, falling back to OpenAI"
                    )
            else:
                logger.warning(
                    "Anthropic selected but no API key configured, falling back to OpenAI"
                )

        # Default to OpenAI
        api_key = self._app_settings.get("openai_api_key", "")
        if not api_key:
            logger.warning(
                "No OpenAI API key configured - LLM features will be disabled until configured via Settings UI"
            )
            self.llm = None
            return

        self.llm = ChatOpenAI(  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
            model=model if provider == "openai" else "gpt-4-turbo",
            temperature=0,
            streaming=True,
            openai_api_key=api_key,
        )
        logger.info(f"Using OpenAI LLM: {self.llm.model_name}")

    async def _get_embedding_model(self):
        """Get embedding model based on database settings."""
        assert self._app_settings is not None  # Set by initialize()
        provider = self._app_settings.get("embedding_provider", "ollama").lower()
        model = self._app_settings.get("embedding_model", "nomic-embed-text")

        if provider == "ollama":
            from langchain_ollama import \
                OllamaEmbeddings  # type: ignore[import-untyped]

            base_url = self._app_settings.get(
                "ollama_base_url", "http://localhost:11434"
            )
            logger.info(f"Using Ollama embeddings: {model} at {base_url}")
            return OllamaEmbeddings(model=model, base_url=base_url)
        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            api_key = self._app_settings.get("openai_api_key", "")
            if not api_key:
                logger.warning("OpenAI embeddings selected but no API key configured")
            logger.info(f"Using OpenAI embeddings: {model}")
            return OpenAIEmbeddings(model=model, openai_api_key=api_key)  # type: ignore[call-arg]
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    async def _load_faiss_indexes(self, embedding_model):
        """Load FAISS indexes from database metadata.

        Uses the index_metadata table to discover available indexes and loads
        only those that are enabled. The path is read directly from the database
        metadata, which was saved by the indexer service.
        """
        assert self._app_settings is not None  # Set by initialize()
        # Try to load from database metadata (preferred)
        if self._index_metadata:
            enabled_indexes = [
                idx for idx in self._index_metadata if idx.get("enabled", True)
            ]

            if enabled_indexes:
                for idx in enabled_indexes:
                    index_name = idx.get("name")
                    if not index_name:
                        continue

                    # Use the path stored in the database by the indexer
                    index_path_str = idx.get("path")
                    if not index_path_str:
                        logger.warning(
                            f"Index {index_name} has no path in metadata, skipping"
                        )
                        continue

                    index_path = Path(index_path_str)
                    if index_path.exists():
                        try:
                            db = FAISS.load_local(
                                str(index_path),
                                embedding_model,
                                allow_dangerous_deserialization=True,
                            )
                            # Use configurable k from settings
                            search_k = self._app_settings.get("search_results_k", 5)
                            self.retrievers[index_name] = db.as_retriever(
                                search_kwargs={"k": search_k}
                            )
                            logger.info(
                                f"Loaded FAISS index: {index_name} from {index_path} (k={search_k})"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to load FAISS index {index_name}: {e}"
                            )
                    else:
                        logger.warning(f"FAISS index path not found: {index_path}")

                if self.retrievers:
                    logger.info(
                        f"Loaded {len(self.retrievers)} FAISS index(es) from database metadata"
                    )
                else:
                    logger.info("No enabled FAISS indexes found in database metadata")
            else:
                logger.info("No indexes found in database metadata")
        else:
            logger.info("No index metadata available (database not initialized)")

    async def _load_index_metadata(self) -> list[dict]:
        """Load index metadata from database for system prompt."""
        from ragtime.indexer.repository import repository

        try:
            metadata_list = await repository.list_index_metadata()
            return [
                {
                    "name": m.name,
                    "path": m.path,
                    "description": getattr(m, "description", ""),
                    "enabled": m.enabled,
                    "search_weight": getattr(m, "searchWeight", 1.0),
                    "document_count": m.documentCount,
                    "chunk_count": m.chunkCount,
                    "source_type": m.sourceType,
                }
                for m in metadata_list
            ]
        except Exception as e:
            logger.warning(f"Failed to load index metadata: {e}")
            return []

    async def _create_agent(self):
        """Create the LangChain agent with tools."""
        assert self._app_settings is not None  # Set by initialize()
        # Skip agent creation if LLM is not configured
        if self.llm is None:
            logger.warning("Skipping agent creation - no LLM configured")
            self.agent_executor = None
            return

        tools = []

        # Add knowledge search tool(s) if we have FAISS retrievers
        if self.retrievers:
            aggregate_search = self._app_settings.get("aggregate_search", True)
            if aggregate_search:
                # Single aggregated search_knowledge tool
                tools.append(self._create_knowledge_search_tool())
                logger.info(
                    f"Added search_knowledge tool for {len(self.retrievers)} index(es)"
                )
            else:
                # Separate search_<index_name> tools for each index
                knowledge_tools = self._create_per_index_search_tools()
                tools.extend(knowledge_tools)
                logger.info(f"Added {len(knowledge_tools)} per-index search tools")

        # Get tools from the new ToolConfig system
        if self._tool_configs:
            config_tools = await self._build_tools_from_configs(
                skip_knowledge_tool=True
            )
            tools.extend(config_tools)
            logger.info(f"Built {len(config_tools)} tools from configurations")
        else:
            # Fallback to legacy enabled_tools system
            app_settings = await get_app_settings()
            enabled_list = app_settings["enabled_tools"]
            if enabled_list:
                legacy_tools = get_enabled_tools(enabled_list)
                tools.extend(legacy_tools)
                logger.info(f"Using legacy tool configuration: {enabled_list}")

        if not tools:
            available = list(get_all_tools().keys())
            logger.warning(
                f"No tools configured. Available tool types: {available}. "
                f"Configure via Tools tab at /indexes/ui?view=tools"
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        # Respect admin-configured iteration limit; fall back to 15 if invalid
        try:
            max_iterations = int(self._app_settings.get("max_iterations", 15))
            if max_iterations < 1:
                max_iterations = 1
        except (TypeError, ValueError):
            max_iterations = 15

        if tools:
            # Use create_tool_calling_agent which works with both OpenAI and Anthropic
            agent = create_tool_calling_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=settings.debug_mode,
                handle_parsing_errors=True,
                max_iterations=max_iterations,  # Allow enough iterations for query refinement
                return_intermediate_steps=settings.debug_mode,
            )
        else:
            # No tools - agent_executor stays None
            self.agent_executor = None

    async def _build_tools_from_configs(
        self, skip_knowledge_tool: bool = False
    ) -> List[Any]:
        """
        Build LangChain tools from ToolConfig entries.

        Creates dynamic tool wrappers for each configured tool instance.

        Args:
            skip_knowledge_tool: If True, don't add the search_knowledge tool
                (used when caller has already added it)
        """
        tools = []

        # Add knowledge search tool if we have FAISS retrievers (unless skipped)
        if self.retrievers and not skip_knowledge_tool:
            tools.append(self._create_knowledge_search_tool())

        for config in self._tool_configs or []:
            tool_type = config.get("tool_type")
            # Convert the user-provided tool name into a tool-safe identifier
            raw_name = (config.get("name", "") or "").strip()
            # Lowercase and replace any non-alphanumeric sequence with a single underscore
            tool_name = re.sub(r"[^a-zA-Z0-9]+", "_", raw_name).strip("_").lower()
            tool_id = config.get("id") or ""

            if tool_type == "postgres":
                tool = await self._create_postgres_tool(config, tool_name, tool_id)
                # Also create schema search tool if schema indexing is enabled
                schema_tool = await self._create_schema_search_tool(
                    config, tool_name, tool_id
                )
                if schema_tool:
                    tools.append(schema_tool)
            elif tool_type == "mssql":
                tool = await self._create_mssql_tool(config, tool_name, tool_id)
                # Also create schema search tool if schema indexing is enabled
                schema_tool = await self._create_schema_search_tool(
                    config, tool_name, tool_id
                )
                if schema_tool:
                    tools.append(schema_tool)
            elif tool_type == "odoo_shell":
                tool = await self._create_odoo_tool(config, tool_name, tool_id)
            elif tool_type == "ssh_shell":
                tool = await self._create_ssh_tool(config, tool_name, tool_id)
            elif tool_type == "filesystem_indexer":
                tool = await self._create_filesystem_tool(config, tool_name, tool_id)
            else:
                logger.warning(f"Unknown tool type: {tool_type}")
                continue

            if tool:
                tools.append(tool)

        return tools

    async def _create_schema_search_tool(
        self, config: dict, tool_name: str, tool_id: str
    ):
        """Create a schema search tool for SQL database tools.

        This tool allows the agent to search the indexed database schema
        to find relevant table/column information before writing queries.
        """
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field

        conn_config = config.get("connection_config", {})

        # Check if schema indexing is enabled
        schema_index_enabled = conn_config.get("schema_index_enabled", False)
        logger.debug(
            f"Schema tool check for {tool_name}: enabled={schema_index_enabled}"
        )
        if not schema_index_enabled:
            return None

        # Check if there are any schema embeddings for this tool
        from ragtime.indexer.schema_service import schema_indexer

        embedding_count = await schema_indexer.get_embedding_count(tool_id, tool_name)
        if embedding_count == 0:
            logger.debug(
                f"Schema indexing enabled but no embeddings found for {tool_name}"
            )
            # Still create the tool - it will just return "no results" until indexed
            # This is better than no tool at all

        # Use tool_name (safe name) for index lookup - matches how trigger_index creates it
        index_name = f"schema_{tool_name}"
        description = config.get("description", "")

        class SchemaSearchInput(BaseModel):
            query: str = Field(
                description="Search query to find relevant tables, columns, or relationships in the database schema"
            )

        async def search_schema(query: str) -> str:
            """Search the database schema for relevant table information."""
            from ragtime.indexer.schema_service import search_schema_index

            logger.debug(f"[{tool_name}] Schema search: {query[:100]}")

            result = await search_schema_index(
                query=query,
                index_name=index_name,
                max_results=5,
            )
            return result

        tool_description = (
            f"Search the schema of the {config.get('name', 'database')} database "
            f"to find table names, column definitions, relationships, and indexes. "
            f"Use this BEFORE writing SQL queries when you need to understand the database structure."
        )
        if description:
            tool_description += f" Database contains: {description}"

        schema_tool = StructuredTool.from_function(
            coroutine=search_schema,
            name=f"search_{tool_name}_schema",
            description=tool_description,
            args_schema=SchemaSearchInput,
        )
        logger.info(f"Created schema search tool: search_{tool_name}_schema")
        return schema_tool

    def _create_knowledge_search_tool(self):
        """Create a tool for on-demand FAISS knowledge search.

        This allows the agent to search the indexed documentation at any point
        during its reasoning, not just at the beginning of the query.
        """
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field

        class KnowledgeSearchInput(BaseModel):
            query: str = Field(
                description="Search query to find relevant documentation, code, or technical information"
            )
            index_name: str = Field(
                default="",
                description="Optional: specific index to search (leave empty to search all indexes)",
            )

        def search_knowledge(query: str, index_name: str = "") -> str:
            """Search indexed documentation for relevant information."""
            results = []
            errors = []

            # Log the search attempt for debugging
            logger.debug(
                f"search_knowledge called with query='{query[:50]}...', index_name='{index_name}'"
            )
            logger.debug(f"Available retrievers: {list(self.retrievers.keys())}")

            # Determine which retrievers to search
            if index_name and index_name in self.retrievers:
                retrievers_to_search = {index_name: self.retrievers[index_name]}
            else:
                retrievers_to_search = self.retrievers

            if not retrievers_to_search:
                logger.warning("No retrievers available for search_knowledge")
                return "No knowledge indexes are currently loaded. Please index some documents first."

            for name, retriever in retrievers_to_search.items():
                try:
                    logger.debug(
                        f"Searching index '{name}' with query: {query[:50]}..."
                    )
                    docs = retriever.invoke(query)
                    logger.debug(f"Index '{name}' returned {len(docs)} documents")
                    for doc in docs:
                        source = doc.metadata.get("source", "unknown")
                        content = (
                            doc.page_content[:500] + "..."
                            if len(doc.page_content) > 500
                            else doc.page_content
                        )
                        results.append(f"[{name}] {source}:\n{content}")
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Error searching {name}: {e}", exc_info=True)
                    # Detect Ollama connectivity issues
                    if (
                        "ollama" in error_msg.lower()
                        or "failed to connect" in error_msg.lower()
                    ):
                        errors.append(
                            f"[{name}] Embedding service unavailable - Cannot connect to Ollama. "
                            "Check that Ollama is running and the URL in Settings is accessible from the server "
                            "(use 'host.docker.internal' instead of 'localhost' when running in Docker)."
                        )
                    else:
                        errors.append(f"[{name}] Search error: {error_msg}")

            if results:
                logger.debug(f"search_knowledge found {len(results)} results")
                return (
                    f"Found {len(results)} relevant documents:\n\n"
                    + "\n\n---\n\n".join(results)
                )

            # Return errors if we had any, otherwise generic no results message
            if errors:
                logger.warning(f"search_knowledge failed with errors: {errors}")
                return "Search failed:\n" + "\n".join(errors)

            logger.debug("search_knowledge found no results")
            return "No relevant documentation found for this query."

        # Build description with available indexes
        index_names = list(self.retrievers.keys())
        description = (
            "Search the indexed documentation and codebase for relevant information. "
            "Use this to find code examples, schema definitions, configuration details, or technical documentation. "
            f"Available indexes: {', '.join(index_names)}. "
            "The query should describe what you're looking for (e.g., 'user authentication implementation', 'database schema for orders')."
        )

        return StructuredTool.from_function(
            func=search_knowledge,
            name="search_knowledge",
            description=description,
            args_schema=KnowledgeSearchInput,
        )

    def _create_per_index_search_tools(self) -> List[Any]:
        """Create separate search tools for each index.

        When aggregate_search is disabled, this creates search_<index_name>
        tools that give the AI granular control over which index to search.
        """
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field

        tools = []

        # Get index metadata for descriptions and weights
        index_weights = {}
        index_descriptions = {}
        for idx in self._index_metadata or []:
            if idx.get("enabled", True):
                name = idx.get("name", "")
                index_weights[name] = idx.get("search_weight", 1.0)
                index_descriptions[name] = idx.get("description", "")

        for index_name, retriever in self.retrievers.items():
            # Create a closure to capture the current index_name and retriever
            def make_search_func(idx_name: str, idx_retriever):
                def search_index(query: str) -> str:
                    """Search this specific index for relevant information."""
                    results = []

                    logger.debug(
                        f"search_{idx_name} called with query='{query[:50]}...'"
                    )

                    try:
                        docs = idx_retriever.invoke(query)
                        logger.debug(
                            f"Index '{idx_name}' returned {len(docs)} documents"
                        )
                        for doc in docs:
                            source = doc.metadata.get("source", "unknown")
                            content = (
                                doc.page_content[:500] + "..."
                                if len(doc.page_content) > 500
                                else doc.page_content
                            )
                            results.append(f"{source}:\n{content}")
                    except Exception as e:
                        error_msg = str(e)
                        logger.warning(
                            f"Error searching {idx_name}: {e}", exc_info=True
                        )
                        if (
                            "ollama" in error_msg.lower()
                            or "failed to connect" in error_msg.lower()
                        ):
                            return (
                                "Embedding service unavailable - Cannot connect to Ollama. "
                                "Check that Ollama is running and accessible."
                            )
                        return f"Search error: {error_msg}"

                    if results:
                        return (
                            f"Found {len(results)} relevant documents:\n\n"
                            + "\n\n---\n\n".join(results)
                        )

                    return (
                        f"No relevant documents found in {idx_name} for query: {query}"
                    )

                return search_index

            # Create input schema for this tool
            class IndexSearchInput(BaseModel):
                query: str = Field(
                    description="Search query to find relevant documentation or code"
                )

            # Build description including the index description and weight hint
            weight = index_weights.get(index_name, 1.0)
            idx_desc = index_descriptions.get(index_name, "")

            tool_description = (
                f"Search the '{index_name}' index for relevant information."
            )
            if idx_desc:
                tool_description += f" {idx_desc}"
            if weight != 1.0:
                if weight > 1.0:
                    tool_description += f" [Priority: High (weight={weight})]"
                elif weight > 0:
                    tool_description += f" [Priority: Low (weight={weight})]"

            # Sanitize index name for tool name (replace invalid chars)
            safe_name = index_name.replace("-", "_").replace(" ", "_").lower()
            tool_name = f"search_{safe_name}"

            tool = StructuredTool.from_function(
                func=make_search_func(index_name, retriever),
                name=tool_name,
                description=tool_description,
                args_schema=IndexSearchInput,
            )
            tools.append(tool)

        return tools

    async def _create_postgres_tool(self, config: dict, tool_name: str, _tool_id: str):
        """Create a PostgreSQL query tool from config."""
        import subprocess

        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field

        conn_config = config.get("connection_config", {})
        timeout = config.get("timeout", 30)
        allow_write = config.get("allow_write", False)
        description = config.get("description", "")

        class PostgresInput(BaseModel):
            query: str = Field(
                default="",
                description="SQL query to execute. Must include LIMIT clause.",
            )
            reason: str = Field(
                default="", description="Brief description of what this query retrieves"
            )

        async def execute_query(query: str = "", reason: str = "", **_: Any) -> str:
            """Execute PostgreSQL query using this tool's configuration."""
            from ragtime.core.security import (sanitize_output,
                                               validate_sql_query)

            # Validate required fields
            if not query or not query.strip():
                return "Error: 'query' parameter is required. Provide a SQL query to execute."
            if not reason:
                reason = "SQL query"

            logger.info(f"[{tool_name}] Query: {reason}")

            # Validate query
            is_safe, validation_reason = validate_sql_query(
                query, enable_write=allow_write
            )
            if not is_safe:
                return f"Error: {validation_reason}"

            # Build command
            host = conn_config.get("host", "")
            port = conn_config.get("port", 5432)
            user = conn_config.get("user", "")
            password = conn_config.get("password", "")
            database = conn_config.get("database", "")
            container = conn_config.get("container", "")

            escaped_query = query.replace("'", "'\\''")

            if host:
                cmd = [
                    "psql",
                    "-h",
                    host,
                    "-p",
                    str(port),
                    "-U",
                    user,
                    "-d",
                    database,
                    "-c",
                    query,
                ]
                env = {"PGPASSWORD": password}
            elif container:
                cmd = [
                    "docker",
                    "exec",
                    "-i",
                    container,
                    "bash",
                    "-c",
                    f'PGPASSWORD="$POSTGRES_PASSWORD" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \'{escaped_query}\'',
                ]
                env = None
            else:
                return "Error: No connection configured"

            try:
                process = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
                    ),
                    timeout=timeout,
                )
                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    return f"Error: {stderr.decode('utf-8', errors='replace').strip()}"

                output = stdout.decode("utf-8", errors="replace").strip()
                return (
                    sanitize_output(output)
                    if output
                    else "Query executed successfully (no results)"
                )

            except asyncio.TimeoutError:
                return f"Error: Query timed out after {timeout}s"
            except Exception as e:
                return f"Error: {str(e)}"

        tool_description = (
            f"Query the {config.get('name', 'PostgreSQL')} database using SQL."
        )
        if description:
            tool_description += f" This database contains: {description}"
        tool_description += " Include LIMIT clause to restrict results. SELECT queries only unless writes are enabled."

        return StructuredTool.from_function(
            coroutine=execute_query,
            name=f"query_{tool_name}",
            description=tool_description,
            args_schema=PostgresInput,
        )

    async def _create_mssql_tool(self, config: dict, tool_name: str, _tool_id: str):
        """Create an MSSQL/SQL Server query tool from config."""
        from ragtime.tools.mssql import create_mssql_tool

        conn_config = config.get("connection_config", {})
        timeout = config.get("timeout", 30)
        max_results = config.get("max_results", 100)
        allow_write = config.get("allow_write", False)
        description = config.get("description", "")

        host = conn_config.get("host", "")
        port = conn_config.get("port", 1433)
        user = conn_config.get("user", "")
        password = conn_config.get("password", "")
        database = conn_config.get("database", "")

        if not host:
            logger.error(f"MSSQL tool {tool_name} missing host configuration")
            return None

        return create_mssql_tool(
            name=config.get("name", tool_name),
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            timeout=timeout,
            max_results=max_results,
            allow_write=allow_write,
            description=description,
        )

    async def _create_odoo_tool(self, config: dict, tool_name: str, _tool_id: str):
        """Create an Odoo shell tool from config (Docker or SSH mode)."""
        import subprocess

        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field

        conn_config = config.get("connection_config", {})
        timeout = config.get("timeout", 60)  # Odoo shell needs more time to initialize
        allow_write = config.get("allow_write", False)
        description = config.get("description", "")
        mode = conn_config.get("mode", "docker")  # docker or ssh

        class OdooInput(BaseModel):
            code: str = Field(
                default="",
                description="Python code to execute in Odoo shell using ORM methods",
            )
            reason: str = Field(
                default="", description="Brief description of what this code does"
            )

        def _build_docker_command(
            container: str, database: str, config_path: str
        ) -> list:
            """Build Docker exec command for Odoo shell."""
            cmd = [
                "docker",
                "exec",
                "-i",
                container,
                "odoo",
                "shell",
                "--no-http",
                "-d",
                database,
            ]
            if config_path:
                cmd.extend(["-c", config_path])
            return cmd

        async def execute_odoo(code: str = "", reason: str = "", **_: Any) -> str:
            """Execute Odoo shell command using this tool's configuration."""
            from ragtime.core.security import (sanitize_output,
                                               validate_odoo_code)

            # Validate required fields
            if not code or not code.strip():
                return "Error: 'code' parameter is required. Provide Python code to execute in the Odoo shell."
            if not reason:
                reason = "Odoo query"

            logger.info(f"[{tool_name}] Odoo ({mode}): {reason}")

            # Validate code
            is_safe, validation_reason = validate_odoo_code(
                code, enable_write_ops=allow_write
            )
            if not is_safe:
                return f"Error: {validation_reason}"

            database = conn_config.get("database", "odoo")
            config_path = conn_config.get("config_path", "")

            # Wrap user code with env setup and error handling
            wrapped_code = f"""
env = self.env
try:
{chr(10).join("    " + line for line in code.strip().split(chr(10)))}
except Exception as e:
    print(f"ODOO_ERROR: {{type(e).__name__}}: {{e}}")
"""
            # Add exit command
            full_input = wrapped_code + "\nexit()\n"

            async def _run_with_cmd(cmd: list) -> str:
                """Execute command and return filtered output."""
                try:
                    process = await asyncio.wait_for(
                        asyncio.create_subprocess_exec(
                            *cmd,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,  # Merge stderr into stdout
                        ),
                        timeout=timeout,
                    )
                    stdout, _ = await process.communicate(input=full_input.encode())
                    output = stdout.decode("utf-8", errors="replace")

                    result = filter_odoo_output(output)
                    return (
                        sanitize_output(result)
                        if result
                        else "Query executed successfully (no output)"
                    )

                except asyncio.TimeoutError:
                    return f"Error: Query timed out after {timeout}s"
                except FileNotFoundError:
                    cmd_name = "SSH" if mode == "ssh" else "Docker"
                    return f"Error: {cmd_name} command not found"
                except Exception as e:
                    logger.exception(f"Odoo shell error: {e}")
                    return f"Error: {str(e)}"

            # Build command based on mode
            if mode == "ssh":
                ssh_host = conn_config.get("ssh_host", "")
                if not ssh_host:
                    return "Error: No SSH host configured"

                # Use Paramiko for SSH connection
                from ragtime.core.ssh import SSHConfig, execute_ssh_command

                ssh_config = SSHConfig(
                    host=ssh_host,
                    port=conn_config.get("ssh_port", 22),
                    user=conn_config.get("ssh_user", ""),
                    password=conn_config.get("ssh_password"),
                    key_path=conn_config.get("ssh_key_path"),
                    key_content=conn_config.get("ssh_key_content"),
                    key_passphrase=conn_config.get("ssh_key_passphrase"),
                    timeout=timeout,
                )

                # Build remote Odoo shell command
                odoo_bin_path = conn_config.get("odoo_bin_path", "odoo-bin")
                odoo_config_path = conn_config.get("config_path", "")
                working_directory = conn_config.get("working_directory", "")
                run_as_user = conn_config.get("run_as_user", "")

                odoo_cmd = f"{odoo_bin_path} shell --no-http -d {database}"
                if odoo_config_path:
                    odoo_cmd = f"{odoo_cmd} -c {odoo_config_path}"
                if run_as_user:
                    odoo_cmd = f"sudo -u {run_as_user} {odoo_cmd}"
                if working_directory:
                    odoo_cmd = f"cd {working_directory} && {odoo_cmd}"

                # Use heredoc to pass code to shell
                remote_command = f"{odoo_cmd} <<'ODOO_EOF'\n{full_input}ODOO_EOF"

                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, lambda: execute_ssh_command(ssh_config, remote_command)
                    )

                    if not result.success and "ODOO_ERROR" not in result.output:
                        return f"Error (exit {result.exit_code}): {result.stderr or result.stdout}"

                    # For SSH, filter with ssh_mode=True to strip STDERR section
                    filtered = filter_odoo_output(result.output, ssh_mode=True)
                    return (
                        sanitize_output(filtered)
                        if filtered
                        else "Query executed successfully (no output)"
                    )

                except Exception as e:
                    logger.exception(f"Odoo SSH error: {e}")
                    return f"Error: {str(e)}"

            else:  # docker mode
                container = conn_config.get("container", "")
                if not container:
                    return "Error: No container configured"
                cmd = _build_docker_command(container, database, config_path)
                return await _run_with_cmd(cmd)

        mode_label = "SSH" if mode == "ssh" else "Docker"
        tool_description = f"Query {config.get('name', 'Odoo')} ERP using Python ORM code ({mode_label} connection)."
        if description:
            tool_description += f" This system contains: {description}"
        tool_description += (
            " Use env['model'].search_read(domain, fields, limit=N) for data retrieval."
        )

        return StructuredTool.from_function(
            coroutine=execute_odoo,
            name=f"odoo_{tool_name}",
            description=tool_description,
            args_schema=OdooInput,
        )

    async def _create_ssh_tool(self, config: dict, tool_name: str, _tool_id: str):
        """Create an SSH shell tool from config."""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field

        conn_config = config.get("connection_config", {})
        timeout = config.get("timeout", 30)
        description = config.get("description", "")

        class SSHInput(BaseModel):
            command: str = Field(
                default="", description="Shell command to execute on the remote server"
            )
            reason: str = Field(
                default="", description="Brief description of what this command does"
            )

        async def execute_ssh(command: str = "", reason: str = "", **_: Any) -> str:
            """Execute SSH command using this tool's configuration."""
            from ragtime.core.security import sanitize_output
            from ragtime.core.ssh import SSHConfig, execute_ssh_command

            # Validate required fields
            if not command or not command.strip():
                return "Error: 'command' parameter is required. Provide a shell command to execute."
            if not reason:
                reason = "SSH command"

            logger.info(f"[{tool_name}] SSH: {reason}")

            host = conn_config.get("host", "")
            port = conn_config.get("port", 22)
            user = conn_config.get("user", "")
            key_path = conn_config.get("key_path")
            key_content = conn_config.get("key_content")
            key_passphrase = conn_config.get("key_passphrase")
            password = conn_config.get("password")
            command_prefix = conn_config.get("command_prefix", "")

            if not host or not user:
                return "Error: Host and user are required"

            full_command = f"{command_prefix}{command}"

            # Build SSH config
            ssh_config = SSHConfig(
                host=host,
                port=port,
                user=user,
                password=password,
                key_path=key_path,
                key_content=key_content,
                key_passphrase=key_passphrase,
                timeout=timeout,
            )

            try:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: execute_ssh_command(ssh_config, full_command)
                )

                if not result.success:
                    return f"Error (exit {result.exit_code}): {result.stderr or result.stdout}"

                output = result.stdout.strip()
                return (
                    sanitize_output(output)
                    if output
                    else "Command executed successfully (no output)"
                )

            except Exception as e:
                return f"Error: {str(e)}"

        tool_description = (
            f"Execute shell commands on {config.get('name', 'remote server')} via SSH."
        )
        if description:
            tool_description += f" This server provides access to: {description}"

        return StructuredTool.from_function(
            coroutine=execute_ssh,
            name=f"ssh_{tool_name}",
            description=tool_description,
            args_schema=SSHInput,
        )

    async def _create_filesystem_tool(self, config: dict, tool_name: str, tool_id: str):
        """Create a filesystem search tool from config."""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field

        from ragtime.tools.filesystem_indexer import search_filesystem_index

        conn_config = config.get("connection_config", {})
        description = config.get("description", "")
        index_name = conn_config.get("index_name", "")

        # Store tool_id for potential future use in logging/tracking
        _tool_id = tool_id  # noqa: F841

        class FilesystemSearchInput(BaseModel):
            query: str = Field(
                description="Natural language search query to find relevant documents/files"
            )
            max_results: int = Field(
                default=10,
                ge=1,
                le=50,
                description="Maximum number of results to return",
            )

        async def search_filesystem(query: str, max_results: int = 10, **_: Any) -> str:
            """Search the filesystem index."""
            logger.info(f"[{tool_name}] Filesystem search: {query[:100]}...")
            return await search_filesystem_index(
                query=query,
                index_name=index_name,
                max_results=max_results,
            )

        tool_description = (
            f"Search indexed documents from {config.get('name', 'filesystem')}."
        )
        if description:
            tool_description += f" {description}"
        if index_name:
            tool_description += f" (Index: {index_name})"

        return StructuredTool.from_function(
            coroutine=search_filesystem,
            name=f"search_{tool_name}",
            description=tool_description,
            args_schema=FilesystemSearchInput,
        )

    def get_context_from_retrievers(
        self, query: str, max_docs: int = 5
    ) -> tuple[str, list[dict]]:
        """
        Retrieve relevant context from all FAISS indexes.

        Args:
            query: The search query.
            max_docs: Maximum documents per index.

        Returns:
            Tuple of (combined context string, list of source metadata).
        """
        all_docs = []
        sources = []
        for name, retriever in self.retrievers.items():
            try:
                docs = retriever.invoke(query)
                for doc in docs[:max_docs]:
                    source = doc.metadata.get("source", "unknown")
                    all_docs.append(f"[{name}:{source}]\n{doc.page_content}")
                    sources.append(
                        {
                            "index": name,
                            "source": source,
                            "preview": (
                                doc.page_content[:200] + "..."
                                if len(doc.page_content) > 200
                                else doc.page_content
                            ),
                        }
                    )
            except Exception as e:
                logger.warning(f"Error retrieving from {name}: {e}")

        context = "\n\n---\n\n".join(all_docs) if all_docs else ""
        return context, sources

    def _build_augmented_input(self, user_message: str) -> str:
        """Build the input for the agent.

        The agent has access to search_knowledge tool to search documentation
        on-demand, so we just pass through the user message.

        Returns:
            The user message (unmodified).
        """
        return user_message

    async def process_query(
        self, user_message: str, chat_history: Optional[List[Any]] = None
    ) -> str:
        """
        Process a user query through the RAG pipeline (non-streaming).

        Args:
            user_message: The user's question or request.
            chat_history: Previous messages in the conversation.

        Returns:
            The assistant's response.
        """
        if chat_history is None:
            chat_history = []

        augmented_input = self._build_augmented_input(user_message)

        try:
            if self.agent_executor:
                # Use agent with tools
                result = await self.agent_executor.ainvoke(
                    {"input": augmented_input, "chat_history": chat_history}
                )
                output = result.get("output", "I couldn't generate a response.")
                # Handle Anthropic-style content blocks (list of dicts with 'text' key)
                if isinstance(output, list):
                    output = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in output
                    )
                return output
            else:
                # Direct LLM call without tools
                if self.llm is None:
                    return (
                        "Error: No LLM configured. Please configure an LLM in Settings."
                    )

                messages: List[BaseMessage] = [
                    SystemMessage(content=self._system_prompt)
                ]
                messages.extend(chat_history)
                messages.append(HumanMessage(content=augmented_input))
                response = await self.llm.ainvoke(messages)
                content = response.content
                return content if isinstance(content, str) else str(content)

        except Exception as e:
            logger.exception("Error processing query")
            return f"I encountered an error processing your request: {str(e)}"

    async def process_query_stream(
        self, user_message: str, chat_history: Optional[List[Any]] = None
    ):
        """
        Process a user query with true token-by-token streaming.

        For agent with tools: executes tool calls first, then streams the final response.
        For direct LLM: streams tokens directly from the LLM.

        Yields:
            dict or str: Structured events for tool calls, or text tokens for content.
            - Tool start: {"type": "tool_start", "tool": "tool_name", "input": {...}}
            - Tool end: {"type": "tool_end", "tool": "tool_name", "output": "..."}
            - Content: str (individual tokens/chunks)
            - Max iterations: {"type": "max_iterations_reached"}
        """
        if chat_history is None:
            chat_history = []

        # Agent will use search_knowledge tool on-demand
        augmented_input = self._build_augmented_input(user_message)

        try:
            if self.agent_executor:
                # Agent with tools: use astream_events for true streaming
                # This streams tool calls and final response tokens
                # Track tool runs to avoid duplicates from nested events
                active_tool_runs: set[str] = set()

                async for event in self.agent_executor.astream_events(
                    {"input": augmented_input, "chat_history": chat_history},
                    version="v2",
                ):
                    kind = event.get("event", "")
                    run_id = event.get("run_id", "")

                    # Emit tool start events - only for new tool runs
                    if kind == "on_tool_start":
                        # Skip if we've already seen this tool run
                        if run_id in active_tool_runs:
                            continue
                        active_tool_runs.add(run_id)

                        tool_name = event.get("name", "unknown")
                        tool_input = event.get("data", {}).get("input", {})
                        yield {
                            "type": "tool_start",
                            "tool": tool_name,
                            "input": tool_input,
                            "run_id": run_id,  # Include run_id for matching with tool_end
                        }

                    # Emit tool end events - only for runs we started
                    elif kind == "on_tool_end":
                        # Skip if we didn't track this tool run starting
                        if run_id not in active_tool_runs:
                            continue
                        active_tool_runs.discard(run_id)

                        tool_name = event.get("name", "unknown")
                        tool_output = event.get("data", {}).get("output", "")
                        # Truncate very long outputs for display
                        if isinstance(tool_output, str) and len(tool_output) > 2000:
                            tool_output = tool_output[:2000] + "... (truncated)"
                        yield {
                            "type": "tool_end",
                            "tool": tool_name,
                            "output": tool_output,
                            "run_id": run_id,  # Include run_id for matching with tool_start
                        }

                    # Stream tokens from the chat model
                    elif kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            content = chunk.content
                            # Handle Anthropic-style content blocks (list of dicts with 'text' key)
                            if isinstance(content, list):
                                content = "".join(
                                    (
                                        block.get("text", "")
                                        if isinstance(block, dict)
                                        else str(block)
                                    )
                                    for block in content
                                )
                            if content:
                                yield content

                    # Detect when agent executor finishes - check for max iterations
                    elif kind == "on_chain_end":
                        # Check if this is the AgentExecutor finishing
                        output = event.get("data", {}).get("output", {})

                        # AgentExecutor sets "Agent stopped due to iteration limit" in output
                        if isinstance(output, dict):
                            agent_output = output.get("output", "")
                            if (
                                "iteration limit" in str(agent_output).lower()
                                or "max iterations" in str(agent_output).lower()
                            ):
                                yield {"type": "max_iterations_reached"}
                            # Also check return_values for the same message
                            return_values = output.get("return_values", {})
                            if isinstance(return_values, dict):
                                rv_output = return_values.get("output", "")
                                if "iteration limit" in str(rv_output).lower():
                                    yield {"type": "max_iterations_reached"}
            else:
                # Direct LLM streaming without tools
                if self.llm is None:
                    yield "Error: No LLM configured. Please configure an LLM in Settings."
                    return

                messages: List[BaseMessage] = [
                    SystemMessage(content=self._system_prompt)
                ]
                messages.extend(chat_history)
                messages.append(HumanMessage(content=augmented_input))

                # Use astream for true token-by-token streaming
                async for chunk in self.llm.astream(messages):
                    if hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        # Handle Anthropic-style content blocks (list of dicts with 'text' key)
                        if isinstance(content, list):
                            content = "".join(
                                (
                                    block.get("text", "")
                                    if isinstance(block, dict)
                                    else str(block)
                                )
                                for block in content
                            )
                        if content:
                            yield content

        except Exception as e:
            logger.exception("Error in streaming query")
            yield f"I encountered an error processing your request: {str(e)}"


# Global RAG components instance
rag = RAGComponents()
