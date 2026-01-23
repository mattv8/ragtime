"""
RAG Components - FAISS Vector Store and LangChain Agent setup.
"""

import asyncio
import os
import re
import resource
import subprocess
import time
from pathlib import Path
from typing import Any, List, Optional, Union

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from ragtime.config import settings
from ragtime.core.app_settings import get_app_settings, get_tool_configs
from ragtime.core.logging import get_logger
from ragtime.core.security import (
    _SSH_ENV_VAR_RE,
    sanitize_output,
    validate_odoo_code,
    validate_sql_query,
    validate_ssh_command,
)
from ragtime.core.sql_utils import add_table_metadata_to_psql_output
from ragtime.core.ssh import (
    SSHConfig,
    build_ssh_tunnel_config,
    execute_ssh_command,
    expand_env_vars_via_ssh,
    ssh_tunnel_config_from_dict,
)
from ragtime.core.tokenization import truncate_to_token_budget
from ragtime.indexer.pdm_service import pdm_indexer, search_pdm_index
from ragtime.indexer.repository import repository
from ragtime.indexer.schema_service import schema_indexer, search_schema_index
from ragtime.tools import get_all_tools, get_enabled_tools
from ragtime.tools.chart import create_chart_tool
from ragtime.tools.datatable import create_datatable_tool
from ragtime.tools.filesystem_indexer import search_filesystem_index
from ragtime.tools.git_history import (
    _is_shallow_repository,
    create_aggregate_git_history_tool,
    create_per_index_git_history_tool,
)
from ragtime.tools.mssql import create_mssql_tool
from ragtime.tools.mysql import create_mysql_tool
from ragtime.tools.odoo_shell import filter_odoo_output

logger = get_logger(__name__)


def get_process_memory_bytes() -> int:
    """Get current process RSS memory in bytes."""
    try:
        # Try reading from /proc for more accurate Linux stats
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # VmRSS is in kB
                    return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, IndexError):
        pass

    # Fallback to resource module (less accurate on some systems)
    try:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KB on Linux, bytes on macOS
        if os.uname().sysname == "Darwin":
            return rusage.ru_maxrss
        return rusage.ru_maxrss * 1024
    except Exception:
        return 0


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


# Additional system prompt section for UI-only features (charts, tables)
# This is appended only for requests from the chat UI, not API/MCP
UI_SYSTEM_PROMPT_ADDITION = """

DATA VISUALIZATION:

You have access to visualization tools that render rich, interactive displays:

**create_chart** - Create Chart.js visualizations
**create_datatable** - Create interactive DataTables with sorting/searching

RULES:
1. NEVER write markdown tables in your response text. ALWAYS use create_datatable instead.
2. Proactively use these tools AFTER you retrieve data. Don't wait to be asked.
3. You can (and should) use BOTH tools together - a chart for visualization AND a datatable for the raw data.
4. You MUST pass the actual data values to these tools - they do not have access to previous query results.

CRITICAL: When calling create_datatable, you must include the 'data' parameter with the actual row values
from your query results. The tool cannot see your previous outputs - you must explicitly pass the data array.

AUTO-VISUALIZE when:
- Query returns numeric data that can be compared -> USE create_chart (bar/line)
- Query returns counts, totals, or aggregations -> USE create_chart + create_datatable
- Query returns percentages or proportions -> USE create_chart (pie/doughnut)
- Query returns tabular data -> USE create_datatable (NEVER markdown tables)
- Data has trends over time -> USE create_chart (line)
- Comparing categories or groups -> USE create_chart (bar)

Chart type selection:
- Bar: Category comparisons (sales by region, counts by status, totals by type)
- Line: Time series, trends, sequential data (daily counts, monthly growth)
- Pie/Doughnut: Parts of whole, market share, distribution (<7 segments)

The tools accept full Chart.js/DataTables configuration objects, giving you
complete control over styling, axes, legends, tooltips, and interactivity.

REMEMBER: create_datatable renders beautiful interactive tables. Markdown tables are ugly.
"""


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
    """Container for RAG components initialized at startup.

    Supports progressive initialization with background FAISS loading:
    - core_ready: LLM and settings loaded, can serve non-indexed queries
    - indexes_ready: All FAISS indexes loaded, full knowledge search available
    - is_ready: Alias for core_ready (API can serve requests)

    UI vs API/MCP:
    - agent_executor: Standard agent for API/MCP requests
    - agent_executor_ui: Agent with UI-only tools (charts) for chat UI requests
    """

    def __init__(self):
        self.retrievers: dict[str, Any] = {}
        self.faiss_dbs: dict[str, Any] = (
            {}
        )  # Raw FAISS vectorstores for dynamic k searches
        self.agent_executor: Optional[AgentExecutor] = None
        self.agent_executor_ui: Optional[AgentExecutor] = (
            None  # UI-only agent with chart tool
        )
        self.llm: Optional[Any] = None  # ChatOpenAI, ChatAnthropic, or ChatOllama
        self._core_ready: bool = False  # LLM/settings ready
        self._indexes_ready: bool = False  # All FAISS indexes loaded
        self._indexes_loading: bool = False  # Background loading in progress
        self._indexes_total: int = 0  # Total indexes to load
        self._indexes_loaded: int = 0  # Indexes loaded so far
        self._app_settings: Optional[dict] = None
        self._tool_configs: Optional[List[dict]] = None
        self._index_metadata: Optional[List[dict]] = None
        self._system_prompt: str = BASE_SYSTEM_PROMPT
        self._system_prompt_ui: str = BASE_SYSTEM_PROMPT  # Includes UI additions
        self._init_lock: asyncio.Lock = asyncio.Lock()
        self._init_in_progress: bool = False
        self._embedding_model: Optional[Any] = None  # Cached for background loading
        # Detailed index loading tracking
        self._index_details: dict[str, dict] = (
            {}
        )  # name -> {status, size_mb, chunk_count, load_time, error}
        self._loading_index: Optional[str] = None  # Currently loading index name

    @property
    def is_ready(self) -> bool:
        """API can serve requests when core is ready."""
        return self._core_ready

    @is_ready.setter
    def is_ready(self, value: bool):
        """Setter for backwards compatibility."""
        self._core_ready = value

    @property
    def indexes_ready(self) -> bool:
        """All FAISS indexes are loaded."""
        return self._indexes_ready

    @property
    def loading_status(self) -> dict:
        """Get current loading status for health endpoint."""
        return {
            "core_ready": self._core_ready,
            "indexes_ready": self._indexes_ready,
            "indexes_loading": self._indexes_loading,
            "indexes_total": self._indexes_total,
            "indexes_loaded": self._indexes_loaded,
            "retrievers_available": list(self.retrievers.keys()),
            "index_details": list(self._index_details.values()),
            "loading_index": self._loading_index,
            "sequential_loading": (
                self._app_settings.get("sequential_index_loading", False)
                if self._app_settings
                else False
            ),
        }

    def unload_index(self, name: str) -> bool:
        """Remove an index from memory.

        Args:
            name: Index name to unload

        Returns:
            True if the index was unloaded, False if not found
        """
        unloaded = False
        if name in self.retrievers:
            del self.retrievers[name]
            logger.info(f"Unloaded index '{name}' from retrievers")
            unloaded = True

        if name in self.faiss_dbs:
            del self.faiss_dbs[name]
            logger.debug(f"Removed index '{name}' from faiss_dbs")

        if name in self._index_details:
            del self._index_details[name]
            logger.debug(f"Removed index '{name}' from index_details tracking")

        return unloaded

    def rename_index(self, old_name: str, new_name: str) -> bool:
        """Rename an index in memory (reuse loaded FAISS data).

        Args:
            old_name: Current index name
            new_name: New index name

        Returns:
            True if the index was renamed, False if not found
        """
        if old_name not in self.retrievers:
            return False

        # Move retriever to new key
        self.retrievers[new_name] = self.retrievers.pop(old_name)
        logger.info(f"Renamed index '{old_name}' to '{new_name}' in retrievers")

        # Also move faiss_dbs
        if old_name in self.faiss_dbs:
            self.faiss_dbs[new_name] = self.faiss_dbs.pop(old_name)
            logger.debug(f"Renamed index in faiss_dbs")

        # Also move index_details tracking
        if old_name in self._index_details:
            self._index_details[new_name] = self._index_details.pop(old_name)
            self._index_details[new_name]["name"] = new_name
            logger.debug(f"Renamed index in index_details tracking")

        return True

    async def rebuild_agent(self) -> None:
        """Rebuild the agent with current tools and retrievers.

        Use this instead of full initialize() when only tool/index changes
        need to be reflected without reloading all indexes from disk.
        """
        await self._create_agent()

    async def initialize(self):
        """Initialize all RAG components.

        Uses a lock to prevent concurrent initializations - if another init
        is in progress, this call will wait for it to complete rather than
        starting a duplicate initialization.
        """
        # Fast path: if init is already in progress, just wait for it
        if self._init_in_progress:
            logger.debug("RAG initialization already in progress, waiting...")
            async with self._init_lock:
                # Lock acquired means the other init finished
                logger.debug("RAG initialization completed by another caller")
                return

        async with self._init_lock:
            # Double-check after acquiring lock
            if self._init_in_progress:
                return

            self._init_in_progress = True
            try:
                await self._do_initialize()
            finally:
                self._init_in_progress = False

    async def _do_initialize(self):
        """Perform the actual RAG initialization.

        Uses a two-phase approach:
        1. Core init (blocking): Load settings, LLM, tools - marks core_ready
        2. Index loading (background): Load FAISS indexes in parallel - marks indexes_ready

        This allows the API to serve requests immediately while indexes load.
        """
        start_time = time.time()
        logger.info("Initializing RAG components (core)...")

        # Load settings from database
        self._app_settings = await get_app_settings()
        self._tool_configs = await get_tool_configs()
        self._index_metadata = await self._load_index_metadata()

        # Build system prompts with tool and index descriptions
        tool_prompt_section = build_tool_system_prompt(self._tool_configs)
        index_prompt_section = build_index_system_prompt(self._index_metadata)

        # Base system prompt (for API/MCP)
        self._system_prompt = (
            BASE_SYSTEM_PROMPT + index_prompt_section + tool_prompt_section
        )

        # UI system prompt (includes chart visualization instructions)
        self._system_prompt_ui = (
            BASE_SYSTEM_PROMPT
            + index_prompt_section
            + tool_prompt_section
            + UI_SYSTEM_PROMPT_ADDITION
        )

        # Initialize LLM based on provider from database settings
        await self._init_llm()

        # Load embedding model (needed for FAISS loading)
        self._embedding_model = await self._get_embedding_model()

        # Create the agent with tools (without FAISS retrievers for now)
        # This allows non-indexed queries to work immediately
        await self._create_agent()

        # Mark core as ready - API can now serve requests
        self._core_ready = True
        core_time = time.time() - start_time
        logger.info(
            f"RAG core initialized in {core_time:.1f}s - API ready (indexes loading in background)"
        )

        # Start background FAISS loading
        asyncio.create_task(self._load_faiss_indexes_background())

    async def _load_faiss_indexes_background(self):
        """Load FAISS indexes in background.

        This runs after core init completes, loading indexes without blocking
        the API. Supports both parallel loading (faster, higher peak RAM) and
        sequential loading (slower, lower peak RAM - loads smallest first).
        """
        start_time = time.time()

        if not self._embedding_model:
            logger.warning("No embedding model available for FAISS loading")
            self._indexes_ready = True
            return

        try:
            self._indexes_loading = True
            sequential = self._app_settings.get("sequential_index_loading", False)
            if sequential:
                await self._load_faiss_indexes_sequential(self._embedding_model)
            else:
                await self._load_faiss_indexes_parallel(self._embedding_model)
        except Exception as e:
            logger.error(f"Error in background FAISS loading: {e}")
        finally:
            self._indexes_loading = False
            self._indexes_ready = True

            # Rebuild agent with updated retrievers
            try:
                await self._create_agent()
            except Exception as e:
                logger.error(f"Failed to rebuild agent after index loading: {e}")

            elapsed = time.time() - start_time
            logger.info(
                f"FAISS indexes loaded in background ({elapsed:.1f}s): "
                f"{len(self.retrievers)} index(es) ready"
            )

    async def _init_llm(self):
        """Initialize LLM based on database settings."""
        assert self._app_settings is not None  # Set by initialize()
        provider = self._app_settings.get("llm_provider", "openai").lower()
        model = self._app_settings.get("llm_model", "gpt-4-turbo")

        if provider == "ollama":
            try:
                # Use LLM-specific Ollama settings if available, otherwise fall back to embedding settings
                base_url = self._app_settings.get(
                    "llm_ollama_base_url",
                    self._app_settings.get("ollama_base_url", "http://localhost:11434"),
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
                    self.llm = ChatAnthropic(
                        model=model,
                        temperature=0,
                        api_key=api_key,
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

        self.llm = ChatOpenAI(
            model=model if provider == "openai" else "gpt-4-turbo",
            temperature=0,
            streaming=True,
            api_key=api_key,
        )
        logger.info(f"Using OpenAI LLM: {self.llm.model_name}")

    async def _get_embedding_model(self):
        """Get embedding model based on database settings."""
        assert self._app_settings is not None  # Set by initialize()
        provider = self._app_settings.get("embedding_provider", "ollama").lower()
        model = self._app_settings.get("embedding_model", "nomic-embed-text")

        if provider == "ollama":
            base_url = self._app_settings.get(
                "ollama_base_url", "http://localhost:11434"
            )
            logger.info(f"Using Ollama embeddings: {model} at {base_url}")
            return OllamaEmbeddings(model=model, base_url=base_url)
        elif provider == "openai":
            api_key = self._app_settings.get("openai_api_key", "")
            if not api_key:
                logger.warning("OpenAI embeddings selected but no API key configured")
            logger.info(f"Using OpenAI embeddings: {model}")
            return OpenAIEmbeddings(model=model, openai_api_key=api_key)  # type: ignore[call-arg]
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    def _create_retriever_from_faiss(self, db: FAISS, index_name: str) -> Any:
        """Create a retriever from a FAISS vectorstore with appropriate settings.

        Supports MMR (Max Marginal Relevance) for result diversification when enabled.
        MMR reduces near-duplicate results by balancing relevance with diversity.

        Args:
            db: FAISS vectorstore instance
            index_name: Name of the index (for logging)

        Returns:
            A retriever configured with current settings
        """
        search_k = self._app_settings.get("search_results_k", 5)
        use_mmr = self._app_settings.get("search_use_mmr", True)
        mmr_lambda = self._app_settings.get("search_mmr_lambda", 0.5)

        if use_mmr:
            # MMR retriever: fetch_k gets more candidates, then MMR selects k diverse ones
            # fetch_k should be larger than k to give MMR choices to diversify from
            fetch_k = max(search_k * 4, 20)  # At least 4x k or 20 candidates
            retriever = db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": search_k,
                    "fetch_k": fetch_k,
                    "lambda_mult": mmr_lambda,  # 0=max diversity, 1=max relevance
                },
            )
            logger.debug(
                f"Created MMR retriever for {index_name} "
                f"(k={search_k}, fetch_k={fetch_k}, lambda={mmr_lambda})"
            )
        else:
            # Standard similarity retriever
            retriever = db.as_retriever(search_kwargs={"k": search_k})
            logger.debug(
                f"Created similarity retriever for {index_name} (k={search_k})"
            )

        return retriever

    async def _load_faiss_indexes(self, embedding_model):
        """Load FAISS indexes from database metadata (sequential, for backwards compat).

        Uses the index_metadata table to discover available indexes and loads
        only those that are enabled. The path is read directly from the database
        metadata, which was saved by the indexer service.

        For parallel loading, use _load_faiss_indexes_parallel instead.
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
                            # Create retriever with MMR support if enabled
                            self.retrievers[index_name] = (
                                self._create_retriever_from_faiss(db, index_name)
                            )
                            self.faiss_dbs[index_name] = (
                                db  # Store for dynamic k searches
                            )
                            logger.info(
                                f"Loaded FAISS index: {index_name} from {index_path}"
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

    async def _load_faiss_indexes_parallel(self, embedding_model):
        """Load FAISS indexes in parallel using asyncio.to_thread.

        This offloads the blocking FAISS.load_local calls to a thread pool,
        allowing multiple indexes to load concurrently and not blocking the
        event loop. Tracks memory usage for each index.
        """
        if not self._index_metadata:
            logger.info("No index metadata available for parallel loading")
            return

        enabled_indexes = [
            idx for idx in self._index_metadata if idx.get("enabled", True)
        ]

        if not enabled_indexes:
            logger.info("No enabled indexes to load")
            return

        self._indexes_total = len(enabled_indexes)
        self._indexes_loaded = 0

        # Get current embedding dimension by probing the embedding model
        # This is more reliable than tracked app_settings when provider changes
        current_embedding_dim = None
        try:
            test_embedding = await asyncio.to_thread(
                embedding_model.embed_query, "test"
            )
            current_embedding_dim = len(test_embedding)
            logger.info(
                f"Detected embedding dimension: {current_embedding_dim} "
                f"(will check indexes for mismatch)"
            )
        except Exception as e:
            # Fall back to tracked dimension if probe fails
            current_embedding_dim = self._app_settings.get("embedding_dimension")
            logger.warning(
                f"Could not probe embedding dimension: {e}. "
                f"Using tracked dimension: {current_embedding_dim}"
            )

        # Initialize index details for all indexes
        for idx in enabled_indexes:
            index_name = idx.get("name")
            if index_name:
                self._index_details[index_name] = {
                    "name": index_name,
                    "status": "pending",
                    "size_mb": (
                        idx.get("size_bytes", 0) / (1024 * 1024)
                        if idx.get("size_bytes")
                        else None
                    ),
                    "chunk_count": idx.get("chunk_count"),
                    "load_time_seconds": None,
                    "error": None,
                }

        async def load_single_index(
            idx: dict,
        ) -> tuple[str, Any, dict] | None:
            """Load a single FAISS index in a thread and measure memory."""
            index_name = idx.get("name")
            if not index_name:
                return None

            # Mark as loading
            if index_name in self._index_details:
                self._index_details[index_name]["status"] = "loading"

            index_path_str = idx.get("path")
            if not index_path_str:
                logger.warning(f"Index {index_name} has no path in metadata, skipping")
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "error"
                    self._index_details[index_name]["error"] = "No path in metadata"
                return None

            index_path = Path(index_path_str)
            if not index_path.exists():
                logger.warning(f"FAISS index path not found: {index_path}")
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "error"
                    self._index_details[index_name][
                        "error"
                    ] = f"Index files missing: {index_path} - re-index required"
                return None

            try:
                start = time.time()
                mem_before = get_process_memory_bytes()

                # Offload blocking I/O to thread pool
                db = await asyncio.to_thread(
                    FAISS.load_local,
                    str(index_path),
                    embedding_model,
                    allow_dangerous_deserialization=True,
                )

                elapsed = time.time() - start
                mem_after = get_process_memory_bytes()

                # Get embedding dimension from the loaded index
                embedding_dim = db.index.d if hasattr(db, "index") else None

                # Check for dimension mismatch before creating retriever
                if (
                    current_embedding_dim
                    and embedding_dim
                    and embedding_dim != current_embedding_dim
                ):
                    mismatch_msg = (
                        f"Embedding dimension mismatch: index has {embedding_dim} dims, "
                        f"but current model produces {current_embedding_dim} dims. "
                        f"Re-index required."
                    )
                    logger.warning(f"Index {index_name}: {mismatch_msg}")
                    if index_name in self._index_details:
                        self._index_details[index_name]["status"] = "error"
                        self._index_details[index_name]["error"] = mismatch_msg
                        self._index_details[index_name][
                            "embedding_dimension"
                        ] = embedding_dim
                    # Return None to skip adding this retriever
                    return None

                # Memory used by this index (approximate - may include GC overhead)
                steady_mem = max(0, mem_after - mem_before)

                # Create retriever with MMR support if enabledfrom ragtime.core.ssh import
                retriever = self._create_retriever_from_faiss(db, index_name)
                logger.info(
                    f"Loaded FAISS index: {index_name} from {index_path} "
                    f"({elapsed:.1f}s, ~{steady_mem / 1024**3:.2f}GB)"
                )

                # Update index detail
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "loaded"
                    self._index_details[index_name]["load_time_seconds"] = elapsed

                memory_stats = {
                    "embedding_dimension": embedding_dim,
                    "steady_memory_bytes": steady_mem,
                    "load_time_seconds": elapsed,
                }

                return (index_name, db, retriever, memory_stats)
            except Exception as e:
                logger.warning(f"Failed to load FAISS index {index_name}: {e}")
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "error"
                    self._index_details[index_name]["error"] = str(e)
                return None

        # Load all indexes in parallel
        logger.info(f"Loading {len(enabled_indexes)} FAISS index(es) in parallel...")
        results = await asyncio.gather(
            *[load_single_index(idx) for idx in enabled_indexes],
            return_exceptions=True,
        )

        # Process results and update memory stats in database
        for result in results:
            if isinstance(result, BaseException):
                logger.warning(f"Index loading exception: {result}")
            elif result is not None:
                index_name, db, retriever, memory_stats = result
                self.retrievers[index_name] = retriever
                self.faiss_dbs[index_name] = db  # Store for dynamic k searches
                self._indexes_loaded += 1

                # Update memory stats in database (best effort)
                try:
                    await repository.update_index_memory_stats(index_name, memory_stats)
                except Exception as e:
                    logger.debug(f"Failed to update memory stats for {index_name}: {e}")

        if self.retrievers:
            logger.info(
                f"Loaded {len(self.retrievers)} FAISS index(es) from database metadata"
            )

    async def _load_faiss_indexes_sequential(self, embedding_model):
        """Load FAISS indexes sequentially, smallest first.

        This reduces peak memory usage compared to parallel loading by loading
        one index at a time. Indexes are sorted by size (smallest first) so
        the system becomes partially functional faster.
        """

        if not self._index_metadata:
            logger.info("No index metadata available for sequential loading")
            return

        enabled_indexes = [
            idx for idx in self._index_metadata if idx.get("enabled", True)
        ]

        if not enabled_indexes:
            logger.info("No enabled indexes to load")
            return

        # Sort by size_bytes (smallest first) for faster initial availability
        enabled_indexes.sort(key=lambda x: x.get("size_bytes", 0))

        self._indexes_total = len(enabled_indexes)
        self._indexes_loaded = 0
        search_k = self._app_settings.get("search_results_k", 5)

        # Get current embedding dimension by probing the embedding model
        # This is more reliable than tracked app_settings when provider changes
        current_embedding_dim = None
        try:
            test_embedding = await asyncio.to_thread(
                embedding_model.embed_query, "test"
            )
            current_embedding_dim = len(test_embedding)
            logger.info(
                f"Detected embedding dimension: {current_embedding_dim} "
                f"(will check indexes for mismatch)"
            )
        except Exception as e:
            # Fall back to tracked dimension if probe fails
            current_embedding_dim = self._app_settings.get("embedding_dimension")
            logger.warning(
                f"Could not probe embedding dimension: {e}. "
                f"Using tracked dimension: {current_embedding_dim}"
            )

        # Initialize index details for all indexes
        for idx in enabled_indexes:
            index_name = idx.get("name")
            if index_name:
                self._index_details[index_name] = {
                    "name": index_name,
                    "status": "pending",
                    "size_mb": (
                        idx.get("size_bytes", 0) / (1024 * 1024)
                        if idx.get("size_bytes")
                        else None
                    ),
                    "chunk_count": idx.get("chunk_count"),
                    "load_time_seconds": None,
                    "error": None,
                }

        logger.info(
            f"Loading {len(enabled_indexes)} FAISS index(es) sequentially "
            "(smallest first)..."
        )

        for idx in enabled_indexes:
            index_name = idx.get("name")
            if not index_name:
                continue

            # Mark as currently loading
            self._loading_index = index_name
            if index_name in self._index_details:
                self._index_details[index_name]["status"] = "loading"

            index_path_str = idx.get("path")
            if not index_path_str:
                logger.warning(f"Index {index_name} has no path in metadata, skipping")
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "error"
                    self._index_details[index_name]["error"] = "No path in metadata"
                continue

            index_path = Path(index_path_str)
            if not index_path.exists():
                logger.warning(f"FAISS index path not found: {index_path}")
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "error"
                    self._index_details[index_name][
                        "error"
                    ] = f"Index files missing: {index_path} - re-index required"
                continue

            try:
                start = time.time()
                mem_before = get_process_memory_bytes()

                # Track peak memory during loading
                peak_mem = mem_before

                def load_and_track_peak():
                    nonlocal peak_mem
                    db = FAISS.load_local(
                        str(index_path),
                        embedding_model,
                        allow_dangerous_deserialization=True,
                    )
                    # Check memory after loading (this is approximate)
                    current_mem = get_process_memory_bytes()
                    peak_mem = max(peak_mem, current_mem)
                    return db

                db = await asyncio.to_thread(load_and_track_peak)

                elapsed = time.time() - start
                mem_after = get_process_memory_bytes()

                # Get embedding dimension from the loaded index
                embedding_dim = db.index.d if hasattr(db, "index") else None

                # Check for dimension mismatch before creating retriever
                if (
                    current_embedding_dim
                    and embedding_dim
                    and embedding_dim != current_embedding_dim
                ):
                    mismatch_msg = (
                        f"Embedding dimension mismatch: index has {embedding_dim} dims, "
                        f"but current model produces {current_embedding_dim} dims. "
                        f"Re-index required."
                    )
                    logger.warning(f"Index {index_name}: {mismatch_msg}")
                    if index_name in self._index_details:
                        self._index_details[index_name]["status"] = "error"
                        self._index_details[index_name]["error"] = mismatch_msg
                        self._index_details[index_name][
                            "embedding_dimension"
                        ] = embedding_dim
                    continue  # Skip this index

                # Calculate memory stats
                steady_mem = max(0, mem_after - mem_before)
                observed_peak = max(0, peak_mem - mem_before)

                # Create retriever with MMR support if enabled
                retriever = self._create_retriever_from_faiss(db, index_name)
                self.retrievers[index_name] = retriever
                self.faiss_dbs[index_name] = db  # Store for dynamic k searches
                self._indexes_loaded += 1

                # Update index detail
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "loaded"
                    self._index_details[index_name]["load_time_seconds"] = elapsed

                logger.info(
                    f"Loaded FAISS index: {index_name} from {index_path} "
                    f"(k={search_k}, {elapsed:.1f}s, ~{steady_mem / 1024**3:.2f}GB)"
                )

                # Update memory stats in database
                try:
                    await repository.update_index_memory_stats(
                        index_name,
                        {
                            "embedding_dimension": embedding_dim,
                            "steady_memory_bytes": steady_mem,
                            "peak_memory_bytes": observed_peak,
                            "load_time_seconds": elapsed,
                        },
                    )
                except Exception as e:
                    logger.debug(f"Failed to update memory stats for {index_name}: {e}")

            except Exception as e:
                logger.warning(f"Failed to load FAISS index {index_name}: {e}")
                if index_name in self._index_details:
                    self._index_details[index_name]["status"] = "error"
                    self._index_details[index_name]["error"] = str(e)

        # Clear loading index indicator
        self._loading_index = None

        if self.retrievers:
            logger.info(
                f"Loaded {len(self.retrievers)} FAISS index(es) from database metadata"
            )

    async def _load_index_metadata(self) -> list[dict]:
        """Load index metadata from database for system prompt."""
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
                    "size_bytes": m.sizeBytes,
                    "embedding_dimension": getattr(m, "embeddingDimension", None),
                    "steady_memory_bytes": getattr(m, "steadyMemoryBytes", None),
                    "peak_memory_bytes": getattr(m, "peakMemoryBytes", None),
                }
                for m in metadata_list
            ]
        except Exception as e:
            logger.warning(f"Failed to load index metadata: {e}")
            return []

    async def _create_agent(self):
        """Create the LangChain agents with tools.

        Creates two agents:
        - agent_executor: Standard agent for API/MCP requests
        - agent_executor_ui: Agent with UI-only tools (charts) for chat UI requests
        """
        assert self._app_settings is not None  # Set by initialize()
        # Skip agent creation if LLM is not configured
        if self.llm is None:
            logger.warning("Skipping agent creation - no LLM configured")
            self.agent_executor = None
            self.agent_executor_ui = None
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

        # Add git history search tool(s) if we have git repos
        git_history_tools = await self._create_git_history_tools()
        if git_history_tools:
            tools.extend(git_history_tools)

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

        # Respect admin-configured iteration limit; fall back to 15 if invalid
        try:
            max_iterations = int(self._app_settings.get("max_iterations", 15))
            if max_iterations < 1:
                max_iterations = 1
        except (TypeError, ValueError):
            max_iterations = 15

        # Create standard agent (for API/MCP)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        if tools:
            # Use create_tool_calling_agent which works with both OpenAI and Anthropic
            agent = create_tool_calling_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=settings.debug_mode,
                handle_parsing_errors=True,
                max_iterations=max_iterations,
                return_intermediate_steps=settings.debug_mode,
            )
        else:
            self.agent_executor = None

        # Create UI agent (with visualization tools and UI prompt)
        ui_tools = tools + [create_chart_tool, create_datatable_tool]

        prompt_ui = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_prompt_ui),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        if ui_tools:
            agent_ui = create_tool_calling_agent(self.llm, ui_tools, prompt_ui)
            self.agent_executor_ui = AgentExecutor(
                agent=agent_ui,
                tools=ui_tools,
                verbose=settings.debug_mode,
                handle_parsing_errors=True,
                max_iterations=max_iterations,
                return_intermediate_steps=settings.debug_mode,
            )
            logger.info("Created UI agent with chart and datatable tools")
        else:
            self.agent_executor_ui = None

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
            elif tool_type == "mysql":
                tool = await self._create_mysql_tool(config, tool_name, tool_id)
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
            elif tool_type == "solidworks_pdm":
                tool = await self._create_pdm_search_tool(config, tool_name, tool_id)
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
        conn_config = config.get("connection_config", {})

        # Check if schema indexing is enabled
        schema_index_enabled = conn_config.get("schema_index_enabled", False)
        logger.debug(
            f"Schema tool check for {tool_name}: enabled={schema_index_enabled}"
        )
        if not schema_index_enabled:
            return None

        # Check if there are any schema embeddings for this tool
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

        The agent can control:
        - k: Number of results to retrieve (default from settings, max 50)
        - max_chars_per_result: How much content to show per result (default 500, 0=unlimited)
        """
        # Build index_name description with available indexes
        index_names = list(self.retrievers.keys())
        index_name_desc = (
            "Optional: specific index to search (leave empty to search all indexes)"
        )
        if index_names:
            index_name_desc += f". Available indexes: {', '.join(index_names)}"

        # Get default k from settings
        default_k = (
            self._app_settings.get("search_results_k", 5) if self._app_settings else 5
        )
        use_mmr = (
            self._app_settings.get("search_use_mmr", True)
            if self._app_settings
            else True
        )
        mmr_lambda = (
            self._app_settings.get("search_mmr_lambda", 0.5)
            if self._app_settings
            else 0.5
        )

        class KnowledgeSearchInput(BaseModel):
            query: str = Field(
                description="Search query to find relevant documentation, code, or technical information"
            )
            index_name: str = Field(
                default="",
                description=index_name_desc,
            )
            k: int = Field(
                default=default_k,
                ge=1,
                le=50,
                description=f"Number of results to retrieve (default: {default_k}). Increase for broader searches, decrease for focused results.",
            )
            max_chars_per_result: int = Field(
                default=500,
                ge=0,
                le=10000,
                description="Maximum characters per result (default: 500). Use 0 for full content when you need complete code/file content. Increase when results are truncated.",
            )

        def search_knowledge(
            query: str,
            index_name: str = "",
            k: int = default_k,
            max_chars_per_result: int = 500,
        ) -> str:
            """Search indexed documentation for relevant information."""
            results = []
            errors = []

            # Clamp parameters
            k = max(1, min(50, k))
            max_chars_per_result = max(0, min(10000, max_chars_per_result))

            # Log the search attempt for debugging
            logger.debug(
                f"search_knowledge called with query='{query[:50]}...', index_name='{index_name}', k={k}, max_chars={max_chars_per_result}"
            )
            logger.debug(f"Available FAISS dbs: {list(self.faiss_dbs.keys())}")

            # Determine which indexes to search
            if index_name and index_name in self.faiss_dbs:
                dbs_to_search = {index_name: self.faiss_dbs[index_name]}
            else:
                dbs_to_search = self.faiss_dbs

            if not dbs_to_search:
                # Check if indexes are still loading
                if self._indexes_loading:
                    logger.info("Knowledge search called while indexes still loading")
                    return (
                        "Knowledge indexes are currently loading in the background. "
                        f"Progress: {self._indexes_loaded}/{self._indexes_total} loaded. "
                        "Please try again in a moment, or use other available tools."
                    )
                logger.warning("No FAISS dbs available for search_knowledge")
                return "No knowledge indexes are currently loaded. Please index some documents first."

            for name, db in dbs_to_search.items():
                try:
                    logger.debug(
                        f"Searching index '{name}' with query: {query[:50]}..., k={k}"
                    )
                    # Use MMR or similarity search based on settings
                    if use_mmr:
                        fetch_k = max(k * 4, 20)  # Get more candidates for diversity
                        docs = db.max_marginal_relevance_search(
                            query, k=k, fetch_k=fetch_k, lambda_mult=mmr_lambda
                        )
                    else:
                        docs = db.similarity_search(query, k=k)

                    logger.debug(f"Index '{name}' returned {len(docs)} documents")
                    for doc in docs:
                        source = doc.metadata.get("source", "unknown")
                        content = doc.page_content
                        # Apply truncation if max_chars_per_result > 0
                        if (
                            max_chars_per_result > 0
                            and len(content) > max_chars_per_result
                        ):
                            content = content[:max_chars_per_result] + "... (truncated)"
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
            "The query should describe what you're looking for. "
            "Use 'k' to control number of results (increase for broader searches). "
            "Use 'max_chars_per_result' to control content length (use 0 for full content when results are truncated)."
        )

        return StructuredTool.from_function(
            func=search_knowledge,
            name="search_knowledge",
            description=description,
            args_schema=KnowledgeSearchInput,
        )

    async def _create_git_history_tools(self) -> List[Any]:
        """Create git history search tool(s) for git-based indexes.

        When aggregate_search is enabled: creates a single search_git_history tool
        When aggregate_search is disabled: creates search_git_history_<name> per index
        """
        tools: List[Any] = []
        index_base = Path(settings.index_data_path)

        # Find all git repos and match them with index metadata
        # Only include repos where git_history_depth != 1 (shallow clone has no history)
        git_repos: List[tuple[str, Path, str]] = []  # (name, path, description)

        if index_base.exists():
            for index_dir in index_base.iterdir():
                if not index_dir.is_dir():
                    continue

                git_repo = index_dir / ".git_repo"
                if git_repo.exists() and (git_repo / ".git").exists():
                    # Get metadata including config_snapshot to check git_history_depth
                    description = ""
                    git_history_depth = 0  # Default to full history
                    for idx in self._index_metadata or []:
                        if idx.get("name") == index_dir.name:
                            description = idx.get("description", "")
                            # Get git_history_depth from config_snapshot
                            config = idx.get("config_snapshot") or {}
                            git_history_depth = config.get("git_history_depth", 0)
                            break

                    # Only expose git history tool if depth != 1
                    # depth=0 means full history, depth>1 means we have commits to search
                    # depth=1 is a shallow clone with only the latest commit (not useful)
                    if git_history_depth == 1:
                        logger.debug(
                            f"Skipping git history tool for {index_dir.name}: "
                            "shallow clone (depth=1 in config)"
                        )
                        continue

                    # Also check actual repo state - it may have minimal history
                    # even if config doesn't reflect this (e.g., cloned externally)
                    # Note: _is_shallow_repository checks commit count, not just
                    # whether it's technically shallow - depth > 1 is still useful
                    if await _is_shallow_repository(git_repo):
                        logger.debug(
                            f"Skipping git history tool for {index_dir.name}: "
                            "minimal commit history (1-2 commits)"
                        )
                        continue

                    git_repos.append((index_dir.name, git_repo, description))

        if not git_repos:
            return []

        aggregate_search = (self._app_settings or {}).get("aggregate_search", True)

        if aggregate_search:
            # Single tool for all git repos - include available repos in description
            repo_names = [name for name, _, _ in git_repos]
            tools.append(create_aggregate_git_history_tool(repo_names))
            logger.info(
                f"Added search_git_history tool for {len(git_repos)} repo(s): {repo_names}"
            )
        else:
            # Separate tool per repo
            for name, repo_path, description in git_repos:
                tool = create_per_index_git_history_tool(name, repo_path, description)
                tools.append(tool)
            logger.info(f"Added {len(tools)} per-index git history search tools")

        return tools

    def _create_per_index_search_tools(self) -> List[Any]:
        """Create separate search tools for each index.

        When aggregate_search is disabled, this creates search_<index_name>
        tools that give the AI granular control over which index to search.

        Supports dynamic k and max_chars_per_result parameters like the aggregate search.
        """
        tools = []

        # Get settings
        default_k = (
            self._app_settings.get("search_results_k", 5) if self._app_settings else 5
        )
        use_mmr = (
            self._app_settings.get("search_use_mmr", True)
            if self._app_settings
            else True
        )
        mmr_lambda = (
            self._app_settings.get("search_mmr_lambda", 0.5)
            if self._app_settings
            else 0.5
        )

        # Get index metadata for descriptions and weights
        index_weights = {}
        index_descriptions = {}
        for idx in self._index_metadata or []:
            if idx.get("enabled", True):
                name = idx.get("name", "")
                index_weights[name] = idx.get("search_weight", 1.0)
                index_descriptions[name] = idx.get("description", "")

        for index_name, db in self.faiss_dbs.items():
            # Create a closure to capture the current index_name and db
            def make_search_func(
                idx_name: str,
                idx_db,
                use_mmr_: bool,
                mmr_lambda_: float,
                default_k_: int,
            ):
                def search_index(
                    query: str,
                    k: int = default_k_,
                    max_chars_per_result: int = 500,
                ) -> str:
                    """Search this specific index for relevant information."""
                    results = []

                    # Clamp parameters
                    k = max(1, min(50, k))
                    max_chars_per_result = max(0, min(10000, max_chars_per_result))

                    logger.debug(
                        f"search_{idx_name} called with query='{query[:50]}...', k={k}, max_chars={max_chars_per_result}"
                    )

                    try:
                        # Use MMR or similarity search based on settings
                        if use_mmr_:
                            fetch_k = max(k * 4, 20)
                            docs = idx_db.max_marginal_relevance_search(
                                query, k=k, fetch_k=fetch_k, lambda_mult=mmr_lambda_
                            )
                        else:
                            docs = idx_db.similarity_search(query, k=k)

                        logger.debug(
                            f"Index '{idx_name}' returned {len(docs)} documents"
                        )
                        for doc in docs:
                            source = doc.metadata.get("source", "unknown")
                            content = doc.page_content
                            # Apply truncation if max_chars_per_result > 0
                            if (
                                max_chars_per_result > 0
                                and len(content) > max_chars_per_result
                            ):
                                content = (
                                    content[:max_chars_per_result] + "... (truncated)"
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

            # Create input schema for this tool with k and max_chars_per_result
            class IndexSearchInput(BaseModel):
                query: str = Field(
                    description="Search query to find relevant documentation or code"
                )
                k: int = Field(
                    default=default_k,
                    ge=1,
                    le=50,
                    description=f"Number of results (default: {default_k}). Increase for broader searches.",
                )
                max_chars_per_result: int = Field(
                    default=500,
                    ge=0,
                    le=10000,
                    description="Max chars per result (default: 500). Use 0 for full content.",
                )

            # Build description including the index description and weight hint
            weight = index_weights.get(index_name, 1.0)
            idx_desc = index_descriptions.get(index_name, "")

            tool_description = (
                f"Search the '{index_name}' index for relevant information. "
                "Use 'k' for result count, 'max_chars_per_result' for content length (0=full)."
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
                func=make_search_func(index_name, db, use_mmr, mmr_lambda, default_k),
                name=tool_name,
                description=tool_description,
                args_schema=IndexSearchInput,
            )
            tools.append(tool)

        return tools

    async def _create_postgres_tool(self, config: dict, tool_name: str, _tool_id: str):
        """Create a PostgreSQL query tool from config."""
        conn_config = config.get("connection_config", {})
        timeout = config.get("timeout", 30)
        allow_write = config.get("allow_write", False)
        description = config.get("description", "")

        host = conn_config.get("host", "")
        port = conn_config.get("port", 5432)

        # Build SSH tunnel config if enabled
        ssh_tunnel_config = build_ssh_tunnel_config(conn_config, host, port)

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
            user = conn_config.get("user", "")
            password = conn_config.get("password", "")
            database = conn_config.get("database", "")
            container = conn_config.get("container", "")

            # SSH tunnel mode uses psycopg2
            if ssh_tunnel_config:

                def run_tunnel_query() -> str:
                    try:
                        import psycopg2  # type: ignore[import-untyped]
                        import psycopg2.extras  # type: ignore[import-untyped]
                    except ImportError:
                        return "Error: psycopg2 package not installed. Install with: pip install psycopg2-binary"

                    tunnel: SSHTunnel | None = None
                    conn = None
                    try:
                        if ssh_tunnel_config is None:
                            return "Error: SSH tunnel configuration is missing"
                        tunnel_cfg = ssh_tunnel_config_from_dict(
                            ssh_tunnel_config, default_remote_port=5432
                        )
                        if not tunnel_cfg:
                            return "Error: Invalid SSH tunnel configuration"

                        tunnel = SSHTunnel(tunnel_cfg)
                        local_port = tunnel.start()

                        conn = psycopg2.connect(
                            host="127.0.0.1",
                            port=local_port,
                            user=user,
                            password=password,
                            dbname=database,
                            connect_timeout=timeout,
                        )
                        cursor = conn.cursor(
                            cursor_factory=psycopg2.extras.RealDictCursor
                        )
                        cursor.execute(query)

                        if cursor.description:
                            rows = cursor.fetchall()
                            if not rows:
                                return "Query executed successfully (no results)"
                            # Format as psql-like output
                            columns = [col.name for col in cursor.description]
                            lines = []
                            lines.append(" | ".join(columns))
                            lines.append("-+-".join(["-" * len(c) for c in columns]))
                            for row in rows:
                                lines.append(
                                    " | ".join(str(row.get(c, "")) for c in columns)
                                )
                            lines.append(
                                f"({len(rows)} row{'s' if len(rows) != 1 else ''})"
                            )
                            output = "\n".join(lines)
                            output = add_table_metadata_to_psql_output(output)
                            return sanitize_output(output)
                        else:
                            return "Query executed successfully (no results)"

                    except Exception as e:
                        return f"Error: {str(e)}"
                    finally:
                        if conn:
                            try:
                                conn.close()
                            except Exception:
                                pass
                        if tunnel:
                            try:
                                tunnel.stop()
                            except Exception:
                                pass

                try:
                    return await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, run_tunnel_query
                        ),
                        timeout=timeout + 5,
                    )
                except asyncio.TimeoutError:
                    return f"Error: Query timed out after {timeout}s"

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
                if not output:
                    return "Query executed successfully (no results)"

                # Add table metadata for UI rendering BEFORE sanitizing
                # so metadata is extracted from complete data
                output = add_table_metadata_to_psql_output(output)

                # Now sanitize the combined output
                output = sanitize_output(output)
                return output

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

        # Build SSH tunnel config if enabled
        ssh_tunnel_config = build_ssh_tunnel_config(conn_config, host, port)
        if not ssh_tunnel_config and not host:
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
            ssh_tunnel_config=ssh_tunnel_config,
        )

    async def _create_mysql_tool(self, config: dict, tool_name: str, _tool_id: str):
        """Create a MySQL/MariaDB query tool from config."""

        conn_config = config.get("connection_config", {})
        timeout = config.get("timeout", 30)
        max_results = config.get("max_results", 100)
        allow_write = config.get("allow_write", False)
        description = config.get("description", "")

        host = conn_config.get("host", "")
        port = conn_config.get("port", 3306)
        user = conn_config.get("user", "")
        password = conn_config.get("password", "")
        database = conn_config.get("database", "")

        # Build SSH tunnel config if enabled
        ssh_tunnel_config = build_ssh_tunnel_config(conn_config, host, port)
        if not ssh_tunnel_config and not host:
            logger.error(f"MySQL tool {tool_name} missing host configuration")
            return None

        return create_mysql_tool(
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
            ssh_tunnel_config=ssh_tunnel_config,
        )

    async def _create_odoo_tool(self, config: dict, tool_name: str, _tool_id: str):
        """Create an Odoo shell tool from config (Docker or SSH mode)."""
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
        conn_config = config.get("connection_config", {})
        timeout = config.get("timeout", 30)
        allow_write = config.get("allow_write", False)
        description = config.get("description", "")
        working_directory = conn_config.get("working_directory", "")

        class SSHInput(BaseModel):
            command: str = Field(
                default="", description="Shell command to execute on the remote server"
            )
            reason: str = Field(
                default="", description="Brief description of what this command does"
            )

        async def execute_ssh(command: str = "", reason: str = "", **_: Any) -> str:
            """Execute SSH command using this tool's configuration."""
            # Validate required fields
            if not command or not command.strip():
                return "Error: 'command' parameter is required. Provide a shell command to execute."
            if not reason:
                reason = "SSH command"

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

            # Build SSH config for potential env var expansion
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

            # If working_directory is set, expand env vars for path validation
            expanded_command = None
            if working_directory:
                # Check if command contains env vars that need expansion (using precompiled pattern)
                if _SSH_ENV_VAR_RE.search(command):
                    # Expand env vars on the remote host
                    loop = asyncio.get_event_loop()
                    expanded_command, expand_error = await loop.run_in_executor(
                        None, lambda: expand_env_vars_via_ssh(ssh_config, command)
                    )
                    if expand_error:
                        return f"Error: {expand_error}"

            # Validate command for dangerous patterns
            is_safe, validation_reason = validate_ssh_command(
                command,
                allow_write=allow_write,
                allowed_directory=working_directory or None,
                expanded_command=expanded_command,
            )
            if not is_safe:
                return f"Error: {validation_reason}"

            logger.info(f"[{tool_name}] SSH: {reason}")

            full_command = f"{command_prefix}{command}"

            try:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: execute_ssh_command(ssh_config, full_command)
                )

                if not result.success:
                    return f"Error (exit {result.exit_code}): {result.stderr or result.stdout}"

                output = result.output.strip()
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
        if not allow_write:
            tool_description += " Read-only mode: write operations are blocked."

        return StructuredTool.from_function(
            coroutine=execute_ssh,
            name=f"ssh_{tool_name}",
            description=tool_description,
            args_schema=SSHInput,
        )

    async def _create_filesystem_tool(self, config: dict, tool_name: str, tool_id: str):
        """Create a filesystem search tool from config."""
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

    async def _create_pdm_search_tool(self, config: dict, tool_name: str, tool_id: str):
        """Create a SolidWorks PDM search tool from config."""
        description = config.get("description", "")

        # Use tool_name for index lookup - matches how trigger_index creates it
        index_name = f"pdm_{tool_name}"

        # Store tool_id for potential future use in logging/tracking
        _tool_id = tool_id  # noqa: F841

        # Check if there are any PDM embeddings for this tool
        embedding_count = await pdm_indexer.get_embedding_count(tool_id, tool_name)
        if embedding_count == 0:
            logger.debug(f"PDM tool configured but no embeddings found for {tool_name}")
            # Still create the tool - it will just return "no results" until indexed

        class PdmSearchInput(BaseModel):
            query: str = Field(
                description=(
                    "Natural language search query to find PDM documents. "
                    "Search for parts, assemblies, drawings by part number, "
                    "material, description, author, folder, or BOM relationships."
                )
            )
            document_type: Optional[str] = Field(
                default=None,
                description=(
                    "Optional filter by document type: SLDPRT (parts), "
                    "SLDASM (assemblies), SLDDRW (drawings), or None for all"
                ),
            )

        async def search_pdm(
            query: str, document_type: Optional[str] = None, **_: Any
        ) -> str:
            """Search the PDM index."""
            logger.info(f"[{tool_name}] PDM search: {query[:100]}...")
            return await search_pdm_index(
                query=query,
                index_name=index_name,
                document_type=document_type,
                max_results=10,
            )

        tool_description = (
            f"Search SolidWorks PDM metadata from {config.get('name', 'PDM vault')}. "
            f"Find parts, assemblies, and drawings by part number, material, "
            f"description, author, folder path, or BOM relationships."
        )
        if description:
            tool_description += f" Database contains: {description}"

        logger.info(f"Created PDM search tool: search_{tool_name}")
        return StructuredTool.from_function(
            coroutine=search_pdm,
            name=f"search_{tool_name}",
            description=tool_description,
            args_schema=PdmSearchInput,
        )

    def get_context_from_retrievers(
        self, query: str, max_docs: int = 5
    ) -> tuple[str, list[dict]]:
        """
        Retrieve relevant context from all FAISS indexes.

        Applies token budget from settings to prevent context overflow.
        When token budget is exceeded, earlier retrieved chunks take priority.

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

        if not all_docs:
            return "", sources

        # Apply token budget if configured (0 = unlimited)
        token_budget = (
            self._app_settings.get("context_token_budget", 4000)
            if self._app_settings
            else 4000
        )

        if token_budget > 0:
            context, actual_tokens = truncate_to_token_budget(
                all_docs, max_tokens=token_budget
            )
            if actual_tokens >= token_budget:
                logger.debug(
                    f"Context truncated to {actual_tokens} tokens (budget: {token_budget})"
                )
        else:
            context = "\n\n---\n\n".join(all_docs)

        return context, sources

    def _build_augmented_input(self, user_message: str) -> str:
        """Build the input for the agent.

        The agent has access to search_knowledge tool to search documentation
        on-demand, so we just pass through the user message.

        Returns:
            The user message (unmodified).
        """
        return user_message

    def _convert_message_to_langchain(self, message: Any) -> Any:
        """
        Convert a Message object (from schemas.py) to LangChain HumanMessage format.

        Handles both string content and multimodal content (text + images).
        LangChain expects multimodal content in this format:
        [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "..."}}
        ]

        Args:
            message: Either a string or Message object with content attribute

        Returns:
            str or list suitable for LangChain HumanMessage content
        """
        # If already a string, return as-is
        if isinstance(message, str):
            return message

        # Get content from Message object
        content = getattr(message, "content", message)

        # If content is a string, return it
        if isinstance(content, str):
            return content

        # If content is a list (multimodal), convert to LangChain format
        if isinstance(content, list):
            langchain_content = []
            for item in content:
                if isinstance(item, dict):
                    # Already in dict format
                    if item.get("type") == "text":
                        langchain_content.append(
                            {"type": "text", "text": item.get("text", "")}
                        )
                    elif item.get("type") == "image_url":
                        langchain_content.append(item)  # Pass through
                elif hasattr(item, "type"):
                    # Pydantic model
                    if item.type == "text":
                        langchain_content.append({"type": "text", "text": item.text})
                    elif item.type == "image_url":
                        langchain_content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": item.image_url.url},
                            }
                        )
            return langchain_content if langchain_content else ""

        # Fallback: convert to string
        return str(content)

    def _strip_images_from_content(self, content: Any) -> Any:
        """
        Strip image_url parts from multimodal content.

        Used to prevent rate limit issues when using tool-calling agents,
        since each agent iteration resends the full input including images.
        Images are replaced with [image attached] placeholder text.

        Args:
            content: String or list of content parts

        Returns:
            Content with images replaced by placeholder text
        """
        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return content

        stripped = []
        image_count = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                image_count += 1
                stripped.append({"type": "text", "text": "[image attached]"})
            else:
                stripped.append(part)

        if image_count > 0:
            logger.info(
                f"Stripped {image_count} image(s) from input to reduce token usage "
                "(tool-calling agents resend input on each iteration)"
            )

        return stripped if stripped else content

    def _extract_text_from_message(self, message: Any) -> str:
        """
        Extract text content from a message (string or multimodal).

        Args:
            message: Either a string or Message object

        Returns:
            Extracted text content
        """
        if isinstance(message, str):
            return message

        # Use get_text_content if available
        if hasattr(message, "get_text_content"):
            return message.get_text_content()

        # Get content attribute
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content

        # Extract from multimodal list
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif hasattr(item, "type") and item.type == "text":
                    text_parts.append(item.text)
            return " ".join(text_parts)

        return str(content)

    async def process_query(
        self, user_message: Union[str, Any], chat_history: Optional[List[Any]] = None
    ) -> str:
        """
        Process a user query through the RAG pipeline (non-streaming).

        Args:
            user_message: The user's question (string or Message object with multimodal content)
            chat_history: Previous messages in the conversation.

        Returns:
            The assistant's response.
        """
        if chat_history is None:
            chat_history = []

        # Extract text for augmentation (search still text-based)
        user_text = self._extract_text_from_message(user_message)
        augmented_input = self._build_augmented_input(user_text)

        # Convert to LangChain format (preserves multimodal content)
        langchain_content = self._convert_message_to_langchain(user_message)

        try:
            if self.agent_executor:
                # Use agent with tools
                # Use langchain_content to support multimodal current message
                result = await self.agent_executor.ainvoke(
                    {"input": langchain_content, "chat_history": chat_history}
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
                # Direct LLM call without tools - use multimodal content
                if self.llm is None:
                    return (
                        "Error: No LLM configured. Please configure an LLM in Settings."
                    )

                messages: List[BaseMessage] = [
                    SystemMessage(content=self._system_prompt)
                ]
                messages.extend(chat_history)
                # Use langchain_content (can be string or multimodal list)
                messages.append(HumanMessage(content=langchain_content))
                response = await self.llm.ainvoke(messages)
                content = response.content
                return content if isinstance(content, str) else str(content)

        except Exception as e:
            logger.exception("Error processing query")
            return f"I encountered an error processing your request: {str(e)}"

    async def process_query_stream(
        self,
        user_message: Union[str, Any],
        chat_history: Optional[List[Any]] = None,
        is_ui: bool = False,
    ):
        """
        Process a user query with true token-by-token streaming.

        For agent with tools: executes tool calls first, then streams the final response.
        For direct LLM: streams tokens directly from the LLM.

        Args:
            user_message: The user's question (string or Message object with multimodal content)
            chat_history: Previous messages in the conversation.
            is_ui: If True, use the UI agent with chart tool and enhanced prompt.
                   If False (default), use the standard agent for API/MCP.

        Yields:
            dict or str: Structured events for tool calls, or text tokens for content.
            - Tool start: {"type": "tool_start", "tool": "tool_name", "input": {...}}
            - Tool end: {"type": "tool_end", "tool": "tool_name", "output": "..."}
            - Content: str (individual tokens/chunks)
            - Max iterations: {"type": "max_iterations_reached"}
        """
        if chat_history is None:
            chat_history = []

        # Extract text for augmentation (search still text-based)
        user_text = self._extract_text_from_message(user_message)
        augmented_input = self._build_augmented_input(user_text)

        # Convert to LangChain format (preserves multimodal content)
        langchain_content = self._convert_message_to_langchain(user_message)

        # Select the appropriate agent executor
        executor = self.agent_executor_ui if is_ui else self.agent_executor
        system_prompt = self._system_prompt_ui if is_ui else self._system_prompt

        try:
            if executor:
                # Agent with tools: use astream_events for true streaming
                # Strip images from input - tool-calling agents resend the full input
                # on each iteration (tool call -> response -> tool call...) which
                # quickly exhausts rate limits. Images are replaced with [image attached].
                agent_input = self._strip_images_from_content(langchain_content)
                # Track tool runs to avoid duplicates from nested events
                active_tool_runs: set[str] = set()

                async for event in executor.astream_events(
                    {"input": agent_input, "chat_history": chat_history},
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
                        # Don't truncate UI visualization tools - their JSON must be complete
                        # Truncate other long outputs for display
                        ui_tools = {"create_chart", "create_datatable"}
                        if (
                            isinstance(tool_output, str)
                            and len(tool_output) > 2000
                            and tool_name not in ui_tools
                        ):
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
                # Direct LLM streaming without tools - use multimodal content
                if self.llm is None:
                    yield "Error: No LLM configured. Please configure an LLM in Settings."
                    return

                messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
                messages.extend(chat_history)
                # Use langchain_content (can be string or multimodal list)
                messages.append(HumanMessage(content=langchain_content))

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
