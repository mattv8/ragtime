"""
RAG Components - FAISS Vector Store and LangChain Agent setup.
"""

from pathlib import Path
from typing import Any, Optional, List

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ragtime.config import settings
from ragtime.core.logging import get_logger
from ragtime.core.app_settings import get_app_settings, get_tool_configs
from ragtime.tools import get_enabled_tools, get_all_tools

logger = get_logger(__name__)

# Base system prompt template - tool descriptions will be appended
BASE_SYSTEM_PROMPT = """You are a helpful AI assistant with access to business data and code documentation.
You help users query and understand their data using natural language.

CAPABILITIES:
1. **Code Knowledge**: You have access to documentation about the codebase and custom modules.
2. **Live Data Queries**: You can query connected systems using the available tools.

GUIDELINES:
- Always explain what you're doing before executing queries
- Format results in a clear, readable way (tables, lists, summaries)
- If a query fails, explain the error and suggest alternatives
- For sensitive data, remind users about data access policies
- Always include appropriate LIMIT clauses to prevent large result sets

When answering questions about the codebase, use your knowledge from the documentation.
When answering questions about live data, use the appropriate query tool."""


def build_tool_system_prompt(tool_configs: List[dict]) -> str:
    """
    Build system prompt section describing available tools.

    Args:
        tool_configs: List of tool configuration dictionaries.

    Returns:
        System prompt section with tool descriptions.
    """
    if not tool_configs:
        return ""

    tool_sections = []
    for config in tool_configs:
        tool_type = config.get("tool_type", "unknown")
        name = config.get("name", "Unnamed Tool")
        description = config.get("description", "")

        # Build type-specific guidance
        if tool_type == "postgres":
            type_hint = "PostgreSQL database - use SQL queries"
        elif tool_type == "odoo_shell":
            type_hint = "Odoo ORM - use Python ORM commands"
        elif tool_type == "ssh_shell":
            type_hint = "SSH shell - use shell commands"
        else:
            type_hint = "Query tool"

        section = f"- **{name}** ({type_hint})"
        if description:
            section += f"\n  {description}"
        tool_sections.append(section)

    return f"""

AVAILABLE TOOLS:
{chr(10).join(tool_sections)}

When a user asks about data, determine which tool is most appropriate based on the descriptions above.
"""


class RAGComponents:
    """Container for RAG components initialized at startup."""

    def __init__(self):
        self.retrievers: dict[str, Any] = {}
        self.agent_executor: Optional[AgentExecutor] = None
        self.llm: Optional[ChatOpenAI] = None
        self.is_ready: bool = False
        self._app_settings: Optional[dict] = None
        self._tool_configs: Optional[List[dict]] = None
        self._system_prompt: str = BASE_SYSTEM_PROMPT

    async def initialize(self):
        """Initialize all RAG components."""
        logger.info("Initializing RAG components...")

        # Load settings from database
        self._app_settings = await get_app_settings()
        self._tool_configs = await get_tool_configs()

        # Build system prompt with tool descriptions
        tool_prompt_section = build_tool_system_prompt(self._tool_configs)
        self._system_prompt = BASE_SYSTEM_PROMPT + tool_prompt_section

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
        provider = self._app_settings.get("llm_provider", "openai").lower()
        model = self._app_settings.get("llm_model", "gpt-4-turbo")

        if provider == "anthropic":
            api_key = self._app_settings.get("anthropic_api_key", "")
            if api_key:
                try:
                    from langchain_anthropic import ChatAnthropic
                    self.llm = ChatAnthropic(
                        model=model,
                        temperature=0,
                        anthropic_api_key=api_key
                    )
                    logger.info(f"Using Anthropic LLM: {model}")
                    return
                except ImportError:
                    logger.warning("langchain-anthropic not installed, falling back to OpenAI")
            else:
                logger.warning("Anthropic selected but no API key configured, falling back to OpenAI")

        # Default to OpenAI
        api_key = self._app_settings.get("openai_api_key", "")
        if not api_key:
            logger.warning("No OpenAI API key configured - LLM features will be disabled until configured via Settings UI")
            self.llm = None
            return

        self.llm = ChatOpenAI(
            model=model if provider == "openai" else "gpt-4-turbo",
            temperature=0,
            streaming=True,
            openai_api_key=api_key
        )
        logger.info(f"Using OpenAI LLM: {self.llm.model_name}")

    async def _get_embedding_model(self):
        """Get embedding model based on database settings."""
        provider = self._app_settings.get("embedding_provider", "ollama").lower()
        model = self._app_settings.get("embedding_model", "nomic-embed-text")

        if provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            base_url = self._app_settings.get("ollama_base_url", "http://localhost:11434")
            logger.info(f"Using Ollama embeddings: {model} at {base_url}")
            return OllamaEmbeddings(model=model, base_url=base_url)
        elif provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            api_key = self._app_settings.get("openai_api_key", "")
            if not api_key:
                logger.warning("OpenAI embeddings selected but no API key configured")
            logger.info(f"Using OpenAI embeddings: {model}")
            return OpenAIEmbeddings(model=model, openai_api_key=api_key)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    async def _load_faiss_indexes(self, embedding_model):
        """Load FAISS indexes from configured paths."""
        # Try multiple base directories
        possible_bases = [
            Path(__file__).resolve().parent.parent.parent,  # /app
            Path("/app"),
            Path.cwd(),
        ]

        index_paths = [p.strip() for p in settings.faiss_index_paths.split(",") if p.strip()]

        for index_path in index_paths:
            loaded = False
            for base_dir in possible_bases:
                full_path = base_dir / index_path
                if full_path.exists():
                    try:
                        db = FAISS.load_local(
                            str(full_path),
                            embedding_model,
                            allow_dangerous_deserialization=True
                        )
                        index_name = Path(index_path).name
                        self.retrievers[index_name] = db.as_retriever(search_kwargs={"k": 5})
                        logger.info(f"Loaded FAISS index: {index_name} from {full_path}")
                        loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load FAISS index {full_path}: {e}")

            if not loaded:
                logger.warning(f"FAISS index not found: {index_path}")

    async def _create_agent(self):
        """Create the LangChain agent with tools."""
        # Skip agent creation if LLM is not configured
        if self.llm is None:
            logger.warning("Skipping agent creation - no LLM configured")
            self.agent_executor = None
            return

        tools = []
        if settings.enable_tools:
            # Get tools from the new ToolConfig system
            if self._tool_configs:
                tools = await self._build_tools_from_configs()
                logger.info(f"Built {len(tools)} tools from configurations")
            else:
                # Fallback to legacy enabled_tools system
                app_settings = await get_app_settings()
                enabled_list = app_settings["enabled_tools"]
                if enabled_list:
                    tools = get_enabled_tools(enabled_list)
                    logger.info(f"Using legacy tool configuration: {enabled_list}")

            if not tools:
                available = list(get_all_tools().keys())
                logger.warning(
                    f"No tools configured. Available tool types: {available}. "
                    f"Configure via Tools tab at /indexes/ui?view=tools"
                )

        prompt = ChatPromptTemplate.from_messages([
            ("system", self._system_prompt),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        if tools:
            agent = create_openai_tools_agent(self.llm, tools, prompt)
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=settings.debug_mode,
                handle_parsing_errors=True,
                max_iterations=5,
                return_intermediate_steps=settings.debug_mode
            )
        else:
            # No tools - agent_executor stays None
            self.agent_executor = None

    async def _build_tools_from_configs(self) -> List[Any]:
        """
        Build LangChain tools from ToolConfig entries.

        Creates dynamic tool wrappers for each configured tool instance.
        """
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field

        tools = []

        for config in self._tool_configs:
            tool_type = config.get("tool_type")
            tool_name = config.get("name", "").replace(" ", "_").lower()
            tool_id = config.get("id")

            if tool_type == "postgres":
                tool = await self._create_postgres_tool(config, tool_name, tool_id)
            elif tool_type == "odoo_shell":
                tool = await self._create_odoo_tool(config, tool_name, tool_id)
            elif tool_type == "ssh_shell":
                tool = await self._create_ssh_tool(config, tool_name, tool_id)
            else:
                logger.warning(f"Unknown tool type: {tool_type}")
                continue

            if tool:
                tools.append(tool)

        return tools

    async def _create_postgres_tool(self, config: dict, tool_name: str, tool_id: str):
        """Create a PostgreSQL query tool from config."""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
        import asyncio
        import subprocess

        conn_config = config.get("connection_config", {})
        max_results = config.get("max_results", 100)
        timeout = config.get("timeout", 30)
        allow_write = config.get("allow_write", False)
        description = config.get("description", "")

        class PostgresInput(BaseModel):
            query: str = Field(description="SQL query to execute. Must include LIMIT clause.")
            reason: str = Field(description="Brief description of what this query retrieves")

        async def execute_query(query: str, reason: str) -> str:
            """Execute PostgreSQL query using this tool's configuration."""
            from ragtime.core.security import validate_sql_query, sanitize_output

            logger.info(f"[{tool_name}] Query: {reason}")

            # Validate query
            is_safe, validation_reason = validate_sql_query(query, enable_write=allow_write)
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
                cmd = ["psql", "-h", host, "-p", str(port), "-U", user, "-d", database, "-c", query]
                env = {"PGPASSWORD": password}
            elif container:
                cmd = [
                    "docker", "exec", "-i", container, "bash", "-c",
                    f'PGPASSWORD="$POSTGRES_PASSWORD" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \'{escaped_query}\''
                ]
                env = None
            else:
                return "Error: No connection configured"

            try:
                process = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env),
                    timeout=timeout
                )
                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    return f"Error: {stderr.decode('utf-8', errors='replace').strip()}"

                output = stdout.decode("utf-8", errors="replace").strip()
                return sanitize_output(output) if output else "Query executed successfully (no results)"

            except asyncio.TimeoutError:
                return f"Error: Query timed out after {timeout}s"
            except Exception as e:
                return f"Error: {str(e)}"

        tool_description = f"Execute SQL queries against {config.get('name', 'PostgreSQL')}."
        if description:
            tool_description += f" {description}"
        tool_description += " Must include LIMIT clause. Only SELECT queries unless write operations are enabled."

        return StructuredTool.from_function(
            coroutine=execute_query,
            name=f"query_{tool_name}",
            description=tool_description,
            args_schema=PostgresInput
        )

    async def _create_odoo_tool(self, config: dict, tool_name: str, tool_id: str):
        """Create an Odoo shell tool from config."""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
        import asyncio
        import subprocess
        import re

        conn_config = config.get("connection_config", {})
        timeout = config.get("timeout", 60)  # Odoo shell needs more time to initialize
        allow_write = config.get("allow_write", False)
        description = config.get("description", "")

        class OdooInput(BaseModel):
            code: str = Field(description="Python code to execute in Odoo shell using ORM methods")
            reason: str = Field(description="Brief description of what this code does")

        async def execute_odoo(code: str, reason: str) -> str:
            """Execute Odoo shell command using this tool's configuration."""
            from ragtime.core.security import validate_odoo_code, sanitize_output

            logger.info(f"[{tool_name}] Odoo: {reason}")

            # Validate code
            is_safe, validation_reason = validate_odoo_code(code, enable_write_ops=allow_write)
            if not is_safe:
                return f"Error: {validation_reason}"

            container = conn_config.get("container", "")
            database = conn_config.get("database", "odoo")
            config_path = conn_config.get("config_path", "")

            if not container:
                return "Error: No container configured"

            # Build the shell command - use standard Odoo shell with stdin piping
            cmd = [
                "docker", "exec", "-i", container,
                "odoo", "shell", "--no-http", "-d", database
            ]
            # Only add config path if specified (some containers use env vars)
            if config_path:
                cmd.extend(["-c", config_path])

            # Wrap user code with env setup and error handling
            wrapped_code = f'''
env = self.env
try:
{chr(10).join("    " + line for line in code.strip().split(chr(10)))}
except Exception as e:
    print(f"ODOO_ERROR: {{type(e).__name__}}: {{e}}")
'''
            # Add exit command
            full_input = wrapped_code + "\nexit()\n"

            try:
                process = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *cmd,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT  # Merge stderr into stdout
                    ),
                    timeout=timeout
                )
                stdout, _ = await process.communicate(input=full_input.encode())
                output = stdout.decode("utf-8", errors="replace")

                # Filter output similar to odoo-shell-exec.py
                result_lines = []
                command_started = False

                # Patterns to skip during initialization
                init_skip_patterns = [
                    r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \d+ INFO',
                    r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \d+ WARNING',
                    r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \d+ DEBUG',
                    r'^/.*\.py:\d+: UserWarning:',
                    r'^\s*import pkg_resources',
                    r'^\s*The pkg_resources package',
                    r'^profiling:.*Cannot open',
                    r'^Python \d+\.\d+\.\d+',
                    r'^IPython.*--',
                    r'^Tip:',
                    r'^In \[\d+\]:',
                    r'^>>>',
                    r'^\.\.\.',
                    r'^odoo:',
                    r'^openerp:',
                    r'^werkzeug:',
                ]

                for line in output.split("\n"):
                    line_stripped = line.rstrip()

                    # Check for our error marker
                    if line_stripped.startswith("ODOO_ERROR:"):
                        return f"Error: {line_stripped[11:].strip()}"

                    # Check for Python exceptions
                    if any(re.match(pattern, line_stripped) for pattern in [
                        r'^Traceback \(most recent call last\):',
                        r'^\w+Error:',
                        r'^\w+Exception:',
                    ]):
                        # Capture the error and remaining output
                        result_lines.append(line_stripped)
                        command_started = True
                        continue

                    # Skip initialization noise
                    if not command_started:
                        if any(re.match(pattern, line_stripped) for pattern in init_skip_patterns):
                            continue
                        # Look for indicators that command output is starting
                        # (non-empty lines that aren't init noise)
                        if line_stripped and not line_stripped.startswith((' ', '\t')):
                            # Check if this looks like actual output
                            if not any(skip in line_stripped.lower() for skip in [
                                'loading', 'loaded', 'initializing', 'registering',
                                'odoo.', 'werkzeug', 'modules.'
                            ]):
                                command_started = True
                                result_lines.append(line_stripped)
                    else:
                        # After command starts, capture everything except profiling noise
                        if not re.match(r'^profiling:.*Cannot open', line_stripped):
                            result_lines.append(line_stripped)

                result = "\n".join(result_lines).strip()

                # Clean up common shell artifacts
                result = re.sub(r'^In \[\d+\]:\s*', '', result, flags=re.MULTILINE)
                result = re.sub(r'^Out\[\d+\]:\s*', '', result, flags=re.MULTILINE)
                result = re.sub(r'^\.\.\.:?\s*', '', result, flags=re.MULTILINE)

                return sanitize_output(result) if result else "Query executed successfully (no output)"

            except asyncio.TimeoutError:
                return f"Error: Query timed out after {timeout}s"
            except FileNotFoundError:
                return "Error: Docker command not found"
            except Exception as e:
                logger.exception(f"Odoo shell error: {e}")
                return f"Error: {str(e)}"

        tool_description = f"Execute Python ORM code in {config.get('name', 'Odoo')} shell."
        if description:
            tool_description += f" {description}"
        tool_description += " Use env['model'].search_read() for queries."

        return StructuredTool.from_function(
            coroutine=execute_odoo,
            name=f"odoo_{tool_name}",
            description=tool_description,
            args_schema=OdooInput
        )

    async def _create_ssh_tool(self, config: dict, tool_name: str, tool_id: str):
        """Create an SSH shell tool from config."""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
        import asyncio
        import subprocess

        conn_config = config.get("connection_config", {})
        timeout = config.get("timeout", 30)
        description = config.get("description", "")

        class SSHInput(BaseModel):
            command: str = Field(description="Shell command to execute on the remote server")
            reason: str = Field(description="Brief description of what this command does")

        async def execute_ssh(command: str, reason: str) -> str:
            """Execute SSH command using this tool's configuration."""
            from ragtime.core.security import sanitize_output

            logger.info(f"[{tool_name}] SSH: {reason}")

            host = conn_config.get("host", "")
            port = conn_config.get("port", 22)
            user = conn_config.get("user", "")
            key_path = conn_config.get("key_path")
            command_prefix = conn_config.get("command_prefix", "")

            if not host or not user:
                return "Error: Host and user are required"

            full_command = f"{command_prefix}{command}"

            cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
            if key_path:
                cmd.extend(["-i", key_path])
            cmd.extend(["-p", str(port), f"{user}@{host}", full_command])

            try:
                process = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE),
                    timeout=timeout
                )
                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    error = stderr.decode('utf-8', errors='replace').strip()
                    return f"Error (exit {process.returncode}): {error}"

                output = stdout.decode("utf-8", errors="replace").strip()
                return sanitize_output(output) if output else "Command executed successfully (no output)"

            except asyncio.TimeoutError:
                return f"Error: Command timed out after {timeout}s"
            except Exception as e:
                return f"Error: {str(e)}"

        tool_description = f"Execute shell commands on {config.get('name', 'remote server')} via SSH."
        if description:
            tool_description += f" {description}"

        return StructuredTool.from_function(
            coroutine=execute_ssh,
            name=f"ssh_{tool_name}",
            description=tool_description,
            args_schema=SSHInput
        )

    def get_context_from_retrievers(self, query: str, max_docs: int = 5) -> str:
        """
        Retrieve relevant context from all FAISS indexes.

        Args:
            query: The search query.
            max_docs: Maximum documents per index.

        Returns:
            Combined context string from all indexes.
        """
        all_docs = []
        for name, retriever in self.retrievers.items():
            try:
                docs = retriever.invoke(query)
                for doc in docs[:max_docs]:
                    source = doc.metadata.get("source", "unknown")
                    all_docs.append(f"[{name}:{source}]\n{doc.page_content}")
            except Exception as e:
                logger.warning(f"Error retrieving from {name}: {e}")

        return "\n\n---\n\n".join(all_docs) if all_docs else ""

    async def process_query(
        self,
        user_message: str,
        chat_history: List[Any] = None
    ) -> str:
        """
        Process a user query through the RAG pipeline.

        Args:
            user_message: The user's question or request.
            chat_history: Previous messages in the conversation.

        Returns:
            The assistant's response.
        """
        if chat_history is None:
            chat_history = []

        # Get relevant context from FAISS
        context = self.get_context_from_retrievers(user_message)

        # Augment the query with context if available
        augmented_input = user_message
        if context:
            augmented_input = f"""Question: {user_message}

Relevant documentation context:
{context}

Please answer the question using the context above and/or query tools if you need live data."""

        try:
            if self.agent_executor and settings.enable_tools:
                # Use agent with tools
                result = await self.agent_executor.ainvoke({
                    "input": augmented_input,
                    "chat_history": chat_history
                })
                return result.get("output", "I couldn't generate a response.")
            else:
                # Direct LLM call without tools
                if self.llm is None:
                    return "Error: No LLM configured. Please configure an LLM in Settings."

                messages = [SystemMessage(content=self._system_prompt)]
                messages.extend(chat_history)
                messages.append(HumanMessage(content=augmented_input))
                response = await self.llm.ainvoke(messages)
                return response.content

        except Exception as e:
            logger.exception("Error processing query")
            return f"I encountered an error processing your request: {str(e)}"


# Global RAG components instance
rag = RAGComponents()
