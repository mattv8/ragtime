# Ragtime

OpenAI-compatible RAG API with LangChain tool calling for business intelligence queries.

## Features

- **OpenAI API Compatible**: Works with OpenWebUI, ChatGPT clients, and any OpenAI-compatible interface
- **RAG with FAISS**: Vector search over your codebase documentation
- **Tool Calling**: Execute Odoo ORM queries and PostgreSQL queries via natural language
- **Security**: Read-only by default with SQL injection and command injection prevention
- **Async**: Non-blocking execution for better performance

## Quick Start

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 2. Run with Docker Compose (Development)

```bash
docker compose up --build
```

> **Tip**: In VS Code, use `Ctrl+Shift+B` to run Docker tasks (start, stop, rebuild) without the terminal.

### 3. Access the Application

| URL | Description |
|-----|-------------|
| http://localhost:8001 | **Indexer UI** (Vite dev server with hot-reload) |
| http://localhost:8000 | API root (serves UI in production) |
| http://localhost:8000/docs | Swagger API documentation |
| http://localhost:8000/health | Health check endpoint |

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ragtime",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Local Development Setup

For local development without Docker, use the setup script:

```bash
bash scripts/setup.sh
```

This script:
- Detects Python 3.12 (or falls back to available Python 3)
- Creates a `.venv` virtual environment
- Installs all dependencies from `requirements.txt`
- Installs dev tools (pylint, black, isort)
- Copies `.env.example` to `.env` if missing

## Architecture

Single container running both the RAG API and Indexer service:

| Service | Port | Purpose |
|---------|------|---------|
| ragtime | 8000 | FastAPI backend (RAG API + Indexer API) |
| ragtime (Vite) | 8001 | Indexer UI with hot-reload (dev only) |
| ragtime-db | 5434 | PostgreSQL (Prisma ORM) |

In development, Vite proxies API requests from port 8001 to port 8000 for seamless hot-reload.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | (required) |
| `OPENAI_MODEL` | OpenAI model to use | `gpt-4-turbo` |
| `LLM_PROVIDER` | LLM provider (`openai` or `anthropic`) | `openai` |
| `FAISS_INDEX_PATHS` | Comma-separated FAISS index paths | `.data/codebase` |
| `ENABLE_TOOLS` | Enable tool calling | `true` |
| `ENABLED_TOOLS` | Comma-separated list of enabled tools | `odoo,postgres` |
| `API_KEY` | API authentication key | (none) |
| `DEBUG_MODE` | Enable debug logging | `false` |

See [.env.example](.env.example) for full list.

## Adding Custom Tools

Tools are **auto-discovered** from `ragtime/tools/`. To add a new tool:

### 1. Create a new file

```bash
cp ragtime/tools/_example_template.py ragtime/tools/my_tool.py
```

### 2. Implement your tool

```python
# ragtime/tools/my_tool.py
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class MyToolInput(BaseModel):
    query: str = Field(description="What to search for")

async def execute_my_tool(query: str) -> str:
    # Your implementation here
    return f"Result for: {query}"

# Name must be: <filename>_tool
my_tool_tool = StructuredTool.from_function(
    coroutine=execute_my_tool,
    name="my_tool",
    description="Description for the LLM...",
    args_schema=MyToolInput
)
```

### 3. Enable the tool

Enable via the Settings panel in the Indexer UI at http://localhost:8001

That's it! The tool will be auto-discovered and available to the LLM.

## Creating FAISS Indexes

The Indexer UI provides an easy way to create FAISS indexes from your codebases.

**Open the Indexer UI:** http://localhost:8001

The UI provides:
- **Upload Tab**: Upload a zip file of your codebase to create an index
- **Git Tab**: Clone and index a Git repository by URL
- **Indexes List**: View, manage, and delete existing indexes
- **Job Status**: Monitor indexing progress in real-time
- **Settings**: Configure LLM provider, embedding model, and enabled tools

## Connecting to OpenWebUI

1. In OpenWebUI, go to Settings → Connections → OpenAI API
2. Add a new connection:
   - API Base URL: `http://ragtime:8000/v1` (or `http://localhost:8000/v1` if running locally)
   - API Key: Your configured API_KEY (or any value if not set)
3. Select "ragtime" as the model

## License

MIT
