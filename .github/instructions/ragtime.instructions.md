---
applyTo: '**'
---
## Architecture

**OpenAI-compatible RAG API** with FastAPI + FAISS + LangChain tool calling.

| Service | Port | Purpose |
|---------|------|---------|
| ragtime | 8000 | RAG API + Indexer API + tool execution |
| ragtime (Vite) | 8001 | Indexer UI with hot-reload (dev only) |
| ragtime-db | 5434 | PostgreSQL (Prisma ORM) |

## Tool Pattern

Tools auto-discovered from `ragtime/tools/`. Must export `<filename>_tool`:

```python
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

class MyToolInput(BaseModel):
    query: str = Field(description="Description for LLM")

async def execute_my_tool(query: str) -> str:
    return result

my_tool_tool = StructuredTool.from_function(
    coroutine=execute_my_tool,
    name="my_tool",
    description="What this tool does",
    args_schema=MyToolInput
)
```

## Prisma

Schema: `prisma/schema.prisma`. Tables: `User`, `Session`, `LdapConfig`, `Conversation`, `IndexJob`, `IndexConfig`, `IndexMetadata`, `AppSettings`, `ToolConfig`.

```bash
python -m prisma generate  # After schema changes
python -m prisma db push   # Dev only
```

## Authentication

- Auth module: `ragtime/core/auth.py` (LDAP + local admin)
- Dependencies: `ragtime/core/security.py` → `get_current_user`, `require_admin`
- Routes: `ragtime/api/auth.py`
- Cookie: `ragtime_session` (httpOnly JWT)
- Local admin usernames stored with `local:` prefix in DB to avoid LDAP collision
- LDAP config stored in `ldap_configs` table, managed via Settings UI

## Tool Configs

Tools are configured dynamically via `tool_configs` table (UI: Tools tab). Each tool config includes:
- `tool_type`: postgres, odoo_shell, ssh_shell
- `connection_config`: JSON with type-specific connection params
- `description`: Presented to LLM for context ("Here's what this tool connects to")
- `enabled`: Toggle tool availability for RAG agent
- Connection test results stored in `last_test_at`, `last_test_result`, `last_test_error`

Tools are dynamically built at runtime from configs in `ragtime/rag/components.py`.

## Key Files

| Path | Purpose |
|------|---------|
| `ragtime/rag/components.py` | RAG agent setup, dynamic tool building |
| `ragtime/config/settings.py` | Pydantic settings from env |
| `ragtime/core/auth.py` | LDAP/local auth, JWT, session management |
| `ragtime/core/security.py` | Auth dependencies, query validation |
| `ragtime/core/database.py` | Prisma connection lifecycle |
| `ragtime/api/auth.py` | Auth API routes |
| `ragtime/indexer/repository.py` | Prisma CRUD for jobs/metadata/settings |
| `ragtime/indexer/service.py` | FAISS index creation logic |

## Conventions

- Async everywhere
- Type hints required (Python 3.12+ syntax)
- Pydantic `BaseModel` with `Field(description=...)` for LLM
- Logging: `from ragtime.core.logging import get_logger`
- **No emojis** in logs, entrypoints, or UI text (use plain text)
