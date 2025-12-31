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

Schema: `prisma/schema.prisma`.

```bash
python -m prisma generate  # After schema changes
python -m prisma db push   # Dev only - applies schema directly
```

### Migrations

Schema changes require migrations for production deployment:

1. **Before creating a new migration**, check for uncommitted migrations:
   ```bash
   git status prisma/migrations/
   ```
   If an uncommitted migration exists, add your changes to that migration's SQL file instead of creating a new one.

2. **Creating migrations** (dev stack must be running):
   ```bash
   docker exec ragtime-dev python -m prisma migrate dev --name descriptive_name
   ```

3. **Manual migration** (when prisma migrate dev fails or for complex DDL):
   - Create folder: `prisma/migrations/YYYYMMDDHHMMSS_name/`
   - Add `migration.sql` with the required DDL statements
   - Test by running: `docker exec ragtime-dev python -m prisma migrate deploy`

4. **Production deployment**:
   ```bash
   docker exec ragtime python -m prisma migrate deploy
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
