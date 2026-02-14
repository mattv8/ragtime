---
applyTo: '**'
---
## Architecture

**OpenAI-compatible RAG API** with FastAPI + dual vector backends (FAISS/pgvector) + LangChain tool calling.

| Service | Port | Purpose |
|---------|------|---------|
| ragtime | 8000 | RAG API + Indexer API + tool execution |
| ragtime (Vite) | 8001 | Indexer UI with hot-reload (dev only) |
| ragtime-db | 5434 | PostgreSQL (Prisma ORM) |

## Development Setup

- Dev stack runs in Docker: `ragtime` (8000/8001), `ragtime-db` (5434)
- Frontend `node_modules` persists in named volume, auto-installs on missing
- After `package.json` changes: `docker exec ragtime-dev sh -c "cd /ragtime/ragtime/frontend && npm ci"`

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

5. **Keep migrations idempotent** - PostgreSQL with pgvector. Use `IF NOT EXISTS`, `ADD VALUE IF NOT EXISTS`, and `DO $$ ... EXCEPTION WHEN duplicate_object` blocks for enums/constraints.

## Authentication

- Cookie: `ragtime_session` (httpOnly JWT)
- Local admin usernames stored with `local:` prefix in DB to avoid LDAP collision
- LDAP config stored in `ldap_config` table (singleton, id="default")

## Secrets Encryption

- Sensitive fields (API keys, passwords, tokens) use Fernet symmetric encryption
- Encryption key auto-generated on first startup, persisted to `data/.encryption_key`
- Encryption key derived from `ENCRYPTION_KEY` setting via SHA256
- Encrypted values prefixed with `enc::` in database
- **Backups**: Use `--include-secret` flag to include the encryption key file

**Encrypted fields:**
- `app_settings`: `openai_api_key`, `anthropic_api_key`, `postgres_password`, `mcp_default_route_password`
- `tool_configs.connection_config`: `password`, `ssh_password`, `ssh_key_passphrase`, `key_passphrase`, `smb_password`, `key_content`, `ssh_key_content`, `ssh_tunnel_password`, `ssh_tunnel_key_content`, `ssh_tunnel_key_passphrase`
- `mcp_route_configs`: `auth_password`
- `index_jobs` / `index_metadata`: `git_token`

## Tool Configs

Dynamic tool instances via `tool_configs` table:
- `tool_type`: postgres, odoo_shell, ssh_shell, filesystem_indexer
- `connection_config`: JSON with type-specific params
- `description`: Presented to LLM for tool selection context
- Tools built at runtime in `ragtime/rag/components.py`

## Embedding Provider Changes

Filesystem indexes can use pgvector or FAISS. Embedding dimension/config tracking applies to pgvector-backed filesystem indexes. Changing embedding provider/model/dimensions:
- `embedding_config_hash` tracks `"{provider}:{model}:{dimensions}"` (e.g., `ollama:nomic-embed-text:default`)
- First index sets `embedding_dimension` and `embedding_config_hash` in `app_settings`
- Subsequent indexes check for mismatch - if changed, **full re-index required** (existing embeddings incompatible)
- Deleting all filesystem indexes clears tracking, allowing fresh start with new provider

## Security Validation

- **SQL queries require LIMIT clause** - `core/security.py:validate_sql_query` rejects SELECT without LIMIT
- **Odoo code validation** - blocks `.write()`, `.create()`, `.unlink()` unless `allow_write=True`
- Filesystem indexer skips zero-byte files and symlinks (cloud placeholder detection)

## Conventions

- Async everywhere
- Type hints required (Python 3.12+ syntax)
- Pydantic `BaseModel` with `Field(description=...)` for LLM
- Logging: `from ragtime.core.logging import get_logger`
- **No emojis** in logs, entrypoints, or UI text (use plain text)
