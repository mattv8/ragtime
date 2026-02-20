---
applyTo: '**'
---
## Big Picture

- FastAPI app in `ragtime/main.py` composes four surfaces: OpenAI API (`/v1/*`), auth (`/auth/*`), indexer/UI (`/indexes/*` + `/`), and MCP (`/mcp*`).
- Runtime services: `ragtime-dev` container (API + Vite + tool execution) and `ragtime-db-dev` (Postgres+pgvector).
- Core startup sequence (`lifespan` in `ragtime/main.py`): connect DB -> `rag.initialize()` -> recover/cleanup index jobs -> start background chat tasks -> start MCP session manager.
- Health/readiness is progressive: `/health` is `initializing` until core LLM/tool setup finishes; FAISS indexes can continue loading in background.

## Data and Retrieval Architecture

- Persistence is Prisma (`ragtime/core/database.py`, schema in `prisma/schema.prisma`) using mapped snake_case tables.
- Upload/Git indexes support both backends:
  - `faiss`: files under `.data/indexes/<name>/`, loaded into memory at startup.
  - `pgvector`: embeddings in DB tables (`filesystem_embeddings`, etc.).
- Schema and PDM indexers are pgvector-only (`schema_embeddings`, `pdm_embeddings`).
- `ragtime/rag/components.py` rebuilds the agent when tools/indexes change; avoid bypassing this when changing retrieval/tool wiring.

## Tooling Pattern (Important)

- Static tools are auto-discovered from `ragtime/tools/` by `ragtime/tools/registry.py`.
- Discovery convention: module `my_tool.py` should export `my_tool_tool` (or `tool`) as a `StructuredTool`.
- Runtime tools come from DB `tool_configs` and are materialized in `ragtime/rag/components.py` (not just filesystem modules).
- Tool schemas must use Pydantic fields with descriptions (LLM-facing contract quality matters).

## Auth, MCP, and User Space

- Session auth uses `ragtime_session` cookie (JWT-backed); `get_session_token` also accepts `Authorization: Bearer`.
- Local admin usernames are stored with `local:` prefix to avoid LDAP collisions.
- MCP HTTP transport (`ragtime/mcp/routes.py`) supports default route plus dynamic custom routes and optional auth/group filters.
- User Space is file-backed under `${INDEX_DATA_PATH}/_userspace/workspaces/*` with git-based snapshots (`ragtime/userspace/service.py`).

## Developer Workflow

- Preferred dev loop: VS Code task `Start Development Stack` (`docker/docker-compose.dev.yml`).
- Dev ports: `8000` API, `8001` Vite UI; in dev, `DEBUG_MODE=true` and hot reload is active.
- If `ragtime/frontend/package.json` changes, run `docker exec ragtime-dev sh -c "cd /ragtime/ragtime/frontend && npm ci"`.
- Prisma workflow after schema edits:
  - `docker exec ragtime-dev python -m prisma generate`
  - Use migrations for durable changes (`python -m prisma migrate dev` / `migrate deploy`), not ad-hoc DB edits.

## Prisma + Migration Gotchas

- Before creating a new migration, check `git status prisma/migrations/`; if an uncommitted migration already exists, append to it instead of creating another.
- For complex DDL or Prisma diff failures, use manual migration folders (`prisma/migrations/<timestamp>_name/migration.sql`) and verify with `docker exec ragtime-dev python -m prisma migrate deploy`.
- Keep migration SQL idempotent for pgvector environments (`IF NOT EXISTS`, safe enum updates, `DO $$ ... EXCEPTION WHEN duplicate_object`).
- In this repo, avoid relying on `db push` for production-like behavior; entrypoint and deploy paths are migration-first.

## Security + Secrets Nuance

- Secrets are encrypted in DB as `enc::...`; key is persisted at `.data/.encryption_key` and must be included in backups with `--include-secret`.
- High-impact encrypted fields include:
  - `app_settings` API keys/passwords
  - credential fields inside `tool_configs.connection_config`
  - MCP route auth passwords
  - stored git tokens in index records.
- Validators in `ragtime/core/security.py` are intentionally strict; adjust there (not in prompting) when tool behavior appears blocked.

## Retrieval/Embedding Compatibility

- Upload/Git can be `faiss` or `pgvector`; schema and PDM remain pgvector-only.
- Embedding provider/model changes can invalidate existing vector compatibility; respect embedding tracking fields in `app_settings` (`embedding_dimension`, `embedding_config_hash`) before reusing indexed data.
- If embedding config changes across existing indexes, prefer full re-index over partial patching.

## Project-Specific Guardrails

- Security validators are strict by design (`ragtime/core/security.py`): SQL read-only checks (including `LIMIT` expectations), Odoo write-op blocking, SSH command filtering.
- Secrets in DB are encrypted (`enc::`), key persisted at `.data/.encryption_key`; backups need `--include-secret` when secret portability matters.
- Keep async style and existing logging pattern (`get_logger` / `setup_logging`), and preserve mapped Prisma field naming conventions.
- No dedicated test suite exists in repo; validate changes with targeted API/UI/container smoke checks.
- House style: keep type hints and Pydantic `Field(description=...)` on LLM-facing schemas; avoid emojis in logs/UI text.
