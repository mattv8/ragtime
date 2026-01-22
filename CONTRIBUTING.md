# Contributing to Ragtime

## Codebase Overview

### Key Entry Points

| Component | Path | Description |
|-----------|------|-------------|
| API routes | `ragtime/api/routes.py` | OpenAI-compatible `/v1/chat/completions` endpoint |
| Auth | `ragtime/api/auth.py` | Login, LDAP, session management |
| MCP server | `ragtime/mcp/server.py` | Model Context Protocol implementation |
| MCP tools | `ragtime/mcp/tools.py` | Tool definitions exposed via MCP |
| Tool registry | `ragtime/tools/registry.py` | Auto-discovery of `<name>_tool` exports |
| Security | `ragtime/core/security.py` | SQL validation, injection prevention |

### Vector Store Implementation

| Index Type | Service | Storage |
|------------|---------|--------|
| Upload/Git | `ragtime/indexer/service.py` | FAISS files in `data/indexes/<name>/` |
| Filesystem | `ragtime/indexer/filesystem_service.py` | pgvector `filesystem_chunks` table |
| Schema | `ragtime/indexer/schema_service.py` | pgvector `schema_embeddings` table |
| PDM | `ragtime/indexer/pdm_service.py` | pgvector `pdm_embeddings` table |

FAISS uses `langchain_community.vectorstores.FAISS`. pgvector uses raw SQL with cosine similarity (`<=>` operator). Embedding models are configured via `ragtime/indexer/vector_utils.py:get_embeddings_model()`.

### Tool Types

| Tool | File | Input Schema |
|------|------|-------------|
| PostgreSQL | `ragtime/tools/postgres.py` | `{query, reason}` |
| MSSQL | `ragtime/tools/mssql.py` | `{query, reason}` |
| MySQL | `ragtime/tools/mysql.py` | `{query, reason}` |
| Odoo | `ragtime/tools/odoo_shell.py` | `{code, reason}` |
| SSH | `ragtime/core/ssh.py` | `{command, reason}` |
| Filesystem search | `ragtime/tools/filesystem_indexer.py` | `{query, max_results}` |
| Git history | `ragtime/tools/git_history.py` | `{action, ...}` |

## End-to-End Example

Minimal ingest → index → query workflow using the REST API. All endpoints require authentication via the `ragtime_session` cookie (obtained from `/auth/login`).

```bash
# 1. Login and capture session cookie
SESSION=$(curl -s -c - -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"YOUR_ADMIN_PASSWORD"}' \
  | grep ragtime_session | awk '{print $NF}')

# 2. Create a FAISS index from a git repository (async job)
curl -X POST http://localhost:8000/indexes/git \
  -H "Cookie: ragtime_session=$SESSION" \
  -H "Content-Type: application/json" \
  -d '{"git_url":"https://github.com/owner/repo.git","name":"my-repo"}'
# Returns: {"job_id":"...","status":"pending"}

# 3. Poll job status until "completed"
curl http://localhost:8000/indexes/jobs/{job_id} \
  -H "Cookie: ragtime_session=$SESSION"

# 4. Query via OpenAI-compatible endpoint (uses API_KEY if set)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ragtime",
    "messages": [{"role":"user","content":"How does the authentication system work?"}],
    "stream": false
  }'
```

## Quick Start

1. **Clone and configure:**
   ```bash
   git clone https://github.com/mattv8/ragtime.git
   cd ragtime
   cp .env.example .env
   ```

2. **Start development stack:**

   **VS Code:** Press `Ctrl+Shift+P` → `Tasks: Run Task` → `Start Development Stack`

   **Command line:**
   ```bash
   docker compose -f docker/docker-compose.dev.yml up --build
   ```

3. **Access services:**
   - **UI (Vite hot-reload):** http://localhost:8001
   - **API:** http://localhost:8000
   - **API Docs:** http://localhost:8000/docs

## Adding Tools

Tools are auto-discovered from `ragtime/tools/`. Copy the template and implement:

```bash
cp ragtime/tools/_example_template.py ragtime/tools/my_tool.py
```

The tool export must be named `<filename>_tool`. Enable via the Tools tab in the UI.

## Database

### Prisma Migrations

After schema changes in `prisma/schema.prisma`:

```bash
docker exec ragtime-dev python -m prisma generate
docker exec ragtime-dev python -m prisma db push
```

### Backup & Restore

```bash
# Backup (streams to stdout)
docker exec ragtime-dev backup > backup.tar.gz
docker exec ragtime-dev backup --include-secret > backup.tar.gz  # Include encryption key
docker exec ragtime-dev backup --db-only > db-only.tar.gz
docker exec ragtime-dev backup --faiss-only > faiss-only.tar.gz

# Restore (copy file into container first)
docker cp backup.tar.gz ragtime-dev:/tmp/backup.tar.gz
docker exec ragtime-dev restore /tmp/backup.tar.gz
docker exec ragtime-dev restore --include-secret /tmp/backup.tar.gz  # Restore encryption key
docker exec ragtime-dev restore --db-only /tmp/backup.tar.gz
```

> **Important:** Backups contain encrypted secrets (API keys, passwords). Use `--include-secret` to include the encryption key file in your backup. Without the encryption key, you will need to re-enter all passwords after restore.

### Reset Database

```bash
# Quick reset
docker exec ragtime-db-dev bash -c \
  'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d postgres \
  -c "DROP DATABASE IF EXISTS ${POSTGRES_DB}; CREATE DATABASE ${POSTGRES_DB};"'


# Full reset (destroys volume)
docker compose -f docker/docker-compose.dev.yml down -v
docker compose -f docker/docker-compose.dev.yml up --build
```

## Troubleshooting

### Database Connection

```bash
docker exec -it ragtime-db-dev bash -c \
  'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}"'
```

### API Auth Testing

```bash
# Get session token
TOKEN=$(docker exec ragtime-dev bash -c 'curl -s -c - -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"$LOCAL_ADMIN_USER\",\"password\":\"$LOCAL_ADMIN_PASSWORD\"}" \
  | grep ragtime_session | awk "{print \$NF}"')

# Make authenticated request
curl -s http://localhost:8000/indexes/conversations \
  -H "Cookie: ragtime_session=$TOKEN" | jq .
```

### Container Logs

```bash
docker logs -f ragtime-dev
docker logs -f ragtime-db-dev
```

## CI/CD

GitHub Actions builds and pushes to Harbor on every branch:
- Branch tag: `:main`, `:feature-xyz`
- Main branch also tagged as `:latest`
- Images signed with Cosign, SBOM attached

**Required secrets:** `HARBOR_USERNAME`, `HARBOR_PASSWORD`, `COSIGN_PRIVATE_KEY`, `COSIGN_PASSWORD`
