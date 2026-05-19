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

### Authentication

- Session auth uses the `ragtime_session` httpOnly cookie; bearer token fallback is handled by `get_session_token()` in `ragtime/core/security.py`.
- Auth endpoints and login/session handling live in `ragtime/api/auth.py`.
- User/session validation dependencies are in `ragtime/core/security.py` (`get_current_user`, `require_admin`).
- Local admin auth is always available; LDAP is optional and configured via admin settings.
- Login is rate-limited; keep this behavior intact when changing auth flows.

### User Space Runtime Scope

| Area | Path(s) | Notes |
|------|---------|-------|
| User Space control plane | `ragtime/userspace/runtime_service.py`, `ragtime/userspace/runtime_routes.py` | Session lifecycle, collab, preview proxying, capability checks |
| Public share routes | `ragtime/main.py` | Canonical public sharing routes (`/{owner}/{slug}`) and token redirect (`/shared/{token}`) |
| Runtime manager/worker data plane | `runtime/manager/**`, `runtime/worker/**`, `runtime/main.py` | Isolated runtime execution, devserver lifecycle, FS and PTY upstreams |
| Bootstrap + launch config | `.ragtime/runtime-bootstrap.json`, `.ragtime/runtime-entrypoint.json` | Bootstrap commands and runtime launch overrides |

### What Workspaces Are

Workspaces are the core User Space unit: an agentic sandbox backed by real project files, membership/roles, conversation context, selected tool access, and a runtime preview session.

- Workspace API and model surface: `ragtime/userspace/routes.py`, `ragtime/userspace/models.py` (`selected_tool_ids`, files, snapshots, sharing).
- Runtime-backed preview lifecycle: `ragtime/userspace/runtime_service.py` with preview proxy routes in `ragtime/userspace/runtime_routes.py`.
- Runtime service mode split: `runtime/main.py` chooses manager/worker behavior via `RUNTIME_SERVICE_MODE`.
- Agent infrastructure access path: `ragtime/mcp/tools.py` dynamically materializes enabled tool configs (`get_tool_configs()`), applies health filtering, and exposes tool execution through MCP.

Current User Space scope is runtime-backed preview + sharing. Keep endpoint contract details in `/docs` and avoid duplicating full route inventories in docs.

When changing sharing behavior, validate both route layers:
- public top-level routes in `ragtime/main.py`
- internal editor preview routes under `/indexes/userspace/shared/*` in `ragtime/userspace/runtime_routes.py`

### Vector Store Implementation

| Index Type | Service | Storage |
|------------|---------|--------|
| Upload/Git | `ragtime/indexer/service.py` | FAISS files in `data/indexes/<name>/` **or** pgvector `filesystem_embeddings` table |
| Filesystem | `ragtime/indexer/filesystem_service.py` | pgvector `filesystem_embeddings` table |
| Schema | `ragtime/indexer/schema_service.py` | pgvector `schema_embeddings` table |
| PDM | `ragtime/indexer/pdm_service.py` | pgvector `pdm_embeddings` table |

FAISS uses `langchain_community.vectorstores.FAISS`. pgvector uses raw SQL with cosine similarity (`<=>` operator). Upload and Git indexes use a unified indexer and can target either backend via `vector_store_type`. Embedding models are configured via `ragtime/indexer/vector_utils.py:get_embeddings_model()`.

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

# 2. Create a git index (defaults to FAISS unless config.vector_store_type is set)
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

  If you use Colima on macOS, add the Docker socket override before starting the dev stack:
  ```bash
  echo 'DOCKER_SOCKET_PATH=$HOME/.colima/default/docker.sock' >> .env
  ```
  Linux and WSL typically do not need this because `/var/run/docker.sock` is already the default.

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

4. **User Space sanity checks (for runtime/share changes):**
  - Verify runtime/session endpoints in `/docs`
  - Verify both public and internal share flows (slug/token + preview proxy)
  - Verify runtime bootstrap stamp behavior (`.ragtime/.runtime-bootstrap.done`) after editing bootstrap config

## Contribution Workflow

Most contributors only need this section. Maintainers handle promotion from `beta` to `main`.

### Quick Path

1. Fork the repo or create a branch from `beta`.
2. Make a focused change.
3. Run the smallest relevant local check you can.
4. Open a pull request into `beta`, not `main`.
5. In the PR, describe what changed and how you tested it.

### Branches

| Branch | Who uses it | Purpose |
|--------|-------------|---------|
| `beta` | Contributors and maintainers | Integration branch for PRs and quality testing. |
| `main` | Maintainers | Stable release branch. Contributors should not target this directly. |
| `feature/*` or `fix/*` | Contributors | Work branches for one focused change. |

### Pull Request Checklist

- Keep the PR focused on one bug fix, feature, or cleanup.
- Explain the behavior change in plain language.
- Add or update tests when behavior changes.
- Mention any database, auth, runtime, or configuration impact.
- For UI changes, include a screenshot or short note about what you checked.

CI runs automatically on every branch and PR. It checks backend tests, frontend build, critical lint failures, and duplicate-code drift.

Useful local checks:

```bash
# Full backend quality gate and test suite
docker build --target python-test -f docker/Dockerfile .

# Frontend typecheck and production build
docker build --target frontend-builder -f docker/Dockerfile .

# Duplicate-code check only
npx --yes jscpd@4.0.5 --config .jscpd.json ragtime runtime tests
```

If a local check is slow or hard to run on your machine, say that in the PR and rely on CI.

### Maintainer Release Notes

Maintainers promote tested `beta` changes to `main`. The repository is configured so `mattv8` and official maintainers can promote `beta` to `main` directly without a PR when needed; outside contributors should still use PRs into `beta`.

Container labels:

- Local Vite development displays a `dev` badge next to the Ragtime brand.
- Non-main container builds display a badge matching the originating branch name, such as `beta`.
- Main builds do not display a branch badge; they display the short commit SHA.

Ragtime does not use manual release version numbers. Build identity is automatic and comes from Git:

- `main` images publish `:main`, `:latest`, and `:<short-sha>`.
- `beta` images publish `:beta`, `:latest-beta`, and `:<short-sha>`.
- The application image label and in-app main build identifier use the same short SHA.

After quality testing `beta`, maintainers can fast-forward or merge `beta` into `main`. CI builds and publishes the new `main` images automatically.

```bash
git fetch origin
git checkout main
git merge --ff-only origin/beta
git push origin main
```

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
docker exec ragtime-dev python -m prisma migrate dev --name describe_the_change

# Production-like validation path
docker exec ragtime-dev python -m prisma migrate deploy
```

Do not rely on `db push` for release-bound changes. Committed migrations are the source of truth.

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

GitHub Actions separates validation from publishing:

- Every branch push runs backend quality checks, the pytest suite, and the frontend build.
- PRs into `beta` run the same required checks.
- Maintainers can promote tested `beta` changes to `main` directly; if a PR is opened into `main`, it must come from `beta`.
- Container images are published only from `beta`, `main`, and manual workflow dispatches.
- Branch tags are published as `:beta` and `:main`; beta also publishes `:latest-beta`; main also publishes `:latest`.
- Every published image also gets a short commit SHA Docker alias.
- Images are signed with Cosign and SBOM artifacts are attached.

**Required secrets:** `HARBOR_USERNAME`, `HARBOR_PASSWORD`, `COSIGN_PRIVATE_KEY`, `COSIGN_PASSWORD`
