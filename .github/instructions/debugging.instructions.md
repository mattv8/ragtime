---
applyTo: '**'
---
# Debugging

## Start With Runtime Health

```bash
# API readiness + index loading status
curl -s http://localhost:8000/health | jq .

# Service logs
docker logs -f ragtime-dev
docker logs -f ragtime-db-dev
docker logs -f runtime-dev      # User Space runtime manager/worker
```

If `/health` shows `status=initializing`, check for startup failures in `ragtime/main.py` lifecycle (DB connect, `rag.initialize()`, index recovery/cleanup).

The dev stack is shared between worktrees and serves one checkout at a time. Before trusting what these commands show, confirm it is serving your checkout; switch it only with `scripts/switch-dev-worktree.sh` (see `.agents/dev-worktree-switching.md`), never with a bare `docker compose` command.

The OpenAPI UI at `/docs` is served only when `DEBUG_MODE=true`.

## Database Access (Prisma/Postgres)

```bash
docker exec -i ragtime-db-dev bash -c \
  'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "YOUR_QUERY"'
```

Useful tables: `users`, `sessions`, `ldap_config`, `conversations`, `chat_tasks`, `index_jobs`, `index_metadata`, `index_configs`, `tool_configs`, `app_settings`.

Prompt debug table (DEBUG mode only): `provider_prompt_debug_records`.

```bash
# Latest prompt-debug rows for one conversation
docker exec -i ragtime-db-dev bash -c \
  'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
   -c "select provider,model,mode,request_kind,created_at from provider_prompt_debug_records where conversation_id = '\''<conversation_id>'\'' order by created_at desc limit 20;"'

# Prompt-debug rows for a specific background task
docker exec -i ragtime-db-dev bash -c \
  'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
   -c "select provider,model,request_kind,created_at from provider_prompt_debug_records where chat_task_id = '\''<task_id>'\'' order by created_at asc;"'

# Compare userspace vs chat prompt payloads (prompt additions + reminders)
docker exec -i ragtime-db-dev bash -c \
  'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
   -c "select mode,left(turn_reminders,120) as reminders,left(prompt_additions,120) as additions,created_at from provider_prompt_debug_records where conversation_id = '\''<conversation_id>'\'' order by created_at desc limit 20;"'
```

## Auth + Session Troubleshooting

```bash
# Detect cookie/security config mismatches
curl -s http://localhost:8000/auth/status | jq .

# Login and capture ragtime_session cookie value
TOKEN=$(docker exec ragtime-dev bash -lc 'curl -s -c - -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"$LOCAL_ADMIN_USER\",\"password\":\"$LOCAL_ADMIN_PASSWORD\"}" \
  | awk "/ragtime_session/ {print \\$NF}"')

# Authenticated admin endpoint smoke test
curl -s http://localhost:8000/indexes/jobs -H "Cookie: ragtime_session=$TOKEN" | jq .
```

If login appears successful but requests are anonymous, check `SESSION_COOKIE_SECURE` vs actual HTTP/HTTPS protocol in `/auth/status`.

## Frontend Testing on Port 8000

When testing against port 8000 (non-Vite production build), frontend changes require manual rebuild:
```bash
docker exec ragtime-dev sh -c "cd /ragtime/ragtime/frontend && npm run build"
```
Port 8001 (Vite dev server) has hot-reload and does not require this step.

## Indexing and Tool Debugging

```bash
# Index jobs + metadata
curl -s http://localhost:8000/indexes/jobs -H "Cookie: ragtime_session=$TOKEN" | jq .
curl -s http://localhost:8000/indexes -H "Cookie: ragtime_session=$TOKEN" | jq .

# Tool config state from DB (enabled + connection config)
docker exec -i ragtime-db-dev bash -c \
  'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
   -c "select id,name,tool_type,enabled from tool_configs order by name;"'
```

For tool-call failures, inspect validators in `ragtime/core/security.py` (`validate_sql_query`, `validate_odoo_code`, `validate_ssh_command`) before changing agent logic.

## Schema/Prisma Issues

```bash
# Regenerate Prisma client after schema changes
docker exec ragtime-dev python -m prisma generate

# Apply committed migrations
docker exec ragtime-dev python -m prisma migrate deploy
```

In this repo, do not rely on `db push` for production-like behavior; migrations are the source of truth.

## MCP Route Debugging

```bash
# MCP debug endpoints (admin auth required): list the MCP tools
curl -s http://localhost:8000/mcp-debug/tools -H "Cookie: ragtime_session=$TOKEN" | jq .

# Include tools that are failing their health check
curl -s "http://localhost:8000/mcp-debug/tools?include_unhealthy=true" -H "Cookie: ragtime_session=$TOKEN" | jq .
```

`POST /mcp-debug/tools/{tool_name}/call` runs one tool directly, with the same admin auth.

When MCP routes behave differently from chat API, check per-route auth/filter settings and cached route servers in `ragtime/mcp/routes.py`.

When a client reports a missing tool, establish its connector URL and caller identity first and check `mcp_request_logs.auth_method`. A list of only `workspace_development*` tools is a development-only connection, not a broken one. Read `.agents/mcp-byoh-vs-custom-routes.md` before changing route auth or recommending a connector URL.

## Area-Specific Procedures

- Chat history or branch failures: `.agents/chat-branch-debugging.md`.
- User Space runtime, workspace files, mounts, or preview proxies: `.agents/userspace-runtime.md`.
- Durable workspace SQLite capture and restore: `.agents/userspace-sqlite-history.md`.
- Slow requests: "Request timing diagnostics" in `CONTRIBUTING.md`.
