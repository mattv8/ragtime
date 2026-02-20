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
```

If `/health` shows `status=initializing`, check for startup failures in `ragtime/main.py` lifecycle (DB connect, `rag.initialize()`, index recovery/cleanup).

## Database Access (Prisma/Postgres)

```bash
docker exec -i ragtime-db-dev bash -c \
  'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "YOUR_QUERY"'
```

Useful tables: `users`, `sessions`, `ldap_config`, `conversations`, `chat_tasks`, `index_jobs`, `index_metadata`, `index_configs`, `tool_configs`, `app_settings`.

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
# MCP debug endpoints (admin auth required)
curl -s http://localhost:8000/mcp-debug/status -H "Cookie: ragtime_session=$TOKEN" | jq .
```

When MCP routes behave differently from chat API, check per-route auth/filter settings and cached route servers in `ragtime/mcp/routes.py`.
