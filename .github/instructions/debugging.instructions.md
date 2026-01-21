---
applyTo: '**'
---
# Debugging

## Database Access
```bash
docker exec -i ragtime-db-dev bash -c \
  'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "YOUR_QUERY"'
```
**Tables**: `users`, `sessions`, `ldap_configs`, `conversations`, `index_jobs`, `index_metadata`, `index_configs`, `app_settings`, `tool_configs`

## Frontend Testing on Port 8000

When testing against port 8000 (non-Vite production build), frontend changes require manual rebuild:
```bash
docker exec ragtime-dev sh -c "cd /ragtime/ragtime/frontend && npm run build"
```
Port 8001 (Vite dev server) has hot-reload and does not require this step.

## API Auth Testing
```bash
# Get token (uses container env vars)
TOKEN=$(docker exec ragtime-dev bash -c 'curl -s -c - -X POST http://localhost:8000/auth/login -H "Content-Type: application/json" -d "{\"username\":\"$LOCAL_ADMIN_USER\",\"password\":\"$LOCAL_ADMIN_PASSWORD\"}" | grep ragtime_session | awk "{print \$NF}"')

# Use token
curl -s http://localhost:8000/indexes/conversations \
  -H "Cookie: ragtime_session=$TOKEN" | jq .
```
