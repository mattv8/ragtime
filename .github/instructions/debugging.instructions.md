---
applyTo: '**'
---
# Database Debugging Queries

These queries help debug indexing jobs, metadata, and application settings. Use when investigating specific bugs.

## Guidelines
- **Token-safe**: Minimal columns; no `SELECT *`; truncate large fields with `LEFT(..., N)`
- Use `ORDER BY created_at` + `LIMIT` to focus on relevant records

## Database Access Pattern

### General Query Example
```bash
# One-off query (PREFERRED - use for all debugging)
docker exec -i ragtime-db-dev bash -c \
  'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "SELECT * FROM index_jobs LIMIT 5;"'
```

**NEVER use interactive shell** (`docker exec -it`)

### Get Database Schema
```bash
# List all tables
docker exec -i ragtime-db-dev bash -c 'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "\dt"'

# Describe table structure
docker exec -i ragtime-db-dev bash -c 'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "\d table_name"'
```

**Tables**: `index_jobs`, `index_metadata`, `index_configs`, `app_settings`, `tool_configs`
**Schema**: See `prisma/schema.prisma` for complete field definitions

## Debugging Tips
1. Check `index_jobs.status` and `error_message` for failed indexing
2. Verify `index_metadata` entries match files in `.data/` directory
3. Use `tool_configs` to confirm tool configurations (connection, enabled state, descriptions)
4. Check `tool_configs.last_test_result` and `last_test_error` for connection issues
