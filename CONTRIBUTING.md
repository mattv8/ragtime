# Contributing to Ragtime

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

# Restore (reads from stdin)
cat backup.tar.gz | docker exec -i ragtime-dev restore
cat backup.tar.gz | docker exec -i ragtime-dev restore --include-secret  # Restore encryption key
cat backup.tar.gz | docker exec -i ragtime-dev restore --db-only
```

> **Important:** Backups contain encrypted secrets (API keys, passwords). Use `--include-secret` to include the encryption key file in your backup. If not using `--include-secret`, the backup command prints the `JWT_SECRET_KEY` - save this! You must set the same key in `.env` before restoring on a new server, or re-enter all passwords after restore.

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
