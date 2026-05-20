# Contributing to Ragtime

## Quick Start

To get started with contributing, follow these bare-minimum steps:
1. **Fork the repository** to your personal GitHub account.
2. **Clone your fork** and configure the environment:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ragtime.git
   cd ragtime
   cp .env.example .env
   # macOS Colima users: echo 'DOCKER_SOCKET_PATH=$HOME/.colima/default/docker.sock' >> .env
   ```
3. **Start the development stack:**
   - **VS Code:** Press `Ctrl+Shift+P` → `Tasks: Run Task` → `Start Development Stack`
   - **Command line:** `docker compose -f docker/docker-compose.dev.yml up --build`
4. **Access the services:**
   - **UI (Vite hot-reload):** http://localhost:8001
   - **API:** http://localhost:8000
   - **API Docs:** http://localhost:8000/docs
5. **Make your changes** on a new branch.
6. **Open a Pull Request** against the **`beta`** branch. All PRs must target `beta` (not `main`).

## UX Style Guide

When contributing to the frontend and user experience, follow these design principles:
- **Clean and Lean:** Keep the interface minimal and uncluttered. If an element doesn't serve a clear purpose, remove it.
- **Discoverability:** Rely on standard, intuitive UX patterns. Users should be able to figure out the system naturally.
- **Guided Setup (Wizards):** Use paginated multi-step wizards for complex workflows or setup steps to guide users sequentially and reduce cognitive load.
- **Tabbed Modals & Grouping:** Categorically group related UI elements. Use tabbed modals, panels, and distinct contextual groups rather than overwhelming users with long, scrolling forms.
- **Consistency:** Reuse existing components from `ragtime/frontend/src/components` and stick to the established design language. Use ubiquitous elements like `lucide-react` for icons to prevent disjointed designs. Avoid introducing new patterns for one-off features.

## Codebase Overview

Ragtime is built to be easily navigated and understood, especially with the help of AI coding agents. The core components are:
- **Core API & Auth:** A FastAPI backend providing OpenAI-compatible endpoints, user authentication, and Model Context Protocol (MCP) server integration.
- **User Space:** Agentic sandboxes/workspaces backed by project files, access roles, and isolated runtime execution.
- **Persistence & Retrieval:** Data is managed via PostgreSQL and Prisma, utilizing pgvector and FAISS for embeddings and retrieval.

## Testing

When contributing new features or modifying existing logic, follow the project's testing principles in `tests/`:
- **Framework:** The project uses Python's `unittest` (`unittest.IsolatedAsyncioTestCase` for async) and runs them using `pytest`.
- **Integration over Isolation:** Prefer testing the actual flow with real infrastructure (like the database) over heavily-mocked unit tests. Mock only external boundaries like LLM providers or third-party APIs.
- **FastAPI Routes:** Test routes using raw `starlette.requests.Request` objects instead of a full ASGI `TestClient` where possible.
- **Run Tests:** Execute `pytest` or target specific files with `pytest tests/test_file.py`.

## Pull Request Checklist

- Keep the PR focused on one bug fix, feature, or cleanup.
- Explain the behavior change in plain language.
- Add or update tests when behavior changes.
- Mention any database, auth, runtime, or configuration impact.
- For UI changes, include a screenshot or short note about what you checked.

### Local Checks
If you want to run validations locally before opening a PR:
```bash
# Full backend quality gate and test suite
docker build --target python-test -f docker/Dockerfile .

# Frontend typecheck and production build
docker build --target frontend-builder -f docker/Dockerfile .
```

## Adding Tools

Tools are auto-discovered from `ragtime/tools/`. Create a new file and implement a `StructuredTool` export:

```python
# ragtime/tools/my_tool.py
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class MyToolInput(BaseModel):
  query: str = Field(description="What to search or process")


async def execute_my_tool(query: str) -> str:
  return f"Processed: {query}"


my_tool_tool = StructuredTool.from_function(
  coroutine=execute_my_tool,
  name="my_tool",
  description="Describe when this tool should be used.",
  args_schema=MyToolInput,
)
```

The module name and export should follow the discovery convention (`my_tool.py` -> `my_tool_tool`). Enable via the Tools tab in the UI.

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
docker exec ragtime-dev backup --files-only > data-only.tar.gz

# Restore (copy file into container first)
docker cp backup.tar.gz ragtime-dev:/tmp/backup.tar.gz
docker exec ragtime-dev restore /tmp/backup.tar.gz
docker exec ragtime-dev restore --include-secret /tmp/backup.tar.gz  # Restore encryption key
docker exec ragtime-dev restore --db-only /tmp/backup.tar.gz
docker exec ragtime-dev restore --mirror-local-admin-from some_user /tmp/backup.tar.gz # Local admin will impersonate
docker exec ragtime-dev restore --files-only --replace-existing-data /tmp/backup.tar.gz
```

> **Important:** Database backups contain encrypted secrets (API keys, passwords), while the encryption key lives in the Ragtime data directory at `.encryption_key`. Use `--include-secret` to include the key file in your backup. Without the key, you will need to re-enter all passwords after restore. `--files-only` backs up the whole Ragtime data directory, including indexes and userspace workspaces.

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
