# Ragtime

Self-hosted, OpenAI-compatible RAG API + MCP server that plugs local knowledge into existing LLM clients.

<p align="center">
  <a href="https://github.com/mattv8/ragtime/actions/workflows/build-container.yml?query=branch%3Amain">
    <img src="https://github.com/mattv8/ragtime/actions/workflows/build-container.yml/badge.svg?branch=main" alt="Build" />
  </a>
  <a href="https://github.com/mattv8/ragtime/actions/workflows/build-container.yml?query=branch%3Amain+is%3Asuccess">
    <img src="https://img.shields.io/badge/Container-signed%20with%20Cosign-0a7cff" alt="Container Signed" />
  </a>
  <a href="https://github.com/mattv8/ragtime/actions/workflows/build-container.yml?query=branch%3Amain+is%3Asuccess">
    <img src="https://img.shields.io/badge/SBOM-SPDX%20artifact-4c1" alt="SBOM" />
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT" />
  </a>
</p>

<p align="center">
  <a href="CONTRIBUTING.md">Contributing Guide</a>
</p>

<div align="center">
  <img src=".github/images/2026-01-12.png" alt="Screenshot 1" height="360" />
  <img src=".github/images/Screenshot%202026-01-12%20110434.png" alt="Screenshot 2" height="360" />
</div>

## Features

- **OpenAI-compatible `/v1/chat/completions`** endpoint with streaming: works with [OpenWebUI](#connecting-to-openwebui), Continue, and any OpenAI client
- **Built-in chat UI** at `/` with tool visualization, interactive charts, and DataTables: no external client required
- **[MCP server](#model-context-protocol-mcp-integration)** (HTTP Streamable + stdio transports) exposing tools to Claude Desktop, VS Code Copilot, Cursor, and JetBrains IDEs with auth
- **Dual vector store**: Choose FAISS or pgvector for Upload/Git indexes; pgvector for schema/PDM and optional filesystem indexing ([details](#vector-store-abstraction))
- **[LangChain tool calling](#tool-configuration)**: PostgreSQL, MSSQL, MySQL/MariaDB, Odoo ORM, SSH shell: read-only by default, write-ops opt-in
- **Auto-discovered tools**: drop a `<name>_tool` StructuredTool in `ragtime/tools/` and it registers at startup (see [CONTRIBUTING.md](CONTRIBUTING.md))
- **[Tool security](#security)**: SQL injection prevention via allowlist patterns, LIMIT enforcement, Odoo code validation, optional write-ops flag
- **Auth**: Local admin + optional LDAP, rate-limited login, encrypted secrets ([Fernet](https://cryptography.io/en/latest/fernet/)), httpOnly session cookies

## Architecture

```mermaid
flowchart LR
  Tools["Tools<br/>(SQL, SSH, Odoo)"] -->|tool runs + results| Ragtime
  Context["Knowledge Sources<br/>(FAISS, pgvector)"] -->|retrieved context| Ragtime
  LLM["LLM Provider<br/>(OpenAI, Anthropic, Ollama)"] -->|LLM API| Ragtime

  subgraph Ragtime["Ragtime"]
    direction TB
    API["/v1/chat/completions"]
    MCP["/mcp"]
    UI["Web UI"]
  end

  Ragtime -->|chat responses| Clients["Clients<br/>(OpenWebUI, Claude, VS Code)"]
  Clients -->|chat queries| Ragtime

  style Ragtime fill:#1a365d,stroke:#3182ce,stroke-width:3px,color:#fff
```

## Vector Store Abstraction

Ragtime uses **two vector backends**: **FAISS** (in-memory, loaded at startup) and **pgvector** (PostgreSQL, persistent). Upload and Git indexes use a unified indexer and can use either backend.

See [Creating Indexes](#creating-indexes) for a detailed breakdown of index types and their storage backends.

FAISS indexes are loaded into memory at startup; pgvector indexes stay in PostgreSQL and use cosine similarity search. Embedding provider (OpenAI or Ollama) is configured once in Settings and applies to all index types. Swapping embedding model or dimensions after initial indexing requires a full re-index.

## Who This Is For / Not For

**Good fit**: Teams that want to query internal docs and databases through existing chat UIs without shipping data to third-party RAG SaaS: especially if you already run OpenWebUI, Claude Desktop, or VS Code with MCP.

**Not a fit**: If you need multi-tenant SaaS, fine-grained RBAC beyond admin/user, or sub-second latency at scale, consider managed solutions or a custom LangServe deployment.

## Table of Contents

- [Quick Start](#quick-start)
- [Tool Configuration](#tool-configuration)
- [Creating Indexes](#creating-indexes)
- [Model Context Protocol (MCP) Integration](#model-context-protocol-mcp-integration)
- [Connecting to OpenWebUI](#connecting-to-openwebui)
- [Security](#security)
- [Updating](#updating)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

### Prerequisites

- Docker and Docker Compose
- A `.env` file with your configuration

### Setup

1. **Create environment file:**

  Create a file named [.env](.env) with the following content (you can also copy from the full example in [.env.example](.env.example)):

   <details>
   <summary>Click to expand .env template</summary>

   ```bash
   # =============================================================================
   # Ragtime RAG API - Environment Configuration
   # =============================================================================
   # Copy this file to .env and fill in your values

   # -----------------------------------------------------------------------------
   # Database Configuration
   # -----------------------------------------------------------------------------
   # PostgreSQL password (used by both database container and ragtime)
   POSTGRES_PASSWORD=changeme

   # -----------------------------------------------------------------------------
   # Authentication Configuration
   # -----------------------------------------------------------------------------
   # Local admin account credentials
   LOCAL_ADMIN_USER=admin
   LOCAL_ADMIN_PASSWORD=changeme_admin

   # -----------------------------------------------------------------------------
   # Server Configuration
   # -----------------------------------------------------------------------------
   # API port (default: 8000)
   PORT=8000

   # CORS allowed origins (comma-separated, or * for all)
   ALLOWED_ORIGINS=*

   # -----------------------------------------------------------------------------
   # Security Configuration
   # -----------------------------------------------------------------------------
   # API Key for OpenAI-compatible endpoint authentication (REQUIRED)
   # Generate with: openssl rand -base64 32
   API_KEY=

   # HTTPS: Enable built-in TLS with self-signed certificate (auto-generated on first run)
   # To use your own certs, place them at ./data/ssl/server.crt and ./data/ssl/server.key
   # ENABLE_HTTPS=true

   # Set to true if behind an HTTPS reverse proxy (nginx, Caddy, Traefik)
   # This marks cookies as Secure. Auto-enabled when ENABLE_HTTPS=true.
   SESSION_COOKIE_SECURE=false

   ############################################################
   # Developer Only - Typically do not modify below this line #
   ############################################################

   # Debug mode (enables verbose logging and hot-reload)
   DEBUG_MODE=false

   # Database URL (auto-configured by docker-compose, override for external DB)
   # DATABASE_URL=postgresql://ragtime:password@hostname:5432/ragtime

   # Runtime manager base URL override. Set to blank/non-http to force local
   # placeholder runtime mode.
   # RUNTIME_MANAGER_URL=http://runtime:8090

   # Optional bearer token for runtime-manager calls
   # RUNTIME_MANAGER_AUTH_TOKEN=

   # Runtime manager HTTP timeout in seconds
   # RUNTIME_MANAGER_TIMEOUT_SECONDS=5

   # Runtime manager pool orchestration settings (runtime container)
   # RUNTIME_POOL_SIZE=2
   # RUNTIME_MAX_POOL_SIZE=6
   # RUNTIME_LEASE_TTL_SECONDS=3600
   # RUNTIME_IDLE_EVICT_SECONDS=1200
   # RUNTIME_RECONCILE_INTERVAL_SECONDS=15
   ```

   </details>

2. **Edit .env** and configure your specific values (see [.env.example](.env.example) for the complete sample file)

3. **Create docker-compose.yml:**

   Create a file named `docker-compose.yml` with the following content:

   <details>
   <summary>Click to expand docker-compose.yml</summary>

   ```yaml
   # =============================================================================
   # Ragtime - Self-Hosted Docker Compose
   # =============================================================================
   # For self-hosted deployment. See README.md for setup instructions.
   #
   # Usage:
   #   1. Create .env file with your configuration
   #   2. docker compose up -d
   #   3. Access at http://localhost:${PORT:-8000}

   services:
     # PostgreSQL database for Prisma persistence
     ragtime-db:
       image: pgvector/pgvector:pg18
       container_name: ragtime-db
       restart: unless-stopped
       environment:
         POSTGRES_USER: ragtime
         POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
         POSTGRES_DB: ragtime
       volumes:
         - ragtime-db-data:/var/lib/postgresql
       networks:
         - ragtime-network
       healthcheck:
         test: ["CMD-SHELL", "pg_isready -U ragtime -d ragtime"]
         interval: 10s
         timeout: 5s
         retries: 5

     # Ragtime RAG API
     ragtime:
       # For older CPUs without X86_V2 support, use the legacy tag:
       # image: hub.docker.visnovsky.us/library/ragtime:legacy
       image: hub.docker.visnovsky.us/library/ragtime:main
       container_name: ragtime
       restart: unless-stopped
       ports:
         - "${PORT:-8000}:8000"
       env_file:
         - .env
       environment:
         # Database connection (uses container network)
         DATABASE_URL: postgresql://ragtime:${POSTGRES_PASSWORD}@ragtime-db:5432/ragtime
         # Recommended defaults
         DEBUG_MODE: "false"
         RUNTIME_MANAGER_URL: ${RUNTIME_MANAGER_URL:-http://runtime:8090}
         RUNTIME_MANAGER_AUTH_TOKEN: ${RUNTIME_MANAGER_AUTH_TOKEN:-runtime-manager-token}
       volumes:
         # Data persistence (indexes, SSL certs, etc.)
         - ./data:/data
         # Docker socket for container exec (optional, for tool execution)
         - /var/run/docker.sock:/var/run/docker.sock:ro
       # Uncomment below if using SMB/NFS mounting inside container (consider mounting via docker volume instead)
       # privileged: true
       # cap_add:
       #   - SYS_ADMIN
       networks:
         - ragtime-network
       depends_on:
         ragtime-db:
           condition: service_healthy
         runtime:
           condition: service_started
       healthcheck:
         test: ["CMD", "sh", "-c", "if [ \"$ENABLE_HTTPS\" = \"true\" ]; then curl -fsk https://localhost:8000/health; else curl -fs http://localhost:8000/health; fi"]
         interval: 30s
         timeout: 10s
         retries: 3
         start_period: 15s

     runtime:
       image: hub.docker.visnovsky.us/library/runtime:main
       container_name: runtime
       restart: unless-stopped
       environment:
         PORT: "8090"
         RUNTIME_SERVICE_MODE: manager
         RUNTIME_MANAGER_AUTH_TOKEN: ${RUNTIME_MANAGER_AUTH_TOKEN:-runtime-manager-token}
         RUNTIME_WORKER_AUTH_TOKEN: ${RUNTIME_WORKER_AUTH_TOKEN:-runtime-worker-token}
         RUNTIME_WORKER_BASE_URL: ${RUNTIME_WORKER_BASE_URL:-http://runtime:8090}
         RUNTIME_WORKSPACE_ROOT: ${RUNTIME_WORKSPACE_ROOT:-/data/_userspace}
         RUNTIME_MAX_SESSIONS: ${RUNTIME_MAX_SESSIONS:-12}
         RUNTIME_LEASE_TTL_SECONDS: ${RUNTIME_LEASE_TTL_SECONDS:-3600}
         RUNTIME_RECONCILE_INTERVAL_SECONDS: ${RUNTIME_RECONCILE_INTERVAL_SECONDS:-15}
       volumes:
         - ./data:/data
       networks:
         - ragtime-network
       healthcheck:
         test: ["CMD", "curl", "-fs", "http://localhost:8090/health"]
         interval: 30s
         timeout: 10s
         retries: 3
         start_period: 5s

   networks:
     ragtime-network:
       driver: bridge

   volumes:
     ragtime-db-data:
   ```

   > **Note:** All configuration variables are loaded from the `.env` file via `env_file`. The `environment` section only overrides `DATABASE_URL` to use the container network.

   </details>

4. **Start the application:**
   ```bash
   docker compose up -d
   ```

5. **Access the application:**
   - Web UI: http://localhost:8000
   - API docs: http://localhost:8000/docs

   Default credentials: `admin` / (set via `LOCAL_ADMIN_PASSWORD` in `.env`)

## Tool Configuration

Before you connect Ragtime to MCP clients, configure tools in the Ragtime web UI so they are ready for use:

1. Open the web UI at http://localhost:8000 and log in with your admin account.
2. Navigate to the **Tools** tab.
3. Click **Add Tool** and select the tool type (PostgreSQL, Odoo, SSH, filesystem indexer, etc.).
4. Fill in connection details (hostnames, credentials, database names, paths) for each tool.
5. Use the built-in test button to verify each tool connection.

Only tools that pass their health checks are exposed to chat and MCP clients. Configure and verify your tools here before following the [MCP Integration](#model-context-protocol-mcp-integration) section below.

## Creating Indexes

The Indexer UI (http://localhost:8000, **Indexes** tab) supports multiple index types:

| Method | Vector Store | Storage | Use Case |
|--------|--------------|---------|----------|
| **Upload** (zip/tar) | FAISS or pgvector | FAISS: `data/indexes/<name>/`<br/>pgvector: `filesystem_embeddings` table | Static codebases, documentation snapshots |
| **Git Clone** | FAISS or pgvector | FAISS: `data/indexes/<name>/`<br/>pgvector: `filesystem_embeddings` table | Repositories with optional private token auth |
| **Filesystem** | pgvector | `filesystem_embeddings` table | Live SMB/NFS shares, Docker volumes, local paths: incremental re-index |
| **Schema** | pgvector | `schema_embeddings` table | Auto-generated from PostgreSQL/MSSQL/MySQL tools (enable in [Tool Configuration](#tool-configuration)) |
| **PDM** | pgvector | `pdm_embeddings` table | SolidWorks PDM metadata via SQL Server |

Jobs run async with progress streaming to the UI.

## Model Context Protocol (MCP) Integration

Ragtime exposes its tools via the [Model Context Protocol](https://modelcontextprotocol.io), allowing AI coding assistants to interact with your databases, execute shell commands, and search your indexed codebases.

### Available MCP Tools

Tools are dynamically exposed based on what you configure in the UI:

| Tool Type | Input Schema |
|-----------|-------------|
| `postgres` | `{query, reason}` |
| `mssql` | `{query, reason}` |
| `mysql` | `{query, reason}` |
| `odoo_shell` | `{code, reason}` |
| `ssh_shell` | `{command, reason}` |
| `filesystem_indexer` | `{query, max_results}` |
| `knowledge_search` | `{query, index_name}` |
| `schema_search` | `{prompt, limit}` |
| `git_history` | `{action, ...}` |

### MCP Server Setup

#### HTTP Transport (Recommended)

Ragtime exposes an MCP endpoint at `/mcp` that supports the Streamable HTTP transport. Add this to your MCP client configuration:

```json
{
	"servers": {
		"ragtime": {
			"url": "http://localhost:8000/mcp",
			"type": "http",
			// If you've enabled MCP authentication in the Ragtime Settings UI (Settings > MCP Configuration), add the `MCP-Password` header:
			// "headers": {
			//   "MCP-Password": "your-mcp-password-here"
			// }
		}
	},
	"inputs": []
}
```

> **NOTE:** For remote access, replace `localhost:8000` with your server URL.

#### Stdio Transport (Alternative)

For local development or environments where HTTP isn't preferred, use stdio transport via Docker:

```json
{
  "mcpServers": {
    "ragtime": {
      "command": "docker",
      "args": ["exec", "-i", "ragtime", "python", "-m", "ragtime.mcp"]
    }
  }
}
```

Replace `ragtime` with your container name if different (find it with `docker ps`).

**Configuration file locations:**
- **[Claude Desktop](https://claude.ai/download)**: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows)
- **[VS Code / Copilot](https://code.visualstudio.com/docs/copilot/chat/mcp-servers)**: User or workspace MCP settings
- **[Cursor](https://docs.cursor.com/context/model-context-protocol)**: `.cursor/mcp.json`
- **[Windsurf](https://docs.codeium.com/windsurf/mcp)**: `~/.codeium/windsurf/mcp_config.json`

## Connecting to OpenWebUI

1. In OpenWebUI, go to **Settings** > **Connections** > **OpenAI API**
2. Add a new connection:
   - **API Base URL**: `http://ragtime:8000/v1` (or `http://localhost:8000/v1` if running locally)
   - **API Key**: Your configured `API_KEY` (or any value if not set)
3. Select your server's model name (default: "ragtime", configurable in Settings > Server Branding)

## Security

Ragtime is designed for self-hosted deployment on trusted networks. Review these recommendations before exposing it beyond localhost:

CI builds each push; main-branch images are Cosign-signed and ship with an SPDX SBOM artifact (linked from the badges above and workflow runs) so you can verify what you pull from the registry.

### Network & Access
- **Run behind a reverse proxy or firewall.** Avoid exposing port 8000 directly to the public internet.
- **Set `API_KEY`** to protect the `/v1/chat/completions` endpoint. When unset (the default), anyone with network access can call the chat API and invoke your configured tools.
- **Restrict `ALLOWED_ORIGINS`** to trusted domains. The default `*` with `allow_credentials=True` permits cross-site requests that carry session cookies, which may be exploitable if the server is publicly reachable.
- **Enable MCP route authentication** via Settings UI if `/mcp` is network-accessible. By default the MCP endpoint is open without auth.
- Set a strong `LOCAL_ADMIN_PASSWORD` when deploying.

### Authentication Security
- **Encryption key is auto-generated** on first startup and stored at `data/.encryption_key`. Include this file in your backups using `backup --include-secret` or your encrypted secrets will be unrecoverable.
- **Rate limiting** protects the login endpoint (5 attempts/minute per IP) to prevent brute-force attacks.

### Debug Mode Warning
**Do not use `DEBUG_MODE=true` outside local development.** When enabled, the `/auth/status` endpoint exposes your admin username and password in plaintext. This is intentional for self-hosted debugging but dangerous if the server is accessible to untrusted users.

### SSH Connections
The SSH tool uses Paramiko with `AutoAddPolicy`, which accepts any host key without verification. This makes SSH connections vulnerable to man-in-the-middle attacks on first connect. Only use the SSH tool on trusted networks or with hosts you have verified out-of-band.

### Docker & Mounts
- The default compose files include mounts for `docker.sock` and optional privileged flags to support advanced tool features (container exec, SSH tunnels, NFS/SMB mounts).
- If you do not need these features, remove or comment out the corresponding lines in your compose file.
- For NFS/SMB filesystem indexing, the container may require elevated privileges. Consider the security implications before enabling `privileged: true` or `SYS_ADMIN` capabilities.

### Third-Party Data Relay
Queries and tool calls may forward your data to external services you configure (OpenAI, Anthropic, Ollama, PostgreSQL, MSSQL, SSH hosts). Only connect to services you trust with your data.

## Updating

To update to the latest version:

```bash
docker compose pull
docker compose up -d
```

## Troubleshooting

### NumPy CPU Compatibility Error

If you see an error like:
```
RuntimeError: NumPy was built with baseline optimizations (X86_V2) but your machine doesn't support (X86_V2)
```

Your CPU lacks the X86_V2 instruction set required by modern NumPy binaries. Use the legacy image instead:

```yaml
image: hub.docker.visnovsky.us/library/ragtime:legacy
```

This image builds NumPy from source without CPU-specific optimizations.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, making changes, and CI/CD details.

## License

MIT: see [LICENSE](LICENSE).
