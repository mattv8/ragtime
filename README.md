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
  <img src=".github/images/Screenshot 2026-02-28 131359.png " alt="Screenshot 1" height="360" />
</div>

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Post-Install Setup](#post-install-setup)
- [Concepts](#concepts)
- [Integrations](#integrations)
- [Operations](#operations)
- [Contributing](#contributing)
- [License](#license)

## Overview

What Ragtime provides and how the main pieces fit together.

### Features

- **Chat UI** built in, with tool visualization, interactive charts, and DataTables: no external client required
- **[Workspaces](#workspaces)** with live previews run in isolated runtime sessions; shared links use clean public URLs (`/{owner}/{slug}`), with optional password-protected full-page access
- **[MCP server](#model-context-protocol-mcp-integration)** (HTTP Streamable + stdio transports) exposing tools to Claude Desktop, VS Code Copilot, Cursor, and JetBrains IDEs with auth
- **OpenAI-compatible API** `/v1/chat/completions` endpoint with streaming: works with [OpenWebUI](#connecting-to-openwebui), Continue, and any OpenAI client
- **Dual vector store**: Choose FAISS or pgvector for Upload/Git indexes; pgvector for schema/PDM and optional filesystem indexing ([details](#vector-store-abstraction))
- **[Tool security](#security)**: SQL injection prevention via allowlist patterns, LIMIT enforcement, Odoo code validation, optional write-ops flag

### Architecture

```mermaid
flowchart LR
  Tools["Tools<br/>(SQL, SSH, Odoo)"] -->|tool runs + results| Ragtime
  Context["Knowledge Sources<br/>(FAISS, pgvector)"] -->|retrieved context| Ragtime
  LLM["LLM Provider<br/>(OpenAI, Anthropic, Ollama, llama.cpp, LM Studio)"] -->|LLM API| Ragtime

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

## Quick Start

### Prerequisites

- Docker and Docker Compose
- A `.env` file with your configuration

### Setup

1. **Create `.env`:**

  Copy [.env.example](.env.example) to [.env](.env).
  The expanded block below is CI-synced from [.env.example](.env.example), so future edits should go there instead of the README snippet:

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

   # Optional comma-separated OAuth callback URLs to trust in addition to built-in
   # loopback/IDE callback handling.
   # OAUTH_TRUSTED_REDIRECT_URIS=https://example-1/oauth/callback,https://example-2/oauth/callback

   # -----------------------------------------------------------------------------
   # Server Configuration
   # -----------------------------------------------------------------------------
   # API port (default: 8000)
   PORT=8000

   # CORS allowed origins (comma-separated)
   # Leave empty to allow loopback-only origins.
   # Example: https://ragtime.example.com,https://chat.example.com
   ALLOWED_ORIGINS=

   # Canonical public Ragtime origin (scheme + host, no trailing slash), for example:
   # https://ragtime.example.com
   # Set it only when Ragtime sees a different origin than users do
   # (for example behind TLS termination) or public URLs must stay fixed.
   # EXTERNAL_BASE_URL=https://ragtime.example.com

   # Optional preview base-domain override for userspace subdomain previews.
   # Leave unset to derive preview hosts from the current Ragtime origin
   # (for example https://ragtime.example.com -> https://<workspace>.ragtime.example.com).
   # Set this when previews should use a separate wildcard host family such as
   # example-userspaces.com. Wildcard DNS/TLS must route *.example-userspaces.com
   # back to Ragtime.
   # USERSPACE_PREVIEW_BASE_DOMAIN=example-userspaces.com

   # -----------------------------------------------------------------------------
   # Security Configuration
   # -----------------------------------------------------------------------------
   # API Key for OpenAI-compatible endpoint authentication (strongly recommended
   # for non-local deployments)
   # Generate with: openssl rand -base64 32
   API_KEY=

   # Optional GitHub OAuth App client ID for GitHub Copilot device auth flow.
   # If not set, Ragtime falls back to the built-in default client id.
   # Create your own OAuth app to control consent screen branding (e.g., app name "Ragtime").
   # Note: this is the Client ID only (not the Client Secret).
   # GITHUB_COPILOT_CLIENT_ID=Ovxxxxxxxxxxxxxxxxxx

   # HTTPS: Enable built-in TLS with self-signed certificate (auto-generated on first run)
   # To use your own certs, place them at ./data/ssl/server.crt and ./data/ssl/server.key
   # ENABLE_HTTPS=true

   # Set to true if behind an HTTPS reverse proxy (nginx, Caddy, Traefik)
   # This marks cookies as Secure. Auto-enabled when ENABLE_HTTPS=true.
   SESSION_COOKIE_SECURE=false

   # -----------------------------------------------------------------------------
   # Runtime Configuration
   # -----------------------------------------------------------------------------

   # Base URL Ragtime uses for outbound calls to the runtime-manager API that
   # creates and controls userspace runtime sessions. In the default Docker setup,
   # this should stay pointed at the internal runtime service.
   # Set to blank/non-http to force local placeholder runtime mode.
   # RUNTIME_MANAGER_URL=http://runtime:8090

   # Optional bearer token for runtime-manager calls
   # RUNTIME_MANAGER_AUTH_TOKEN=

   # Optional bearer token for preview proxy calls to the runtime worker
   # RUNTIME_WORKER_AUTH_TOKEN=

   # Maximum concurrent runtime sessions in running/starting state (default: 12)
   # RUNTIME_MAX_SESSIONS=12

    # Chat web search uses the bundled internal SearXNG service by default.
    # Set TAVILY_API_KEY to use Tavily instead (get a key at https://app.tavily.com/home).
    # TAVILY_API_KEY=

   ############################################################
   # Developer Only - Typically do not modify below this line #
   ############################################################

   # Debug mode (enables verbose logging and hot-reload)
   DEBUG_MODE=false

   # Database URL (auto-configured by docker-compose, override for external DB)
   # DATABASE_URL=postgresql://ragtime:password@hostname:5432/ragtime
   ```

   </details>

2. **Edit `.env`** with your actual values.

3. **Create `docker-compose.yml`** if you want the standalone self-hosted compose setup shown below:

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
         RUNTIME_WORKER_AUTH_TOKEN: ${RUNTIME_WORKER_AUTH_TOKEN:-runtime-worker-token}
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
       # Uncomment below to enable full runtime sandbox isolation (pivot_root + mount namespace)
       # privileged: true
       # cap_add:
       #   - SYS_ADMIN
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
  - API docs: http://localhost:8000/docs (available when `DEBUG_MODE=true`)

   Default credentials: `admin` / (set via `LOCAL_ADMIN_PASSWORD` in `.env`)

## Post-Install Setup

After the stack is running, configure the capabilities Ragtime will expose.

### Tool Configuration

<div align="center">
  <img src=".github/images/Screenshot 2026-02-28 131951.png" alt="Screenshot 2" height="360" />
</div>

Configure tools in the Ragtime web UI before connecting MCP clients. Enabled tools are available to chat; MCP exposure additionally applies heartbeat health filtering.

1. Open the web UI at http://localhost:8000 and log in with your admin account.
2. Navigate to the **Tools** tab.
3. Click **Add Tool** and select the tool type (PostgreSQL, Odoo, SSH, filesystem indexer, etc.).
4. Fill in connection details (hostnames, credentials, database names, paths) for each tool.
5. Use the built-in test button to verify each tool connection.

Once the required tools are healthy, continue with [MCP Integration](#model-context-protocol-mcp-integration).

### Creating Indexes

The Indexer UI (http://localhost:8000, **Indexes** tab) supports multiple index types:

| Method | Vector Store | Storage | Use Case |
|--------|--------------|---------|----------|
| **Upload** (zip/tar) | FAISS or pgvector | FAISS: `data/indexes/<name>/`<br/>pgvector: `filesystem_embeddings` table | Static codebases, documentation snapshots |
| **Git Clone** | FAISS or pgvector | FAISS: `data/indexes/<name>/`<br/>pgvector: `filesystem_embeddings` table | Repositories with optional private token auth |
| **Filesystem** | FAISS or pgvector | FAISS: `data/indexes/<name>/`<br/>pgvector: `filesystem_embeddings` table | Live SMB/NFS shares, Docker volumes, local paths: incremental re-index |
| **Schema** | pgvector | `schema_embeddings` table | Auto-generated from PostgreSQL/MSSQL/MySQL tools (enable in [Tool Configuration](#tool-configuration)) |
| **PDM** | pgvector | `pdm_embeddings` table | SolidWorks PDM metadata via SQL Server |

Jobs run async with progress streaming to the UI.

### Preview DNS Setup (Reverse Proxy Deployments)

This deployment step configures userspace preview subdomains. For workspace behavior context, see [Workspaces](#workspaces).

Userspace previews use per-workspace subdomains. By default, Ragtime derives the preview host from its public origin: `https://ragtime.example.com` becomes `https://<workspace>.ragtime.example.com`. Set `USERSPACE_PREVIEW_BASE_DOMAIN` only if previews should use a different wildcard domain such as `example-userspaces.com`.

1. Choose whether to keep the derived preview host family or override it explicitly:
  - Derived default: `https://ragtime.example.com` -> `https://<workspace>.ragtime.example.com`
  - Explicit override: `USERSPACE_PREVIEW_BASE_DOMAIN=example-userspaces.com` -> `https://<workspace>.example-userspaces.com`
2. Add a wildcard DNS record for the chosen preview base domain:
  - `*.ragtime.example.com` or `*.example-userspaces.com` -> your public Ragtime entrypoint (A, AAAA, or CNAME)
3. Ensure your reverse proxy accepts wildcard hosts for that domain and routes them to Ragtime.
4. Ensure TLS certificates cover the wildcard domain if you use HTTPS.
5. Set `.env` variables only when needed:

```bash
# Set this when the public Ragtime origin differs from what the app sees on
# incoming requests, such as behind TLS termination.
EXTERNAL_BASE_URL=https://ragtime.example.com

# Set this only if previews should use a different wildcard host family than
# the main Ragtime host.
USERSPACE_PREVIEW_BASE_DOMAIN=example-userspaces.com
```

Notes:
- Leave `EXTERNAL_BASE_URL` unset unless Ragtime sees a different origin than users do, or you need public URLs pinned to a canonical host.
- Keep `SESSION_COOKIE_SECURE=true` when traffic is HTTPS at the edge.
- Forward `Host` and `X-Forwarded-Proto` headers through the proxy.
- In local development with `DEBUG_MODE=true`, Ragtime uses `userspace-preview.lvh.me` automatically.

## Concepts

Core concepts that affect how Ragtime is deployed and used.

### Workspaces

**Workspaces** combine files, conversations, selected infrastructure tools, and an isolated runtime preview session in one agentic sandbox. They are Replit-like, but wired into Ragtime's tools and indexed context so agents can work against live systems.

- Previews are runtime-only and proxied from session-managed devservers.
- Public sharing uses direct routes (`/{owner}/{slug}` and `/shared/{token}`); both route families launch shared previews.
- Password-protected shares are handled server-side with a full-page prompt.
- Preview DNS and reverse-proxy setup is in [Preview DNS Setup (Reverse Proxy Deployments)](#preview-dns-setup-reverse-proxy-deployments).

### Vector Store Abstraction

Ragtime uses **two vector backends**: **FAISS** (in-memory, loaded at startup) and **pgvector** (PostgreSQL, persistent). Upload, Git, and Filesystem indexes can use either backend.

See [Creating Indexes](#creating-indexes) for a detailed breakdown of index types and their storage backends.

FAISS indexes are loaded into memory at startup; pgvector indexes stay in PostgreSQL and use cosine similarity search. Embedding provider (OpenAI, Ollama, llama.cpp, or LM Studio) is configured once in Settings and applies to all index types. Swapping embedding model or dimensions after initial indexing requires a full re-index.

## Integrations

How to connect external clients and coding assistants to Ragtime.

### Model Context Protocol (MCP) Integration

Ragtime exposes its tools via the [Model Context Protocol](https://modelcontextprotocol.io), allowing AI coding assistants to interact with your databases, execute shell commands, and search your indexed codebases.

By default, MCP is disabled until you enable it in Settings. If disabled, `/mcp` responds with HTTP 503.

#### Available MCP Tools

Tools are dynamically exposed based on what you configure in the UI.
The schemas below list the primary fields; several tools also accept optional fields such as `timeout`.

| Tool Type | Input Schema |
|-----------|-------------|
| `postgres` | `{query, reason}` |
| `mssql` | `{query, reason}` |
| `mysql` | `{query, reason}` |
| `influxdb` | `{query, reason}` |
| `odoo_shell` | `{code, reason}` |
| `ssh_shell` | `{command, reason}` |
| `filesystem_indexer` | `{query, max_results}` |
| `solidworks_pdm` | `{query, document_type}` |
| `knowledge_search` | `{query, index_name}` |
| `schema_search` | `{prompt, limit}` |
| `git_history` | `{action, ...}` |

#### MCP Server Setup

##### HTTP Transport (Recommended)

Ragtime exposes an MCP endpoint at `/mcp` that supports the Streamable HTTP transport. Add this to your MCP client configuration:

```json
{
	"servers": {
		"ragtime": {
			"url": "http://localhost:8000/mcp",
			"type": "http",
      // If you've enabled MCP authentication in the Ragtime Settings UI (Settings > MCP Configuration), add headers for your configured auth method.
      // Password mode supports either `MCP-Password` or `Authorization: Bearer <password>`:
			// "headers": {
      //   "MCP-Password": "your-mcp-password-here"
      //   // or: "Authorization": "Bearer your-mcp-password-here"
			// }
		}
	},
	"inputs": []
}
```

> **NOTE:** For remote access, replace `localhost:8000` with your server URL.

##### Stdio Transport (Alternative)

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

### Connecting to OpenWebUI

1. In OpenWebUI, go to **Settings** > **Connections** > **OpenAI API**
2. Add a new connection:
   - **API Base URL**: `http://ragtime:8000/v1` (or `http://localhost:8000/v1` if running locally)
   - **API Key**: Your configured `API_KEY` (or any value if not set)
3. Select your server's model name (default: "ragtime", configurable in Settings > Server Branding)

## Operations

Security, maintenance, and troubleshooting guidance for self-hosted deployments.

### Security

Ragtime is designed for self-hosted deployment on trusted networks. Review these recommendations before exposing it beyond localhost:

CI builds each push; main-branch images are Cosign-signed and ship with an SPDX SBOM artifact (linked from the badges above and workflow runs) so you can verify what you pull from the registry.

#### Network & Access
- **Run behind a reverse proxy or firewall.** Avoid exposing port 8000 directly to the public internet.
- **Set `API_KEY`** to protect the `/v1/chat/completions` endpoint. When unset (the default), anyone with network access can call the chat API and invoke your configured tools.
- **Restrict `ALLOWED_ORIGINS`** to trusted domains. The application default is loopback-only when unset; avoid using `*` in network-accessible deployments because it is permissive with `allow_credentials=True`.
- **Enable MCP route authentication** via Settings UI if `/mcp` is network-accessible. MCP is disabled by default; when MCP is enabled, the default route is open unless you turn on route authentication.
- MCP authentication supports password-based headers (including `MCP-Password` and bearer form) and OAuth2/client_credentials route modes.
- Set a strong `LOCAL_ADMIN_PASSWORD` when deploying.

#### Authentication Security
- **Encryption key is auto-generated** on first startup and stored at `data/.encryption_key`. Include this file in your backups using `backup --include-secret` or your encrypted secrets will be unrecoverable.
- **Rate limiting** protects the login endpoint (5 attempts/minute per IP) to prevent brute-force attacks. In `DEBUG_MODE=true`, rate limiting is disabled for local testing.

#### Debug Mode Warning
**Do not use `DEBUG_MODE=true` outside local development.** When enabled, the `/auth/status` endpoint exposes your admin username and password in plaintext (including unauthenticated callers). This is intentional for self-hosted debugging but dangerous if the server is accessible to untrusted users.

#### SSH Connections
The SSH tool uses Paramiko with `AutoAddPolicy`, which accepts any host key without verification. This makes SSH connections vulnerable to man-in-the-middle attacks on first connect. Only use the SSH tool on trusted networks or with hosts you have verified out-of-band.

#### Docker & Mounts
- The default compose files include mounts for `docker.sock` and optional privileged flags to support advanced tool features (container exec, SSH tunnels, NFS/SMB mounts).
- If you do not need these features, remove or comment out the corresponding lines in your compose file.
- For NFS/SMB filesystem indexing, the container may require elevated privileges. Consider the security implications before enabling `privileged: true` or `SYS_ADMIN` capabilities.

#### Third-Party Data Relay
Queries and tool calls may forward your data to external services you configure (OpenAI, Anthropic, Ollama, llama.cpp, LM Studio, PostgreSQL, MSSQL, SSH hosts). Only connect to services you trust with your data.

### Updating

To update to the latest version:

```bash
docker compose pull
docker compose up -d
```

### Troubleshooting

#### SSH Tools Fail from Colima on macOS but Hosts Are Reachable

SSH checks run from inside the Colima VM, so macOS-reachable hosts may still fail if Colima's user-mode networking is stale. To confirm, test TCP connectivity from each layer:

```bash
nc -vz -w 3 <HOST_IP> 22                                                                      # macOS
colima ssh -- bash -lc 'timeout 4 bash -lc "</dev/tcp/<HOST_IP>/22" && echo open || echo failed'  # Colima VM
docker exec ragtime-dev bash -lc 'timeout 4 bash -lc "</dev/tcp/<HOST_IP>/22" && echo open || echo failed'  # container
```

If only macOS can connect, restart Colima and the stack:

```bash
colima stop && colima start
docker compose -f docker/docker-compose.dev.yml up --build
```

If the problem persists, kill stale Lima usernet helpers: `ps ax | grep 'limactl usernet'` and keep only the PID in `~/.colima/_lima/_networks/user-v2/usernet_user-v2.pid`.

#### NumPy CPU Compatibility Error

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
