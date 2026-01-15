# Ragtime

OpenAI-compatible RAG API and MCP server with LangChain tool calling for business intelligence queries.

<p align="center">
  <a href="https://ragtime.dev.visnovsky.us"><strong>🚀 Live Demo (coming soon)</strong></a><br />
  <a href="CONTRIBUTING.md"><strong>📄 Contributing Guide</strong></a>
</p>

<p align="center">
  <a href="https://github.com/mattv8/ragtime/actions/workflows/build-container.yml">
    <img src="https://github.com/mattv8/ragtime/actions/workflows/build-container.yml/badge.svg?branch=main" alt="Build" />
  </a>
  <a href="https://github.com/mattv8/ragtime/actions/workflows/build-container.yml">
    <img src="https://img.shields.io/badge/Container-signed%20with%20Cosign-0a7cff" alt="Container Signed" />
  </a>
  <a href="https://github.com/mattv8/ragtime/actions/workflows/build-container.yml?query=branch%3Amain+event%3Apush">
    <img src="https://img.shields.io/badge/SBOM-SPDX%20artifact-4c1" alt="SBOM" />
  </a>
</p>

<div align="center">
  <img src=".github/images/2026-01-12.png" alt="Screenshot 1" height="360" />
  <img src=".github/images/Screenshot%202026-01-12%20110434.png" alt="Screenshot 2" height="360" />
</div>

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Security](#security)
- [Tool Configuration](#tool-configuration)
- [Model Context Protocol (MCP) Integration](#model-context-protocol-mcp-integration)
  - [MCP Server Setup](#mcp-server-setup)
    - [HTTP Transport (Recommended)](#http-transport-recommended)
    - [Stdio Transport (Alternative)](#stdio-transport-alternative)
- [Creating FAISS Indexes](#creating-faiss-indexes)
- [Connecting to OpenWebUI](#connecting-to-openwebui)
- [License](#license)
- [Updating](#updating)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Features

- **OpenAI API Compatible**: Works with OpenWebUI, ChatGPT clients, and any OpenAI-compatible interface
- **MCP Server**: Integrates with Claude and other MCP-compatible clients
- **RAG with FAISS**: Vector search over your codebase documentation
- **Tool Calling**: Execute Odoo ORM queries and PostgreSQL queries via natural language
- **Security**: Read-only by default with SQL injection and command injection prevention

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
       healthcheck:
         # -k allows self-signed certs; checks http first, falls back gracefully
         test: ["CMD", "sh", "-c", "curl -fsk https://localhost:8000/health 2>/dev/null || curl -fs http://localhost:8000/health"]
         interval: 30s
         timeout: 10s
         retries: 3
         start_period: 15s

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

## Tool Configuration

Before you connect Ragtime to MCP clients, configure tools in the Ragtime web UI so they are ready for use:

1. Open the web UI at http://localhost:8000 and log in with your admin account.
2. Navigate to the **Tools** tab.
3. Click **Add Tool** and select the tool type (PostgreSQL, Odoo, SSH, filesystem indexer, etc.).
4. Fill in connection details (hostnames, credentials, database names, paths) for each tool.
5. Use the built-in test button to verify each tool connection.

Only tools that pass their health checks are exposed to chat and MCP clients. Configure and verify your tools here before following the MCP Integration section below.

## Connecting to OpenWebUI

1. In OpenWebUI, go to **Settings** > **Connections** > **OpenAI API**
2. Add a new connection:
   - **API Base URL**: `http://ragtime:8000/v1` (or `http://localhost:8000/v1` if running locally)
   - **API Key**: Your configured `API_KEY` (or any value if not set)
3. Select your server's model name (default: "ragtime", configurable in Settings > Server Branding)

## Model Context Protocol (MCP) Integration

Ragtime exposes its tools via the [Model Context Protocol](https://modelcontextprotocol.io), allowing AI coding assistants to interact with your databases, execute shell commands, and search your indexed codebases.

### Available MCP Tools

When you configure tool connections in the Ragtime UI, they become available to MCP clients:

- **PostgreSQL Queries** - Execute read-only SQL queries against configured databases
- **Odoo Shell** - Run ORM queries against Odoo instances
- **SSH Shell** - Execute commands on remote servers via SSH
- **Filesystem Search** - Semantic search across indexed codebases and documentation
- **Knowledge Search** - Search FAISS vector indexes created in the UI

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
- **Claude Desktop**: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows)
- **VS Code / Copilot**: User or workspace MCP settings
- **Cursor**: `.cursor/mcp.json`
- **Windsurf**: `~/.codeium/windsurf/mcp_config.json`

## Creating FAISS Indexes

The Indexer UI provides an easy way to create FAISS indexes from your codebases.

1. **Open the Web UI:** http://localhost:8000
2. Navigate to the **Indexes** tab
3. Use the **Upload** or **Git** tabs to create indexes from your code

The UI provides:
- **Upload Tab**: Upload a zip file of your codebase to create an index
- **Git Tab**: Clone and index a Git repository by URL
- **Indexes List**: View, manage, and delete existing indexes
- **Job Status**: Monitor indexing progress in real-time
- **Settings**: Configure LLM provider, embedding model, and enabled tools

## License

MIT

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
