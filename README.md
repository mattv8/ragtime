# Ragtime

OpenAI-compatible RAG API with LangChain tool calling for business intelligence queries.

🚀 **[Live Demo](https://ragtime.dev.visnovsky.us)**
📄 **[Contributing Guide](CONTRIBUTING.md)**

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Connecting to OpenWebUI](#connecting-to-openwebui)
- [Model Context Protocol (MCP) Integration](#model-context-protocol-mcp-integration)
  - [MCP Server Setup](#mcp-server-setup)
    - [HTTP Transport (Recommended)](#http-transport-recommended)
    - [Stdio Transport (Alternative)](#stdio-transport-alternative)
  - [Tool Configuration](#tool-configuration)
- [Creating FAISS Indexes](#creating-faiss-indexes)
- [License](#license)
- [Updating](#updating)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Features

- **OpenAI API Compatible**: Works with OpenWebUI, ChatGPT clients, and any OpenAI-compatible interface
- **RAG with FAISS**: Vector search over your codebase documentation
- **Tool Calling**: Execute Odoo ORM queries and PostgreSQL queries via natural language
- **Security**: Read-only by default with SQL injection and command injection prevention
- **Async**: Non-blocking execution for better performance

## Quick Start

### Prerequisites

- Docker and Docker Compose
- A `.env` file with your configuration

### Setup

1. **Create environment file:**

   Create a file named `.env` with the following content:

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

   # JWT secret key for session tokens (auto-generated if not set)
   # IMPORTANT: Set this in production to persist sessions across restarts
   # JWT_SECRET_KEY=your-secret-key-here

   # -----------------------------------------------------------------------------
   # Server Configuration
   # -----------------------------------------------------------------------------
   # API port (default: 8000)
   PORT=8000

   # API Key for OpenAI-compatible endpoint authentication (leave empty to disable)
   # API_KEY=

   # CORS allowed origins (comma-separated, or * for all)
   ALLOWED_ORIGINS=*

   # Set to true if running behind HTTPS reverse proxy
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

2. **Edit `.env`** and configure your specific values (see [Environment Variables](#environment-variables) section)

3. **Create docker-compose.yml:**

   Create a file named `docker-compose.yml` with the following content:

   <details>
   <summary>Click to expand docker-compose.yml</summary>

   ```yaml
   # =============================================================================
   # Ragtime - Production Docker Compose
   # =============================================================================
   # For self-hosted deployment. See README.md for setup instructions.
   #
   # Usage:
   #   1. Create .env file with your configuration
   #   2. docker compose up -d
   #   3. Access at http://localhost:8000

   services:
     # PostgreSQL database for Prisma persistence
     ragtime-db:
       image: pgvector/pgvector:pg17
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
       # For older CPUs without X86_V2 support, use the legacy image:
       # image: hub.docker.visnovsky.us/library/ragtime:main
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
         # Production settings
         DEBUG_MODE: "false"
         SESSION_COOKIE_SECURE: "${SESSION_COOKIE_SECURE:-false}"
       volumes:
         # FAISS index data persistence
         - ./data:/app/data
         # Docker socket for container exec (optional, for tool execution)
         - /var/run/docker.sock:/var/run/docker.sock:ro
       networks:
         - ragtime-network
       depends_on:
         ragtime-db:
           condition: service_healthy
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
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

   **Note:** All configuration variables are loaded from the `.env` file via `env_file`. The `environment` section only overrides `DATABASE_URL` to use the container network.

   </details>

4. **Start the application:**
   ```bash
   docker compose up -d
   ```

5. **Access the application:**
   - Web UI: http://localhost:8000
   - API docs: http://localhost:8000/docs

   Default credentials: `admin` / (set via `LOCAL_ADMIN_PASSWORD` in `.env`)

### Updating

To update to the latest version:

```bash
docker compose pull
docker compose up -d
```

## Environment Variables

### Database Configuration
- `POSTGRES_PASSWORD` - PostgreSQL password (required, used by both containers)

### Authentication
- `LOCAL_ADMIN_USER` - Local admin username (default: `admin`)
- `LOCAL_ADMIN_PASSWORD` - Local admin password (required for first setup)
- `JWT_SECRET_KEY` - Secret key for JWT signing (auto-generated if not set, set in production)

### Server Configuration
- `PORT` - API port (default: `8000`)
- `API_KEY` - Optional API key for external authentication
- `ALLOWED_ORIGINS` - CORS allowed origins (default: `*`)
- `SESSION_COOKIE_SECURE` - Set to `true` if behind HTTPS reverse proxy

### Debug
- `DEBUG_MODE` - Enable debug logging (default: `false`)

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
  "mcpServers": {
    "ragtime": {
      "url": "http://localhost:8000/mcp",
      "type": "http"
    }
  }
}
```

For remote access, replace `localhost:8000` with your server URL.

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

### Tool Configuration

Before tools appear in your MCP client, configure them in the Ragtime UI:

1. Navigate to the **Tools** tab at http://localhost:8000
2. Click **Add Tool** and select the tool type
3. Configure connection details (hostname, credentials, etc.)
4. Test the connection to verify it works
5. The tool will automatically appear in your MCP client

**Tool Health Monitoring**: Ragtime only exposes tools that pass a heartbeat check. Offline or unreachable tools are automatically hidden from MCP clients.

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
