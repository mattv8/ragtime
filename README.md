# Ragtime

OpenAI-compatible RAG API with LangChain tool calling for business intelligence queries.

**[Live Demo](https://ragtime.dev.visnovsky.us)**
**[Contributing Guide](CONTRIBUTING.md)**

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

   # UI/Vite port (default: 8001)
   API_PORT=8001

   # API Key for external authentication (leave empty to disable)
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
       image: postgres:latest
       container_name: ragtime-db
       restart: unless-stopped
       environment:
         POSTGRES_USER: ragtime
         POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-ragtime_prod}
         POSTGRES_DB: ragtime
       volumes:
         - ragtime-db-data:/var/lib/postgresql/data
       networks:
         - ragtime-network
       healthcheck:
         test: ["CMD-SHELL", "pg_isready -U ragtime -d ragtime"]
         interval: 10s
         timeout: 5s
         retries: 5

     # Ragtime RAG API
     ragtime:
       image: hub.docker.visnovsky.us/library/ragtime:main
       container_name: ragtime
       restart: unless-stopped
       ports:
         - "${PORT:-8000}:8000"
         - "${API_PORT:-8001}:${API_PORT:-8001}"
       env_file:
         - .env
       environment:
         # Database connection (uses container network)
         DATABASE_URL: postgresql://ragtime:${POSTGRES_PASSWORD:-ragtime_prod}@ragtime-db:5432/ragtime
         # Production settings
         DEBUG_MODE: "false"
         SESSION_COOKIE_SECURE: "${SESSION_COOKIE_SECURE:-false}"
         API_PORT: "${API_PORT:-8001}"
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
- `API_PORT` - UI/Vite port (default: `8001`)
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
3. Select "ragtime" as the model

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, making changes, and CI/CD details.
