# =============================================================================
# Ragtime - Production Dockerfile
# =============================================================================
# Multi-stage build for unified RAG API + Indexer container

# =============================================================================
# Stage 1: Build frontend
# =============================================================================
FROM node:20-slim AS frontend-builder

WORKDIR /frontend

# Copy frontend manifest files
COPY ragtime/frontend/package*.json ./

# Install dependencies with lockfile
RUN npm ci --loglevel=error

# Copy the rest of the frontend
COPY ragtime/frontend/ ./

# Build the frontend
RUN npm run build

# =============================================================================
# Stage 2: Build Python dependencies
# =============================================================================
FROM python:3.12-slim AS python-builder

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive
# Set LEGACY_CPU=1 to build NumPy from source for older CPUs without X86_V2 support
ARG LEGACY_CPU=0

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
# For legacy CPU support, use NumPy 1.x which has better compatibility with older CPUs
# Install all deps first, then force-reinstall legacy packages to override numpy 2.x
COPY requirements.txt requirements-legacy.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$LEGACY_CPU" = "1" ]; then \
        echo "Installing all dependencies..." && \
        pip install --no-cache-dir -r /tmp/requirements.txt && \
        echo "Force-reinstalling legacy CPU packages (NumPy 1.x, rawpy <0.22)..." && \
        pip install --no-cache-dir --force-reinstall --no-deps -r /tmp/requirements-legacy.txt; \
    else \
        pip install --no-cache-dir -r /tmp/requirements.txt; \
    fi

# =============================================================================
# Stage 3: Production image
# =============================================================================
FROM python:3.12-slim AS production

# Labels
LABEL maintainer="elastropy.com"
LABEL description="Ragtime - OpenAI-compatible RAG server with tool calling and indexer"
LABEL version="2.0.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/ragtime \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    dos2unix \
    git \
    gnupg \
    docker.io \
    libmagic1 \
    openssl \
    openssh-client \
    nfs-common \
    cifs-utils \
    smbclient \
    libnfs14 \
    libnfs-utils \
    libnfs-dev \
    nodejs \
    npm \
    tesseract-ocr \
    libraw-dev \
    # Add PostgreSQL APT repository for version 18 client
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/keyrings/postgresql.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/postgresql.gpg] http://apt.postgresql.org/pub/repos/apt $(. /etc/os-release && echo \"$VERSION_CODENAME\")-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client-18 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r ragtime && useradd -r -g ragtime ragtime

# Copy virtual environment from builder
COPY --from=python-builder /opt/venv /opt/venv

# Create working directory
WORKDIR /ragtime

# Copy Prisma schema and migrations, then generate client
COPY prisma/ /ragtime/prisma/
RUN cd /ragtime && python -m prisma generate

# Copy application code (includes indexer module)
COPY ragtime/ /ragtime/ragtime/

# Copy built frontend from the builder stage
# Code expects it at ragtime/frontend/dist (see indexer/routes.py DIST_DIR)
COPY --from=frontend-builder /frontend/dist /ragtime/ragtime/frontend/dist/

# Copy entrypoint and scripts
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/scripts/backup-restore.sh /docker-scripts/backup-restore.sh
COPY docker/scripts/init-pgvector.sh /docker-scripts/init-pgvector.sh
COPY docker/scripts/functions.sh /docker-scripts/functions.sh

# Create data directory
RUN mkdir -p /ragtime/data/faiss_index && \
    chown -R ragtime:ragtime /ragtime

# Fix line endings (Windows CRLF -> Unix LF) and make scripts executable
RUN dos2unix /entrypoint.sh /docker-scripts/backup-restore.sh /docker-scripts/init-pgvector.sh /docker-scripts/functions.sh \
    && chmod +x /entrypoint.sh /docker-scripts/backup-restore.sh /docker-scripts/init-pgvector.sh /docker-scripts/functions.sh \
    && ln -sf /docker-scripts/backup-restore.sh /usr/local/bin/backup \
    && ln -sf /docker-scripts/backup-restore.sh /usr/local/bin/restore

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user (commented out for docker access)
# USER ragtime

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "ragtime.main:app", "--host", "0.0.0.0", "--port", "8000"]
