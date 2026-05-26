# syntax=docker/dockerfile:1.7

# =============================================================================
# Ragtime - Production Dockerfile
# =============================================================================
# Multi-stage build for unified RAG API + Indexer container

# =============================================================================
# Stage 1: Build frontend
# =============================================================================
FROM node:24-alpine3.22 AS frontend-builder

ARG ENVIRONMENT=main
ARG APP_VERSION=dev
ENV VITE_RAGTIME_ENVIRONMENT=${ENVIRONMENT}
ENV VITE_RAGTIME_VERSION=${APP_VERSION}

WORKDIR /frontend

RUN npm install -g npm@latest

# Copy frontend manifest files
COPY ragtime/frontend/package*.json ./

# Install dependencies with lockfile
RUN --mount=type=cache,id=ragtime-npm-cache,target=/root/.npm,sharing=locked \
    npm ci --loglevel=error

# Copy the rest of the frontend
COPY ragtime/frontend/ ./

# Build the frontend
RUN npm run build

# =============================================================================
# Stage 2: Build Python dependencies
# =============================================================================
FROM python:3.12-slim-trixie AS python-builder

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive
# Set LEGACY_CPU=1 to use dependency pins compatible with older CPUs without X86_V2 support
ARG LEGACY_CPU=0

# Install build dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies from pyproject groups
# For legacy CPU support, constrain the full app install up front so pip never
# selects wheels that require newer CPU instructions, then force-reinstall the
# legacy overrides to keep the final environment honest.
COPY pyproject.toml /tmp/pyproject.toml
COPY docker/scripts/install_deps_from_pyproject.py /tmp/install_deps_from_pyproject.py
RUN --mount=type=cache,id=ragtime-pip-cache,target=/root/.cache/pip,sharing=locked \
    python /tmp/install_deps_from_pyproject.py /tmp/pyproject.toml app /tmp/requirements.app.txt && \
    if [ "$LEGACY_CPU" = "1" ]; then \
        python /tmp/install_deps_from_pyproject.py /tmp/pyproject.toml legacy /tmp/requirements.legacy.txt && \
        python /tmp/install_deps_from_pyproject.py /tmp/pyproject.toml legacy-constraints /tmp/constraints.legacy.txt && \
        echo "Installing all dependencies with legacy CPU constraints..." && \
        pip install -c /tmp/constraints.legacy.txt -r /tmp/requirements.app.txt && \
        echo "Force-reinstalling legacy CPU packages..." && \
        pip install --force-reinstall --no-deps -r /tmp/requirements.legacy.txt && \
        python -c "from importlib.metadata import version; assert version('faiss-cpu') == '1.13.0', version('faiss-cpu'); import faiss; print('legacy faiss-cpu import ok', version('faiss-cpu'))"; \
    else \
        pip install -r /tmp/requirements.app.txt; \
    fi && \
    python -c "import chonkie, pandas"

# =============================================================================
# Stage 3: Python CI base
# =============================================================================
FROM python-builder AS python-ci-base

ENV PYTHONPATH=/ragtime \
    INDEX_DATA_PATH=/tmp/ragtime-data \
    PYTHONPYCACHEPREFIX=/tmp/pycache \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,id=ragtime-pip-cache,target=/root/.cache/pip,sharing=locked \
    python /tmp/install_deps_from_pyproject.py /tmp/pyproject.toml test /tmp/requirements.test.txt && \
    pip install -r /tmp/requirements.test.txt

RUN command -v ruff >/dev/null && python -m ruff --version && \
    python -m mypy --version && \
    python -m pyright --version

WORKDIR /ragtime
RUN ln -s /opt/venv /ragtime/.venv
COPY pyproject.toml /ragtime/pyproject.toml
COPY .jscpd.json /ragtime/.jscpd.json
COPY docker/Dockerfile /ragtime/docker/Dockerfile
COPY docker/scripts/ /ragtime/docker/scripts/
COPY prisma/ /ragtime/prisma/
COPY typings/ /ragtime/typings/
RUN --mount=type=cache,id=ragtime-npm-cache,target=/root/.npm,sharing=locked \
    --mount=type=cache,id=ragtime-prisma-cache,target=/root/.cache/prisma-python,sharing=locked \
    python -m prisma generate
COPY ragtime/ /ragtime/ragtime/
COPY runtime/ /ragtime/runtime/
COPY tests/ /ragtime/tests/

FROM python-ci-base AS python-compile-check

RUN mkdir -p "$INDEX_DATA_PATH" && \
    python -m compileall ragtime/userspace/runtime_service.py runtime/manager runtime/main.py

FROM python-ci-base AS python-ruff-check

RUN python -m ruff check --select E9,F821,F822,F823 ragtime runtime tests

FROM python-ci-base AS python-ruff-style-check

RUN python -m ruff format --check docker/scripts ragtime runtime tests && \
    python -m ruff check --select I docker/scripts ragtime runtime tests && \
    python docker/scripts/fix_inline_imports.py docker --check && \
    python docker/scripts/fix_inline_imports.py ragtime --check && \
    python docker/scripts/fix_inline_imports.py runtime --check && \
    python docker/scripts/fix_inline_imports.py tests --check

FROM python-ci-base AS python-duplicate-check

RUN npx --yes jscpd@4.0.5 --config .jscpd.json ragtime runtime tests

FROM python-ci-base AS python-mypy-check

RUN python -m mypy --config-file pyproject.toml docker/scripts ragtime runtime tests

FROM python-ci-base AS python-pyright-check

RUN python -m pyright --pythonpath .venv/bin/python

FROM python-ci-base AS python-pytest

RUN mkdir -p "$INDEX_DATA_PATH" && \
    python -m pytest tests

# =============================================================================
# Stage 4: Python release/test gate
# =============================================================================
FROM python-ci-base AS python-test

RUN mkdir -p "$INDEX_DATA_PATH" && \
    python -m compileall ragtime/userspace/runtime_service.py runtime/manager runtime/main.py && \
    python -m ruff check --select E9,F821,F822,F823 ragtime runtime tests && \
    python -m ruff format --check docker/scripts ragtime runtime tests && \
    python -m ruff check --select I docker/scripts ragtime runtime tests && \
    python docker/scripts/fix_inline_imports.py docker --check && \
    python docker/scripts/fix_inline_imports.py ragtime --check && \
    python docker/scripts/fix_inline_imports.py runtime --check && \
    python docker/scripts/fix_inline_imports.py tests --check && \
    npx --yes jscpd@4.0.5 --config .jscpd.json ragtime runtime tests && \
    python -m mypy --config-file pyproject.toml docker/scripts ragtime runtime tests && \
    python -m pyright --pythonpath .venv/bin/python && \
    python -m pytest tests

# =============================================================================
# Stage 5: Production image
# =============================================================================
FROM python:3.12-slim-trixie AS production

ARG APP_VERSION=dev

# Labels
LABEL maintainer="elastropy.com"
LABEL description="Ragtime - OpenAI-compatible RAG server with tool calling and indexer"
LABEL version="${APP_VERSION}"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/ragtime \
    RAGTIME_VERSION=${APP_VERSION} \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
# tini: lightweight init for zombie process reaping (uvicorn as PID 1 doesn't reap children)
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl \
    tini \
    dos2unix \
    git \
    gnupg \
    docker.io \
    libmagic1t64 \
    openssl \
    openssh-client \
    rsync \
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
RUN --mount=type=cache,id=ragtime-npm-cache,target=/root/.npm,sharing=locked \
    python -m prisma generate

# Install frontend node modules for runtime tooling (Playwright screenshots)
COPY ragtime/frontend/package.json ragtime/frontend/package-lock.json* /ragtime/ragtime/frontend/
RUN --mount=type=cache,id=ragtime-npm-cache,target=/root/.npm,sharing=locked \
    cd /ragtime/ragtime/frontend \
    && npm ci --loglevel=error \
    && npx playwright install --with-deps chromium

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
    CMD ["curl", "-f", "http://localhost:8000/health"]

# Switch to non-root user (commented out for docker access)
# USER ragtime

# Use tini as PID 1 to reap zombie processes (healthcheck curl, git subprocesses)
ENTRYPOINT ["tini", "--", "/entrypoint.sh"]
CMD ["uvicorn", "ragtime.main:app", "--host", "0.0.0.0", "--port", "8000"]
