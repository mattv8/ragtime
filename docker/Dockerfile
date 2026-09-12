# syntax=docker/dockerfile:1.7

# Dependency-image arguments deliberately default to local stages. This keeps
# ordinary and legacy local builds self-contained; CI can replace them with
# immutable Harbor digest references.
ARG FRONTEND_DEPS_IMAGE=frontend-deps
ARG PYTHON_CI_DEPS_IMAGE=python-ci-deps
ARG PRODUCTION_DEPS_IMAGE=production-deps

# =============================================================================
# Exportable dependency stages. These contain manifests, dependencies and
# generated Prisma schema only--never application, test, workflow or helper
# source. They are published by base-images.yml for linux/amd64 CI builds.
# =============================================================================
FROM node:24-alpine3.22 AS frontend-deps
WORKDIR /frontend
COPY ragtime/frontend/package.json ragtime/frontend/package-lock.json ./
RUN --mount=type=cache,id=ragtime-npm-cache,target=/root/.npm,sharing=locked \
    npm ci --loglevel=error

FROM python:3.12-slim-trixie AS python-builder
ARG DEBIAN_FRONTEND=noninteractive
ARG LEGACY_CPU=0
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml /tmp/pyproject.toml
COPY docker/scripts/install_deps_from_pyproject.py /tmp/install_deps_from_pyproject.py
RUN --mount=type=cache,id=ragtime-pip-cache,target=/root/.cache/pip,sharing=locked \
    python /tmp/install_deps_from_pyproject.py /tmp/pyproject.toml app /tmp/requirements.app.txt && \
    if [ "$LEGACY_CPU" = "1" ]; then \
        python /tmp/install_deps_from_pyproject.py /tmp/pyproject.toml legacy /tmp/requirements.legacy.txt && \
        python /tmp/install_deps_from_pyproject.py /tmp/pyproject.toml legacy-constraints /tmp/constraints.legacy.txt && \
        pip install -c /tmp/constraints.legacy.txt -r /tmp/requirements.app.txt && \
        pip install --force-reinstall --no-deps -r /tmp/requirements.legacy.txt && \
        python -c "from importlib.metadata import version; assert version('faiss-cpu') == '1.13.0', version('faiss-cpu'); import faiss; print('legacy faiss-cpu import ok', version('faiss-cpu'))"; \
    else \
        pip install -r /tmp/requirements.app.txt; \
    fi && \
    python -c "import anydoc, chonkie; assert callable(anydoc.to_markdown_bytes); assert anydoc.format_from_extension('csv') is not None"

FROM python-builder AS python-ci-deps
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
RUN --mount=type=cache,id=ragtime-npm-cache,target=/root/.npm,sharing=locked \
    npm install --global jscpd@4.0.5
RUN command -v ruff >/dev/null && python -m ruff --version && \
    python -m mypy --version && python -m pyright --version && jscpd --version
WORKDIR /ragtime
RUN ln -s /opt/venv /ragtime/.venv
COPY prisma/schema.prisma /ragtime/prisma/schema.prisma
RUN --mount=type=cache,id=ragtime-prisma-cache,target=/root/.cache/prisma-python,sharing=locked \
    python -m prisma generate

FROM python:3.12-slim-trixie AS production-deps
ARG LEGACY_CPU=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/ragtime \
    PATH="/opt/venv/bin:$PATH"
COPY --from=python-builder /opt/venv /opt/venv
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl tini dos2unix git gnupg docker.io libmagic1t64 openssl openssh-client rsync \
    nfs-common cifs-utils smbclient libnfs14 libnfs-utils libnfs-dev nodejs npm \
    tesseract-ocr libraw-dev \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/keyrings/postgresql.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/postgresql.gpg] http://apt.postgresql.org/pub/repos/apt $(. /etc/os-release && echo \"$VERSION_CODENAME\")-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update && apt-get install -y --no-install-recommends postgresql-client-18 \
    && rm -rf /var/lib/apt/lists/* && apt-get clean
RUN groupadd -r ragtime && useradd -r -g ragtime ragtime && \
    mkdir -p /ragtime/data/faiss_index /ragtime/ragtime/frontend && \
    chown -R ragtime:ragtime /ragtime
COPY --chown=ragtime:ragtime ragtime/frontend/package.json ragtime/frontend/package-lock.json /ragtime/ragtime/frontend/
RUN --mount=type=cache,id=ragtime-npm-cache,target=/root/.npm,sharing=locked \
    cd /ragtime/ragtime/frontend && npm ci --loglevel=error && npx playwright install --with-deps chromium && \
    chown -R ragtime:ragtime /ragtime/ragtime/frontend
RUN --mount=type=cache,id=ragtime-npm-cache,target=/root/.npm,sharing=locked \
    if [ "$LEGACY_CPU" = "1" ]; then npm install -g @anthropic-ai/claude-code@2.1.112; \
    else npm install -g @anthropic-ai/claude-code@2.1.186; fi && claude --version

# =============================================================================
# Source assembly and CI gates. A selected external CI base already contains a
# generated client for the same schema hash, so this path never regenerates it.
# =============================================================================
FROM ${FRONTEND_DEPS_IMAGE} AS frontend-base
COPY ragtime/frontend/ ./

FROM frontend-base AS frontend-builder
ARG ENVIRONMENT=main
ARG APP_VERSION=dev
ENV VITE_RAGTIME_ENVIRONMENT=${ENVIRONMENT} VITE_RAGTIME_VERSION=${APP_VERSION}
RUN npm run build

FROM frontend-base AS frontend-format-check
RUN npm run format:check

FROM frontend-base AS frontend-lint
ARG ESLINT_SCOPE=all
RUN set -eu; case "$ESLINT_SCOPE" in \
    all) npm run lint -- --max-warnings 0 ;; \
    none) echo "Skipping ESLint: no changed frontend files matched selector." ;; \
    files:*) node -e "const { spawnSync } = require('node:child_process'); const prefix = 'ragtime/frontend/'; const scope = process.env.ESLINT_SCOPE || 'all'; const paths = Buffer.from(scope.slice('files:'.length), 'base64').toString('utf8').split('\\0').filter(Boolean); if (!paths.length) process.exit(0); const frontendPaths = paths.map((path) => { if (!path.startsWith(prefix)) { console.error('Unsupported ESLINT_SCOPE path: ' + path); process.exit(1); } return path.slice(prefix.length); }); const result = spawnSync('./node_modules/.bin/eslint', [...frontendPaths, '--max-warnings', '0'], { stdio: 'inherit' }); process.exit(result.status ?? 1);" ;; \
    *) echo "Unsupported ESLINT_SCOPE: $ESLINT_SCOPE" >&2; exit 1 ;; esac

FROM ${PYTHON_CI_DEPS_IMAGE} AS python-ci-base
WORKDIR /ragtime
COPY pyproject.toml .jscpd.json /ragtime/
COPY docker/Dockerfile /ragtime/docker/Dockerfile
COPY docker/entrypoint.sh /ragtime/docker/entrypoint.sh
COPY docker/scripts/ /ragtime/docker/scripts/
COPY scripts/chat_branch_fixture.py /ragtime/scripts/chat_branch_fixture.py
COPY prisma/ /ragtime/prisma/
COPY typings/ /ragtime/typings/
COPY ragtime/ /ragtime/ragtime/
COPY runtime/ /ragtime/runtime/
COPY tests/ /ragtime/tests/
COPY .github/workflows/ /ragtime/.github/workflows/

FROM python-ci-base AS python-compile-check
RUN mkdir -p "$INDEX_DATA_PATH" && python -m compileall ragtime/userspace/runtime_service.py runtime/manager runtime/main.py
FROM python-ci-base AS python-ruff-check
RUN python -m ruff check --select E9,F821,F822,F823 ragtime runtime tests
FROM python-ci-base AS python-ruff-style-check
RUN python -m ruff format --check docker/scripts ragtime runtime tests && python -m ruff check --select I docker/scripts ragtime runtime tests && \
    python docker/scripts/fix_inline_imports.py docker --check && python docker/scripts/fix_inline_imports.py ragtime --check && \
    python docker/scripts/fix_inline_imports.py runtime --check && python docker/scripts/fix_inline_imports.py tests --check

# Fast style checks intentionally remain independent of the heavyweight CI
# dependency image so they do not pull application/test dependencies.
FROM python:3.12-slim-trixie AS python-quality-base
ENV INDEX_DATA_PATH=/tmp/ragtime-data PYTHONPYCACHEPREFIX=/tmp/pycache
WORKDIR /ragtime
COPY pyproject.toml /ragtime/pyproject.toml
RUN RUFF_REQUIREMENT="$(python -c 'import tomllib; from pathlib import Path; dependencies = tomllib.loads(Path("pyproject.toml").read_text())["project"]["optional-dependencies"]["test"]; print(next(dependency for dependency in dependencies if dependency.lower().startswith("ruff")))')" && \
    python -m pip install --no-cache-dir "$RUFF_REQUIREMENT"
COPY docker/scripts/ /ragtime/docker/scripts/
COPY ragtime/ /ragtime/ragtime/
COPY runtime/ /ragtime/runtime/
COPY tests/ /ragtime/tests/
FROM python-quality-base AS python-quality-check
ARG RUFF_SCOPE=all
RUN python docker/scripts/run_scoped_ruff.py && mkdir -p "$INDEX_DATA_PATH" && \
    python -m compileall ragtime/userspace/runtime_service.py runtime/manager runtime/main.py && \
    python docker/scripts/fix_inline_imports.py docker --check && python docker/scripts/fix_inline_imports.py ragtime --check && \
    python docker/scripts/fix_inline_imports.py runtime --check && python docker/scripts/fix_inline_imports.py tests --check
FROM python-ci-base AS python-duplicate-check
RUN jscpd --config .jscpd.json ragtime runtime tests
FROM python-ci-base AS python-mypy-check
ARG MYPY_SCOPE=all
RUN --mount=type=cache,id=ragtime-mypy-cache,target=/ragtime/.mypy_cache/3.12,sharing=locked python docker/scripts/run_scoped_mypy.py
FROM python-ci-base AS python-pyright-check
RUN python -m pyright --pythonpath .venv/bin/python
FROM python-ci-base AS python-pytest
RUN mkdir -p "$INDEX_DATA_PATH" && python -m pytest tests
FROM python-ci-base AS python-test
ARG MYPY_SCOPE=all
RUN mkdir -p "$INDEX_DATA_PATH" && python -m compileall ragtime/userspace/runtime_service.py runtime/manager runtime/main.py && \
    python -m ruff check --select E9,F821,F822,F823 ragtime runtime tests && python -m ruff format --check docker/scripts ragtime runtime tests && \
    python -m ruff check --select I docker/scripts ragtime runtime tests && python docker/scripts/fix_inline_imports.py docker --check && \
    python docker/scripts/fix_inline_imports.py ragtime --check && python docker/scripts/fix_inline_imports.py runtime --check && \
    python docker/scripts/fix_inline_imports.py tests --check && jscpd --config .jscpd.json ragtime runtime tests
RUN --mount=type=cache,id=ragtime-mypy-cache,target=/ragtime/.mypy_cache/3.12,sharing=locked python docker/scripts/run_scoped_mypy.py
RUN python -m pyright --pythonpath .venv/bin/python && python -m pytest tests

# The production assembly intentionally regenerates Prisma after copying the
# current schema and migrations; production-deps contains no application source.
FROM ${PRODUCTION_DEPS_IMAGE} AS production
ARG LEGACY_CPU=0
WORKDIR /ragtime
COPY --chown=ragtime:ragtime prisma/ /ragtime/prisma/
RUN python -m prisma generate && chown -R ragtime:ragtime /ragtime/prisma
COPY --chown=ragtime:ragtime ragtime/ /ragtime/ragtime/
COPY --chown=ragtime:ragtime scripts/chat_branch_fixture.py /ragtime/scripts/chat_branch_fixture.py
COPY --from=frontend-builder --chown=ragtime:ragtime /frontend/dist /ragtime/ragtime/frontend/dist/
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/scripts/backup-restore.sh /docker-scripts/backup-restore.sh
COPY docker/scripts/init-pgvector.sh /docker-scripts/init-pgvector.sh
COPY docker/scripts/functions.sh /docker-scripts/functions.sh
COPY docker/scripts/export.sh /docker-scripts/export.sh
COPY docker/scripts/import.sh /docker-scripts/import.sh
RUN dos2unix /entrypoint.sh /docker-scripts/backup-restore.sh /docker-scripts/init-pgvector.sh /docker-scripts/functions.sh /docker-scripts/export.sh /docker-scripts/import.sh && \
    chmod +x /entrypoint.sh /docker-scripts/backup-restore.sh /docker-scripts/init-pgvector.sh /docker-scripts/functions.sh /docker-scripts/export.sh /docker-scripts/import.sh && \
    ln -sf /docker-scripts/backup-restore.sh /usr/local/bin/backup && ln -sf /docker-scripts/backup-restore.sh /usr/local/bin/restore && \
    ln -sf /docker-scripts/export.sh /usr/local/bin/export && ln -sf /docker-scripts/import.sh /usr/local/bin/import
ARG APP_VERSION=dev
LABEL maintainer="elastropy.com" description="Ragtime - OpenAI-compatible RAG server with tool calling and indexer" version="${APP_VERSION}"
ENV RAGTIME_VERSION=${APP_VERSION} RAGTIME_LEGACY_CPU=${LEGACY_CPU}
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD ["curl", "-f", "http://localhost:8000/health"]
ENTRYPOINT ["tini", "--", "/entrypoint.sh"]
CMD ["uvicorn", "ragtime.main:app", "--host", "0.0.0.0", "--port", "8000"]
