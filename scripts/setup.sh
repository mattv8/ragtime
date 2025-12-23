#!/bin/bash
# =============================================================================
# Ragtime Dev Environment Setup
# =============================================================================
# Run: ./scripts/setup.sh

set -e

PYTHON_VERSION="3.12"
NODE_REQUIRED_MAJOR=18
VENV_DIR=".venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_ROOT/ragtime/frontend"

cd "$PROJECT_ROOT"

echo "Setting up Ragtime development environment..."

# Check for Python
if command -v python${PYTHON_VERSION} &> /dev/null; then
    PYTHON_CMD="python${PYTHON_VERSION}"
elif command -v python3 &> /dev/null; then
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$PY_VER" == "$PYTHON_VERSION" ]]; then
        PYTHON_CMD="python3"
    else
        echo "Warning: Found Python $PY_VER, expected $PYTHON_VERSION"
        PYTHON_CMD="python3"
    fi
else
    echo "Error: Python 3 not found. Please install Python $PYTHON_VERSION"
    exit 1
fi

echo "Using: $($PYTHON_CMD --version)"

# Create virtual environment
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating virtual environment..."
    rm -rf "$VENV_DIR" 2>/dev/null || true
    $PYTHON_CMD -m venv "$VENV_DIR"
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        echo "Error: Failed to create virtual environment"
        exit 1
    fi
fi

# Activate and install
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip -q

echo "Installing dependencies..."
pip install -r requirements.txt -q
pip install -r requirements-indexer.txt -q

echo "Installing dev tools..."
pip install pylint black isort -q

# Generate Prisma client
echo "Generating Prisma client..."
python -m prisma generate

# Frontend dependencies/build for indexer UI
if [ -d "$FRONTEND_DIR" ]; then
    if ! command -v node >/dev/null; then
        echo "Error: Node.js ${NODE_REQUIRED_MAJOR}+ is required to build the indexer UI."
        echo "Install Node.js and re-run this script."
        exit 1
    fi

    if ! command -v npm >/dev/null; then
        echo "Error: npm was not found even though Node.js is installed."
        exit 1
    fi

    NODE_MAJOR=$(node -v | sed 's/^v//' | cut -d. -f1)
    if [ "$NODE_MAJOR" -lt "$NODE_REQUIRED_MAJOR" ]; then
        echo "Error: Detected Node.js v${NODE_MAJOR}. Please install version ${NODE_REQUIRED_MAJOR} or newer."
        exit 1
    fi

    echo "Installing frontend dependencies..."
    pushd "$FRONTEND_DIR" >/dev/null
    if [ -f package-lock.json ]; then
        npm ci --loglevel=error
    else
        npm install --loglevel=error
    fi

    echo "Building indexer UI..."
    npm run build
    popd >/dev/null
fi

# Setup .env
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

echo ""
echo "Development environment ready!"
echo ""
echo "Activate: source .venv/bin/activate"
echo "Run:      uvicorn ragtime.main:app --reload"
