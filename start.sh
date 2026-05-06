#!/bin/bash

# Skillevate Analyses Backend - Startup Script
# Boots the unified FastAPI service (parsing + JD analysis + analyses CRUD).

set -euo pipefail

# Always run from the script's own directory so relative paths (.venv,
# requirements.txt, app/) resolve regardless of the caller's cwd.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

echo "🚀 Skillevate Analyses Backend - Startup"
echo "========================================"

# pydantic-core ships wheels for 3.10–3.13; 3.14 falls back to Rust builds that fail.
resolve_python() {
    local cmd minor
    for cmd in python3.13 python3.12 python3.11 python3.10; do
        if command -v "$cmd" &>/dev/null; then
            echo "$cmd"
            return 0
        fi
    done
    if command -v python3 &>/dev/null; then
        minor=$(python3 -c 'import sys; print(sys.version_info[1])')
        if [ "${minor:-99}" -le 13 ]; then
            echo python3
            return 0
        fi
    fi
    return 1
}

PYTHON_CMD=""
if ! PYTHON_CMD=$(resolve_python); then
    echo " Error: Need Python 3.10–3.13 (pydantic-core has no wheels for 3.14+)."
    echo "   Install e.g. via Homebrew: brew install python@3.12"
    echo "   Then: rm -rf .venv && ./start.sh"
    exit 1
fi

# Drop a stale venv (e.g. one accidentally created with python3.14) so we
# recreate with the resolved interpreter.
if [ -x ".venv/bin/python" ]; then
    existing_minor=$(.venv/bin/python -c 'import sys; print(sys.version_info[1])')
    if [ "${existing_minor:-99}" -ge 14 ]; then
        echo "♻️  Removing .venv (Python 3.${existing_minor}: pydantic-core cannot build on this interpreter)."
        rm -rf .venv
    fi
fi

if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment with $($PYTHON_CMD --version)..."
    "$PYTHON_CMD" -m venv .venv
fi

echo "🔧 Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# main.py / app.db.repository load .env via python-dotenv at import time, so we
# don't need to source it here. We only need PORT/HOST for uvicorn flags.
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
MONGO_DB="${MONGODB_DATABASE:-(see .env)}"

echo ""
echo "#######  Setup complete!"
echo ""
echo "🌐 Starting Skillevate Analyses Backend..."
echo "   - Local:    http://localhost:${PORT}"
echo "   - API Docs: http://localhost:${PORT}/docs"
echo "   - Health:   http://localhost:${PORT}/health"
echo "   - Mongo DB: ${MONGO_DB}"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# Reload only on changes to our app code. Uvicorn's watcher receives absolute
# paths from watchfiles, so a relative `--reload-exclude .venv` silently does
# nothing. Pin the watch roots to source files we actually edit.
APP_DIR="$(pwd)"
exec python -m uvicorn main:app \
    --reload \
    --reload-dir "$APP_DIR/app" \
    --reload-include "main.py" \
    --reload-exclude "$APP_DIR/.venv" \
    --host "$HOST" \
    --port "$PORT"
