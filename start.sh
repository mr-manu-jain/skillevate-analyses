#!/bin/bash

# Skillevate Analyses Backend - Startup Script
# Boots the unified FastAPI service (parsing + JD analysis + analyses CRUD).
# Also bootstraps local RAG from backend files: Ollama, embeddings model,
# chat model, and one-time / idempotent vector index ingestion.
#
# Optional env:
#   SKIP_RAG_BOOTSTRAP=1  — skip RAG / Ollama steps (API still starts).
#   OLLAMA_BASE_URL       — default http://127.0.0.1:11434
#                             (note: backend rag.embedder defaults to localhost unless you change it).

set -euo pipefail

# Always run from the script's own directory so relative paths (.venv,
# requirements.txt, app/) resolve regardless of the caller's cwd.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL%/}"

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

bootstrap_rag() {
    if [ "${SKIP_RAG_BOOTSTRAP:-0}" = "1" ]; then
        echo "⏭️  SKIP_RAG_BOOTSTRAP=1 — skipping RAG / Ollama bootstrap."
        return 0
    fi

    ollama_api_ok() {
        curl -sf "${OLLAMA_BASE_URL}/api/tags" >/dev/null
    }

    if ! ollama_api_ok; then
        if command -v ollama &>/dev/null; then
            echo "🦙 Ollama not reachable at ${OLLAMA_BASE_URL} — starting \`ollama serve\` in the background..."
            # Only auto-start when using default local URL (avoids spawning a second daemon blindly).
            if [ "$OLLAMA_BASE_URL" = "http://127.0.0.1:11434" ] || [ "$OLLAMA_BASE_URL" = "http://localhost:11434" ]; then
                ollama serve >/dev/null 2>&1 &
                local waited=0
                while ! ollama_api_ok; do
                    if [ "$waited" -ge 90 ]; then
                        echo " Error: Ollama did not become ready within 90s. Start it manually: ollama serve"
                        exit 1
                    fi
                    sleep 1
                    waited=$((waited + 1))
                done
                echo "   Ollama is up."
            else
                echo " Error: Ollama not reachable at ${OLLAMA_BASE_URL}. Start it there, then re-run."
                exit 1
            fi
        else
            echo " Error: Ollama is not running and the \`ollama\` CLI is not installed."
            echo "   Install: https://ollama.com  then: ollama serve"
            exit 1
        fi
    else
        echo "🦙 Ollama OK at ${OLLAMA_BASE_URL}"
    fi

    # Ollama CLI uses OLLAMA_HOST (host:port, no scheme) for list/pull.
    _ollama_host="${OLLAMA_BASE_URL#http://}"
    _ollama_host="${_ollama_host#https://}"
    export OLLAMA_HOST="$_ollama_host"

    model_missing() {
        # $1 = substring to match in `ollama list` output (e.g. nomic-embed-text, llama3.1)
        ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qF "$1"
    }

    if model_missing "nomic-embed-text"; then
        echo "⬇️  Pulling embedding model nomic-embed-text (needed for RAG)..."
        ollama pull nomic-embed-text
    else
        echo "   Embedding model nomic-embed-text: already present."
    fi

    # Chat model for JD/resume paths (matches app/llm/factory.py default unless .env overrides).
    CHAT_MODEL="$(python -c "
from pathlib import Path
import os
from dotenv import load_dotenv
p = Path(r'''$SCRIPT_DIR''') / '.env'
load_dotenv(p) if p.exists() else None
print(os.getenv('OLLAMA_MODEL', 'llama3.1'))
")"
    if model_missing "$CHAT_MODEL"; then
        echo "⬇️  Pulling chat model ${CHAT_MODEL} (OLLAMA_MODEL from .env or default)..."
        ollama pull "$CHAT_MODEL"
    else
        echo "   Chat model ${CHAT_MODEL}: already present."
    fi

    echo "🧠 RAG ingest from backend files (idempotent — skips already-built FAISS indexes)..."
    # Run as a module so `from app.rag...` imports resolve correctly.
    # When executing a file directly (python app/rag/ingest.py), Python's
    # sys.path is rooted at `app/rag/`, which breaks the top-level `app.*`
    # import path.
    PYTHONPATH="$SCRIPT_DIR" python -m app.rag.ingest
}

bootstrap_rag

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
if [ "${SKIP_RAG_BOOTSTRAP:-0}" != "1" ]; then
    echo "   - RAG:      backend internal files (Ollama ${OLLAMA_BASE_URL})"
fi
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
