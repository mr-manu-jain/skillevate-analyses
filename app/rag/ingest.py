"""
Backend RAG ingestion script (one-time, idempotent).

Mirrors the behavior of the standalone `skillevate-rag/ingest.py` script, but
uses the backend's internal `app.rag.store.SkillStore` implementation.
"""

import argparse
import sys

from app.rag.embedder import OllamaEmbedder
from app.rag.store import SkillStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Skillevate backend RAG knowledge base")
    parser.add_argument("--skills-only", action="store_true")
    parser.add_argument("--roles-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if already populated")
    parser.add_argument("--stats", action="store_true", help="Print stats and exit")
    args = parser.parse_args()

    # Health check first (avoid starting embeddings when Ollama/model isn't ready)
    print("[ingest] Checking Ollama connection...")
    embedder = OllamaEmbedder()
    if not embedder.health_check():
        print(
            "[ingest] ERROR: Ollama is not running or nomic-embed-text is not pulled.\n"
            "  Run: ollama serve &\n"
            "       ollama pull nomic-embed-text"
        )
        sys.exit(1)
    print("[ingest] Ollama OK — nomic-embed-text available.")

    store = SkillStore()

    if args.stats:
        stats = store.stats()
        print(f"[ingest] Collection stats: {stats}")
        return

    if not args.roles_only:
        try:
            n = store.ingest_skills(force=args.force)
            if n > 0:
                print(f"[ingest] Ingested {n} skills.")
        except FileNotFoundError as e:
            print(f"[ingest] WARNING: {e}")
            print("[ingest] Skipping skills ingestion. Download ESCO CSV to enable.")

    if not args.skills_only:
        n = store.ingest_roles(force=args.force)
        if n > 0:
            print(f"[ingest] Ingested {n} role profiles.")

    print("[ingest] Done. Stats:", store.stats())


if __name__ == "__main__":
    main()

