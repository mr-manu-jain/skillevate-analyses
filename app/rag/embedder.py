"""
embedder.py
-----------
Local embedding using nomic-embed-text via Ollama.
No external API calls. No API keys needed.

Setup (one-time):
    ollama pull nomic-embed-text
"""

import httpx
from typing import List

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"


class OllamaEmbedder:
    """
    Wraps Ollama's /api/embeddings endpoint.
    Produces 768-dim vectors from nomic-embed-text.
    """

    def __init__(self, model: str = EMBEDDING_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url
        self._client = httpx.Client(timeout=30.0)

    def embed(self, text: str) -> List[float]:
        """Embed a single string. Returns a list of floats."""
        response = self._client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings. Returns a list of embedding vectors."""
        return [self.embed(text) for text in texts]

    def health_check(self) -> bool:
        """Returns True if Ollama is reachable and the model is available."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()