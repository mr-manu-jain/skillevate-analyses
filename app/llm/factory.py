"""
Process-wide LangChain LLM instances built from environment variables.

Env:
  OLLAMA_BASE_URL   — default http://localhost:11434
  OLLAMA_MODEL      — default phi3:mini
  OLLAMA_TIMEOUT    — httpx timeout seconds for Ollama clients (default 180)
  GROQ_API_KEY      — Groq API key (also supported by Groq SDK / ChatGroq)
  GROQ_MODEL        — default llama-3.1-8b-instant
  GROQ_TIMEOUT      — request timeout seconds for Groq (default 60)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama, OllamaLLM

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "phi3:mini")


def _ollama_client_timeout() -> float:
    return float(os.getenv("OLLAMA_TIMEOUT", "180"))


def _ollama_client_kwargs() -> dict:
    return {"timeout": _ollama_client_timeout()}


@lru_cache(maxsize=1)
def get_ollama_completion_llm() -> OllamaLLM:
    """Completion-style Ollama (`/api/generate`) for legacy JSON prompts."""
    return OllamaLLM(
        model=_ollama_model(),
        base_url=_ollama_base_url(),
        client_kwargs=_ollama_client_kwargs(),
    )


@lru_cache(maxsize=1)
def get_chat_ollama() -> ChatOllama:
    """Chat-style Ollama for message-based calls."""
    return ChatOllama(
        model=_ollama_model(),
        base_url=_ollama_base_url(),
        client_kwargs=_ollama_client_kwargs(),
    )


@lru_cache(maxsize=1)
def get_chat_groq() -> ChatGroq:
    """Groq chat model; reads ``GROQ_API_KEY`` from the environment."""
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        request_timeout=float(os.getenv("GROQ_TIMEOUT", "60")),
    )
