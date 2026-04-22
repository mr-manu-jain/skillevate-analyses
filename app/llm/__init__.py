"""Shared LangChain LLM construction (Ollama + Groq)."""

from app.llm.factory import get_chat_groq, get_chat_ollama, get_ollama_completion_llm

__all__ = ["get_ollama_completion_llm", "get_chat_ollama", "get_chat_groq"]
