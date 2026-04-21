"""
retriever.py
------------
High-level retrieval interface used by both microservices.

This is the only file the two services need to import directly.
It abstracts the store and provides clean, typed methods.

Usage:
    from rag.retriever import Retriever

    r = Retriever()
    skills_context = r.get_skills_context("apache kafka stream processing")
    role_context   = r.get_role_context("senior data engineer at google")
"""

from typing import List, Dict, Optional
from rag.store import SkillStore


class Retriever:
    """
    Clean retrieval interface for the Skillevate RAG layer.

    Methods:
        get_skills_context(query)  → formatted context string for LLM prompt
        get_role_context(query)    → formatted context string for LLM prompt
        get_skills_raw(query)      → raw list of result dicts (for inspection)
        get_role_raw(query)        → raw list of result dicts (for inspection)
    """

    def __init__(self, store: Optional[SkillStore] = None):
        self._store = store or SkillStore()

    # ------------------------------------------------------------------
    # Primary interface — returns formatted strings ready for LLM prompts
    # ------------------------------------------------------------------

    def get_skills_context(
        self,
        query: str,
        n_results: int = 12,
        distance_threshold: float = 0.6,
    ) -> str:
        """
        Retrieve top-k skills relevant to a query.

        Returns a formatted string block suitable for injection into an
        LLM prompt as context. Filters out low-relevance results.

        Args:
            query: free-form text (e.g. a JD paragraph or skill phrase)
            n_results: max number of skills to retrieve
            distance_threshold: cosine distance cutoff (lower = more similar)
                                 0.0 = identical, 1.0 = unrelated

        Returns:
            Multi-line string, e.g.:
                Relevant skills from knowledge base:
                - Python (also: python3, python programming)
                - Apache Kafka (also: kafka, event streaming)
                ...
        """
        results = self._store.query_skills(query, n_results=n_results)
        filtered = [r for r in results if r["distance"] <= distance_threshold]

        if not filtered:
            return "No relevant skills found in knowledge base."

        lines = ["Relevant skills from knowledge base:"]
        for r in filtered:
            # metadata fields are flat on the result dict (no nested "metadata" key)
            label = r.get("label", "")
            alt   = r.get("alt_labels", "")
            entry = f"- {label}"
            if alt:
                # Show first 3 alt labels only to keep context concise
                alt_list = [a.strip() for a in alt.split(",") if a.strip()][:3]
                entry += f" (also known as: {', '.join(alt_list)})"
            lines.append(entry)

        return "\n".join(lines)

    def get_role_context(
        self,
        query: str,
        n_results: int = 3,
        distance_threshold: float = 0.7,
    ) -> str:
        """
        Retrieve role profiles relevant to a target role query.

        Returns a formatted string block for LLM prompt injection.

        Args:
            query: target role string (e.g. "senior data engineer at google")
            n_results: max number of role profiles to retrieve
            distance_threshold: cosine distance cutoff

        Returns:
            Multi-line string with role profiles as context.
        """
        results = self._store.query_roles(query, n_results=n_results)
        filtered = [r for r in results if r["distance"] <= distance_threshold]

        if not filtered:
            return "No relevant role profiles found in knowledge base."

        lines = ["Relevant role profiles from knowledge base:"]
        for r in filtered:
            # roles store the full document string in the "document" metadata key
            lines.append(r.get("document", ""))
            lines.append("")  # blank line between profiles

        return "\n".join(lines).strip()

    # ------------------------------------------------------------------
    # Raw access — useful for debugging and unit tests
    # ------------------------------------------------------------------

    def get_skills_raw(self, query: str, n_results: int = 10) -> List[Dict]:
        """Return raw list of {document, metadata, distance} dicts."""
        return self._store.query_skills(query, n_results=n_results)

    def get_role_raw(self, query: str, n_results: int = 5) -> List[Dict]:
        """Return raw list of {document, metadata, distance} dicts."""
        return self._store.query_roles(query, n_results=n_results)

    def stats(self) -> Dict[str, int]:
        """Return document counts per collection."""
        return self._store.stats()