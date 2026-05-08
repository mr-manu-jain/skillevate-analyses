"""
store.py
--------
FAISS-backed vector store for the Skillevate RAG layer.

Two indexes are maintained:
  - skills : ESCO skill entries (label + description + alt labels)
  - roles  : Tech role profiles (role name + typical skill set + context)

Each index is saved as two files:
  faiss_index/skills.index  + skills_meta.json
  faiss_index/roles.index   + roles_meta.json

Usage:
    from app.rag.store import SkillStore
    store = SkillStore()
    store.ingest_skills()     # one-time
    store.ingest_roles()
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np

from .embedder import OllamaEmbedder

# Paths
RAG_ROOT = Path(__file__).parent
KB_DIR = RAG_ROOT / "knowledge_base"
INDEX_DIR = RAG_ROOT / "faiss_index"

SKILLS_CSV = KB_DIR / "esco_skills.csv"
ROLES_JSON = KB_DIR / "role_profiles.json"

SKILLS_INDEX_FILE = "skills.index"
SKILLS_META_FILE  = "skills_meta.json"
ROLES_INDEX_FILE  = "roles.index"
ROLES_META_FILE   = "roles_meta.json"

BATCH_SIZE    = 64
EMBEDDING_DIM = 768   # nomic-embed-text output dimension


class SkillStore:
    """
    Manages two FAISS IndexFlatIP indexes: skills and roles.
    Vectors are L2-normalised before storage so that inner product == cosine similarity.
    Embeddings are generated locally via OllamaEmbedder — no external API calls.
    """

    def __init__(self, index_dir: str = str(INDEX_DIR)):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = OllamaEmbedder()

        self.skills_index, self.skills_meta = self._load_or_create(
            SKILLS_INDEX_FILE, SKILLS_META_FILE
        )
        self.roles_index, self.roles_meta = self._load_or_create(
            ROLES_INDEX_FILE, ROLES_META_FILE
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_skills(self, force: bool = False) -> int:
        """
        Ingest ESCO skills CSV into the FAISS skills index.
        Skips if already populated unless force=True.

        Expected CSV columns: conceptUri, preferredLabel, altLabels, description

        Returns number of entries ingested.
        """
        if len(self.skills_meta) > 0 and not force:
            print(f"[store] skills index already has {len(self.skills_meta)} entries, skipping.")
            return 0

        if not SKILLS_CSV.exists():
            raise FileNotFoundError(
                f"ESCO skills CSV not found at {SKILLS_CSV}.\n"
                "Download from: https://esco.ec.europa.eu/en/use-esco/download\n"
                "Select: Skills/competences → CSV (English)"
            )

        rows = self._load_skills_csv()
        print(f"[store] ingesting {len(rows)} skills into FAISS...")

        self.skills_index = self._new_index()
        self.skills_meta = []

        ingested = 0
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            docs  = [r["document"] for r in batch]
            metas = [r["metadata"] for r in batch]

            vecs = self.embedder.embed_batch(docs)
            self.skills_index.add(self._to_matrix(vecs))
            self.skills_meta.extend(metas)

            ingested += len(batch)
            print(f"  {ingested}/{len(rows)} skills ingested", end="\r")

        print(f"\n[store] done. {ingested} skills in index.")
        self._save(self.skills_index, self.skills_meta, SKILLS_INDEX_FILE, SKILLS_META_FILE)
        return ingested

    def ingest_roles(self, force: bool = False) -> int:
        """
        Ingest role profiles JSON into the FAISS roles index.
        Skips if already populated unless force=True.

        Returns number of entries ingested.
        """
        if len(self.roles_meta) > 0 and not force:
            print(f"[store] roles index already has {len(self.roles_meta)} entries, skipping.")
            return 0

        if not ROLES_JSON.exists():
            raise FileNotFoundError(f"Role profiles JSON not found at {ROLES_JSON}.")

        with open(ROLES_JSON) as f:
            roles = json.load(f)

        print(f"[store] ingesting {len(roles)} role profiles...")

        self.roles_index = self._new_index()
        self.roles_meta  = []

        for role in roles:
            doc = self._role_to_document(role)
            vec = self.embedder.embed(doc)
            self.roles_index.add(self._to_matrix([vec]))
            self.roles_meta.append({
                "document": doc,
                "title":    role["title"],
                "domain":   role.get("domain", ""),
            })

        print(f"[store] done. {len(roles)} roles in index.")
        self._save(self.roles_index, self.roles_meta, ROLES_INDEX_FILE, ROLES_META_FILE)
        return len(roles)

    # ------------------------------------------------------------------
    # Query helpers  (called by retriever.py)
    # ------------------------------------------------------------------

    def query_skills(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Return top-n skill entries closest to query string."""
        if not self.skills_meta:
            return []
        k   = min(n_results, len(self.skills_meta))
        vec = self.embedder.embed(query)
        D, I = self.skills_index.search(self._to_matrix([vec]), k)
        return self._format_results(D[0], I[0], self.skills_meta)

    def query_roles(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Return top-n role profiles closest to query string."""
        if not self.roles_meta:
            return []
        k   = min(n_results, len(self.roles_meta))
        vec = self.embedder.embed(query)
        D, I = self.roles_index.search(self._to_matrix([vec]), k)
        return self._format_results(D[0], I[0], self.roles_meta)

    def stats(self) -> Dict[str, int]:
        return {"skills": len(self.skills_meta), "roles": len(self.roles_meta)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_index(self) -> faiss.IndexFlatIP:
        """
        IndexFlatIP performs exact inner-product search.
        After L2 normalisation, inner product equals cosine similarity.
        Scores are in [-1, 1] — we convert to cosine distance in _format_results.
        """
        return faiss.IndexFlatIP(EMBEDDING_DIM)

    @staticmethod
    def _to_matrix(embeddings: List[List[float]]) -> np.ndarray:
        """Convert embedding list to L2-normalised float32 numpy matrix."""
        mat = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(mat)
        return mat

    def _load_or_create(self, index_file: str, meta_file: str):
        """Load index + metadata from disk if they exist, else return empty."""
        ip = self.index_dir / index_file
        mp = self.index_dir / meta_file
        if ip.exists() and mp.exists():
            idx = faiss.read_index(str(ip))
            with open(mp) as f:
                meta = json.load(f)
            print(f"[store] loaded {mp.name} ({len(meta)} entries)")
            return idx, meta
        return self._new_index(), []

    def _save(self, index, meta: List[Dict], index_file: str, meta_file: str):
        """Persist FAISS index and metadata JSON to disk."""
        faiss.write_index(index, str(self.index_dir / index_file))
        with open(self.index_dir / meta_file, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[store] saved {index_file} + {meta_file}")

    # ------------------------------------------------------------------
    # CSV / JSON parsing
    # ------------------------------------------------------------------

    def _load_skills_csv(self) -> List[Dict]:
        """
        Parse ESCO CSV into embeddable document + metadata dicts.

        Filtering strategy:
          - Keep 'cross-sector' and 'transversal' skills only.
            These are transferable skills that appear across industries
            (programming, data analysis, project management, etc.).
            Covers ~3-4k skills — exactly what a tech job platform needs.
          - Drop 'occupation-specific' and 'sector-specific' skills.
            These are narrow trade skills that will never appear in a tech JD
            (e.g. 'ensure coquille uniformity', 'handle fish harvesting waste').

        Result: ~3-4k entries instead of ~13k — faster ingestion, cleaner retrieval.
        """
        KEEP_LEVELS = {"cross-sector", "transversal", ""}

        rows = []
        skipped = 0
        with open(SKILLS_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                reuse_level = row.get("reuseLevel", "").strip().lower()
                if reuse_level not in KEEP_LEVELS:
                    skipped += 1
                    continue

                label = row.get("preferredLabel", "").strip()
                alt   = row.get("altLabels", "").strip().replace("\n", ", ")
                desc  = row.get("description", "").strip()

                if not label:
                    continue

                parts = [f"Skill: {label}"]
                if alt:
                    parts.append(f"Also known as: {alt}")
                if desc:
                    parts.append(f"Description: {desc[:300]}")

                rows.append({
                    "document": " | ".join(parts),
                    "metadata": {
                        "label":       label,
                        "alt_labels":  alt[:500] if alt else "",
                        "reuse_level": reuse_level,
                    },
                })

        print(f"[store] filtered {skipped} occupation/sector-specific skills, kept {len(rows)} cross-sector skills.")
        return rows

    @staticmethod
    def _role_to_document(role: Dict) -> str:
        """Serialise a role profile dict into an embeddable string."""
        parts = [f"Role: {role['title']}"]
        if role.get("domain"):
            parts.append(f"Domain: {role['domain']}")
        if role.get("description"):
            parts.append(f"Description: {role['description']}")
        if role.get("required_skills"):
            parts.append(f"Required skills: {', '.join(role['required_skills'])}")
        if role.get("preferred_skills"):
            parts.append(f"Preferred skills: {', '.join(role['preferred_skills'])}")
        return " | ".join(parts)

    @staticmethod
    def _format_results(
        distances: np.ndarray,
        indices:   np.ndarray,
        meta:      List[Dict],
    ) -> List[Dict[str, Any]]:
        """
        Convert raw FAISS output into clean result dicts.
        Converts inner-product similarity → cosine distance (1 - sim)
        so that the retriever's distance_threshold convention stays consistent:
            0.0 = identical   1.0 = completely unrelated
        """
        out = []
        for sim, idx in zip(distances, indices):
            if idx == -1:
                continue
            entry = dict(meta[idx])
            entry["distance"] = round(float(1.0 - sim), 4)
            out.append(entry)
        return out