# parser/fallback.py

import re
import json
import fitz  # pymupdf
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough

from app.llm.factory import get_ollama_completion_llm

load_dotenv(dotenv_path=".env")

# ── RAG import ────────────────────────────────────────────────────────────────
try:
    from rag.retriever import Retriever
    _retriever = Retriever()
    RAG_AVAILABLE = True
    print("[fallback] RAG retriever loaded.")
except ImportError:
    _retriever = None
    RAG_AVAILABLE = False
    print("[fallback] WARNING: skillevate-rag not installed. RAG unavailable.")

# ── Full Fallback — RAG + Ollama ──────────────────────────────────────────────

FULL_PARSE_PROMPT = """You are a resume parser. Extract all information from this
resume text and return ONLY valid JSON with no explanation, no markdown, no code
fences. Just raw JSON.

Relevant skills from a standard tech skills taxonomy (use these to help identify
skills in the resume):
{rag_context}

Use exactly this schema:
{{
  "basic_details": {{
    "name": null,
    "email": null,
    "phone": null,
    "linkedin": null,
    "github": null,
    "portfolio": null,
    "location": null
  }},
  "skills": {{
    "strong": [],
    "listed": []
  }},
  "raw_sections": {{
    "summary": "",
    "experience": "",
    "education": "",
    "projects": "",
    "skills": "",
    "certifications": "",
    "leadership": ""
  }}
}}

Skill classification rules:
- strong: skill appears in skills section AND is demonstrated/used in experience or projects
- listed: skill appears in skills section only, with no backing in experience or projects

Return null for any field not found. Return empty string for missing sections.

Resume text:
{resume_text}"""


def gemini_full_fallback(pdf_path: str) -> dict:
    """
    RAG + Ollama fallback for resumes where local extraction quality is 'bad'.
    Replaces the previous Gemini image-based fallback.

    Extracts raw text from the PDF using pymupdf, retrieves relevant skill context
    from the ESCO knowledge base, then runs Ollama locally.
    Function signature unchanged — callers in main.py need no updates.
    """
    print("[fallback] Triggering RAG+Ollama full fallback...")

    # Step 1 — extract text from PDF using pymupdf
    try:
        doc         = fitz.open(str(pdf_path))
        resume_text = "\n".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        print(f"[fallback] PDF text extraction failed: {e}")
        return _empty_result()

    if not resume_text.strip():
        print("[fallback] No text extracted from PDF.")
        return _empty_result()

    # Step 2 — retrieve relevant skill context from ESCO knowledge base
    rag_context = ""
    if RAG_AVAILABLE:
        try:
            # Use the skills section or first 600 chars as retrieval query
            query       = resume_text[:600]
            rag_context = _retriever.get_skills_context(query, n_results=20)
        except Exception as e:
            print(f"[fallback] RAG retrieval failed (continuing without context): {e}")
            rag_context = "No skill taxonomy context available."
    else:
        rag_context = "No skill taxonomy context available."

    # Step 3 — build prompt and run Ollama
    prompt = FULL_PARSE_PROMPT.format(
        rag_context=rag_context,
        resume_text=resume_text[:4000],
    )

    try:
        generate = RunnablePassthrough() | get_ollama_completion_llm().bind(
            temperature=0,
            num_predict=1000,
        )
        raw = generate.invoke(prompt).strip()

        # Strip markdown fences if model added them
        clean = re.sub(r'```json|```', '', raw).strip()
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        return _empty_result()

    except Exception as e:
        print(f"[fallback] Ollama full fallback failed: {e}")
        return _empty_result()


# ── Skill Enrichment — RAG + Ollama ──────────────────────────────────────────

ENRICH_PROMPT = """You are a technical skill extractor analyzing software engineering work experience.

Already identified skills (do NOT repeat these): {already_found}

Work experience text to analyze:
{work_text}

Task: Extract ONLY technical skills, tools, and technologies that are explicitly named
or clearly demonstrated in the work experience text above.

STRICT rules:
- ONLY extract skills that appear by name in the text above (e.g. "Spark", "AWS", "Docker")
- Do NOT invent or infer skills not mentioned in the text
- Do NOT include physical/non-tech skills (no "pipeline installation", "warehouse operations")
- Do NOT repeat skills already in the already_found list
- Include: programming languages, frameworks, cloud platforms, databases, DevOps tools
- Max 10 skills, lowercase

Return ONLY a valid JSON array of strings. Example: ["spark", "aws", "docker"]
No explanation. No markdown. Just the JSON array."""


def gemini_enrich_skills(work_text: str, already_found: list) -> list:
    """
    Ollama skill enrichment from experience/projects text.
    For short work text snippets, direct extraction works better than
    RAG context injection (which can introduce noise from unrelated skills).
    Function signature unchanged — callers need no updates.
    """
    print("[fallback] Enriching skills with Ollama...")

    prompt = ENRICH_PROMPT.format(
        already_found=already_found[:40],
        work_text=work_text[:2000],
    )

    try:
        generate = RunnablePassthrough() | get_ollama_completion_llm().bind(
            temperature=0,
            num_predict=300,
        )
        raw = generate.invoke(prompt).strip()
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            return [s.strip().lower() for s in json.loads(match.group(0)) if s]

    except Exception as e:
        print(f"[fallback] Ollama skill enrichment failed: {e}")

    return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_result() -> dict:
    return {
        "basic_details": {
            "name": None, "email": None, "phone": None,
            "linkedin": None, "github": None,
            "portfolio": None, "location": None,
        },
        "skills":       {"strong": [], "listed": []},
        "raw_sections": {},
    }