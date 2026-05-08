# parser/jd_extractor.py

import re
import json
import pdfplumber
import fitz
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough

from app.llm.factory import get_ollama_completion_llm

load_dotenv(dotenv_path=".env")

# ── RAG import ────────────────────────────────────────────────────────────────
# Backend-internal RAG (no dependency on the separate skillevate-rag repo)
try:
    from app.rag.retriever import Retriever
    _retriever = Retriever()
    RAG_AVAILABLE = True
    print("[jd_extractor] RAG retriever loaded.")
except ImportError:
    _retriever = None
    RAG_AVAILABLE = False
    print("[jd_extractor] WARNING: backend RAG retriever unavailable.")

# ── Prompt ────────────────────────────────────────────────────────────────────

# ── JD Cleaner ───────────────────────────────────────────────────────────────

# UI metadata tokens that appear when users copy-paste from job boards (LinkedIn, Simplify etc.)
_UI_METADATA_TOKENS = {
    "corporate_fare", "info_outline", "info", "place", "work_history",
    "bookmark_border", "share", "expand_more", "more_vert", "arrow_back",
    "chevron_right", "open_in_new", "schedule", "location_on",
}

def _strip_ui_metadata(text: str) -> str:
    """
    Remove UI artifact tokens that appear when users copy-paste from job boards
    like LinkedIn, Simplify, Google Jobs, etc.
    These are icon names and metadata labels injected by the site's HTML.
    """
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        # Drop lines that are pure UI tokens (short, no spaces, not a real sentence)
        if stripped.lower() in _UI_METADATA_TOKENS:
            continue
        # Drop lines that look like icon names (single word, starts lowercase, no punctuation)
        if re.match(r'^[a-z_]+$', stripped) and len(stripped) < 20:
            continue
        # Drop lines that are just semicolons + metadata like "; +10 more"
        if re.match(r'^[;,]\s*[+]?\d+\s*more$', stripped):
            continue
        lines.append(line)
    return "\n".join(lines)


# Patterns that signal the start of legal/EEO boilerplate — everything after
# these lines is noise and should be stripped before sending to Ollama.
_BOILERPLATE_SIGNALS = [
    "equal opportunity",
    "affirmative action",
    "eeo policy",
    "accommodation",
    "applicant and candidate privacy",
    "recruitment agencies",
    "we are committed to building a workforce",
    "information collected and processed",
    "disability, age, genetic",
]

def _strip_boilerplate(text: str) -> str:
    """
    Remove legal/EEO boilerplate from the bottom of a JD.
    Skills are always in the top portion — this noise confuses Ollama.
    """
    lines = text.split("\n")
    cutoff = len(lines)
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(signal in lower for signal in _BOILERPLATE_SIGNALS):
            cutoff = i
            break
    cleaned = "\n".join(lines[:cutoff]).strip()
    # If we stripped too aggressively (< 100 chars left), return original
    return cleaned if len(cleaned) > 100 else text


JD_EXTRACTION_PROMPT = """Extract technical skills from the job description below.

Skills taxonomy context (use to recognise skill names):
{rag_context}

OUTPUT FORMAT — return ONLY this JSON, no markdown, no explanation:
{{
  "role_title": "job title here",
  "company": "company name or null",
  "role_area": "broad category (e.g., Backend, Frontend, DevOps, Data Science, Data Engineering)",
  "required": ["skill1", "skill2"],
  "preferred": ["skill3", "skill4"]
}}

EXTRACTION RULES — read carefully:
1. Each skill must be 1-4 words: "python", "machine learning", "unix/linux", "computer vision"
2. Never output a sentence or phrase as a skill
3. Never output degree requirements: skip "bachelor's degree", "master's degree", "years of experience"
4. Split compound lists: "Python, C++, Java" → ["python", "c++", "java"]
5. skills from "Minimum qualifications" section → put in required list
6. Skills from "Preferred qualifications" section → put in preferred list

EXAMPLE INPUT:
  Minimum qualifications: Python or Java. Unix/Linux experience.
  Preferred qualifications: Machine learning, NLP, Computer Vision.
  Bachelor's degree required. 2 years experience.

CORRECT OUTPUT:
{{"role_title": "Software Engineer", "company": null,
  "required": ["python", "java", "unix/linux"],
  "preferred": ["machine learning", "nlp", "computer vision"]}}

WRONG OUTPUT (never do this):
{{"required": ["experience with python or java programming", "bachelor\'s degree or equivalent"],
  "preferred": ["machine learning algorithms and tools knowledge"]}}

Now extract from this job description:
{jd_text}"""


# ── Main Entry Point ──────────────────────────────────────────────────────────

def extract_from_jd(source: str, inference: str = "rag") -> dict:
    """
    source:    raw JD text, or path to PDF/DOCX file
    inference: "rag" (default, local) | "groq" | "gemini" (legacy, kept for
               backwards compat — both now route to RAG)

    Returns:
    {
        "source":     "jd_text | jd_pdf",
        "role_title": "...",
        "company":    "...",
        "required":   [...],
        "preferred":  [...]
    }
    """
    text = _resolve_text(source)

    if not text.strip():
        return _empty_result("jd_text")

    # Strip UI metadata artifacts (icon names from job board copy-paste)
    text = _strip_ui_metadata(text)
    result = _extract_with_rag(text)
    result["source"] = "jd_pdf" if len(source) <= 500 and Path(source).exists() else "jd_text"
    return result


# ── Text Resolution ───────────────────────────────────────────────────────────

def _resolve_text(source: str) -> str:
    """
    If source looks like a file path → extract text from file.
    Otherwise treat as raw JD text.
    """
    if len(source) > 500:
        return source

    path = Path(source)
    try:
        if path.exists():
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                return _extract_pdf(path)
            elif suffix in (".docx", ".doc"):
                return _extract_docx(path)
    except OSError:
        pass

    return source


def _extract_pdf(path: Path) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        try:
            doc  = fitz.open(str(path))
            text = "\n".join(p.get_text() for p in doc)
            doc.close()
            return text
        except Exception as e:
            print(f"[jd_extractor] PDF extraction failed: {e}")
            return ""


def _extract_docx(path: Path) -> str:
    try:
        import docx
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        print(f"[jd_extractor] DOCX extraction failed: {e}")
        return ""


# ── Smart JD Preprocessing ────────────────────────────────────────────────────

# Phrases that signal the start of legal/boilerplate sections — stop reading here
_BOILERPLATE_MARKERS = [
    "equal opportunity", "eeo policy", "affirmative action",
    "accommodation", "applicant privacy", "recruitment agenc",
    "google is proud", "information collected and processed",
    "to all recruitment", "we are committed to building a workforce",
    "criminal histories consistent", "basis protected by law",
]

# Section headers that signal skill-relevant content
_QUAL_MARKERS = [
    "minimum qualifications", "required qualifications",
    "basic qualifications", "preferred qualifications",
    "what you'll need", "requirements", "responsibilities",
    "what we're looking for", "you have", "you bring",
    "skills", "experience", "about you",
]


def _strip_boilerplate(text: str) -> str:
    """
    Remove legal/EEO boilerplate from the bottom of JD text.
    Finds the first boilerplate marker and truncates there.
    """
    lower = text.lower()
    cutoff = len(text)
    for marker in _BOILERPLATE_MARKERS:
        idx = lower.find(marker)
        if idx != -1 and idx < cutoff:
            cutoff = idx
    return text[:cutoff].strip()


def _extract_qualifications_text(text: str) -> str:
    """
    Extract the most skill-dense portion of the JD for use as RAG query.
    Looks for qualifications/requirements sections first.
    Falls back to the first 800 chars of the cleaned text.
    """
    clean = _strip_boilerplate(text)
    lower = clean.lower()

    # Find the earliest qualifications section
    best_idx = len(clean)
    for marker in _QUAL_MARKERS:
        idx = lower.find(marker)
        if idx != -1 and idx < best_idx:
            best_idx = idx

    if best_idx < len(clean):
        # Take up to 800 chars from the qualifications section
        return clean[best_idx:best_idx + 800]

    # No section found — use first 800 chars of cleaned text
    return clean[:800]



# ── RAG + Ollama Extraction ───────────────────────────────────────────────────

def _extract_with_rag(jd_text: str) -> dict:
    """
    1. Build a smart retrieval query from the JD — strips legal boilerplate,
       focuses on the qualifications and responsibilities sections.
    2. Retrieve relevant skills from the ESCO FAISS index as context.
    3. Run Ollama Phi-3 Mini locally — no external API, no cost.
    """
    # Step 1 — build a smart retrieval query
    # Problem: users often paste full JDs including legal boilerplate at the bottom.
    # Strategy: find the qualifications section if it exists, otherwise use
    # the first meaningful chunk. Strip common boilerplate phrases.
    rag_context = ""
    if RAG_AVAILABLE:
        try:
            query = _extract_qualifications_text(jd_text)
            rag_context = _retriever.get_skills_context(query, n_results=20)
        except Exception as e:
            print(f"[jd_extractor] RAG retrieval failed (continuing without context): {e}")
            rag_context = "RAG context unavailable."
    else:
        rag_context = "No skill taxonomy context available."

    # Step 2 — strip boilerplate before sending to Ollama
    clean_jd = _strip_boilerplate(jd_text)

    # Step 3 — build prompt with context injected
    prompt = JD_EXTRACTION_PROMPT.format(
        rag_context=rag_context,
        jd_text=clean_jd[:3000],
    )

    # Step 3 — run Ollama locally (LangChain completion + LCEL)
    try:
        generate = RunnablePassthrough() | get_ollama_completion_llm().bind(
            format="json",
            options={
                "temperature": 0,
                "num_predict": 1200,
            },
        )
        raw = generate.invoke(prompt).strip()
        return _parse_response(raw)

    except Exception as e:
        print(f"[jd_extractor] Ollama extraction failed: {e}")
        return _empty_result("jd_text")


# ── Response Parser ───────────────────────────────────────────────────────────

def _parse_response(raw: str) -> dict:
    try:
        clean = re.sub(r'```json|```', '', raw).strip()
        # Extract the first JSON object if Ollama added extra text
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            clean = match.group(0)
        data = json.loads(clean)

        def clean_list(items):
            """
            Coerce all items to strings, strip, lowercase.
            Post-process filter: remove degree requirements and non-skill phrases
            that small LLMs (phi3:mini) sometimes output despite prompt instructions.
            """
            # Phrases that indicate a degree/experience requirement, not a skill
            DEGREE_SIGNALS = [
                "bachelor", "master", "phd", "doctorate", "degree",
                "years of experience", "year of experience",
                "equivalent practical experience",
                "or equivalent", "industry setting",
            ]
            # If a skill string contains any of these, drop it
            def is_skill(s: str) -> bool:
                sl = s.lower()
                # Drop if it contains a degree signal
                if any(sig in sl for sig in DEGREE_SIGNALS):
                    return False
                # Drop if it's too long to be a skill (> 6 words = likely a sentence)
                if len(sl.split()) > 6:
                    return False
                # Drop if it's too short to mean anything
                if len(sl.strip()) < 2:
                    return False
                return True

            return [
                str(s).strip().lower()
                for s in items
                if isinstance(s, str) and str(s).strip() and is_skill(str(s).strip())
            ]

        return {
            "role_title": data.get("role_title"),
            "role_area":  data.get("role_area"),
            "company":    data.get("company"),
            "required":   clean_list(data.get("required",  [])),
            "preferred":  clean_list(data.get("preferred", [])),
        }
    except Exception as e:
        print(f"[jd_extractor] Response parsing failed: {e}")
        return _empty_result("jd_text")


def _empty_result(source: str) -> dict:
    return {
        "source":     source,
        "role_title": None,
        "role_area":  None,
        "company":    None,
        "required":   [],
        "preferred":  [],
    }