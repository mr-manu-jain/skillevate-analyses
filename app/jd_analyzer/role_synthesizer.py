# parser/role_synthesizer.py

import re
import json
import httpx
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

# ── RAG import ────────────────────────────────────────────────────────────────
try:
    from rag.retriever import Retriever
    _retriever = Retriever()
    RAG_AVAILABLE = True
    print("[role_synthesizer] RAG retriever loaded.")
except ImportError:
    _retriever = None
    RAG_AVAILABLE = False
    print("[role_synthesizer] WARNING: skillevate-rag not installed. RAG unavailable.")

# ── Ollama config ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "phi3:mini")


# ── Prompt ────────────────────────────────────────────────────────────────────

ROLE_SYNTHESIS_PROMPT = """You are a senior technical recruiter with deep knowledge of
the tech industry. A candidate wants to target the following role:

"{role}"

Here is context from a tech role knowledge base to help you:
{role_context}

Here are relevant skills from a standard tech skills taxonomy:
{skills_context}

Based on this context and your knowledge of this role:

1. What skills are REQUIRED — non-negotiable for this role
2. What skills are PREFERRED — would differentiate a candidate

Consider: technical skills, tools, frameworks, domain knowledge, methodologies.
Do NOT include soft skills like communication or teamwork.
Be specific and realistic — what would actually appear in a real job posting.

Return ONLY valid JSON, no explanation, no markdown:
{{
  "role_title":     "normalized role title",
  "company":        "company name or null if generic",
  "seniority":      "junior | mid | senior | staff | principal",
  "required":       ["skill1", "skill2"],
  "preferred":      ["skill3", "skill4"],
  "domain_context": "one line description of what this role does"
}}

All skills must be lowercase.
You MUST include AT LEAST 10 required skills and AT LEAST 8 preferred skills.
Use the role knowledge base context above as your primary source — expand from there.
Do not stop early. Fill all required and preferred slots."""


# ── Main Entry Point ──────────────────────────────────────────────────────────

def synthesize_role(role: str) -> dict:
    """
    Synthesizes a skill profile for a target role string using RAG + Ollama.
    Fully local — no external API calls, no cost.

    Output schema is identical to the previous Gemini implementation so
    gap_analyzer.py requires no changes.
    """
    if not role.strip():
        return _empty_result(role)

    print(f"[role_synthesizer] Synthesizing: '{role}'")

    # Step 1 — retrieve closest role profile + skills from knowledge base
    role_context   = ""
    skills_context = ""
    if RAG_AVAILABLE:
        try:
            # Get role profiles — these contain curated required/preferred skill lists
            role_context   = _retriever.get_role_context(role, n_results=2)
            # Get broader skill context using the role string as query
            skills_context = _retriever.get_skills_context(role, n_results=20)

            # Also inject raw role profile data directly so Ollama sees the skill lists
            raw_roles = _retriever.get_role_raw(role, n_results=1)
            if raw_roles:
                top = raw_roles[0]
                role_context = f"Closest matching role profile:\n{top.get('document', '')}\n\n{role_context}"
        except Exception as e:
            print(f"[role_synthesizer] RAG retrieval failed (continuing without context): {e}")
            role_context   = "No role profile available."
            skills_context = "No skill taxonomy context available."
    else:
        role_context   = "No role profile available."
        skills_context = "No skill taxonomy context available."

    # Step 2 — build prompt
    prompt = ROLE_SYNTHESIS_PROMPT.format(
        role=role.strip(),
        role_context=role_context,
        skills_context=skills_context,
    )

    # Step 3 — run Ollama locally
    try:
        print(f"[role_synthesizer] Running Ollama ({OLLAMA_MODEL})...")
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 1200},
            },
            timeout=90.0,
        )
        response.raise_for_status()
        raw    = response.json().get("response", "").strip()
        result = _parse_response(raw, role)
        result["inference_engine"] = f"rag+ollama-{OLLAMA_MODEL}"
        return result

    except Exception as e:
        print(f"[role_synthesizer] Ollama failed: {e}")
        empty = _empty_result(role)
        empty["inference_engine"] = "none"
        return empty


# ── Response Parser ───────────────────────────────────────────────────────────

def _parse_response(raw: str, original_role: str) -> dict:
    try:
        clean = re.sub(r'```json|```', '', raw).strip()
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            clean = match.group(0)
        data = json.loads(clean)
        return {
            "source":         "target_role",
            "original_input": original_role,
            "role_title":     data.get("role_title"),
            "company":        data.get("company"),
            "seniority":      data.get("seniority"),
            "domain_context": data.get("domain_context"),
            "required":       [s.strip().lower() for s in data.get("required",  [])],
            "preferred":      [s.strip().lower() for s in data.get("preferred", [])],
        }
    except Exception as e:
        print(f"[role_synthesizer] Parse failed: {e}")
        return _empty_result(original_role)


def _empty_result(role: str) -> dict:
    return {
        "source":           "target_role",
        "original_input":   role,
        "role_title":       None,
        "company":          None,
        "seniority":        None,
        "domain_context":   None,
        "required":         [],
        "preferred":        [],
        "inference_engine": None,
    }



# # parser/role_synthesizer.py

# import re
# import json
# import httpx
# import os
# from dotenv import load_dotenv

# load_dotenv(dotenv_path=".env")

# # ── RAG import ────────────────────────────────────────────────────────────────
# try:
#     from rag.retriever import Retriever
#     _retriever = Retriever()
#     RAG_AVAILABLE = True
#     print("[role_synthesizer] RAG retriever loaded.")
# except ImportError:
#     _retriever = None
#     RAG_AVAILABLE = False
#     print("[role_synthesizer] WARNING: skillevate-rag not installed. RAG unavailable.")

# # ── Ollama config ─────────────────────────────────────────────────────────────
# OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "phi3:mini")


# # ── Prompt ────────────────────────────────────────────────────────────────────

# ROLE_SYNTHESIS_PROMPT = """You are a senior technical recruiter with deep knowledge of
# the tech industry. A candidate wants to target the following role:

# "{role}"

# Here is context from a tech role knowledge base to help you:
# {role_context}

# Here are relevant skills from a standard tech skills taxonomy:
# {skills_context}

# Based on this context and your knowledge of this role:

# 1. What skills are REQUIRED — non-negotiable for this role
# 2. What skills are PREFERRED — would differentiate a candidate

# Consider: technical skills, tools, frameworks, domain knowledge, methodologies.
# Do NOT include soft skills like communication or teamwork.
# Be specific and realistic — what would actually appear in a real job posting.

# Return ONLY valid JSON, no explanation, no markdown:
# {{
#   "role_title":     "normalized role title",
#   "company":        "company name or null if generic",
#   "seniority":      "junior | mid | senior | staff | principal",
#   "required":       ["skill1", "skill2"],
#   "preferred":      ["skill3", "skill4"],
#   "domain_context": "one line description of what this role does"
# }}

# All skills must be lowercase. Include 10-20 required and 8-15 preferred."""


# # ── Main Entry Point ──────────────────────────────────────────────────────────

# def synthesize_role(role: str) -> dict:
#     """
#     Synthesizes a skill profile for a target role string using RAG + Ollama.
#     Fully local — no external API calls, no cost.

#     Output schema is identical to the previous Gemini implementation so
#     gap_analyzer.py requires no changes.
#     """
#     if not role.strip():
#         return _empty_result(role)

#     print(f"[role_synthesizer] Synthesizing: '{role}'")

#     # Step 1 — retrieve closest role profile from knowledge base
#     role_context   = ""
#     skills_context = ""
#     if RAG_AVAILABLE:
#         try:
#             role_context   = _retriever.get_role_context(role, n_results=2)
#             skills_context = _retriever.get_skills_context(role, n_results=15)
#         except Exception as e:
#             print(f"[role_synthesizer] RAG retrieval failed (continuing without context): {e}")
#             role_context   = "No role profile available."
#             skills_context = "No skill taxonomy context available."
#     else:
#         role_context   = "No role profile available."
#         skills_context = "No skill taxonomy context available."

#     # Step 2 — build prompt
#     prompt = ROLE_SYNTHESIS_PROMPT.format(
#         role=role.strip(),
#         role_context=role_context,
#         skills_context=skills_context,
#     )

#     # Step 3 — run Ollama locally
#     try:
#         print(f"[role_synthesizer] Running Ollama ({OLLAMA_MODEL})...")
#         response = httpx.post(
#             f"{OLLAMA_BASE_URL}/api/generate",
#             json={
#                 "model":  OLLAMA_MODEL,
#                 "prompt": prompt,
#                 "stream": False,
#                 "options": {"temperature": 0.2, "num_predict": 700},
#             },
#             timeout=90.0,
#         )
#         response.raise_for_status()
#         raw    = response.json().get("response", "").strip()
#         result = _parse_response(raw, role)
#         result["inference_engine"] = f"rag+ollama-{OLLAMA_MODEL}"
#         return result

#     except Exception as e:
#         print(f"[role_synthesizer] Ollama failed: {e}")
#         empty = _empty_result(role)
#         empty["inference_engine"] = "none"
#         return empty


# # ── Response Parser ───────────────────────────────────────────────────────────

# def _parse_response(raw: str, original_role: str) -> dict:
#     try:
#         clean = re.sub(r'```json|```', '', raw).strip()
#         match = re.search(r'\{.*\}', clean, re.DOTALL)
#         if match:
#             clean = match.group(0)
#         data = json.loads(clean)
#         return {
#             "source":         "target_role",
#             "original_input": original_role,
#             "role_title":     data.get("role_title"),
#             "company":        data.get("company"),
#             "seniority":      data.get("seniority"),
#             "domain_context": data.get("domain_context"),
#             "required":       [s.strip().lower() for s in data.get("required",  [])],
#             "preferred":      [s.strip().lower() for s in data.get("preferred", [])],
#         }
#     except Exception as e:
#         print(f"[role_synthesizer] Parse failed: {e}")
#         return _empty_result(original_role)


# def _empty_result(role: str) -> dict:
#     return {
#         "source":           "target_role",
#         "original_input":   role,
#         "role_title":       None,
#         "company":          None,
#         "seniority":        None,
#         "domain_context":   None,
#         "required":         [],
#         "preferred":        [],
#         "inference_engine": None,
#     }