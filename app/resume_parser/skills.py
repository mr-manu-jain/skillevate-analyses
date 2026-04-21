# parser/skills.py

import re
import json
import ollama
from rapidfuzz import fuzz

# ── Skill Ontology ────────────────────────────────────────────────────────────
# Each skill maps to a normalized name + what it implies
# Extend this freely — this is your core matching dictionary

ONTOLOGY = {
    # Languages
    "python":       {"category": "programming",  "infers": ["scripting", "automation"]},
    "sql":          {"category": "database",      "infers": ["data querying", "relational databases"]},
    "scala":        {"category": "programming",   "infers": ["jvm", "functional programming"]},
    "java":         {"category": "programming",   "infers": ["oop", "jvm"]},
    "javascript":   {"category": "programming",   "infers": ["web development"]},
    "typescript":   {"category": "programming",   "infers": ["web development", "javascript"]},
    "r":            {"category": "programming",   "infers": ["statistical analysis"]},

    # Big Data
    "spark":        {"category": "big data",      "infers": ["distributed computing", "large-scale data processing"]},
    "pyspark":      {"category": "big data",      "infers": ["spark", "distributed computing"]},
    "kafka":        {"category": "streaming",     "infers": ["event streaming", "real-time processing"]},
    "flink":        {"category": "streaming",     "infers": ["real-time processing", "stateful computations"]},
    "hadoop":       {"category": "big data",      "infers": ["distributed storage", "mapreduce"]},
    "hive":         {"category": "big data",      "infers": ["sql-on-hadoop", "data warehousing"]},
    "airflow":      {"category": "orchestration", "infers": ["workflow automation", "pipeline scheduling"]},
    "databricks":   {"category": "big data",      "infers": ["spark", "lakehouse", "mlflow"]},
    "dbt":          {"category": "data engineering", "infers": ["data transformation", "analytics engineering"]},
    "delta lake":   {"category": "data engineering", "infers": ["acid transactions", "lakehouse"]},

    # Cloud
    "aws":          {"category": "cloud",         "infers": ["cloud infrastructure", "iaas"]},
    "gcp":          {"category": "cloud",         "infers": ["cloud infrastructure", "iaas"]},
    "azure":        {"category": "cloud",         "infers": ["cloud infrastructure", "iaas"]},
    "bigquery":     {"category": "cloud",         "infers": ["data warehousing", "gcp"]},
    "azure synapse":{"category": "cloud",         "infers": ["data warehousing", "azure"]},
    "data factory": {"category": "cloud",         "infers": ["etl", "azure"]},
    "terraform":    {"category": "devops",        "infers": ["infrastructure as code", "cloud provisioning"]},
    "docker":       {"category": "devops",        "infers": ["containerization", "microservices"]},
    "kubernetes":   {"category": "devops",        "infers": ["container orchestration", "cloud-native"]},

    # Data Engineering
    "etl":          {"category": "data engineering", "infers": ["data pipelines", "data integration"]},
    "elt":          {"category": "data engineering", "infers": ["data pipelines", "data integration"]},
    "data modeling":{"category": "data engineering", "infers": ["schema design", "data architecture"]},
    "data pipeline":{"category": "data engineering", "infers": ["etl", "data integration"]},

    # Databases
    "postgresql":   {"category": "database",      "infers": ["relational databases", "sql"]},
    "mysql":        {"category": "database",      "infers": ["relational databases", "sql"]},
    "mongodb":      {"category": "database",      "infers": ["nosql", "document store"]},
    "redis":        {"category": "database",      "infers": ["caching", "in-memory database"]},
    "elasticsearch":{"category": "database",      "infers": ["full-text search", "elk stack"]},

    # ML / AI
    "machine learning": {"category": "ml",        "infers": ["model training", "statistical modeling"]},
    "deep learning":    {"category": "ml",        "infers": ["neural networks", "pytorch", "tensorflow"]},
    "mlflow":           {"category": "mlops",     "infers": ["experiment tracking", "model registry"]},
    "scikit-learn":     {"category": "ml",        "infers": ["machine learning", "python"]},

    # Observability
    "grafana":      {"category": "observability", "infers": ["monitoring", "dashboarding"]},
    "elk stack":    {"category": "observability", "infers": ["logging", "elasticsearch"]},

    # Soft / Domain
    "a/b testing":  {"category": "analytics",    "infers": ["statistics", "experimentation"]},
    "data warehousing": {"category": "data engineering", "infers": ["olap", "dimensional modeling"]},
# Add these to ONTOLOGY dict in skills.py

"kafka":        {"category": "streaming",      "infers": ["event streaming", "real-time processing"]},
"flink":        {"category": "streaming",      "infers": ["real-time processing"]},
"docker":       {"category": "devops",         "infers": ["containerization", "microservices"]},
"kubernetes":   {"category": "devops",         "infers": ["container orchestration", "cloud-native"]},
"terraform":    {"category": "devops",         "infers": ["infrastructure as code"]},
"snowflake":    {"category": "data warehouse", "infers": ["cloud data warehousing", "sql"]},
"langchain":    {"category": "ai/llm",         "infers": ["llm", "rag", "generative ai"]},
"langgraph":    {"category": "ai/llm",         "infers": ["agentic ai", "llm orchestration"]},
"faiss":        {"category": "ai/ml",          "infers": ["vector database", "semantic search"]},
"rag":          {"category": "ai/llm",         "infers": ["retrieval augmented generation", "llm"]},
"pytorch":      {"category": "ml",             "infers": ["deep learning", "neural networks"]},
"tensorflow":   {"category": "ml",             "infers": ["deep learning", "neural networks"]},
"scikit-learn": {"category": "ml",             "infers": ["machine learning", "python"]},
"hugging face": {"category": "ai/llm",         "infers": ["transformers", "nlp", "llm"]},
"prometheus":   {"category": "observability",  "infers": ["monitoring", "metrics"]},
"tableau":      {"category": "analytics",      "infers": ["data visualization", "bi"]},
"power bi":     {"category": "analytics",      "infers": ["data visualization", "bi"]},
"postgresql":   {"category": "database",       "infers": ["relational databases", "sql"]},
"mysql":        {"category": "database",       "infers": ["relational databases", "sql"]},
"redis":        {"category": "database",       "infers": ["caching", "in-memory database"]},
"mongodb":      {"category": "database",       "infers": ["nosql", "document store"]},
"git":          {"category": "devops",         "infers": ["version control", "collaboration"]},
"ci/cd":        {"category": "devops",         "infers": ["continuous integration", "automation"]},
"nlp":          {"category": "ml",             "infers": ["natural language processing", "text processing"]},
"generative ai":{"category": "ai/llm",         "infers": ["llm", "prompt engineering"]},
"looker studio":{"category": "analytics",      "infers": ["data visualization", "bi", "gcp"]},
"u-sql":        {"category": "big data",       "infers": ["azure data lake", "sql"]},
"scd type 2":   {"category": "data engineering","infers": ["slowly changing dimensions", "data warehousing"]},
}


# ── Main Entry Point ──────────────────────────────────────────────────────────

def extract_skills(sections: dict, inference: str = "ollama") -> dict:
    skills_text     = sections.get("skills", "")
    experience_text = sections.get("experience", "")
    projects_text   = sections.get("projects", "")
    summary_text    = sections.get("summary", "")

    full_body = " ".join([experience_text, projects_text, summary_text]).lower()
    work_text = experience_text + "\n" + projects_text

    listed_skills = _parse_skills_section(skills_text)
    ontology_hits = _ontology_match(full_body)

    if inference == "ollama":
        inferred = _infer_with_ollama(work_text, listed_skills + ontology_hits)
    elif inference == "groq":
        inferred = _infer_with_groq(work_text, listed_skills + ontology_hits)
    else:
        inferred = []

    return _score_skills(listed_skills, ontology_hits, inferred, full_body)
    """
    Returns:
    {
        "strong": [...],   # claimed in skills section + found anywhere in resume body
                           # OR inferred by Ollama from experience
        "listed": [...]    # claimed in skills section but no backing found anywhere
    }
    """
    skills_text     = sections.get("skills", "")
    experience_text = sections.get("experience", "")
    projects_text   = sections.get("projects", "")
    summary_text    = sections.get("summary", "")

    # Full resume body — everything except the skills section itself
    full_body = " ".join([experience_text, projects_text, summary_text]).lower()
    # Step 1 — Parse skills section
    listed_skills = _parse_skills_section(skills_text)

    # Step 2 — Ontology match across full body
    ontology_hits = _ontology_match(full_body)

    # Step 3 — Ollama inferred skills from experience + projects
    work_text = experience_text + "\n" + projects_text
    
    if inference == "ollama":
        inferred = _infer_with_ollama(work_text, listed_skills + ontology_hits)
    elif inference == "groq":
        inferred = _infer_with_groq(work_text, listed_skills + ontology_hits)
    else:
        inferred = []
    
    #inferred  = _infer_with_ollama(work_text, listed_skills + ontology_hits)

    # Step 4 — Score and bucket
    return _score_skills(listed_skills, ontology_hits, inferred, full_body)
    """
    Returns:
    {
        "strong": [...],   # in skills section + backed by experience OR inferred
        "listed": [...]    # in skills section only, no backing found
    }
    """
    skills_text     = sections.get("skills", "")
    experience_text = sections.get("experience", "")
    projects_text   = sections.get("projects", "")
    work_text       = experience_text + "\n" + projects_text

    # Step 1 — Extract from skills section
    listed_skills = _parse_skills_section(skills_text)

    # Step 2 — Ontology match across all sections
    ontology_hits = _ontology_match(work_text)

    # Step 3 — Ollama inferred skills from experience
    inferred = _infer_with_ollama(experience_text, listed_skills + ontology_hits)

    # Step 4 — Score and bucket
    return _score_skills(listed_skills, ontology_hits, inferred)


# ── Step 1: Parse Skills Section ──────────────────────────────────────────────

def _parse_skills_section(text: str) -> list:
    skills = []
    lines = text.splitlines()

    for line in lines:
        line = line.strip().strip('• \t')
        if not line:
            continue

        # Remove category label e.g. "Data Engineering & Big Data:" or "Cloud & DevOps:"
        line = re.sub(r'^[A-Za-z\s,&/]+:\s*', '', line)

        # Flatten parenthetical sub-groups e.g. "GCP (BigQuery, Cloud Storage)" 
        # → "GCP BigQuery Cloud Storage"
        line = re.sub(r'\(([^)]+)\)', r', \1,', line)

        # Split on commas and any leftover bullet-like whitespace clusters
        parts = re.split(r',|\s{2,}', line)

        for part in parts:
            part = part.strip().strip('•-– \t()/\\')
            if not part or len(part) < 2 or len(part) > 50:
                continue
            # Skip leftover category-looking fragments
            if re.search(r'\b(devops|big data|warehousing|governance)\b', part, re.IGNORECASE):
                continue
            # Skip fragments that are clearly broken (contain lone &, :, etc.)
            if re.fullmatch(r'[&:/\\|]+', part):
                continue
            skills.append(_normalize(part))

    return list(dict.fromkeys(filter(None, skills)))


    """
    Extracts individual skills from the skills section.
    Handles comma-separated, bullet-separated, and category: skills formats.
    """
    # Remove category labels like "Data Engineering & Big Data:"
    text = re.sub(r'^[A-Z][^:]{2,40}:\s*', '', text, flags=re.MULTILINE)

    # Split on common delimiters
    raw = re.split(r'[,•\|/\n]+', text)

    skills = []
    for item in raw:
        item = item.strip().strip('•-– \t')
        if item and 2 <= len(item) <= 60:
            skills.append(_normalize(item))

    return list(dict.fromkeys(filter(None, skills)))  # dedupe, preserve order


# ── Step 2: Ontology Matching ─────────────────────────────────────────────────

def _ontology_match(text: str) -> list:
    """
    Finds ontology skills mentioned in experience/projects text.
    Uses fuzzy matching to catch minor variations.
    """
    text_lower = text.lower()
    found = []

    for skill, meta in ONTOLOGY.items():
        # Exact match first
        if skill in text_lower:
            found.append(_normalize(skill))
            # Add inferred skills too
            for inf in meta.get("infers", []):
                found.append(_normalize(inf))
        else:
            # Fuzzy match for skills > 4 chars
            if len(skill) > 4:
                score = fuzz.partial_ratio(skill, text_lower)
                if score >= 90:
                    found.append(_normalize(skill))

    return list(dict.fromkeys(found))


# ── Step 3: Ollama Inference ──────────────────────────────────────────────────

def _infer_with_ollama(experience_text: str, already_found: list) -> list:
    """
    Uses Phi-3 Mini locally to infer skills demonstrated in experience
    that aren't already captured by ontology matching.
    """
    if not experience_text.strip():
        return []

    prompt = f"""You are a technical recruiter analyzing a work experience section.
Identify skills that are clearly DEMONSTRATED through the work described,
but are NOT in this already-found list: {already_found[:30]}

Focus on: technical skills, domain expertise, methodologies, tools.
Do NOT include soft skills or generic terms like "teamwork".

Work Experience:
{experience_text[:1500]}

Respond ONLY with a valid JSON array of strings. Example: ["skill1", "skill2"]
No explanation. No markdown. Just the JSON array."""

    try:
        response = ollama.chat(
            model="phi3:mini",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0, "num_predict": 200}
        )
        raw = response["message"]["content"].strip()
        # Extract JSON array even if model adds extra text
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            return [_normalize(s) for s in json.loads(match.group(0)) if s]
    except Exception as e:
        print(f"[skills] Ollama inference failed: {e}")

    return []

def _infer_with_groq(experience_text: str, already_found: list) -> list:
    """
    Uses Groq (Llama 3.1 8B) to infer skills from experience.
    Faster than Ollama, free tier, open-source model.
    """
    if not experience_text.strip():
        return []

    try:
        from groq import Groq
        import os
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=".env")

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        prompt = f"""You are a technical recruiter analyzing a work experience section.
Identify skills that are clearly DEMONSTRATED through the work described,
but are NOT in this already-found list: {already_found[:30]}

Focus on: technical skills, domain expertise, methodologies, tools.
Do NOT include soft skills or generic terms like "teamwork".

Work Experience:
{experience_text[:1500]}

Respond ONLY with a valid JSON array of strings. Example: ["skill1", "skill2"]
No explanation. No markdown. Just the JSON array."""

        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )

        raw   = response.choices[0].message.content.strip()
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            return [s.strip().lower() for s in json.loads(match.group(0)) if s]

    except Exception as e:
        print(f"[skills] Groq inference failed: {e}")

    return []
# ── Step 4: Score & Bucket ────────────────────────────────────────────────────

def _score_skills(listed: list, ontology_hits: list, inferred: list, full_body: str) -> dict:
    """
    Strong: claimed skill appears anywhere in resume body (experience + projects + summary)
            OR was inferred by Ollama
    Listed: claimed but no evidence found anywhere in resume body
    """
    backed_set   = set(ontology_hits) | set(inferred)
    strong       = set()
    weak         = set()

    for skill in listed:
        # Check 1: direct word match in full body
        pattern    = r'\b' + re.escape(skill) + r'\b'
        in_body    = bool(re.search(pattern, full_body, re.IGNORECASE))

        # Check 2: ontology/inferred match (fuzzy)
        in_backed  = skill in backed_set or any(
            fuzz.ratio(skill, b) >= 85 for b in backed_set
        )

        if in_body or in_backed:
            strong.add(skill)
        else:
            weak.add(skill)

    # Skills inferred from experience/projects but not in skills section → strong
    inferred_only = (set(ontology_hits) | set(inferred)) - set(listed)
    strong.update(inferred_only)

    return {
        "strong": sorted(strong),
        "listed": sorted(weak),
    }

# ── Utility ───────────────────────────────────────────────────────────────────

def _normalize(skill: str) -> str:
    return skill.strip().lower()