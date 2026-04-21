# parser/gap_analyzer.py

from rapidfuzz import fuzz


# ── Normalization ─────────────────────────────────────────────────────────────

def _normalize(skill: str) -> str:
    return skill.strip().lower()

def _normalize_list(skills: list) -> list:
    return [_normalize(s) for s in skills if s]


# ── Fuzzy Match ───────────────────────────────────────────────────────────────

def _fuzzy_match(skill: str, skill_set: list, threshold: int = 82) -> bool:
    """
    Returns True if skill fuzzy-matches any skill in the set.
    Handles variations like "apache spark" vs "spark", "gcp" vs "google cloud platform"
    """
    for s in skill_set:
        if fuzz.ratio(skill, s) >= threshold:
            return True
        # Also check if one contains the other (handles "kubernetes (gke)" vs "kubernetes")
        if skill in s or s in skill:
            return True
    return False


def _find_match(skill: str, skill_set: list, threshold: int = 82) -> str | None:
    """Returns the matching skill from the set, or None."""
    for s in skill_set:
        if fuzz.ratio(skill, s) >= threshold:
            return s
        if skill in s or s in skill:
            return s
    return None


# ── Main Entry Point ──────────────────────────────────────────────────────────

def analyze_gap(resume_skills: dict, jd_skills: dict) -> dict:
    """
    resume_skills: output from ResumeParser
        {
            "strong": [...],
            "listed": [...]
        }

    jd_skills: output from jd_extractor or role_synthesizer
        {
            "source":     "jd_text | jd_pdf | target_role",
            "role_title": "...",
            "company":    "...",
            "required":   [...],
            "preferred":  [...]
        }

    Returns full gap analysis dict.
    """
    # Normalize all skill lists
    strong   = _normalize_list(resume_skills.get("strong", []))
    listed   = _normalize_list(resume_skills.get("listed", []))
    required = _normalize_list(jd_skills.get("required", []))
    preferred= _normalize_list(jd_skills.get("preferred", []))
    all_resume = strong + listed

    matched_strong    = []
    matched_listed    = []
    missing_critical  = []
    missing_preferred = []

    # ── Required skills bucketing ─────────────────────────────────────────────
    for skill in required:
        if _fuzzy_match(skill, strong):
            matched_strong.append({
                "skill":      skill,
                "importance": "required",
                "match_type": "strong"
            })
        elif _fuzzy_match(skill, listed):
            matched_listed.append({
                "skill":      skill,
                "importance": "required",
                "match_type": "listed"
            })
        else:
            missing_critical.append({
                "skill":      skill,
                "importance": "required",
                "category":   _infer_category(skill)
            })

    # ── Preferred skills bucketing ────────────────────────────────────────────
    for skill in preferred:
        if _fuzzy_match(skill, strong) or _fuzzy_match(skill, listed):
            matched_strong.append({
                "skill":      skill,
                "importance": "preferred",
                "match_type": "strong" if _fuzzy_match(skill, strong) else "listed"
            })
        else:
            missing_preferred.append({
                "skill":      skill,
                "importance": "preferred",
                "category":   _infer_category(skill)
            })

    # ── Bonus skills (resume has but JD doesn't mention) ─────────────────────
    all_jd = required + preferred
    bonus_skills = []
    for skill in strong:
        if not _fuzzy_match(skill, all_jd):
            bonus_skills.append({
                "skill":    skill,
                "category": _infer_category(skill)
            })

    # ── Match score ───────────────────────────────────────────────────────────
    score = _calculate_score(
        matched_strong, matched_listed,
        missing_critical, required
    )

    return {
        "role_title":         jd_skills.get("role_title"),
        "company":            jd_skills.get("company"),
        "source":             jd_skills.get("source"),
        "matched_strong":     matched_strong,
        "matched_listed":     matched_listed,
        "missing_critical":   missing_critical,
        "missing_preferred":  missing_preferred,
        "bonus_skills":       bonus_skills,
        "match_score":        score,
        "readiness_level":    _readiness_level(score),
        "summary": {
            "total_required":      len(required),
            "total_preferred":     len(preferred),
            "strong_matches":      len([m for m in matched_strong if m["importance"] == "required"]),
            "listed_matches":      len(matched_listed),
            "critical_gaps":       len(missing_critical),
            "preferred_gaps":      len(missing_preferred),
            "bonus_count":         len(bonus_skills),
        }
    }


# ── Scoring ───────────────────────────────────────────────────────────────────

def _calculate_score(matched_strong, matched_listed, missing_critical, required) -> int:
    """
    Score = (strong_required * 1.0 + listed_required * 0.5) / total_required * 100
    Capped at 100.
    """
    if not required:
        return 0

    strong_req = len([m for m in matched_strong if m["importance"] == "required"])
    listed_req = len([m for m in matched_listed if m["importance"] == "required"])
    total_req  = len(required)

    score = ((strong_req * 1.0) + (listed_req * 0.5)) / total_req * 100
    return min(round(score), 100)


def _readiness_level(score: int) -> str:
    if score >= 75:
        return "strong_candidate"
    if score >= 50:
        return "partial_match"
    return "significant_gaps"


# ── Category Inference ────────────────────────────────────────────────────────

CATEGORY_MAP = {
    "programming":      ["python", "java", "scala", "go", "r", "sql", "javascript",
                         "typescript", "c++", "rust"],
    "big_data":         ["spark", "pyspark", "hadoop", "flink", "kafka", "hive",
                         "presto", "druid", "beam", "dataflow"],
    "cloud":            ["aws", "gcp", "azure", "s3", "ec2", "bigquery", "cloud storage",
                         "pub/sub", "cloud composer", "gke", "eks"],
    "devops":           ["docker", "kubernetes", "terraform", "ci/cd", "git",
                         "linux", "jenkins", "helm"],
    "data_engineering": ["etl", "elt", "airflow", "dbt", "delta lake", "data modeling",
                         "data pipeline", "data warehouse", "data lake", "data quality"],
    "ml_ai":            ["machine learning", "deep learning", "mlops", "tensorflow",
                         "pytorch", "scikit-learn", "mlflow", "llm", "rag", "faiss"],
    "database":         ["postgresql", "mysql", "mongodb", "redis", "cassandra",
                         "hbase", "elasticsearch", "bigtable", "firestore"],
    "analytics":        ["tableau", "looker", "power bi", "data visualization",
                         "dashboarding", "reporting"],
}

def _infer_category(skill: str) -> str:
    skill_lower = skill.lower()
    for category, keywords in CATEGORY_MAP.items():
        for kw in keywords:
            if kw in skill_lower or skill_lower in kw:
                return category
    return "general"