# parser/details.py

import re
import spacy

_nlp = None

def _get_nlp():
    """Lazy load spaCy — only once, reused across calls."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ── Regex Patterns ────────────────────────────────────────────────────────────

_EMAIL    = re.compile(r'[\w.+\-]+@[\w\-]+\.[a-zA-Z]{2,}')
_PHONE    = re.compile(r'[\+]?[\d][\d\s\-\(\)\.]{7,15}[\d]')
_LINKEDIN = re.compile(r'(?:linkedin\.com/in/)([\w\-]+)/?', re.IGNORECASE)
_GITHUB   = re.compile(r'(?:github\.com/)([\w\-]+)/?', re.IGNORECASE)
_PORTFOLIO = re.compile(
    r'(?:portfolio|website|web|site)\s*[:\-]?\s*(https?://[\w./\-]+|www\.[\w./\-]+)',
    re.IGNORECASE
)
_URL      = re.compile(r'(https?://[\w./\-]+|www\.[\w./\-]+)')


# ── Main Extractor ────────────────────────────────────────────────────────────

def extract_details(header_text: str, summary_text: str = "") -> dict:
    """
    Extracts basic details from the header section.
    Falls back to summary for any missing fields.
    """
    combined = header_text + "\n" + summary_text

    name      = _extract_name(header_text)
    email     = _extract_email(combined)
    phone     = _extract_phone(combined)
    linkedin  = _extract_linkedin(combined)
    github    = _extract_github(combined)
    portfolio = _extract_portfolio(combined)
    location  = _extract_location(header_text)

    return {
        "name":      name,
        "email":     email,
        "phone":     phone,
        "linkedin":  linkedin,
        "github":    github,
        "portfolio": portfolio,
        "location":  location,
    }


# ── Individual Extractors ─────────────────────────────────────────────────────

def _extract_name(text: str) -> str | None:
    """
    Name is almost always the first non-empty line of the header.
    Validate with spaCy as a fallback check.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None

    first_line = lines[0]

    # If first line looks like a name (2-4 words, no special chars, no digits)
    if _looks_like_name(first_line):
        return first_line

    # Fallback: use spaCy NER on first 300 chars
    nlp = _get_nlp()
    doc = nlp(text[:300])
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return persons[0] if persons else first_line


def _looks_like_name(text: str) -> bool:
    words = text.strip().split()
    if not (1 < len(words) <= 5):
        return False
    if re.search(r'[\d@:/\\]', text):
        return False
    if any(w.lower() in ("email", "phone", "mobile", "linkedin", "github") for w in words):
        return False
    return True


def _extract_email(text: str) -> str | None:
    match = _EMAIL.search(text)
    return match.group(0).lower() if match else None


def _extract_phone(text: str) -> str | None:
    # Remove emails first to avoid partial matches
    cleaned = _EMAIL.sub('', text)
    match   = _PHONE.search(cleaned)
    if match:
        raw = match.group(0).strip()
        # Basic sanity — must have at least 7 digits
        if len(re.sub(r'\D', '', raw)) >= 7:
            return raw
    return None


def _extract_linkedin(text: str) -> str | None:
    match = _LINKEDIN.search(text)
    if match:
        return f"linkedin.com/in/{match.group(1)}"
    return None


def _extract_github(text: str) -> str | None:
    match = _GITHUB.search(text)
    if match:
        username = match.group(1)
        # Exclude common false positives
        if username.lower() not in ("features", "pricing", "about", "login"):
            return f"github.com/{username}"
    return None


def _extract_portfolio(text: str) -> str | None:
    # Try labeled portfolio first
    match = _PORTFOLIO.search(text)
    if match:
        return match.group(1)

    # Find all URLs, exclude known platforms
    known = re.compile(r'linkedin\.com|github\.com|gmail\.com|yahoo\.com', re.IGNORECASE)
    urls  = _URL.findall(text)
    for url in urls:
        if not known.search(url):
            return url
    return None


def _extract_location(text: str) -> str | None:
    """
    Looks for City, State or City, Country patterns.
    """
    pattern = re.compile(
        r'\b([A-Z][a-zA-Z\s]+),\s*([A-Z]{2}|[A-Z][a-zA-Z]+)\b'
    )
    # Avoid matching names — look past the first line
    lines = text.splitlines()
    search_text = "\n".join(lines[1:])  # skip name line
    match = pattern.search(search_text)
    return match.group(0) if match else None
