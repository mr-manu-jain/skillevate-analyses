# parser/sections.py  — replace entire file

import re
from typing import Optional


SECTION_PATTERNS = [
    (r"work experience|professional experience|employment history|experience", "experience"),
    (r"education|academic background|qualifications", "education"),
    (r"technical skills|skills|core competencies|competencies|expertise", "skills"),
    (r"projects|personal projects|academic projects|key projects", "projects"),
    (r"summary|professional summary|profile|objective|about me", "summary"),
    (r"certifications?|certificates|licenses", "certifications"),
    (r"awards?|honors?|achievements?|accomplishments?", "awards"),
    (r"publications?|research|papers", "publications"),
    (r"volunteer(?:ing)?|community", "volunteering"),
    (r"leadership(?:\s*&\s*recognitions?)?|recognitions?|activities", "leadership"),
    (r"languages?", "languages"),
    (r"interests?|hobbies", "interests"),
]

# Build a combined regex for inline detection
_INLINE_PATTERN = re.compile(
    r'(?:^|\n)(' + '|'.join(p for p, _ in SECTION_PATTERNS) + r')[\s:]*(?:\n|$)',
    re.IGNORECASE
)


def split_sections(text: str) -> dict:
    lines        = text.splitlines()
    sections     = {}
    current_key  = "header"
    buffer       = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            buffer.append("")
            continue

        detected = _detect_header(stripped)

        if detected and detected != current_key:
            sections[current_key] = _clean_buffer(buffer)
            current_key = detected
            buffer = []
        else:
            buffer.append(line)

    if buffer:
        sections[current_key] = _clean_buffer(buffer)

    # ── Secondary pass: catch headers embedded mid-line ────────────────────
    sections = _split_inline_headers(sections)

    return sections


def _split_inline_headers(sections: dict) -> dict:
    """
    Some section headers appear at the end of a previous section's content
    rather than on their own line (e.g. 'Leadership & Recognitions' appended
    to the certifications block). This pass catches those.
    """
    result = {}
    for key, content in sections.items():
        # Look for known headers embedded in the content
        parts = _split_on_embedded_headers(content)
        if len(parts) == 1:
            result[key] = content
        else:
            result[key] = parts[0][1]
            for part_key, part_content in parts[1:]:
                result[part_key] = part_content
    return result


def _split_on_embedded_headers(text: str) -> list:
    """
    Returns list of (section_key, content) tuples if embedded headers found.
    """
    lines  = text.splitlines()
    result = []
    current_key = None
    buffer = []

    for line in lines:
        stripped = line.strip()
        detected = _detect_header(stripped) if stripped else None

        if detected and current_key is not None and detected != current_key:
            result.append((current_key, _clean_buffer(buffer)))
            current_key = detected
            buffer = []
        else:
            if current_key is None:
                current_key = "__content__"
            buffer.append(line)

    if buffer:
        result.append((current_key, _clean_buffer(buffer)))

    return result


def _detect_header(line: str) -> Optional[str]:
    if len(line) > 60:
        return None
    if line.count(",") > 1:
        return None
    if re.search(r"\d{4}\s*[-]\s*(\d{4}|present)", line, re.IGNORECASE):
        return None

    line_lower = line.lower().strip("•:-– \t")

    for pattern, key in SECTION_PATTERNS:
        if re.fullmatch(pattern, line_lower):
            return key

    return None


def _clean_buffer(buffer: list) -> str:
    return "\n".join(buffer).strip()

# # parser/sections.py

# import re
# from typing import Optional


# # ── Known Section Headers ─────────────────────────────────────────────────────
# # Order matters — more specific patterns first

# SECTION_PATTERNS = [
#     # Experience variants
#     (r"work experience|professional experience|employment history|experience", "experience"),
#     # Education
#     (r"education|academic background|qualifications", "education"),
#     # Skills
#     (r"technical skills|skills|core competencies|competencies|expertise", "skills"),
#     # Projects
#     (r"projects|personal projects|academic projects|key projects", "projects"),
#     # Summary
#     (r"summary|professional summary|profile|objective|about me", "summary"),
#     # Certifications
#     (r"certifications|certificates|licenses", "certifications"),
#     # Awards
#     (r"awards|honors|achievements|accomplishments", "awards"),
#     # Publications
#     (r"publications|research|papers", "publications"),
#     # Volunteering
#     (r"volunteer|volunteering|community", "volunteering"),
#     (r"leadership|leadership\s*&\s*recognitions?|recognitions?|activities", "leadership"),
#     (r"languages?", "languages"),
#     (r"interests?|hobbies", "interests"),
# ]


# # ── Section Splitter ──────────────────────────────────────────────────────────

# def split_sections(text: str) -> dict:
#     """
#     Splits raw resume text into named sections.
#     Returns dict like:
#     {
#         "header":      "John Doe\nEmail...",
#         "summary":     "Data Engineer with 4 years...",
#         "experience":  "Microsoft Xbox...",
#         "education":   "San Jose State...",
#         "skills":      "Python, Spark...",
#         ...
#     }
#     """
#     lines        = text.splitlines()
#     sections     = {}
#     current_key  = "header"
#     buffer       = []

#     for line in lines:
#         stripped = line.strip()
#         if not stripped:
#             buffer.append("")
#             continue

#         detected = _detect_header(stripped)

#         if detected and stripped != current_key:
#             # Save current buffer into current section
#             sections[current_key] = _clean_buffer(buffer)
#             current_key = detected
#             buffer = []
#         else:
#             buffer.append(line)

#     # Don't forget the last section
#     if buffer:
#         sections[current_key] = _clean_buffer(buffer)

#     return sections


# # ── Header Detection ──────────────────────────────────────────────────────────

# def _detect_header(line: str) -> Optional[str]:
#     """
#     Returns section key if line looks like a section header, else None.
#     A header is:
#       - Short (< 50 chars)
#       - Matches a known pattern
#       - Not a sentence (no commas, no long phrases)
#     """
#     if len(line) > 50:
#         return None
#     if line.count(",") > 1:
#         return None
#     # Skip lines that look like job titles or dates
#     if re.search(r"\d{4}\s*[-–]\s*(\d{4}|present)", line, re.IGNORECASE):
#         return None

#     line_lower = line.lower().strip("•:-– \t")

#     for pattern, key in SECTION_PATTERNS:
#         if re.fullmatch(pattern, line_lower):
#             return key

#     return None


# # ── Buffer Cleaner ────────────────────────────────────────────────────────────

# def _clean_buffer(buffer: list) -> str:
#     """Strip leading/trailing blank lines from a section buffer."""
#     text = "\n".join(buffer)
#     return text.strip()