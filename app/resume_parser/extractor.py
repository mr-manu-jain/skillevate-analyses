# parser/extractor.py

import pdfplumber
import fitz  # pymupdf
from pathlib import Path
import re
# ── Quality Thresholds ────────────────────────────────────────────────────────
MIN_LINES   = 10   # fewer than this → suspicious
MIN_CHARS   = 200  # fewer than this → almost certainly a bad extract


# ── Main Extractor ────────────────────────────────────────────────────────────
# parser/extractor.py  — add this function at the bottom

import re

def clean_text(text: str) -> str:
    """
    Cleans common PDF extraction artifacts.
    """
    # Remove (cid:XX) font encoding artifacts
    text = re.sub(r'\(cid:\d+\)', '', text)

    # Normalize bullet characters
    text = re.sub(r'[•·▪▸◦‣⁃]', '•', text)

    # Normalize dashes
    text = re.sub(r'[–—―]', '-', text)

    # Remove non-printable characters (except newlines and tabs)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', '', text)

    # Collapse 3+ blank lines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()


def extract_text(pdf_path: str) -> dict:
    """
    Primary entry point.
    Tries pdfplumber first, falls back to pymupdf.
    Returns a dict with text, method used, and quality signal.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # --- Attempt 1: pdfplumber ---
    raw_text, method = _try_pdfplumber(path)
    # --- Attempt 2: pymupdf fallback ---
    if _quality(raw_text) == "bad":
        pymupdf_text, _ = _try_pymupdf(path)
        if _quality(pymupdf_text) != "bad":
            raw_text   = pymupdf_text
            method = "pymupdf"

    cleaned = clean_text(raw_text)
    quality = _quality(cleaned)

    return {
        "text":    cleaned,
        "method":  method,
        "quality": quality,   # "good" | "partial" | "bad"
        "lines":   len(cleaned.splitlines()),
        "chars":   len(cleaned),
    }


# ── Extraction Backends ───────────────────────────────────────────────────────

def _try_pdfplumber(path: Path) -> tuple[str, str]:
    try:
        with pdfplumber.open(path) as pdf:
            pages = []
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if page_text:
                    pages.append(page_text)
            return "\n".join(pages), "pdfplumber"
    except Exception as e:
        print(f"[extractor] pdfplumber failed: {e}")
        return "", "pdfplumber_failed"


def _try_pymupdf(path: Path) -> tuple[str, str]:
    try:
        doc  = fitz.open(str(path))
        pages = []
        for page in doc:
            pages.append(page.get_text("text"))
        doc.close()
        return "\n".join(pages), "pymupdf"
    except Exception as e:
        print(f"[extractor] pymupdf failed: {e}")
        return "", "pymupdf_failed"


# ── Quality Scorer ────────────────────────────────────────────────────────────

def _quality(text: str) -> str:
    lines = [l for l in text.splitlines() if l.strip()]
    chars = len(text.strip())

    if chars >= MIN_CHARS and len(lines) >= MIN_LINES:
        return "good"
    if chars >= MIN_CHARS // 2 or len(lines) >= MIN_LINES // 2:
        return "partial"
    return "bad"