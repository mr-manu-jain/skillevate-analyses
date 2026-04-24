from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import tempfile
import os

# Updated import to reflect the new monolith structure
from app.resume_parser.main import ResumeParser

# Initialize APIRouter instead of FastAPI
router = APIRouter()

# Instantiate the parser once
parser = ResumeParser()

@router.post("/parse")
async def parse_resume(
    file:      UploadFile = File(...),
    inference: str        = Form(default="ollama"),
    enrich:    bool       = Form(default=False)
):
    """
    Parse a resume PDF and return structured JSON.
    inference: ollama | groq | none
    enrich:    true = also run Gemini Flash skill enrichment
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = parser.parse(tmp_path, inference=inference, enrich=enrich)
    finally:
        os.unlink(tmp_path)

    return result

@router.post("/resume/skills")
async def parse_skills_only(
    file:      UploadFile = File(...),
    inference: str        = Form(default="ollama"),
    enrich:    bool       = Form(default=False)
):
    """
    Parse resume and return ONLY the skills section.
    Lighter response for JD analyzer integration.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = parser.parse(tmp_path, inference=inference, enrich=enrich)
    finally:
        os.unlink(tmp_path)

    return {
        "name":   result["basic_details"]["name"],
        "skills": result["skills"]
    }