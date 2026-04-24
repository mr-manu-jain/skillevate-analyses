from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
import json

# Updated imports to reflect the new monolith structure
from app.jd_analyzer.jd_extractor import extract_from_jd
from app.jd_analyzer.role_synthesizer import synthesize_role
from app.jd_analyzer.gap_analyzer import analyze_gap

# Initialize APIRouter instead of FastAPI
router = APIRouter()

# ── Request / Response Models ─────────────────────────────────────────────────
class DirectAnalyzeRequest(BaseModel):
    resume_skills: dict  # The JSON from /parse/skills
    jd_skills: dict      # The JSON from /extract


class AnalyzeTextRequest(BaseModel):
    jd_text:       Optional[str] = None
    target_role:   Optional[str] = None
    inference:     Optional[str] = "rag"   # Defaulted to local RAG
    resume_skills: Optional[dict] = None   # {"strong": [...], "listed": [...]}

# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/jobs/skills")
def extract_jd_skills(request: AnalyzeTextRequest):
    """
    Extract skills from JD text or synthesize from target role.
    Does NOT run gap analysis — just returns the skill profile.
    """
    if not request.jd_text and not request.target_role:
        raise HTTPException(
            status_code=400,
            detail="Provide either jd_text or target_role"
        )

    if request.target_role:
        result = synthesize_role(request.target_role)
    else:
        result = extract_from_jd(request.jd_text, inference=request.inference)

    return result

@router.post("/jobs/skills/file")
async def extract_jd_from_pdf(
    file:      UploadFile = File(...),
    inference: str        = Form(default="rag")
):
    """
    Extract skills from an uploaded JD PDF.
    """
    if not file.filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF or DOCX files supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = extract_from_jd(tmp_path, inference=inference)
    finally:
        os.unlink(tmp_path)

    return result

@router.post("/analyze")
def analyze(request: AnalyzeTextRequest):
    """
    Full pipeline — extract JD skills + run gap analysis against resume.
    Requires resume_skills in the request body.
    """
    if not request.jd_text and not request.target_role:
        raise HTTPException(
            status_code=400,
            detail="Provide either jd_text or target_role"
        )
    if not request.resume_skills:
        raise HTTPException(
            status_code=400,
            detail="Provide resume_skills: {strong: [...], listed: [...]}"
        )

    # Step 1 — Get JD skill profile
    if request.target_role:
        jd_skills = synthesize_role(request.target_role)
    else:
        jd_skills = extract_from_jd(request.jd_text, inference=request.inference)

    # Step 2 — Run gap analysis
    gap = analyze_gap(request.resume_skills, jd_skills)

    return {
        "jd_skills":    jd_skills,
        "gap_analysis": gap
    }

@router.post("/analyze/pdf")
async def analyze_pdf(
    file:          UploadFile = File(...),
    inference:     str        = Form(default="rag"),
    resume_skills: str        = Form(...)   # JSON string
):
    """
    Full pipeline with PDF JD upload + gap analysis.
    resume_skills: JSON string {"strong": [...], "listed": [...]}
    """
    try:
        skills = json.loads(resume_skills)
    except Exception:
        raise HTTPException(status_code=400, detail="resume_skills must be valid JSON")

    if not file.filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF or DOCX files supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        jd_skills = extract_from_jd(tmp_path, inference=inference)
        gap       = analyze_gap(skills, jd_skills)
    finally:
        os.unlink(tmp_path)

    return {
        "jd_skills":    jd_skills,
        "gap_analysis": gap
    }

@router.post("/analysis/gap")
def analyze_direct(request: DirectAnalyzeRequest):
    """
    Lightning-fast gap analysis. 
    Requires PRE-EXTRACTED resume_skills and jd_skills.
    Uses zero LLM inference.
    """
    # Step 1: Pass the pre-extracted JSONs directly into the gap analyzer
    gap = analyze_gap(request.resume_skills, request.jd_skills)

    # Step 2: Return the standard response format
    return {
        "jd_skills":    request.jd_skills,
        "gap_analysis": gap
    }