from fastapi import FastAPI
from app.resume_parser.router import router as resume_parser_router
from app.jd_analyzer.router import router as jd_analyzer_router

app = FastAPI(
    title="Skillevate Unified API",
    version="2.0.0",
    description="Monolithic backend for Skillevate encompassing parsing, analysis, and RAG."
)

# Mount the routers with distinct prefixes
app.include_router(resume_parser_router, prefix="/api/v1/parser", tags=["Resume Parser"])
app.include_router(jd_analyzer_router, prefix="/api/v1/analyzer", tags=["JD Analyzer"])

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0", "architecture": "monolith"}