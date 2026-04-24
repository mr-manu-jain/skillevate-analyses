import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.resume_parser.router import router as resume_parser_router
from app.jd_analyzer.router import router as jd_analyzer_router

app = FastAPI(
    title="Skillevate Unified API",
    version="2.0.0",
    description="Monolithic backend for Skillevate encompassing parsing, analysis, and RAG."
)

# Global CORS config so browser preflight (OPTIONS) works for all endpoints.
# Set CORS_ALLOW_ORIGINS as comma-separated origins in production.
cors_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
allow_origins = ["*"] if cors_origins.strip() == "*" else [o.strip() for o in cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False if "*" in allow_origins else True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the routers with distinct prefixes
app.include_router(resume_parser_router, prefix="/api", tags=["Resumes"])
app.include_router(jd_analyzer_router, prefix="/api", tags=["Jobs & Analysis"])

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0", "architecture": "monolith"}