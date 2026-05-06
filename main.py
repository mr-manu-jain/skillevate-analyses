import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.db.repository import close_mongo_connection, connect_to_mongo
from app.db.router import router as analyses_router
from app.jd_analyzer.router import router as jd_analyzer_router
from app.resume_parser.router import router as resume_parser_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Manage the MongoDB client lifecycle.

    Connection failure is logged but does NOT crash the server — non-DB
    endpoints (parsing, JD extraction, gap analysis) must remain usable for
    local development without Mongo. The repository functions will lazily
    re-attempt the connection on the first request that needs it.
    """
    try:
        await connect_to_mongo()
    except Exception as exc:  # noqa: BLE001 — startup must not crash on Mongo issues
        logger.warning(
            "[startup] MongoDB connection failed; analyses endpoints will retry "
            "lazily. Reason: %s",
            exc,
        )

    try:
        yield
    finally:
        try:
            await close_mongo_connection()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[shutdown] Error closing MongoDB client: %s", exc)


app = FastAPI(
    title="Skillevate Unified API",
    version="2.0.0",
    description="Monolithic backend for Skillevate encompassing parsing, analysis, and RAG.",
    lifespan=lifespan,
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
app.include_router(analyses_router, prefix="/api", tags=["Analyses"])


@app.get("/")
def root():
    """Landing page for browser checks; API lives under /api and docs at /docs."""
    return {
        "service": "Skillevate Unified API",
        "version": "2.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "health": "/health",
        "health_mongo": "/health/mongo",
        "api_prefix": "/api",
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0", "architecture": "monolith"}


@app.get("/health/mongo")
async def mongo_health_check():
    """
    Verifies MongoDB connectivity by running a real ping command.
    Returns 503 when Mongo is unreachable or misconfigured.
    """
    try:
        db = await connect_to_mongo()
        ping = await db.command("ping")
        return {
            "status": "ok",
            "mongodb": "connected",
            "database": db.name,
            "ping": ping.get("ok", 0),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={"status": "error", "mongodb": "disconnected", "reason": str(exc)},
        ) from exc
