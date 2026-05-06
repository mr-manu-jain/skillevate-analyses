"""
HTTP surface for the ``analyses`` collection.

Endpoints
---------
POST   /analyses                        — Persist a fresh gap-analysis result. Demotes
                                          any previously-latest document for the same
                                          ``user_id`` to ``is_latest=False`` before
                                          inserting the new one with ``is_latest=True``.
GET    /analyses/{user_id}/latest       — Return the single ``is_latest=True`` document
                                          for a user (404 if none).
GET    /analyses/{user_id}              — Return all analyses for a user, newest first.

The collection / connection management lives in :mod:`app.db.repository`. This module
is intentionally thin: validation, ordering guarantees, response shaping. No business
logic for matching or scoring belongs here.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, HTTPException, status
from pydantic import ValidationError
from pymongo.errors import PyMongoError

from app.db.models import GapAnalysisDocument
from app.db.repository import (
    ANALYSES_COLLECTION,
    connect_to_mongo,
    save_gap_analysis,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _jsonify_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a Mongo document to a JSON-serialisable dict in-place safe form.

    - ``ObjectId`` values become strings.
    - ``datetime`` values become ISO-8601 strings.
    - Recurses into nested dicts and lists.
    """
    if doc is None:
        return doc

    def _convert(value: Any) -> Any:
        if isinstance(value, ObjectId):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_convert(v) for v in value]
        return value

    return _convert(doc)


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("/analyses", status_code=status.HTTP_201_CREATED)
async def create_analysis(payload: GapAnalysisDocument) -> dict[str, Any]:
    """
    Persist a new gap-analysis document.

    Atomically demotes any prior ``is_latest=True`` rows for the same ``user_id``
    before inserting the new document with ``is_latest=True``. The
    ``recommendations`` field is forced to ``[]`` by the model — that field is
    owned by a downstream microservice.
    """
    try:
        db = await connect_to_mongo()
    except Exception as exc:  # noqa: BLE001 — surface as 503 with detail
        logger.exception("[analyses] MongoDB connection failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unavailable: {exc}",
        ) from exc

    # Force is_latest=True regardless of what the caller submitted.
    payload.is_latest = True

    try:
        await db[ANALYSES_COLLECTION].update_many(
            {"user_id": payload.user_id, "is_latest": True},
            {"$set": {"is_latest": False}},
        )

        inserted_id = await save_gap_analysis(db, payload.model_dump(by_alias=True))
    except ValidationError as exc:
        # save_gap_analysis re-validates; surface as 422 to match FastAPI conventions.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors(),
        ) from exc
    except PyMongoError as exc:
        logger.exception("[analyses] Mongo write failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database write failed: {exc}",
        ) from exc

    return {
        "id": str(inserted_id),
        "user_id": payload.user_id,
        "is_latest": True,
    }


@router.get("/analyses/{user_id}/latest")
async def get_latest_analysis(user_id: str) -> dict[str, Any]:
    """Return the single ``is_latest=True`` analysis for ``user_id`` (404 if none)."""
    try:
        db = await connect_to_mongo()
    except Exception as exc:  # noqa: BLE001
        logger.exception("[analyses] MongoDB connection failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unavailable: {exc}",
        ) from exc

    try:
        doc = await db[ANALYSES_COLLECTION].find_one(
            {"user_id": user_id, "is_latest": True}
        )
    except PyMongoError as exc:
        logger.exception("[analyses] Mongo read failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database read failed: {exc}",
        ) from exc

    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No latest analysis for user_id={user_id!r}",
        )

    return _jsonify_doc(doc)


@router.get("/analyses/{user_id}")
async def list_analyses(user_id: str) -> list[dict[str, Any]]:
    """Return every analysis for ``user_id``, sorted newest first by ``_id``."""
    try:
        db = await connect_to_mongo()
    except Exception as exc:  # noqa: BLE001
        logger.exception("[analyses] MongoDB connection failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unavailable: {exc}",
        ) from exc

    try:
        cursor = db[ANALYSES_COLLECTION].find({"user_id": user_id}).sort("_id", -1)
        docs = await cursor.to_list(length=200)
    except PyMongoError as exc:
        logger.exception("[analyses] Mongo read failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database read failed: {exc}",
        ) from exc

    return [_jsonify_doc(doc) for doc in docs]
