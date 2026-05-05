"""
MongoDB Atlas repository for gap analysis results.

Environment
-----------
MONGODB_URI       Atlas SRV connection string. Required.
MONGODB_DATABASE  Target database (default ``skillevate_user``).
MONGODB_TIMEOUT   Server selection timeout in seconds (default ``10``).
APP_NAME          Optional app name reported to the server.

Lifecycle
---------
Wire ``connect_to_mongo`` and ``close_mongo_connection`` into the FastAPI
lifespan, then call ``get_database()`` from request handlers, e.g.::

    from contextlib import asynccontextmanager
    from app.db.repository import (
        connect_to_mongo, close_mongo_connection, get_database, save_gap_analysis,
    )

    @asynccontextmanager
    async def lifespan(app):
        await connect_to_mongo()
        yield
        await close_mongo_connection()

    app = FastAPI(lifespan=lifespan)

    @app.post("/analyses")
    async def create(payload: dict):
        return {"id": str(await save_gap_analysis(get_database(), payload))}
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from bson import ObjectId
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pydantic import ValidationError

from app.db.models import GapAnalysisDocument

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

logger = logging.getLogger(__name__)

ANALYSES_COLLECTION = "analyses"
DEFAULT_DATABASE = "skillevate_user"

_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None


# ── Connection lifecycle ──────────────────────────────────────────────────────

async def connect_to_mongo() -> AsyncIOMotorDatabase:
    """Create the Motor client and verify the connection. Idempotent."""
    global _client, _database

    if _database is not None:
        return _database

    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "MONGODB_URI is not set. Configure it in the environment / .env file."
        )

    db_name = os.getenv("MONGODB_DATABASE", DEFAULT_DATABASE)
    timeout_ms = int(float(os.getenv("MONGODB_TIMEOUT", "10")) * 1000)
    app_name = os.getenv("APP_NAME")

    client_kwargs: dict = {"serverSelectionTimeoutMS": timeout_ms}
    if app_name:
        client_kwargs["appname"] = app_name

    _client = AsyncIOMotorClient(uri, **client_kwargs)

    # Force a round trip so misconfigurations surface at startup, not on first write.
    await _client.admin.command("ping")
    _database = _client[db_name]
    logger.info("[db] Connected to MongoDB Atlas database %r", db_name)
    return _database


async def close_mongo_connection() -> None:
    """Close the Motor client and clear cached handles."""
    global _client, _database
    if _client is not None:
        _client.close()
        logger.info("[db] MongoDB client closed")
    _client = None
    _database = None


def get_database() -> AsyncIOMotorDatabase:
    """Return the cached database handle. Raises if ``connect_to_mongo`` wasn't called."""
    if _database is None:
        raise RuntimeError(
            "MongoDB is not initialised. Call connect_to_mongo() during app startup."
        )
    return _database


# ── Repository operations ────────────────────────────────────────────────────

async def save_gap_analysis(
    db: AsyncIOMotorDatabase,
    analysis_data: dict,
) -> ObjectId:
    """
    Validate ``analysis_data`` against :class:`GapAnalysisDocument` and insert it
    into the ``analyses`` collection.

    Any ``recommendations`` payload supplied by the caller is discarded — that
    field is owned by a downstream microservice and is always persisted as ``[]``.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
        Target database handle (typically returned by :func:`get_database`).
    analysis_data : dict
        Raw document matching the public JSON schema for an analysis result.

    Returns
    -------
    ObjectId
        The ``_id`` of the inserted document.

    Raises
    ------
    pydantic.ValidationError
        If the payload does not conform to the schema.
    pymongo.errors.PyMongoError
        On any underlying driver / server error.
    """
    try:
        document = GapAnalysisDocument.model_validate(analysis_data)
    except ValidationError:
        logger.exception("[db] Gap analysis payload failed validation")
        raise

    payload = document.to_mongo()

    result = await db[ANALYSES_COLLECTION].insert_one(payload)
    logger.info(
        "[db] Inserted gap analysis %s for user %s",
        result.inserted_id,
        document.user_id,
    )
    return result.inserted_id
