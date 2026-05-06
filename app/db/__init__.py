"""MongoDB Atlas data access layer (async via Motor)."""

from app.db.models import (
    GapAnalysisDocument,
    GapItem,
    JdMetadata,
    ResultsBlock,
    ResumeMetadata,
)
from app.db.repository import (
    close_mongo_connection,
    connect_to_mongo,
    get_database,
    save_gap_analysis,
)

__all__ = [
    "GapAnalysisDocument",
    "GapItem",
    "JdMetadata",
    "ResultsBlock",
    "ResumeMetadata",
    "close_mongo_connection",
    "connect_to_mongo",
    "get_database",
    "save_gap_analysis",
]
