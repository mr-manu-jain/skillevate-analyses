"""
Pydantic v2 schemas mirroring the `analyses` collection document shape.

Notes
-----
- ``_id`` is a real BSON ``ObjectId``. We expose it as ``id`` in Python and
  alias it back to ``_id`` when dumping for Mongo. It is optional on input —
  if omitted at insertion time, MongoDB will generate one server-side.
- ``user_id`` follows the Auth0 ``sub`` shape (e.g. ``"google-oauth2|107..."``)
  so it is a plain string, NOT an ``ObjectId``. The validator still accepts a
  raw ``ObjectId``/``$oid`` dict and coerces to ``str`` for forward-compat.
- ``upload_date`` accepts a Python ``datetime``, an ISO‑8601 string (with or
  without a trailing ``Z``), or the BSON Extended JSON form
  ``{"$date": "2024-01-01T00:00:00.000Z"}``.
- ``recommendations`` is intentionally locked to ``[]``. A separate
  microservice owns this field — anything supplied to this service is dropped.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, List, Optional

from bson import ObjectId
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
)


# ── Reusable validators ───────────────────────────────────────────────────────

def _coerce_object_id(value: Any) -> ObjectId:
    """Accept ObjectId, ``{"$oid": "..."}``, or a 24-char hex string."""
    if isinstance(value, ObjectId):
        return value
    if isinstance(value, dict) and "$oid" in value:
        return ObjectId(value["$oid"])
    if isinstance(value, str) and ObjectId.is_valid(value):
        return ObjectId(value)
    raise ValueError(f"Invalid ObjectId: {value!r}")


def _coerce_user_id(value: Any) -> str:
    """Accept Auth0-style strings or a raw ObjectId/extended-JSON dict."""
    if isinstance(value, str):
        return value
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict) and "$oid" in value:
        return str(value["$oid"])
    raise ValueError(f"Invalid user_id: {value!r}")


def _coerce_datetime(value: Any) -> datetime:
    """Accept datetime, ISO string, or ``{"$date": "..."}`` Extended JSON."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, dict) and "$date" in value:
        value = value["$date"]
    if isinstance(value, str):
        # ``fromisoformat`` in 3.11+ accepts ``Z`` directly; normalise for older runtimes too.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise ValueError(f"Invalid datetime: {value!r}")


PyObjectId = Annotated[ObjectId, BeforeValidator(_coerce_object_id)]
UserIdStr = Annotated[str, BeforeValidator(_coerce_user_id)]
IsoDatetime = Annotated[datetime, BeforeValidator(_coerce_datetime)]


# ── Sub-documents ─────────────────────────────────────────────────────────────

class ResumeMetadata(BaseModel):
    filename: str
    upload_date: IsoDatetime

    model_config = ConfigDict(extra="ignore")


class JdMetadata(BaseModel):
    title: str
    raw_text: str

    model_config = ConfigDict(extra="ignore")


class GapItem(BaseModel):
    skill: str
    preferences: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")


class ResultsBlock(BaseModel):
    match_score: int = Field(ge=0, le=100)
    gaps: List[GapItem] = Field(default_factory=list)
    # Owned by a downstream microservice — always emitted as [] from this service.
    recommendations: List[Any] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    @field_validator("recommendations", mode="before")
    @classmethod
    def _force_empty_recommendations(cls, _value: Any) -> list:
        return []


# ── Root document ─────────────────────────────────────────────────────────────

class GapAnalysisDocument(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    user_id: UserIdStr
    is_latest: bool = True
    resume_metadata: ResumeMetadata
    jd_metadata: JdMetadata
    results: ResultsBlock

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        extra="ignore",
        json_encoders={ObjectId: str, datetime: lambda v: v.isoformat()},
    )

    def to_mongo(self) -> dict:
        """
        Return a dict ready for ``insert_one``.

        - Keeps ObjectId / datetime as native BSON types (Mongo handles them).
        - Drops ``_id`` if it was not explicitly provided so the server assigns one.
        - Guarantees ``recommendations == []`` regardless of input.
        """
        data = self.model_dump(by_alias=True, exclude_none=False)
        if data.get("_id") is None:
            data.pop("_id", None)
        # Defence in depth — should already be [] thanks to the field validator.
        data["results"]["recommendations"] = []
        return data
