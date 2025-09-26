"""Canonical JSON helpers and error translation for the API."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Iterable, Mapping

from fastapi import HTTPException, Response, status


CANONICAL_SEPARATORS = (",", ":")


def canonical_dumps(payload: Any) -> str:
    """Serialize *payload* using a deterministic JSON representation."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=CANONICAL_SEPARATORS,
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_hash(payload: Any) -> str:
    """Return the SHA-256 hash of the canonical JSON encoding of *payload*."""

    return sha256(canonical_dumps(payload).encode("utf-8")).hexdigest()


def canonical_response(payload: Mapping[str, Any]) -> Response:
    """Build a Response object with canonical JSON encoding."""

    body = canonical_dumps(payload)
    return Response(content=body, media_type="application/json")


@dataclass(slots=True)
class ErrorMapping:
    status_code: int
    detail: str


VALIDATION_ERROR = ErrorMapping(status.HTTP_400_BAD_REQUEST, "invalid_request")
UNSUPPORTED_ERROR = ErrorMapping(status.HTTP_422_UNPROCESSABLE_ENTITY, "unsupported_configuration")
COST_GUARD_ERROR = ErrorMapping(status.HTTP_429_TOO_MANY_REQUESTS, "cost_guard_triggered")
NOT_FOUND_ERROR = ErrorMapping(status.HTTP_404_NOT_FOUND, "capsule_not_found")
CONFLICT_ERROR = ErrorMapping(status.HTTP_409_CONFLICT, "build_conflict")


def http_error(mapping: ErrorMapping, *, headers: Mapping[str, str] | None = None) -> HTTPException:
    """Construct a FastAPI HTTPException from an :class:`ErrorMapping`."""

    return HTTPException(status_code=mapping.status_code, detail=mapping.detail, headers=headers)


def safe_float(value: float) -> float:
    """Return a float that is JSON friendly."""

    if math.isnan(value) or math.isinf(value):
        raise ValueError("non-finite float encountered")
    return float(value)


def monotonic_timestamp() -> float:
    """Helper used in tests to control time mocking."""

    return time.time()

