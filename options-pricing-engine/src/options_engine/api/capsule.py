"""In-memory replay capsule management."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from threading import Lock
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

from numpy.random import SeedSequence

from .codec import canonical_dumps, canonical_hash


DEFAULT_BUILD_ID = "local-dev"
IDEMPOTENCY_TTL_SECONDS = 600
MC_MAX_PATHS = 262_144
MC_BATCH_AGGREGATE_LIMIT = 1_500_000


def derive_seed_lineage(*, seed_prefix: str | None, base_hash: str, index: int = 0) -> str:
    """Deterministically derive the seed lineage string for a request."""

    if seed_prefix:
        return f"{seed_prefix.upper()}:{index}"
    return f"{base_hash}:{index}"


def lineage_to_seed_sequence(lineage: str) -> SeedSequence:
    """Convert the lineage string into a stable :class:`SeedSequence`."""

    digest = sha256(lineage.encode("utf-8")).digest()
    seed_int = int.from_bytes(digest[:8], "big", signed=False)
    return SeedSequence(seed_int)


def compute_capsule_id(
    *,
    request_payload: Mapping[str, Any],
    model_config: Mapping[str, Any],
    surface_id: str | None,
    seed_lineage: str,
    build_id: str,
) -> str:
    """Compute the deterministic capsule identifier."""

    components = (
        canonical_dumps(request_payload),
        canonical_dumps(model_config),
        surface_id or "",
        seed_lineage,
        build_id,
    )
    sha = sha256()
    for component in components:
        sha.update(component.encode("utf-8"))
    return sha.hexdigest()


@dataclass(slots=True)
class CapsuleRecord:
    capsule_id: str
    request_payload: Dict[str, Any]
    response_payload: Dict[str, Any]
    seed_lineage: str
    model_used: Dict[str, Any]
    build_id: str
    timestamp: datetime


class CapsuleStore:
    """Simple in-memory capsule store used for replay tests."""

    def __init__(self) -> None:
        self._records: MutableMapping[str, CapsuleRecord] = {}
        self._lock = Lock()

    def save(self, record: CapsuleRecord) -> None:
        with self._lock:
            self._records[record.capsule_id] = record

    def get(self, capsule_id: str) -> CapsuleRecord | None:
        with self._lock:
            record = self._records.get(capsule_id)
            if record is None:
                return None
            return CapsuleRecord(
                capsule_id=record.capsule_id,
                request_payload=dict(record.request_payload),
                response_payload=dict(record.response_payload),
                seed_lineage=record.seed_lineage,
                model_used=dict(record.model_used),
                build_id=record.build_id,
                timestamp=record.timestamp,
            )


class IdempotencyCache:
    """In-memory cache keyed by idempotency token and request hash."""

    def __init__(self) -> None:
        self._entries: MutableMapping[Tuple[str, str], Tuple[float, str]] = {}
        self._lock = Lock()

    def get(self, key: str, request_hash: str, *, now: float | None = None) -> Optional[str]:
        moment = now or datetime.now(UTC).timestamp()
        with self._lock:
            payload = self._entries.get((key, request_hash))
            if not payload:
                return None
            expires_at, body = payload
            if expires_at <= moment:
                self._entries.pop((key, request_hash), None)
                return None
            return body

    def put(self, key: str, request_hash: str, body: str, *, now: float | None = None) -> None:
        moment = now or datetime.now(UTC).timestamp()
        expires_at = moment + IDEMPOTENCY_TTL_SECONDS
        with self._lock:
            self._entries[(key, request_hash)] = (expires_at, body)


CAPSULE_STORE = CapsuleStore()
IDEMPOTENCY_CACHE = IdempotencyCache()


def build_capsule_record(
    *,
    request_payload: Mapping[str, Any],
    response_payload: Mapping[str, Any],
    model_used: Mapping[str, Any],
    seed_lineage: str,
    build_id: str,
) -> CapsuleRecord:
    capsule_id = compute_capsule_id(
        request_payload=request_payload,
        model_config=model_used,
        surface_id=None,
        seed_lineage=seed_lineage,
        build_id=build_id,
    )
    return CapsuleRecord(
        capsule_id=capsule_id,
        request_payload=dict(request_payload),
        response_payload=dict(response_payload),
        seed_lineage=seed_lineage,
        model_used=dict(model_used),
        build_id=build_id,
        timestamp=datetime.now(UTC),
    )

