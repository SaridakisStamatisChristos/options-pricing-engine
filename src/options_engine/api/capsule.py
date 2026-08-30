"""In-memory replay capsule management."""

from __future__ import annotations

import math
import re
import time
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from numbers import Integral, Real
from threading import Lock
from typing import Any

from numpy.random import SeedSequence

from .codec import canonical_dumps

DEFAULT_BUILD_ID = "local-dev"
IDEMPOTENCY_TTL_SECONDS = 600
CAPSULE_TTL_SECONDS = 24 * 60 * 60
MAX_STORE_ENTRIES = 10_000
MAX_STORE_BYTES = 64 * 1024 * 1024
MAX_STORE_TTL_SECONDS = 7 * 24 * 60 * 60
MC_MAX_PATHS = 262_144
MC_BATCH_AGGREGATE_LIMIT = 1_500_000
LSMC_BATCH_AGGREGATE_WORK_LIMIT = 5_000_000
_IDEMPOTENCY_KEY_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}")


class IdempotencyConflictError(ValueError):
    """Raised when one idempotency key is reused for a different request."""


def _bounded_integer(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalised = int(value)
    if not minimum <= normalised <= maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
    return normalised


def _copy_record(record: CapsuleRecord) -> CapsuleRecord:
    return CapsuleRecord(
        capsule_id=record.capsule_id,
        request_payload=deepcopy(record.request_payload),
        response_payload=deepcopy(record.response_payload),
        seed_lineage=record.seed_lineage,
        model_used=deepcopy(record.model_used),
        build_id=record.build_id,
        timestamp=record.timestamp,
    )


def _record_size(record: CapsuleRecord) -> int:
    payload = {
        "capsule_id": record.capsule_id,
        "request": record.request_payload,
        "response": record.response_payload,
        "seed_lineage": record.seed_lineage,
        "model": record.model_used,
        "build_id": record.build_id,
        "timestamp": record.timestamp.isoformat(),
    }
    return len(canonical_dumps(payload).encode("utf-8"))


def _validate_digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal digest")
    return value


def _monotonic_moment(value: float | None) -> float:
    if value is None:
        return time.monotonic()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("now must be a real number")
    moment = float(value)
    if not math.isfinite(moment):
        raise ValueError("now must be finite")
    return moment


def derive_seed_lineage(*, seed_prefix: str | None, base_hash: str, index: int = 0) -> str:
    """Deterministically derive the seed lineage string for a request."""

    _validate_digest(base_hash, name="base_hash")
    index = _bounded_integer("index", index, minimum=0, maximum=1_000_000)
    if seed_prefix is not None:
        if (
            not isinstance(seed_prefix, str)
            or not 1 <= len(seed_prefix) <= 64
            or any(
                character
                not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-"
                for character in seed_prefix
            )
        ):
            raise ValueError("seed_prefix contains unsupported characters or length")
        return f"{seed_prefix.upper()}:{index}"
    return f"{base_hash}:{index}"


def lineage_to_seed_sequence(lineage: str) -> SeedSequence:
    """Convert the lineage string into a stable :class:`SeedSequence`."""

    if not isinstance(lineage, str) or not 1 <= len(lineage) <= 256:
        raise ValueError("lineage must contain between 1 and 256 characters")
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

    if not isinstance(request_payload, Mapping) or not isinstance(model_config, Mapping):
        raise TypeError("request_payload and model_config must be mappings")
    if surface_id is not None and (
        not isinstance(surface_id, str)
        or not 1 <= len(surface_id) <= 128
        or any(ord(character) < 32 for character in surface_id)
    ):
        raise ValueError("surface_id must be a printable string of at most 128 characters")
    if (
        not isinstance(seed_lineage, str)
        or not 1 <= len(seed_lineage) <= 256
        or any(ord(character) < 32 for character in seed_lineage)
    ):
        raise ValueError("seed_lineage must contain between 1 and 256 characters")
    if (
        not isinstance(build_id, str)
        or not 1 <= len(build_id) <= 128
        or any(ord(character) < 32 for character in build_id)
    ):
        raise ValueError("build_id must contain between 1 and 128 characters")
    components = (
        canonical_dumps(request_payload),
        canonical_dumps(model_config),
        surface_id or "",
        seed_lineage,
        build_id,
    )
    sha = sha256()
    for component in components:
        encoded = component.encode("utf-8")
        sha.update(len(encoded).to_bytes(8, "big"))
        sha.update(encoded)
    return sha.hexdigest()


@dataclass(frozen=True, slots=True)
class CapsuleRecord:
    capsule_id: str
    request_payload: dict[str, Any]
    response_payload: dict[str, Any]
    seed_lineage: str
    model_used: dict[str, Any]
    build_id: str
    timestamp: datetime


class CapsuleStore:
    """Bounded, expiring in-process capsule store.

    Production deployments use one API worker unless an external state backend is
    configured; this fallback cannot grow without limit.
    """

    def __init__(
        self,
        *,
        max_entries: int = MAX_STORE_ENTRIES,
        ttl_seconds: int = CAPSULE_TTL_SECONDS,
        max_bytes: int = MAX_STORE_BYTES,
    ) -> None:
        self._records: OrderedDict[str, tuple[CapsuleRecord, int, float]] = OrderedDict()
        self._lock = Lock()
        self._max_entries = _bounded_integer("max_entries", max_entries, minimum=1, maximum=100_000)
        self._ttl_seconds = _bounded_integer(
            "ttl_seconds",
            ttl_seconds,
            minimum=1,
            maximum=MAX_STORE_TTL_SECONDS,
        )
        self._max_bytes = _bounded_integer(
            "max_bytes", max_bytes, minimum=1_024, maximum=1_073_741_824
        )
        self._current_bytes = 0

    def _purge_locked(self, moment: float) -> None:
        expired = [
            key
            for key, (_value, _size, expires_at) in self._records.items()
            if expires_at <= moment
        ]
        for key in expired:
            removed = self._records.pop(key, None)
            if removed is not None:
                self._current_bytes -= removed[1]

    def save(self, record: CapsuleRecord) -> None:
        if not isinstance(record, CapsuleRecord):
            raise TypeError("record must be a CapsuleRecord")
        _validate_digest(record.capsule_id, name="capsule_id")
        if not isinstance(record.request_payload, dict) or not isinstance(
            record.response_payload, dict
        ):
            raise TypeError("capsule request and response payloads must be dictionaries")
        if not isinstance(record.model_used, dict):
            raise TypeError("capsule model_used must be a dictionary")
        if not isinstance(record.timestamp, datetime):
            raise TypeError("record timestamp must be a datetime")
        if record.timestamp.tzinfo is None or record.timestamp.utcoffset() is None:
            raise ValueError("record timestamp must be timezone-aware")
        stored = _copy_record(record)
        expected_id = compute_capsule_id(
            request_payload=stored.request_payload,
            model_config=stored.model_used,
            surface_id=None,
            seed_lineage=stored.seed_lineage,
            build_id=stored.build_id,
        )
        if stored.capsule_id != expected_id:
            raise ValueError("capsule_id does not match the record configuration")
        size = _record_size(stored)
        if size > self._max_bytes:
            raise ValueError("capsule record exceeds the store byte budget")
        moment = datetime.now(UTC)
        if stored.timestamp <= moment - timedelta(seconds=self._ttl_seconds):
            raise ValueError("capsule record is already expired")
        if stored.timestamp > moment + timedelta(minutes=5):
            raise ValueError("capsule record timestamp is too far in the future")
        remaining_ttl = (
            stored.timestamp + timedelta(seconds=self._ttl_seconds) - moment
        ).total_seconds()
        expires_at = time.monotonic() + remaining_ttl
        with self._lock:
            self._purge_locked(time.monotonic())
            previous = self._records.pop(stored.capsule_id, None)
            if previous is not None:
                self._current_bytes -= previous[1]
            self._records[stored.capsule_id] = (stored, size, expires_at)
            self._current_bytes += size
            while len(self._records) > self._max_entries or self._current_bytes > self._max_bytes:
                _key, (_removed, removed_size, _expires_at) = self._records.popitem(last=False)
                self._current_bytes -= removed_size

    def get(self, capsule_id: str) -> CapsuleRecord | None:
        try:
            _validate_digest(capsule_id, name="capsule_id")
        except ValueError:
            return None
        with self._lock:
            self._purge_locked(time.monotonic())
            item = self._records.get(capsule_id)
            if item is None:
                return None
            self._records.move_to_end(capsule_id)
            return _copy_record(item[0])


class IdempotencyCache:
    """In-memory cache keyed by idempotency token and request hash."""

    def __init__(
        self,
        *,
        max_entries: int = MAX_STORE_ENTRIES,
        ttl_seconds: int = IDEMPOTENCY_TTL_SECONDS,
        max_bytes: int = MAX_STORE_BYTES,
    ) -> None:
        self._entries: OrderedDict[str, tuple[str, float, str, int]] = OrderedDict()
        self._lock = Lock()
        self._max_entries = _bounded_integer("max_entries", max_entries, minimum=1, maximum=100_000)
        self._ttl_seconds = _bounded_integer(
            "ttl_seconds",
            ttl_seconds,
            minimum=1,
            maximum=MAX_STORE_TTL_SECONDS,
        )
        self._max_bytes = _bounded_integer(
            "max_bytes", max_bytes, minimum=1_024, maximum=1_073_741_824
        )
        self._current_bytes = 0

    def _purge_locked(self, moment: float) -> None:
        expired = [
            key
            for key, (_request_hash, expires_at, _body, _size) in self._entries.items()
            if expires_at <= moment
        ]
        for key in expired:
            removed = self._entries.pop(key, None)
            if removed is not None:
                self._current_bytes -= removed[3]

    def get(self, key: str, request_hash: str, *, now: float | None = None) -> str | None:
        if not isinstance(key, str) or _IDEMPOTENCY_KEY_PATTERN.fullmatch(key) is None:
            raise ValueError("idempotency key contains unsupported characters or length")
        _validate_digest(request_hash, name="request_hash")
        moment = _monotonic_moment(now)
        with self._lock:
            self._purge_locked(moment)
            payload = self._entries.get(key)
            if not payload:
                return None
            stored_hash, _expires_at, body, _size = payload
            if stored_hash != request_hash:
                raise IdempotencyConflictError(
                    "idempotency key was already used for a different request"
                )
            self._entries.move_to_end(key)
            return body

    def put(self, key: str, request_hash: str, body: str, *, now: float | None = None) -> None:
        if not isinstance(key, str) or _IDEMPOTENCY_KEY_PATTERN.fullmatch(key) is None:
            raise ValueError("idempotency key contains unsupported characters or length")
        _validate_digest(request_hash, name="request_hash")
        if not isinstance(body, str):
            raise TypeError("body must be a string")
        moment = _monotonic_moment(now)
        size = len(key.encode("utf-8")) + len(request_hash) + len(body.encode("utf-8"))
        if size > self._max_bytes:
            raise ValueError("idempotency response exceeds the cache byte budget")
        expires_at = moment + self._ttl_seconds
        with self._lock:
            self._purge_locked(moment)
            previous = self._entries.get(key)
            if previous is not None:
                if previous[0] != request_hash:
                    raise IdempotencyConflictError(
                        "idempotency key was already used for a different request"
                    )
                self._entries.move_to_end(key)
                return
            self._entries[key] = (request_hash, expires_at, body, size)
            self._current_bytes += size
            while len(self._entries) > self._max_entries or self._current_bytes > self._max_bytes:
                _removed_key, (_hash, _expiry, _body, removed_size) = self._entries.popitem(
                    last=False
                )
                self._current_bytes -= removed_size


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
        request_payload=deepcopy(dict(request_payload)),
        response_payload=deepcopy(dict(response_payload)),
        seed_lineage=seed_lineage,
        model_used=deepcopy(dict(model_used)),
        build_id=build_id,
        timestamp=datetime.now(UTC),
    )
