"""Tests for bounded replay and idempotency state."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from options_engine.api.capsule import (
    CapsuleRecord,
    CapsuleStore,
    IdempotencyCache,
    IdempotencyConflictError,
    compute_capsule_id,
)


def _record(identifier: str, *, payload: str = "value") -> CapsuleRecord:
    request_payload = {"label": identifier, "nested": {"payload": payload}}
    model_used = {"family": "black_scholes"}
    capsule_id = compute_capsule_id(
        request_payload=request_payload,
        model_config=model_used,
        surface_id=None,
        seed_lineage="seed",
        build_id="test",
    )
    return CapsuleRecord(
        capsule_id=capsule_id,
        request_payload=request_payload,
        response_payload={"price": 10.0},
        seed_lineage="seed",
        model_used=model_used,
        build_id="test",
        timestamp=datetime.now(UTC),
    )


def test_capsule_store_defensively_copies_nested_payloads() -> None:
    store = CapsuleStore(max_entries=2, max_bytes=4_096)
    record = _record("a" * 64)
    store.save(record)

    record.request_payload["nested"]["payload"] = "mutated"
    fetched = store.get(record.capsule_id)
    assert fetched is not None
    assert fetched.request_payload["nested"]["payload"] == "value"

    fetched.request_payload["nested"]["payload"] = "also-mutated"
    fetched_again = store.get(record.capsule_id)
    assert fetched_again is not None
    assert fetched_again.request_payload["nested"]["payload"] == "value"


def test_capsule_store_enforces_entry_and_byte_budgets() -> None:
    store = CapsuleStore(max_entries=1, max_bytes=2_048)
    first = _record("a" * 64, payload="x" * 500)
    second = _record("b" * 64, payload="y" * 500)

    store.save(first)
    store.save(second)

    assert store.get(first.capsule_id) is None
    assert store.get(second.capsule_id) is not None


def test_idempotency_cache_rejects_key_reuse_for_different_request() -> None:
    cache = IdempotencyCache(max_entries=2, max_bytes=2_048)
    first_hash = "a" * 64
    second_hash = "b" * 64

    cache.put("key", first_hash, '{"price":1}', now=10.0)
    assert cache.get("key", first_hash, now=11.0) == '{"price":1}'
    with pytest.raises(IdempotencyConflictError):
        cache.get("key", second_hash, now=11.0)
    with pytest.raises(IdempotencyConflictError):
        cache.put("key", second_hash, '{"price":2}', now=11.0)


def test_idempotency_cache_expires_and_releases_key() -> None:
    cache = IdempotencyCache(ttl_seconds=10, max_bytes=2_048)
    first_hash = "a" * 64
    second_hash = "b" * 64
    cache.put("key", first_hash, "first", now=10.0)

    assert cache.get("key", first_hash, now=20.0) is None
    cache.put("key", second_hash, "second", now=20.0)
    assert cache.get("key", second_hash, now=20.0) == "second"


def test_capsule_id_commits_to_field_boundaries() -> None:
    baseline = compute_capsule_id(
        request_payload={"contract": "ab"},
        model_config={"name": "c"},
        surface_id=None,
        seed_lineage="seed",
        build_id="build",
    )
    shifted = compute_capsule_id(
        request_payload={"contract": "a"},
        model_config={"name": "bc"},
        surface_id=None,
        seed_lineage="seed",
        build_id="build",
    )

    assert baseline != shifted
