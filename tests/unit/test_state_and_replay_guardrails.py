"""Boundary tests for bounded process state and deterministic replay."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from numpy.random import SeedSequence

from options_engine.api import capsule
from options_engine.api.capsule import (
    CapsuleRecord,
    CapsuleStore,
    IdempotencyCache,
    build_capsule_record,
    compute_capsule_id,
    derive_seed_lineage,
    lineage_to_seed_sequence,
)
from options_engine.core import replay
from options_engine.core.replay import ReplayCapsule, build_replay_capsule


def _record(
    label: str = "record",
    *,
    request_payload: dict[str, Any] | None = None,
    response_payload: dict[str, Any] | None = None,
    model_used: dict[str, Any] | None = None,
    seed_lineage: str = "seed:0",
    build_id: str = "test-build",
    timestamp: datetime | None = None,
) -> CapsuleRecord:
    request = request_payload if request_payload is not None else {"label": label}
    response = response_payload if response_payload is not None else {"price": 1.25}
    model = model_used if model_used is not None else {"family": "monte_carlo"}
    identifier = compute_capsule_id(
        request_payload=request,
        model_config=model,
        surface_id=None,
        seed_lineage=seed_lineage,
        build_id=build_id,
    )
    return CapsuleRecord(
        capsule_id=identifier,
        request_payload=request,
        response_payload=response,
        seed_lineage=seed_lineage,
        model_used=model,
        build_id=build_id,
        timestamp=timestamp or datetime.now(UTC),
    )


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_bounded_integer_rejects_non_integral_values(value: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        capsule._bounded_integer("limit", value, minimum=1, maximum=2)


@pytest.mark.parametrize("value", [0, 3])
def test_bounded_integer_rejects_out_of_range_values(value: int) -> None:
    with pytest.raises(ValueError, match="within"):
        capsule._bounded_integer("limit", value, minimum=1, maximum=2)


@pytest.mark.parametrize("digest", [None, "a" * 63, "A" * 64, "g" * 64])
def test_digest_validation_is_strict(digest: object) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        capsule._validate_digest(digest, name="digest")


def test_seed_lineage_derivation_is_stable_and_bounded() -> None:
    digest = "a" * 64
    assert derive_seed_lineage(seed_prefix=None, base_hash=digest, index=2) == f"{digest}:2"
    assert derive_seed_lineage(seed_prefix="desk-1", base_hash=digest) == "DESK-1:0"
    assert lineage_to_seed_sequence("DESK-1:0").generate_state(4).tolist() == (
        lineage_to_seed_sequence("DESK-1:0").generate_state(4).tolist()
    )


@pytest.mark.parametrize("prefix", ["", "x" * 65, "bad prefix", "bad/seed", 7])
def test_seed_lineage_rejects_malformed_prefixes(prefix: object) -> None:
    with pytest.raises(ValueError, match="seed_prefix"):
        derive_seed_lineage(seed_prefix=prefix, base_hash="a" * 64)  # type: ignore[arg-type]


@pytest.mark.parametrize("lineage", ["", "x" * 257, 42])
def test_seed_sequence_rejects_malformed_lineage(lineage: object) -> None:
    with pytest.raises(ValueError, match="lineage"):
        lineage_to_seed_sequence(lineage)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"request_payload": [], "model_config": {}},
        {"request_payload": {}, "model_config": []},
    ],
)
def test_capsule_id_requires_mapping_components(kwargs: dict[str, object]) -> None:
    with pytest.raises(TypeError, match="mappings"):
        compute_capsule_id(
            **kwargs,  # type: ignore[arg-type]
            surface_id=None,
            seed_lineage="seed",
            build_id="build",
        )


@pytest.mark.parametrize("surface_id", ["", "x" * 129, "bad\nvalue", 1])
def test_capsule_id_rejects_malformed_surface_id(surface_id: object) -> None:
    with pytest.raises(ValueError, match="surface_id"):
        compute_capsule_id(
            request_payload={},
            model_config={},
            surface_id=surface_id,  # type: ignore[arg-type]
            seed_lineage="seed",
            build_id="build",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed_lineage", "", "seed_lineage"),
        ("seed_lineage", "bad\nseed", "seed_lineage"),
        ("build_id", "", "build_id"),
        ("build_id", "x" * 129, "build_id"),
        ("build_id", "bad\nbuild", "build_id"),
    ],
)
def test_capsule_id_rejects_malformed_identity_fields(field: str, value: str, message: str) -> None:
    values = {"seed_lineage": "seed", "build_id": "build", field: value}
    with pytest.raises(ValueError, match=message):
        compute_capsule_id(
            request_payload={},
            model_config={},
            surface_id=None,
            seed_lineage=values["seed_lineage"],
            build_id=values["build_id"],
        )


def test_capsule_id_commits_to_surface_and_build() -> None:
    common = {
        "request_payload": {"request": 1},
        "model_config": {"model": "mc"},
        "seed_lineage": "seed",
    }
    first = compute_capsule_id(**common, surface_id="surface-a", build_id="one")
    second = compute_capsule_id(**common, surface_id="surface-b", build_id="one")
    third = compute_capsule_id(**common, surface_id="surface-a", build_id="two")
    assert len({first, second, third}) == 3


@pytest.mark.parametrize(
    ("name", "value", "exception"),
    [
        ("max_entries", True, TypeError),
        ("max_entries", 0, ValueError),
        ("ttl_seconds", 0, ValueError),
        ("ttl_seconds", capsule.MAX_STORE_TTL_SECONDS + 1, ValueError),
        ("max_bytes", 1_023, ValueError),
    ],
)
def test_state_store_configuration_is_bounded(
    name: str, value: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        CapsuleStore(**{name: value})  # type: ignore[arg-type]
    with pytest.raises(exception):
        IdempotencyCache(**{name: value})  # type: ignore[arg-type]


def test_capsule_store_rejects_malformed_records() -> None:
    store = CapsuleStore(max_bytes=4_096)
    valid = _record()
    with pytest.raises(TypeError, match="CapsuleRecord"):
        store.save(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="capsule_id"):
        store.save(replace(valid, capsule_id="bad"))
    with pytest.raises(TypeError, match="payloads"):
        store.save(replace(valid, request_payload=[]))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="model_used"):
        store.save(replace(valid, model_used=[]))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="datetime"):
        store.save(replace(valid, timestamp="today"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="timezone-aware"):
        store.save(replace(valid, timestamp=datetime.now()))
    with pytest.raises(ValueError, match="does not match"):
        store.save(replace(valid, response_payload={"price": 9.0}, capsule_id="b" * 64))


def test_capsule_store_rejects_oversized_expired_and_future_records() -> None:
    with pytest.raises(ValueError, match="byte budget"):
        CapsuleStore(max_bytes=1_024).save(_record(request_payload={"payload": "x" * 2_000}))

    short_store = CapsuleStore(ttl_seconds=2, max_bytes=4_096)
    with pytest.raises(ValueError, match="already expired"):
        short_store.save(_record(timestamp=datetime.now(UTC) - timedelta(seconds=3)))
    with pytest.raises(ValueError, match="future"):
        short_store.save(_record(timestamp=datetime.now(UTC) + timedelta(minutes=6)))


def test_capsule_store_updates_and_expires_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [10.0]
    monkeypatch.setattr(capsule.time, "monotonic", lambda: clock[0])
    store = CapsuleStore(ttl_seconds=1, max_entries=2, max_bytes=4_096)
    record = _record()
    store.save(record)
    store.save(record)
    assert store.get(record.capsule_id) is not None
    clock[0] = 12.0
    assert store.get(record.capsule_id) is None
    assert store.get("not-a-digest") is None


def test_capsule_store_evicts_to_meet_byte_budget() -> None:
    store = CapsuleStore(max_entries=10, max_bytes=1_024)
    first = _record("first", request_payload={"payload": "a" * 350})
    second = _record("second", request_payload={"payload": "b" * 350})
    store.save(first)
    store.save(second)
    assert store.get(first.capsule_id) is None
    assert store.get(second.capsule_id) is not None


def test_build_capsule_record_is_isolated_from_callers() -> None:
    request = {"nested": {"value": 1}}
    response = {"price": 2.0}
    model = {"family": "mc"}
    record = build_capsule_record(
        request_payload=request,
        response_payload=response,
        model_used=model,
        seed_lineage="seed",
        build_id="build",
    )
    request["nested"]["value"] = 99
    response["price"] = 99.0
    model["family"] = "changed"
    assert record.request_payload["nested"]["value"] == 1
    assert record.response_payload["price"] == 2.0
    assert record.model_used["family"] == "mc"


@pytest.mark.parametrize("now", [True, math.nan, math.inf, "now"])
def test_idempotency_cache_rejects_invalid_clock(now: object) -> None:
    cache = IdempotencyCache(max_bytes=2_048)
    exception = TypeError if isinstance(now, (bool, str)) else ValueError
    with pytest.raises(exception):
        cache.get("key", "a" * 64, now=now)  # type: ignore[arg-type]


def test_idempotency_cache_validates_keys_hashes_and_bodies() -> None:
    cache = IdempotencyCache(max_bytes=1_024)
    for key in ("", "space key", "x" * 129):
        with pytest.raises(ValueError, match="key"):
            cache.get(key, "a" * 64, now=1.0)
        with pytest.raises(ValueError, match="key"):
            cache.put(key, "a" * 64, "body", now=1.0)
    with pytest.raises(ValueError, match="request_hash"):
        cache.get("key", "bad", now=1.0)
    with pytest.raises(TypeError, match="body"):
        cache.put("key", "a" * 64, b"body", now=1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="byte budget"):
        cache.put("key", "a" * 64, "x" * 2_000, now=1.0)


def test_idempotency_cache_is_lru_bounded_and_duplicate_put_is_noop() -> None:
    cache = IdempotencyCache(max_entries=2, max_bytes=2_048)
    cache.put("one", "a" * 64, "first", now=1.0)
    cache.put("one", "a" * 64, "ignored", now=2.0)
    assert cache.get("one", "a" * 64, now=2.0) == "first"
    cache.put("two", "b" * 64, "second", now=2.0)
    cache.put("three", "c" * 64, "third", now=2.0)
    assert cache.get("one", "a" * 64, now=2.0) is None
    assert cache.get("missing", "d" * 64, now=2.0) is None


def test_idempotency_cache_evicts_by_bytes_and_releases_expired_keys() -> None:
    cache = IdempotencyCache(max_entries=10, max_bytes=1_024, ttl_seconds=2)
    cache.put("one", "a" * 64, "a" * 500, now=1.0)
    cache.put("two", "b" * 64, "b" * 500, now=1.0)
    assert cache.get("one", "a" * 64, now=1.0) is None
    assert cache.get("two", "b" * 64, now=3.0) is None
    cache.put("two", "c" * 64, "new", now=3.0)
    assert cache.get("two", "c" * 64, now=3.0) == "new"


def test_replay_payload_normalisation_rejects_unsafe_shapes() -> None:
    with pytest.raises(TypeError, match="keys"):
        replay._normalise_payload({1: "value"})
    with pytest.raises(TypeError, match="unsupported type"):
        replay._normalise_payload({"value": object()})
    with pytest.raises(ValueError, match="NaN"):
        replay._normalise_payload({"value": math.inf})
    with pytest.raises(ValueError, match="257-bit"):
        replay._normalise_payload({"value": 2**257})

    nested: object = "leaf"
    for _ in range(34):
        nested = [nested]
    with pytest.raises(ValueError, match="nesting"):
        replay._normalise_payload(nested)


def test_replay_payload_size_and_item_budgets() -> None:
    with pytest.raises(ValueError, match="byte limit"):
        replay._encode_payload({"value": "x" * replay._MAX_CAPSULE_BYTES})
    with pytest.raises(ValueError, match="sequence"):
        replay._normalise_payload([None] * (replay._MAX_COLLECTION_ITEMS + 1))
    with pytest.raises(ValueError, match="total item"):
        replay._normalise_payload([None] * replay._MAX_COLLECTION_ITEMS)


@pytest.mark.parametrize("capsule_id", [1, None])
def test_replay_capsule_requires_string_identifier(capsule_id: object) -> None:
    with pytest.raises(TypeError, match="capsule_id"):
        ReplayCapsule(capsule_id, {})  # type: ignore[arg-type]


@pytest.mark.parametrize("capsule_id", ["", "A" * 64, "g" * 64])
def test_replay_capsule_requires_canonical_identifier(capsule_id: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        ReplayCapsule(capsule_id, {})


def test_replay_capsule_is_immutable_by_interface_and_verifies_integrity() -> None:
    capsule_value = build_replay_capsule(
        seed_sequence=SeedSequence(7),
        model_name="monte_carlo",
        model_config={"paths": 128},
        request={"contract": {"strike": 100.0}},
        surface_id="surface-1",
    )
    assert capsule_value.verify_integrity() is True
    payload = capsule_value.payload
    payload["model"]["name"] = "changed"
    assert capsule_value.payload["model"]["name"] == "monte_carlo"
    assert ReplayCapsule("0" * 64, {"value": 1}).verify_integrity() is False
    with pytest.raises(TypeError, match="mapping"):
        ReplayCapsule("0" * 64, [])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "seed",
    [
        None,
        {"entropy": 1, "spawn_key": "bad"},
        {"entropy": 1, "spawn_key": [True]},
        {"entropy": 1, "spawn_key": [-1]},
        {"entropy": 1, "spawn_key": [0] * 65},
        {"entropy": []},
        {"entropy": [1] * 65},
        {"entropy": "bad"},
        {"entropy": 1, "pool_size": True},
        {"entropy": 1, "pool_size": 3},
        {"entropy": 1, "pool_size": 65},
    ],
)
def test_replay_capsule_rejects_malformed_seed_metadata(seed: object) -> None:
    payload = {} if seed is None else {"seed": seed}
    assert ReplayCapsule("0" * 64, payload).resolve_seed_sequence() is None


def test_replay_capsule_resolves_integer_and_sequence_entropy() -> None:
    integer = ReplayCapsule(
        "0" * 64,
        {"seed": {"entropy": 7, "spawn_key": [1, 2]}},
    ).resolve_seed_sequence()
    sequence = ReplayCapsule(
        "0" * 64,
        {"seed": {"entropy": [7, 8], "spawn_key": [], "pool_size": 8}},
    ).resolve_seed_sequence()
    assert integer is not None and integer.spawn_key == (1, 2) and integer.pool_size == 4
    assert sequence is not None and sequence.entropy == [7, 8] and sequence.pool_size == 8


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("seed_sequence", 7, TypeError),
        ("model_name", 7, TypeError),
        ("model_name", " ", ValueError),
        ("model_name", "x" * 129, ValueError),
        ("model_name", "bad\x7fname", ValueError),
        ("model_config", [], TypeError),
        ("request", [], TypeError),
        ("surface_id", "", ValueError),
        ("surface_id", "x" * 129, ValueError),
        ("surface_id", "bad\nsurface", ValueError),
    ],
)
def test_build_replay_capsule_validates_boundaries(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    values: dict[str, object] = {
        "seed_sequence": None,
        "model_name": "monte_carlo",
        "model_config": {},
        "request": {},
        "surface_id": None,
    }
    values[field] = value
    with pytest.raises(exception):
        build_replay_capsule(**values)  # type: ignore[arg-type]
