"""Utilities for deterministic replay of pricing model evaluations."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from numbers import Integral
from typing import Any

import numpy as np
from numpy.random import SeedSequence

_MAX_CAPSULE_BYTES = 1_048_576
_MAX_COLLECTION_ITEMS = 100_000
_MAX_NESTING_DEPTH = 32


def _normalise_payload(
    value: Any,
    *,
    depth: int = 0,
    _items_seen: list[int] | None = None,
) -> Any:
    """Recursively normalise values so JSON encoding is stable."""

    if _items_seen is None:
        _items_seen = [0]
    _items_seen[0] += 1
    if _items_seen[0] > _MAX_COLLECTION_ITEMS:
        raise ValueError("Capsule payload exceeds the supported total item limit")
    if depth > _MAX_NESTING_DEPTH:
        raise ValueError(f"Capsule payload nesting exceeds {_MAX_NESTING_DEPTH} levels")
    if isinstance(value, Mapping):
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise ValueError("Capsule mapping exceeds the supported item limit")
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Capsule mapping keys must be strings")
        return {
            key: _normalise_payload(
                sub_value,
                depth=depth + 1,
                _items_seen=_items_seen,
            )
            for key, sub_value in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise ValueError("Capsule sequence exceeds the supported item limit")
        return [
            _normalise_payload(item, depth=depth + 1, _items_seen=_items_seen) for item in value
        ]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Capsule payload cannot contain NaN or infinite floats")
        return value
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**256:
        raise ValueError("Capsule integers exceed the supported 257-bit range")
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Capsule payload contains unsupported type {type(value).__name__}")


def _encode_payload(payload: Mapping[str, Any]) -> str:
    normalised = _normalise_payload(payload)
    encoded = json.dumps(normalised, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > _MAX_CAPSULE_BYTES:
        raise ValueError(f"Capsule payload exceeds the {_MAX_CAPSULE_BYTES}-byte limit")
    return encoded


class ReplayCapsule:
    """Immutable-by-interface container describing a deterministic pricing run."""

    __slots__ = ("_capsule_id", "_payload")

    def __init__(self, capsule_id: str, payload: Mapping[str, Any]) -> None:
        if not isinstance(capsule_id, str):
            raise TypeError("capsule_id must be a string")
        if len(capsule_id) != 64 or any(
            character not in "0123456789abcdef" for character in capsule_id
        ):
            raise ValueError("capsule_id must be a lowercase SHA-256 digest")
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        normalized = _normalise_payload(payload)
        _encode_payload(normalized)
        self._capsule_id = capsule_id
        self._payload = deepcopy(normalized)

    @property
    def capsule_id(self) -> str:
        return self._capsule_id

    @property
    def payload(self) -> dict[str, Any]:
        """Return an isolated copy so callers cannot invalidate the capsule hash."""

        return deepcopy(self._payload)

    def to_json(self) -> str:
        """Return a canonical JSON representation of the capsule payload."""

        return _encode_payload(self._payload)

    def verify_integrity(self) -> bool:
        """Return whether the identifier matches the canonical payload digest."""

        expected = hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
        return hmac.compare_digest(self._capsule_id, expected)

    def resolve_seed_sequence(self) -> SeedSequence | None:
        """Reconstruct the :class:`SeedSequence` encoded in the capsule."""

        seed_info = self._payload.get("seed")
        if not seed_info or not isinstance(seed_info, Mapping):
            return None

        raw_spawn_key = seed_info.get("spawn_key", ())
        if (
            not isinstance(raw_spawn_key, (list, tuple))
            or len(raw_spawn_key) > 64
            or any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not 0 <= int(value) <= 2**32 - 1
                for value in raw_spawn_key
            )
        ):
            return None
        spawn_key = tuple(int(value) for value in raw_spawn_key)
        entropy_raw = seed_info.get("entropy")
        pool_size = seed_info.get("pool_size")
        if (
            isinstance(entropy_raw, Integral)
            and not isinstance(entropy_raw, bool)
            and 0 <= int(entropy_raw) <= 2**128 - 1
        ):
            entropy: int | Sequence[int] = int(entropy_raw)
        elif isinstance(entropy_raw, (list, tuple)) and all(
            isinstance(value, Integral)
            and not isinstance(value, bool)
            and 0 <= int(value) <= 2**128 - 1
            for value in entropy_raw
        ):
            if not entropy_raw or len(entropy_raw) > 64:
                return None
            entropy = [int(value) for value in entropy_raw]
        else:
            return None
        if pool_size is None:
            resolved_pool_size = 4
        elif (
            isinstance(pool_size, bool)
            or not isinstance(pool_size, Integral)
            or not 4 <= int(pool_size) <= 64
        ):
            return None
        else:
            resolved_pool_size = int(pool_size)
        try:
            return SeedSequence(
                entropy,
                spawn_key=spawn_key,
                pool_size=resolved_pool_size,
            )
        except ValueError:
            return None


def build_replay_capsule(
    *,
    seed_sequence: SeedSequence | None,
    model_name: str,
    model_config: Mapping[str, Any],
    request: Mapping[str, Any],
    surface_id: str | None = None,
) -> ReplayCapsule:
    """Build a :class:`ReplayCapsule` for the provided pricing invocation."""

    if seed_sequence is not None and not isinstance(seed_sequence, SeedSequence):
        raise TypeError("seed_sequence must be a numpy.random.SeedSequence or None")
    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")
    model_name = model_name.strip()
    if (
        not model_name
        or len(model_name) > 128
        or any(ord(character) < 32 or ord(character) == 127 for character in model_name)
    ):
        raise ValueError("model_name must contain between 1 and 128 characters")
    if not isinstance(model_config, Mapping) or not isinstance(request, Mapping):
        raise TypeError("model_config and request must be mappings")
    if surface_id is not None and (
        not isinstance(surface_id, str)
        or not surface_id
        or len(surface_id) > 128
        or any(ord(character) < 32 or ord(character) == 127 for character in surface_id)
    ):
        raise ValueError("surface_id must contain between 1 and 128 characters")

    seed_info: dict[str, Any] | None = None
    if seed_sequence is not None:
        raw_entropy = seed_sequence.entropy
        entropy: int | list[int]
        if raw_entropy is None:  # pragma: no cover - NumPy always materializes entropy
            raise ValueError("seed sequence does not contain replayable entropy")
        if (
            isinstance(raw_entropy, Integral)
            and not isinstance(raw_entropy, bool)
            and 0 <= int(raw_entropy) <= 2**128 - 1
        ):
            entropy = int(raw_entropy)
        elif isinstance(raw_entropy, (Sequence, np.ndarray)):
            raw_values = list(raw_entropy)
            if not 1 <= len(raw_values) <= 64 or any(
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not 0 <= int(value) <= 2**128 - 1
                for value in raw_values
            ):
                raise ValueError("seed sequence entropy is outside the replayable domain")
            entropy = [int(value) for value in raw_values]
        else:
            raise ValueError("seed sequence entropy is outside the replayable domain")
        spawn_key = tuple(seed_sequence.spawn_key)
        pool_size = int(seed_sequence.pool_size)
        if len(spawn_key) > 64 or any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= int(value) <= 2**32 - 1
            for value in spawn_key
        ):
            raise ValueError("seed sequence spawn key is outside the replayable domain")
        if not 4 <= pool_size <= 64:
            raise ValueError("seed sequence pool size is outside the replayable domain")
        seed_info = {
            "entropy": entropy,
            "spawn_key": [int(value) for value in spawn_key],
            "pool_size": pool_size,
        }

    payload: dict[str, Any] = {
        "model": {"name": model_name, "config": dict(model_config)},
        "request": dict(request),
    }

    if seed_info is not None:
        payload["seed"] = seed_info

    if surface_id is not None:
        payload["surface_id"] = surface_id

    encoded = _encode_payload(payload)
    capsule_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return ReplayCapsule(capsule_id=capsule_id, payload=payload)
