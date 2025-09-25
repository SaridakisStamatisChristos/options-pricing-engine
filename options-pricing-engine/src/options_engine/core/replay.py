"""Utilities for deterministic replay of pricing model evaluations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from numpy.random import SeedSequence


def _normalise_payload(value: Any) -> Any:
    """Recursively normalise values so JSON encoding is stable."""

    if isinstance(value, dict):
        return {str(key): _normalise_payload(sub_value) for key, sub_value in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_normalise_payload(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Capsule payload cannot contain NaN or infinite floats")
    return value


def _encode_payload(payload: Mapping[str, Any]) -> str:
    normalised = _normalise_payload(payload)
    return json.dumps(normalised, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class ReplayCapsule:
    """Container describing a deterministic pricing run."""

    capsule_id: str
    payload: Dict[str, Any]

    def to_json(self) -> str:
        """Return a canonical JSON representation of the capsule payload."""

        return _encode_payload(self.payload)

    def resolve_seed_sequence(self) -> Optional[SeedSequence]:
        """Reconstruct the :class:`SeedSequence` encoded in the capsule."""

        seed_info = self.payload.get("seed")
        if not seed_info:
            return None

        spawn_key = tuple(int(value) for value in seed_info.get("spawn_key", ()))
        entropy = seed_info.get("entropy")
        pool_size = seed_info.get("pool_size")
        if entropy is None:
            return None
        kwargs = {"spawn_key": spawn_key}
        if pool_size is not None:
            kwargs["pool_size"] = int(pool_size)
        return SeedSequence(entropy, **kwargs)


def build_replay_capsule(
    *,
    seed_sequence: Optional[SeedSequence],
    model_name: str,
    model_config: Mapping[str, Any],
    request: Mapping[str, Any],
    surface_id: Optional[str] = None,
) -> ReplayCapsule:
    """Build a :class:`ReplayCapsule` for the provided pricing invocation."""

    seed_info: Optional[Dict[str, Any]] = None
    if seed_sequence is not None:
        seed_info = {
            "entropy": int(seed_sequence.entropy),
            "spawn_key": list(int(value) for value in seed_sequence.spawn_key),
            "pool_size": int(seed_sequence.pool_size),
        }

    payload: Dict[str, Any] = {
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
