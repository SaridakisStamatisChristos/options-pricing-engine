"""Stable test-client adapter for the FastAPI integration suite."""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import FastAPI
from fastapi.testclient import TestClient


class SimpleTestClient(TestClient):
    """Expose the historical fixture API on Starlette's supported client.

    The prior hand-written ASGI transport called private router lifecycle APIs,
    which broke as soon as Starlette modernised its lifespan implementation.
    ``TestClient`` owns the portal and lifespan protocol and therefore tests the
    same startup/shutdown behaviour used by an ASGI server.
    """

    def __init__(
        self,
        app: FastAPI,
        *,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(app, headers=dict(default_headers or {}))


__all__ = ["SimpleTestClient"]
