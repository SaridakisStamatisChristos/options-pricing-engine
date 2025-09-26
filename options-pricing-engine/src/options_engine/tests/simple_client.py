"""Minimal ASGI test client used for integration tests without external deps."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx


@dataclass(slots=True)
class SimpleResponse:
    status_code: int
    headers: Dict[str, str]
    content: bytes

    def json(self) -> Any:
        return json.loads(self.content.decode())

    @property
    def text(self) -> str:
        return self.content.decode()


class SimpleTestClient(AbstractContextManager["SimpleTestClient"]):
    def __init__(self, app: FastAPI) -> None:
        self._app = app
        self._client: TestClient | None = None

    def __enter__(self) -> "SimpleTestClient":
        self._client = TestClient(self._app)
        self._client.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        assert self._client is not None
        self._client.__exit__(exc_type, exc, tb)
        self._client = None
        return None

    def get(self, path: str, *, headers: Optional[Mapping[str, str]] = None) -> SimpleResponse:
        assert self._client is not None
        response = self._client.get(path, headers=headers)
        return _to_simple_response(response)

    def post(
        self,
        path: str,
        *,
        json: Any | None = None,
        data: Mapping[str, Any] | None = None,
        content: bytes | None = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> SimpleResponse:
        assert self._client is not None
        if json is not None:
            response = self._client.post(path, json=json, headers=headers)
        elif content is not None:
            response = self._client.post(path, content=content, headers=headers)
        elif data is not None:
            response = self._client.post(path, json=data, headers=headers)
        else:
            response = self._client.post(path, headers=headers)
        return _to_simple_response(response)


def _to_simple_response(response: httpx.Response) -> SimpleResponse:
    raw_headers = dict(response.headers)
    headers: Dict[str, str] = {}
    for key, value in raw_headers.items():
        headers[key] = value
        canonical = "-".join(part.capitalize() for part in key.split("-"))
        headers.setdefault(canonical, value)
        if key.lower() == "x-request-id":
            headers.setdefault("X-Request-ID", value)
    return SimpleResponse(status_code=response.status_code, headers=headers, content=response.content)


__all__ = ["SimpleTestClient", "SimpleResponse"]
