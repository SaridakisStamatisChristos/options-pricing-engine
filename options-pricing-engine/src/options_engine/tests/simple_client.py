"""Minimal ASGI test client used for integration tests without external deps."""

from __future__ import annotations

import asyncio
import json
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from fastapi import FastAPI


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
    def __init__(
        self,
        app: FastAPI,
        *,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        self._app = app
        self._default_headers = dict(default_headers or {})

    def __enter__(self) -> "SimpleTestClient":
        asyncio.run(self._startup())
        return self

    def __exit__(self, exc_type, exc, tb) -> Optional[bool]:
        asyncio.run(self._shutdown())
        return None

    async def _startup(self) -> None:
        await self._app.router.startup()

    async def _shutdown(self) -> None:
        await self._app.router.shutdown()

    def get(self, path: str, *, headers: Optional[Mapping[str, str]] = None) -> SimpleResponse:
        return asyncio.run(self._request("GET", path, headers=headers))

    def post(
        self,
        path: str,
        *,
        json: Any | None = None,
        data: Mapping[str, Any] | None = None,
        content: bytes | None = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> SimpleResponse:
        body: bytes
        send_headers = dict(headers or {})
        if json is not None:
            body = json_dump_bytes(json)
            send_headers.setdefault("content-type", "application/json")
        elif content is not None:
            body = content
        elif data is not None:
            body = json_dump_bytes(data)
            send_headers.setdefault("content-type", "application/json")
        else:
            body = b""
        return asyncio.run(self._request("POST", path, headers=send_headers, body=body))

    async def _request(
        self,
        method: str,
        path: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        body: bytes = b"",
    ) -> SimpleResponse:
        header_items = dict(self._default_headers)
        for key, value in (headers or {}).items():
            if value is None:
                header_items.pop(key, None)
            else:
                header_items[key] = value
        header_items.setdefault("host", "testserver")
        header_items.setdefault("content-length", str(len(body)))
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [(k.lower().encode(), v.encode()) for k, v in header_items.items()],
            "client": ("testclient", 0),
        }

        response_headers: list[tuple[bytes, bytes]] = []
        response_status = 500
        body_parts: list[bytes] = []
        body_sent = False

        async def receive() -> Mapping[str, Any]:
            nonlocal body_sent
            if body_sent:
                await asyncio.sleep(0)
                return {"type": "http.disconnect"}
            body_sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message: Mapping[str, Any]) -> None:
            nonlocal response_status, response_headers
            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                body_parts.append(message.get("body", b""))

        await self._app(scope, receive, send)
        headers_dict: Dict[str, str] = {}
        for raw_key, raw_value in response_headers:
            key = raw_key.decode()
            value = raw_value.decode()
            headers_dict[key] = value
            canonical = "-".join(part.capitalize() for part in key.split("-"))
            headers_dict.setdefault(canonical, value)
            if key.lower() == "x-request-id":
                headers_dict.setdefault("X-Request-ID", value)
        return SimpleResponse(status_code=response_status, headers=headers_dict, content=b"".join(body_parts))


def json_dump_bytes(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()


__all__ = ["SimpleTestClient", "SimpleResponse"]
