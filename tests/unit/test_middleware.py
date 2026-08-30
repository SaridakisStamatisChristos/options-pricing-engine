"""Low-level tests for HTTP framing middleware."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from options_engine.api.middleware import BodySizeLimitMiddleware


def _scope(headers: list[tuple[bytes, bytes]]) -> dict[str, Any]:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/test",
        "raw_path": b"/test",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "state": {},
    }


async def _invoke(
    *,
    headers: list[tuple[bytes, bytes]],
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[bytes]]:
    delivered: list[bytes] = []

    async def downstream(
        _scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        message = await receive()
        delivered.append(message.get("body", b""))
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    queue = iter(messages)

    async def receive() -> dict[str, Any]:
        return next(queue)

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    middleware = BodySizeLimitMiddleware(downstream, max_body_bytes=1024)
    await middleware(_scope(headers), receive, send)  # type: ignore[arg-type]
    return sent, delivered


def _response(sent: list[dict[str, Any]]) -> tuple[int, dict[str, str]]:
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return start["status"], json.loads(body) if body else {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "messages"),
    [
        (
            [(b"content-length", b"2"), (b"content-length", b"2")],
            [{"type": "http.request", "body": b"ab", "more_body": False}],
        ),
        (
            [(b"content-length", b"2"), (b"transfer-encoding", b"chunked")],
            [{"type": "http.request", "body": b"ab", "more_body": False}],
        ),
        (
            [(b"content-length", b"+2")],
            [{"type": "http.request", "body": b"ab", "more_body": False}],
        ),
        (
            [(b"content-length", b"3")],
            [{"type": "http.request", "body": b"ab", "more_body": False}],
        ),
        (
            [(b"content-length", b"1")],
            [{"type": "http.request", "body": b"ab", "more_body": False}],
        ),
        (
            [(b"transfer-encoding", b"gzip, chunked")],
            [{"type": "http.request", "body": b"ab", "more_body": False}],
        ),
    ],
)
async def test_ambiguous_or_mismatched_http_framing_is_rejected(
    headers: list[tuple[bytes, bytes]],
    messages: list[dict[str, Any]],
) -> None:
    sent, delivered = await _invoke(headers=headers, messages=messages)

    status, payload = _response(sent)
    assert status == 400
    assert payload == {"detail": "Invalid request framing"}
    assert delivered == []


@pytest.mark.asyncio
async def test_valid_chunked_body_is_bounded_and_replayed_once() -> None:
    sent, delivered = await _invoke(
        headers=[(b"transfer-encoding", b"chunked")],
        messages=[
            {"type": "http.request", "body": b"ab", "more_body": True},
            {"type": "http.request", "body": b"cd", "more_body": False},
        ],
    )

    status, payload = _response(sent)
    assert status == 204
    assert payload == {}
    assert delivered == [b"abcd"]
