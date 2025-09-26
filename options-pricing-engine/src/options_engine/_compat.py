"""Compatibility helpers for third-party packages."""

from __future__ import annotations

import inspect
import typing

import httpx

try:  # pragma: no cover - optional dependency in runtime contexts
    from fastapi import testclient as fastapi_testclient
    from starlette import testclient as starlette_testclient
except Exception:  # pragma: no cover - nothing to patch if imports fail
    fastapi_testclient = None  # type: ignore[assignment]
    starlette_testclient = None  # type: ignore[assignment]
else:
    fastapi_testclient = typing.cast(typing.Any, fastapi_testclient)
    starlette_testclient = typing.cast(typing.Any, starlette_testclient)


def patch_starlette_testclient() -> None:
    """Ensure Starlette's TestClient works with the bundled httpx version."""

    if starlette_testclient is None:
        return

    if "app" in inspect.signature(httpx.Client.__init__).parameters:
        # httpx still exposes the legacy signature expected by Starlette.
        return

    def _patched_init(self, app, base_url: str = "http://testserver", raise_server_exceptions: bool = True, root_path: str = "", backend: typing.Literal["asyncio", "trio"] = "asyncio", backend_options: dict[str, typing.Any] | None = None, cookies: httpx._types.CookieTypes | None = None, headers: dict[str, str] | None = None, follow_redirects: bool = True) -> None:  # type: ignore[override]
        self.async_backend = starlette_testclient._AsyncBackend(  # type: ignore[attr-defined]
            backend=backend,
            backend_options=backend_options or {},
        )
        if starlette_testclient._is_asgi3(app):  # type: ignore[attr-defined]
            asgi_app = app
        else:
            app = typing.cast(starlette_testclient.ASGI2App, app)  # type: ignore[attr-defined]
            asgi_app = starlette_testclient._WrapASGI2(app)  # type: ignore[attr-defined]
        self.app = asgi_app
        self.app_state: dict[str, typing.Any] = {}
        transport = starlette_testclient._TestClientTransport(  # type: ignore[attr-defined]
            self.app,
            portal_factory=self._portal_factory,
            raise_server_exceptions=raise_server_exceptions,
            root_path=root_path,
            app_state=self.app_state,
        )
        if headers is None:
            headers = {}
        headers.setdefault("user-agent", "testclient")
        httpx.Client.__init__(  # pylint: disable=non-parent-init-called
            self,
            base_url=base_url,
            headers=headers,
            transport=transport,
            follow_redirects=follow_redirects,
            cookies=cookies,
        )

    starlette_testclient.TestClient.__init__ = _patched_init  # type: ignore[attr-defined]
    if fastapi_testclient is not None:
        fastapi_testclient.TestClient = starlette_testclient.TestClient


patch_starlette_testclient()

__all__ = ["patch_starlette_testclient"]
