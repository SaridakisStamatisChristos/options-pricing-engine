"""Compatibility helpers for third-party packages."""

from __future__ import annotations

import inspect
import sys
import typing
from calendar import timegm
from datetime import UTC, datetime

import httpx

try:  # pragma: no cover - optional dependency in runtime contexts
    import python_multipart
except ModuleNotFoundError:  # pragma: no cover - dependency may be missing in minimal installs
    python_multipart = None  # type: ignore[assignment]
else:  # pragma: no cover - executed when python-multipart is available
    # Starlette still imports the legacy ``multipart`` package name which now emits a
    # PendingDeprecationWarning.  Pre-register the modern module to keep imports quiet.
    sys.modules.setdefault("multipart", python_multipart)

try:  # pragma: no cover - optional dependency in runtime contexts
    from jose import jwt as jose_jwt
    from jose.exceptions import ExpiredSignatureError, JWTClaimsError
except Exception:  # pragma: no cover - skip JWT patches when python-jose is unavailable
    jose_jwt = None  # type: ignore[assignment]
else:
    jose_jwt = typing.cast(typing.Any, jose_jwt)

try:  # pragma: no cover - optional dependency in runtime contexts
    from fastapi import testclient as fastapi_testclient
    from starlette import testclient as starlette_testclient
except Exception:  # pragma: no cover - nothing to patch if imports fail
    fastapi_testclient = None  # type: ignore[assignment]
    starlette_testclient = None  # type: ignore[assignment]
else:
    fastapi_testclient = typing.cast(typing.Any, fastapi_testclient)
    starlette_testclient = typing.cast(typing.Any, starlette_testclient)


def _patch_jose_time_validation() -> None:
    """Replace python-jose datetime helpers with timezone-aware implementations."""

    if jose_jwt is None:  # pragma: no cover - environment without python-jose
        return

    def _aware_now_seconds() -> int:
        """Return the current UTC timestamp using timezone-aware datetimes."""

        return timegm(datetime.now(UTC).utctimetuple())

    def _patched_validate_nbf(claims, leeway=0):  # type: ignore[override]
        if "nbf" not in claims:
            return
        try:
            nbf = int(claims["nbf"])
        except ValueError:
            raise JWTClaimsError("Not Before claim (nbf) must be an integer.")
        now = _aware_now_seconds()
        if nbf > (now + leeway):
            raise JWTClaimsError("The token is not yet valid (nbf)")

    def _patched_validate_exp(claims, leeway=0):  # type: ignore[override]
        if "exp" not in claims:
            return
        try:
            exp = int(claims["exp"])
        except ValueError:
            raise JWTClaimsError("Expiration Time claim (exp) must be an integer.")
        now = _aware_now_seconds()
        if exp < (now - leeway):
            raise ExpiredSignatureError("Signature has expired.")

    jose_jwt._validate_nbf = _patched_validate_nbf  # type: ignore[attr-defined]
    jose_jwt._validate_exp = _patched_validate_exp  # type: ignore[attr-defined]


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
_patch_jose_time_validation()

__all__ = ["patch_starlette_testclient"]
