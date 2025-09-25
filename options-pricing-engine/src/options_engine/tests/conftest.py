"""Test configuration and environment bootstrapping."""

from __future__ import annotations

import base64
import os
from typing import Iterator

import pytest

from options_engine.api.config import get_settings
from options_engine.api.fastapi_app import create_app
from options_engine.api.security import _get_authenticator
from options_engine.tests.simple_client import SimpleTestClient
from options_engine.security import oidc


FAKE_KID = "test-key"
FAKE_SECRET = "dev-secret-value-at-least-32-bytes!!!"
FAKE_JWKS = {
    "keys": [
        {
            "kty": "oct",
            "kid": FAKE_KID,
            "k": base64.urlsafe_b64encode(FAKE_SECRET.encode()).rstrip(b"=").decode(),
            "alg": "HS256",
        }
    ]
}


os.environ.setdefault("OIDC_ISSUER", "https://issuer.test")
os.environ.setdefault("OIDC_AUDIENCE", "options-pricing-engine")
os.environ.setdefault("DEV_JWT_SECRET", FAKE_SECRET)
os.environ.setdefault("OPE_ALLOWED_HOSTS", "testserver,localhost,127.0.0.1")
os.environ.setdefault("OPE_ALLOWED_ORIGINS", "http://testserver")
os.environ.setdefault("OPE_THREADS", "2")
os.environ.setdefault("OPE_THREAD_QUEUE_MAX", "4")
os.environ.setdefault("OPE_THREAD_QUEUE_TIMEOUT_SECONDS", "0.1")


@pytest.fixture(autouse=True, scope="session")
def _configure_security() -> Iterator[None]:
    """Provide a deterministic JWKS for tests and reset configuration caches."""

    monkeypatcher = pytest.MonkeyPatch()
    monkeypatcher.setattr(oidc, "_fetch_jwks", lambda _: FAKE_JWKS)
    monkeypatcher.setenv("DEV_JWT_SECRET", FAKE_SECRET)
    monkeypatcher.setenv("OIDC_JWKS_URL", "")
    get_settings.cache_clear()
    _get_authenticator.cache_clear()
    yield
    monkeypatcher.undo()
    get_settings.cache_clear()
    _get_authenticator.cache_clear()


@pytest.fixture(scope="session")
def client() -> Iterator[SimpleTestClient]:
    """Return an ASGI test client backed by a fresh FastAPI application."""

    app = create_app()
    with SimpleTestClient(app) as test_client:
        yield test_client
