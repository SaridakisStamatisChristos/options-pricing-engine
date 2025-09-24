"""Test configuration and environment bootstrapping."""

from __future__ import annotations

import os
from typing import Iterator

import pytest


os.environ.setdefault("OPE_JWT_SECRET", "test-secret")
os.environ.setdefault("OPE_ALLOWED_HOSTS", "testserver,localhost,127.0.0.1")
os.environ.setdefault("OPE_ALLOWED_ORIGINS", "http://testserver")
os.environ.setdefault("OPE_THREADS", "2")
os.environ.setdefault("OPE_THREAD_QUEUE_MAX", "4")
os.environ.setdefault("OPE_THREAD_QUEUE_TIMEOUT_SECONDS", "0.1")

from options_engine.api.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _reset_settings_cache() -> Iterator[None]:
    """Ensure each test session initialises configuration from test env vars."""

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
