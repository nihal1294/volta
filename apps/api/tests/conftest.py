from __future__ import annotations

import os
import socket
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Collection must not pick up provider credentials from a developer's .env.
os.environ["PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "unused-test-key"
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:9/v1"

from volta_api import config  # noqa: E402

_runtime = TemporaryDirectory(prefix="volta-tests-")
_runtime_root = Path(_runtime.name)
for name in ("uploads", "runs", "fixtures"):
    (_runtime_root / name).mkdir()
config.settings = replace(
    config.Settings.from_env(
        env={
            "PROVIDER": "openai",
            "OPENAI_API_KEY": "unused-test-key",
            "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
        },
        dotenv_path=_runtime_root / "absent.env",
    ),
    uploads_root=_runtime_root / "uploads",
    artifact_root=_runtime_root / "runs",
    fixture_root=_runtime_root / "fixtures",
)


@pytest.fixture(autouse=True)
def block_external_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args, **kwargs):
        raise AssertionError("Tests must use fake workers or HTTP mock transports")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture(scope="session", autouse=True)
def temporary_runtime():
    yield
    _runtime.cleanup()
