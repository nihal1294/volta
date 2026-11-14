from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from volta_api.config import settings
from volta_api.session import Session


@pytest.mark.parametrize(
    "filename",
    [
        "../../outside.wav",
        "/tmp/outside.wav",
        r"C:\outside\clip.wav",
        "nested/path/clip.mp3",
        "clip.<img-onerror=x>.wav",
        "no-extension",
    ],
)
def test_file_turn_uses_generated_path_inside_upload_directory(tmp_path, filename):
    observed = []

    class WebSocketStub:
        async def send_text(self, text):
            pass

    class TranscriberStub:
        async def request_stream(self, action, payload, **kwargs):
            path = Path(payload["path"])
            assert action == "transcribe_file"
            assert path.parent == tmp_path
            assert path.name != filename
            assert path.read_bytes() == b"synthetic-audio"
            observed.append(path)
            yield {"event": "final", "data": {"text": ""}}

    session = Session(
        WebSocketStub(),
        replace(settings, uploads_root=tmp_path),
        {"stt": TranscriberStub()},
    )
    session.file_name = filename
    session.file_bytes = bytearray(b"synthetic-audio")
    asyncio.run(session._run_file_turn())
    assert len(observed) == 1
    assert not observed[0].exists()
