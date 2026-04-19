from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[3] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

_stt_stream_protocol = importlib.import_module("stt_stream_protocol")
StreamSessionProtocol = _stt_stream_protocol.StreamSessionProtocol
StreamSessionState = _stt_stream_protocol.StreamSessionState


def test_stt_worker_accepts_stream_snapshot() -> None:
    protocol = StreamSessionProtocol()
    protocol.start(
        "s1",
        StreamSessionState(transcript="hello investor", start_time=time.perf_counter()),
    )

    response = protocol.snapshot("s1")
    assert response["ok"] is True
    assert response["text"] == "hello investor"


def test_stt_worker_commit_returns_final_text() -> None:
    protocol = StreamSessionProtocol()
    protocol.start(
        "s1",
        StreamSessionState(
            transcript="give me the numbers", start_time=time.perf_counter()
        ),
    )

    response = protocol.commit("s1")
    assert response["ok"] is True
    assert response["text"] == "give me the numbers"
