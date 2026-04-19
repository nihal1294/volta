from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class StreamSessionState:
    transcript: str
    start_time: float
    first_partial_ms: float | None = None
    first_word_ms: float | None = None


class StreamSessionProtocol:
    def __init__(self) -> None:
        self.sessions: dict[str, StreamSessionState] = {}

    def start(self, stream_id: str, state: StreamSessionState) -> dict[str, object]:
        self.sessions[stream_id] = state
        return {"ok": True}

    def snapshot(self, stream_id: str, now: float | None = None) -> dict[str, object]:
        session = self.sessions.get(stream_id)
        if session is None:
            return {"ok": False, "ignored": True, "text": ""}
        elapsed = max(
            (now if now is not None else time.perf_counter()) - session.start_time, 0.0
        )
        return {
            "ok": True,
            "text": session.transcript.strip(),
            "first_partial_ms": session.first_partial_ms,
            "first_word_ms": session.first_word_ms,
            "processing_ms": round(elapsed * 1000, 1),
        }

    def commit(self, stream_id: str, now: float | None = None) -> dict[str, object]:
        session = self.sessions.pop(stream_id, None)
        if session is None:
            return {"ok": False, "ignored": True, "text": ""}
        elapsed = max(
            (now if now is not None else time.perf_counter()) - session.start_time, 0.0
        )
        words = len(session.transcript.split())
        return {
            "ok": True,
            "text": session.transcript.strip(),
            "first_partial_ms": session.first_partial_ms,
            "first_word_ms": session.first_word_ms,
            "processing_ms": round(elapsed * 1000, 1),
            "words_per_sec": round(words / elapsed, 3) if elapsed > 0 else 0.0,
        }

    def reset(self, stream_id: str) -> dict[str, object]:
        self.sessions.pop(stream_id, None)
        return {"ok": True}
