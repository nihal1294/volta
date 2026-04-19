from __future__ import annotations

from enum import StrEnum
from typing import Any


class TurnState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


def server_event(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"type": event_type, **data}


DEFAULT_METRICS = {
    "stt_first_partial_ms": None,
    "stt_first_word_ms": None,
    "stt_final_ms": None,
    "stt_words_per_sec": None,
    "llm_ttft_ms": None,
    "llm_tokens_per_sec": None,
    "tts_first_audio_ms": None,
    "tts_realtime_factor": None,
    "turn_total_ms": None,
}
