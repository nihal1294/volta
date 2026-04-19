from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SessionPhase(StrEnum):
    CONNECTED = "connected"
    LISTENING = "listening"
    USER_UTTERANCE_OPEN = "user_utterance_open"
    ASSISTANT_PLANNING = "assistant_planning"
    ASSISTANT_GENERATING = "assistant_generating"
    ASSISTANT_SPEAKING = "assistant_speaking"
    OVERLAP_LISTENING = "overlap_listening"
    COMMITTING_USER_TURN = "committing_user_turn"
    RECOVERING = "recovering"


@dataclass(slots=True)
class TranscriptSnapshot:
    raw_text: str = ""
    stable_text: str = ""
    unstable_text: str = ""
    revision: int = 0


@dataclass(slots=True)
class UserTurnCommit:
    turn_id: str
    text: str
    started_at_ms: float
    committed_at_ms: float


@dataclass(slots=True)
class AssistantDecision:
    name: str
    mode: str = "stream"
    interruptible: bool = True
    reason: str | None = None
