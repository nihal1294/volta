from __future__ import annotations

from volta_api.runtime_types import (
    AssistantDecision,
    SessionPhase,
    TranscriptSnapshot,
    UserTurnCommit,
)


def test_transcript_snapshot_defaults() -> None:
    snapshot = TranscriptSnapshot()

    assert snapshot.raw_text == ""
    assert snapshot.stable_text == ""
    assert snapshot.unstable_text == ""
    assert snapshot.revision == 0


def test_runtime_types_capture_expected_values() -> None:
    decision = AssistantDecision(name="speak")
    commit = UserTurnCommit(
        turn_id="turn-1",
        text="give me the numbers",
        started_at_ms=10.0,
        committed_at_ms=20.0,
    )

    assert SessionPhase.LISTENING == "listening"
    assert decision.mode == "stream"
    assert decision.interruptible is True
    assert commit.text == "give me the numbers"
