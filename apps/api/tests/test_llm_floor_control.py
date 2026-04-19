from __future__ import annotations

from volta_api.public_session import (
    should_start_tts_for_action,
    should_yield_to_llm_action,
)


def test_llm_yield_action_stops_playback() -> None:
    assert should_yield_to_llm_action({"name": "yield_to_user"}) is True


def test_llm_wait_action_keeps_playback_running() -> None:
    assert should_yield_to_llm_action({"name": "wait"}) is False


def test_continue_speaking_still_routes_to_tts() -> None:
    assert should_start_tts_for_action({"name": "continue_speaking"}) is True
