from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from volta_api.config import settings
from volta_api.public_session import PublicRealtimeSession
from volta_api.runtime_types import UserTurnCommit


class FakeWebSocket:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def send_text(self, text: str) -> None:
        self.messages.append(json.loads(text))


class FakeWorker:
    def __init__(
        self,
        *,
        chunk_events: list[list[dict[str, Any]]] | None = None,
        snapshot_texts: list[str] | None = None,
        commit_text: str = "",
        stream_events_by_action: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.request_once_calls: list[tuple[str, dict[str, Any]]] = []
        self.request_stream_calls: list[tuple[str, dict[str, Any]]] = []
        self._chunk_events = iter(chunk_events or [])
        self._snapshot_texts = iter(snapshot_texts or [])
        self._commit_text = commit_text
        self._stream_events_by_action = stream_events_by_action or {}

    async def request_once(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.request_once_calls.append((action, payload))
        if action == "stream_snapshot":
            return {"ok": True, "text": next(self._snapshot_texts, "")}
        if action == "stream_commit":
            return {"ok": True, "text": self._commit_text}
        return {"ok": True}

    async def request_stream(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ):
        self.request_stream_calls.append((action, payload))
        if action == "stream_chunk":
            events = next(self._chunk_events, [])
        else:
            events = self._stream_events_by_action.get(action, [])
        for event in events:
            yield event


def _pcm16_b64(audio: np.ndarray) -> str:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    return base64.b64encode(pcm16.tobytes()).decode("ascii")


def _binary_pcm16_frame(
    audio: np.ndarray, *, sequence: int = 1, sample_rate: int = 24000
) -> bytes:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    frame = bytearray(16 + pcm16.nbytes)
    frame[:4] = b"RVL1"
    frame[4:8] = sequence.to_bytes(4, "little", signed=False)
    frame[8:12] = sample_rate.to_bytes(4, "little", signed=False)
    frame[12:14] = (1).to_bytes(2, "little", signed=False)
    frame[14:16] = (1).to_bytes(2, "little", signed=False)
    frame[16:] = pcm16.tobytes()
    return bytes(frame)


def test_public_session_streams_audio_without_rms_gate() -> None:
    async def scenario() -> None:
        websocket = FakeWebSocket()
        stt = FakeWorker(chunk_events=[[]], snapshot_texts=[""])
        session = PublicRealtimeSession(
            websocket,
            replace(
                settings, commit_stability_window_ms=400, max_open_utterance_ms=8000
            ),
            {"stt": stt, "llm": FakeWorker(), "tts": FakeWorker()},
        )

        await session.handle(
            {
                "type": "input_audio_buffer.append",
                "audio": _pcm16_b64(np.zeros(2400, dtype=np.float32)),
                "encoding": "pcm16",
                "sample_rate": 24000,
            }
        )

        assert [call[0] for call in stt.request_once_calls] == [
            "stream_start",
            "stream_snapshot",
        ]
        assert [call[0] for call in stt.request_stream_calls] == ["stream_chunk"]
        event_types = [message["type"] for message in websocket.messages]
        assert "input_audio_buffer.speech_started" not in event_types
        assert "input_audio_buffer.speech_stopped" not in event_types

    asyncio.run(scenario())


def test_public_session_uses_openai_specific_commit_windows() -> None:
    websocket = FakeWebSocket()
    session = PublicRealtimeSession(
        websocket,
        replace(
            settings,
            pipeline_provider="openai",
            openai_commit_stability_window_ms=2200,
            openai_max_open_utterance_ms=12000,
            commit_stability_window_ms=400,
            max_open_utterance_ms=8000,
        ),
        {"stt": FakeWorker(), "llm": FakeWorker(), "tts": FakeWorker()},
    )

    assert session.transcript_stabilizer.stability_window_ms == 2200
    assert session.transcript_stabilizer.max_open_ms == 12000


def test_public_session_emits_full_partial_text() -> None:
    async def scenario() -> None:
        websocket = FakeWebSocket()
        stt = FakeWorker(
            chunk_events=[
                [
                    {
                        "event": "partial",
                        "data": {
                            "text": "give me the numbers",
                            "delta": "give me the numbers",
                        },
                    }
                ]
            ],
            snapshot_texts=["give me the numbers"],
        )
        session = PublicRealtimeSession(
            websocket,
            replace(
                settings,
                pipeline_provider="local",
                commit_stability_window_ms=400,
                max_open_utterance_ms=8000,
            ),
            {"stt": stt, "llm": FakeWorker(), "tts": FakeWorker()},
        )
        session.history = [
            {"role": "user", "content": "previous turn"},
            {"role": "assistant", "content": "ignored"},
        ]

        await session.handle(
            {
                "type": "input_audio_buffer.append",
                "audio": _pcm16_b64(np.ones(2400, dtype=np.float32) * 0.05),
                "encoding": "pcm16",
                "sample_rate": 24000,
            }
        )

        partial_event = next(
            message
            for message in websocket.messages
            if message["type"] == "conversation.item.input_audio_transcription.delta"
        )
        assert partial_event["delta"] == "give me the numbers"
        assert partial_event["text"] == "give me the numbers"
        assert partial_event["full_text"] == "previous turn\n\ngive me the numbers"

    asyncio.run(scenario())


def test_public_session_accepts_binary_audio_frames() -> None:
    async def scenario() -> None:
        websocket = FakeWebSocket()
        stt = FakeWorker(chunk_events=[[]], snapshot_texts=[""])
        session = PublicRealtimeSession(
            websocket,
            replace(
                settings, commit_stability_window_ms=400, max_open_utterance_ms=8000
            ),
            {"stt": stt, "llm": FakeWorker(), "tts": FakeWorker()},
        )

        await session.handle_binary_audio(
            _binary_pcm16_frame(np.zeros(2400, dtype=np.float32))
        )

        assert [call[0] for call in stt.request_once_calls] == [
            "stream_start",
            "stream_snapshot",
        ]
        assert [call[0] for call in stt.request_stream_calls] == ["stream_chunk"]

    asyncio.run(scenario())


def test_public_session_commits_from_transcript_stability(tmp_path: Path) -> None:
    async def scenario() -> None:
        websocket = FakeWebSocket()
        stt = FakeWorker(
            chunk_events=[
                [
                    {
                        "event": "partial",
                        "data": {
                            "text": "give me the numbers",
                            "delta": "give me the numbers",
                        },
                    }
                ],
                [],
            ],
            snapshot_texts=["give me the numbers", "give me the numbers"],
            commit_text="give me the numbers",
        )
        session = PublicRealtimeSession(
            websocket,
            replace(
                settings,
                pipeline_provider="local",
                artifact_root=tmp_path,
                commit_stability_window_ms=400,
                max_open_utterance_ms=8000,
            ),
            {"stt": stt, "llm": FakeWorker(), "tts": FakeWorker()},
        )

        timestamps = iter([0.0, 0.0, 0.0, 700.0, 700.0, 700.0])
        session._now_ms = lambda: next(timestamps, 700.0)  # type: ignore[method-assign]
        committed_turns: list[str] = []

        async def fake_run_responses(text: str, *, committed: bool) -> None:
            assert committed is True
            committed_turns.append(text)

        session._run_responses = fake_run_responses  # type: ignore[method-assign]

        payload = {
            "type": "input_audio_buffer.append",
            "audio": _pcm16_b64(np.ones(2400, dtype=np.float32) * 0.05),
            "encoding": "pcm16",
            "sample_rate": 24000,
        }
        await session.handle(payload)
        await session.handle(payload)
        await asyncio.sleep(0)

        assert committed_turns == ["give me the numbers"]
        assert [call[0] for call in stt.request_stream_calls] == [
            "stream_chunk",
            "stream_chunk",
        ]
        assert [call[0] for call in stt.request_once_calls] == [
            "stream_start",
            "stream_snapshot",
            "stream_snapshot",
            "stream_commit",
        ]
        run_dir = tmp_path / session.session_id
        assert any((run_dir / "input").glob("*.wav"))
        assert (run_dir / "input" / "full_input.wav").is_file()
        assert (run_dir / "stt" / "commits.jsonl").is_file()
        assert (run_dir / "stt" / "final_transcript.txt").read_text(
            encoding="utf-8"
        ) == "give me the numbers"

    asyncio.run(scenario())


def test_session_update_accepts_persona_override(tmp_path: Path) -> None:
    async def scenario() -> None:
        websocket = FakeWebSocket()
        session = PublicRealtimeSession(
            websocket,
            replace(settings, artifact_root=tmp_path),
            {"stt": FakeWorker(), "llm": FakeWorker(), "tts": FakeWorker()},
        )

        await session.handle(
            {
                "type": "session.update",
                "session": {
                    "voice": "Aiden",
                    "tts_instruct": "Sharper delivery.",
                    "persona_text": "You are a ruthless televised investor.",
                    "output_audio_format": "pcm16",
                },
            }
        )

        assert session.voice == "Aiden"
        assert session.instruct == "Sharper delivery."
        assert session.persona_text == "You are a ruthless televised investor."
        assert session.output_audio_format == "pcm16"
        updated = websocket.messages[-1]
        assert updated["type"] == "session.updated"
        assert (
            updated["session"]["persona_text"]
            == "You are a ruthless televised investor."
        )
        assert (tmp_path / session.session_id / "llm" / "persona.txt").read_text(
            encoding="utf-8"
        ) == "You are a ruthless televised investor."

    asyncio.run(scenario())


def test_turn_mode_still_emits_tts_audio(tmp_path: Path) -> None:
    async def scenario() -> None:
        websocket = FakeWebSocket()
        llm = FakeWorker(
            stream_events_by_action={
                "generate": [
                    {
                        "event": "action",
                        "data": {
                            "name": "speak",
                            "mode": "turn",
                            "interruptible": True,
                        },
                    },
                    {"event": "token", "data": {"text": "That pitch is weak."}},
                    {"event": "metrics", "data": {"generation_tps": 42.0}},
                ]
            }
        )
        tts = FakeWorker(
            stream_events_by_action={
                "synthesize": [
                    {
                        "event": "audio",
                        "data": {
                            "pcm16_b64": _pcm16_b64(
                                np.ones(2400, dtype=np.float32) * 0.1
                            ),
                            "sample_rate": 24000,
                            "samples": 2400,
                            "segment_idx": 0,
                            "real_time_factor": 0.5,
                            "is_final_chunk": True,
                        },
                    },
                    {"event": "metrics", "data": {"real_time_factor": 0.5}},
                ]
            }
        )
        session = PublicRealtimeSession(
            websocket,
            replace(settings, artifact_root=tmp_path),
            {"stt": FakeWorker(), "llm": llm, "tts": tts},
        )
        session.output_audio_format = "pcm16"

        await session._run_single_response("Numbers are fuzzy.")

        event_types = [message["type"] for message in websocket.messages]
        assert "response.audio.delta" in event_types
        assert "response.audio.done" in event_types
        tts_calls = [
            call for call in tts.request_stream_calls if call[0] == "synthesize"
        ]
        assert len(tts_calls) == 1
        assert tts_calls[0][1]["text"] == "That pitch is weak."

    asyncio.run(scenario())


def test_commit_queues_next_turn_while_response_active(tmp_path: Path) -> None:
    async def scenario() -> None:
        websocket = FakeWebSocket()
        stt = FakeWorker(commit_text="wait, the margin is lower than that")
        session = PublicRealtimeSession(
            websocket,
            replace(settings, artifact_root=tmp_path),
            {"stt": stt, "llm": FakeWorker(), "tts": FakeWorker()},
        )
        session.capture_session_id = "capture-1"
        session.current_turn_audio.append(np.ones(2400, dtype=np.float32) * 0.05)
        session.response_task = asyncio.create_task(asyncio.sleep(60))
        session.response_interruptible = True
        interrupt_calls: list[str] = []

        async def fake_interrupt(reason: str) -> None:
            interrupt_calls.append(reason)

        session._interrupt_active_response = fake_interrupt  # type: ignore[method-assign]

        await session._commit_user_turn(
            UserTurnCommit(
                turn_id="turn-1",
                text="wait, the margin",
                started_at_ms=0.0,
                committed_at_ms=800.0,
            )
        )

        assert session.pending_transcript == "wait, the margin is lower than that"
        assert session.pending_transcript_committed is True
        assert interrupt_calls == []

        session.response_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await session.response_task

    asyncio.run(scenario())


def test_history_is_preserved_across_multiple_committed_turns(tmp_path: Path) -> None:
    async def scenario() -> None:
        websocket = FakeWebSocket()
        llm = FakeWorker(
            stream_events_by_action={
                "generate": [
                    {
                        "event": "action",
                        "data": {
                            "name": "speak",
                            "mode": "turn",
                            "interruptible": True,
                        },
                    },
                    {"event": "token", "data": {"text": "Let me stop you there."}},
                    {"event": "metrics", "data": {"generation_tps": 33.0}},
                ]
            }
        )
        tts = FakeWorker(stream_events_by_action={"synthesize": []})
        session = PublicRealtimeSession(
            websocket,
            replace(settings, artifact_root=tmp_path),
            {"stt": FakeWorker(), "llm": llm, "tts": tts},
        )

        await session._run_single_response("first user turn", committed=True)
        await session._run_single_response("second user turn", committed=True)

        assert session.history == [
            {"role": "user", "content": "first user turn"},
            {"role": "assistant", "content": "Let me stop you there."},
            {"role": "user", "content": "second user turn"},
            {"role": "assistant", "content": "Let me stop you there."},
        ]
        assert len(llm.request_stream_calls) == 2
        first_history = llm.request_stream_calls[0][1]["history"]
        second_history = llm.request_stream_calls[1][1]["history"]
        assert first_history == [{"role": "user", "content": "first user turn"}]
        assert second_history == [
            {"role": "user", "content": "first user turn"},
            {"role": "assistant", "content": "Let me stop you there."},
            {"role": "user", "content": "second user turn"},
        ]

    asyncio.run(scenario())
