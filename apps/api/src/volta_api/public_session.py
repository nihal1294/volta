from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
import uuid
from typing import Any

import numpy as np
from fastapi import WebSocket

from .artifacts import ArtifactStore
from .audio_streams import (
    OpusInputStream,
    OpusOutputStream,
    SAMPLE_RATE,
    float32_to_pcm16_b64,
    pcm16_b64_to_float32_array,
    resample_linear,
)
from .config import Settings
from .runtime_types import SessionPhase, TranscriptSnapshot, UserTurnCommit
from .transcript_stabilizer import TranscriptStabilizer
from .worker_client import WorkerLike

FRAME_MAGIC = b"RVL1"
FRAME_HEADER_BYTES = 16


def merge_transcript(current: str, incoming: str) -> str:
    incoming = incoming.strip()
    if not incoming:
        return current
    if not current:
        return incoming
    return f"{current.rstrip()} {incoming}"


def _action_name(action: dict[str, Any]) -> str:
    return str(action.get("name", "")).strip()


def should_yield_to_llm_action(action: dict[str, Any]) -> bool:
    return _action_name(action) == "yield_to_user"


def should_start_tts_for_action(action: dict[str, Any]) -> bool:
    return _action_name(action) in {"speak", "continue_speaking"}


class AudioAccumulator:
    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._chunks: list[np.ndarray] = []

    def append(self, audio: np.ndarray) -> None:
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        if chunk.size:
            self._chunks.append(chunk)

    def clear(self) -> None:
        self._chunks.clear()

    def to_array(self) -> np.ndarray:
        if not self._chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self._chunks).astype(np.float32)


class PublicRealtimeSession:
    def __init__(
        self,
        websocket: WebSocket,
        settings: Settings,
        workers: dict[str, WorkerLike],
    ) -> None:
        self.websocket = websocket
        self.settings = settings
        self.workers = workers
        self.session_id = str(uuid.uuid4())
        self.voice = settings.default_tts_voice
        self.instruct = settings.tts_instruct
        self.persona_text = settings.llm_system_prompt_file.read_text(
            encoding="utf-8"
        ).strip()
        self.output_audio_format = "opus"
        self.history: list[dict[str, str]] = []
        self.send_lock = asyncio.Lock()
        self.closed = False
        self.artifacts = ArtifactStore(settings.artifact_root)
        self.run_handle = self.artifacts.open_run(self.session_id)

        self.audio_in = OpusInputStream(SAMPLE_RATE)
        self.phase = SessionPhase.CONNECTED
        self.capture_counter = 0
        self.capture_session_id: str | None = None
        self.current_transcript = ""
        self.current_turn_audio = AudioAccumulator(sample_rate=SAMPLE_RATE)
        self.full_input_audio = AudioAccumulator(sample_rate=SAMPLE_RATE)
        stability_window_ms = (
            settings.openai_commit_stability_window_ms
            if settings.use_openai_provider
            else settings.commit_stability_window_ms
        )
        max_open_ms = (
            settings.openai_max_open_utterance_ms
            if settings.use_openai_provider
            else settings.max_open_utterance_ms
        )
        self.transcript_stabilizer = TranscriptStabilizer(
            stability_window_ms=stability_window_ms,
            max_open_ms=max_open_ms,
        )

        self.response_task: asyncio.Task[None] | None = None
        self.response_id: str | None = None
        self.response_tts_task: asyncio.Task[None] | None = None
        self.response_tts_queue: asyncio.Queue[str | None] | None = None
        self.response_tts_enabled = False
        self.response_audio_done_sent = False
        self.response_interruptible = True
        self.pending_transcript = ""
        self.pending_transcript_committed = False
        self.current_response_interrupted = False
        self.current_response_input_text = ""
        self.current_response_audio: list[np.ndarray] = []
        self.current_tts_chunk_index = 0
        self.current_turn_metrics: dict[str, Any] = {
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
        self.response_started_at_ms: float | None = None

        self.artifacts.append_event(
            self.run_handle,
            "session.started",
            {"session_id": self.session_id},
        )

    async def handle(self, message: dict[str, Any]) -> None:
        message_type = str(message.get("type", ""))
        if message_type == "session.update":
            session = message.get("session", {})
            if isinstance(session, dict):
                requested_voice = session.get("voice")
                if isinstance(requested_voice, str) and requested_voice.strip():
                    self.voice = requested_voice.strip()
                requested_instruct = session.get("tts_instruct")
                if isinstance(requested_instruct, str) and requested_instruct.strip():
                    self.instruct = requested_instruct.strip()
                requested_persona = session.get("persona_text")
                if isinstance(requested_persona, str) and requested_persona.strip():
                    self.persona_text = requested_persona.strip()
                requested_audio_format = session.get("output_audio_format")
                if requested_audio_format in {"opus", "pcm16"}:
                    self.output_audio_format = str(requested_audio_format)
            if self.settings.save_llm_outputs and self.persona_text:
                self.artifacts.write_text(
                    self.run_handle, "llm/persona.txt", self.persona_text
                )
            self.phase = SessionPhase.LISTENING
            await self._emit(
                {
                    "type": "session.updated",
                    "session": {
                        "id": self.session_id,
                        "voice": self.voice,
                        "tts_instruct": self.instruct,
                        "persona_text": self.persona_text,
                        "output_audio_format": self.output_audio_format,
                    },
                }
            )
            return
        if message_type == "input_audio_buffer.append":
            audio_b64 = message.get("audio")
            if isinstance(audio_b64, str):
                await self._handle_audio_append(
                    audio_b64,
                    encoding=str(message.get("encoding", "opus")),
                    sample_rate=int(message.get("sample_rate", SAMPLE_RATE)),
                )
            return
        if message_type == "input_audio_buffer.commit":
            await self._force_commit_current_turn()
            return

    async def handle_binary_audio(self, frame: bytes) -> None:
        decoded = self._decode_binary_audio_frame(frame)
        if decoded.size == 0:
            return
        await self._handle_decoded_audio(decoded)

    async def close(self) -> None:
        self.closed = True
        if self.response_task and not self.response_task.done():
            await self._interrupt_active_response("session_close")
            try:
                await asyncio.wait_for(asyncio.shield(self.response_task), timeout=10)
            except asyncio.TimeoutError, asyncio.CancelledError:
                self.response_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.response_task
        if self.response_tts_task and not self.response_tts_task.done():
            self.response_tts_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.response_tts_task
        if self.capture_session_id:
            with contextlib.suppress(Exception):
                await self.workers["stt"].request_once(
                    "stream_reset",
                    {"stream_id": self.capture_session_id},
                    timeout=5,
                )
        with contextlib.suppress(Exception):
            await self.workers["llm"].request_once(
                "session_reset",
                {"session_id": self.session_id},
                timeout=5,
            )
        self._persist_full_input_audio()

    async def _handle_audio_append(
        self,
        audio_b64: str,
        encoding: str,
        sample_rate: int,
    ) -> None:
        decoded = self._decode_audio(
            audio_b64, encoding=encoding, sample_rate=sample_rate
        )
        if decoded.size == 0:
            return
        await self._handle_decoded_audio(decoded)

    async def _handle_decoded_audio(self, decoded: np.ndarray) -> None:
        await self._ensure_capture_session()
        self.current_turn_audio.append(decoded)
        self.full_input_audio.append(decoded)
        snapshot = await self._push_audio_to_stt(decoded)
        self._record_snapshot(snapshot)

        commit = self.transcript_stabilizer.maybe_commit(now_ms=self._now_ms())
        if commit is not None:
            await self._commit_user_turn(commit)

    def _decode_audio(
        self,
        audio_b64: str,
        *,
        encoding: str,
        sample_rate: int,
    ) -> np.ndarray:
        if encoding == "pcm16":
            decoded = pcm16_b64_to_float32_array(audio_b64)
            return resample_linear(decoded, sample_rate, SAMPLE_RATE)
        return self.audio_in.append_b64(audio_b64)

    def _decode_binary_audio_frame(self, frame: bytes) -> np.ndarray:
        if len(frame) < FRAME_HEADER_BYTES or frame[:4] != FRAME_MAGIC:
            raise ValueError("invalid audio frame header")
        sample_rate = int.from_bytes(frame[8:12], "little", signed=False)
        encoding = int.from_bytes(frame[14:16], "little", signed=False)
        if encoding != 1:
            raise ValueError(f"unsupported audio frame encoding: {encoding}")
        pcm = (
            np.frombuffer(frame[FRAME_HEADER_BYTES:], dtype="<i2").astype(np.float32)
            / 32768.0
        )
        return resample_linear(pcm, sample_rate, SAMPLE_RATE)

    async def _ensure_capture_session(self) -> None:
        if self.capture_session_id is not None:
            return
        self.capture_counter += 1
        self.capture_session_id = f"{self.session_id}-stt-{self.capture_counter}"
        self.phase = SessionPhase.USER_UTTERANCE_OPEN
        await self.workers["stt"].request_once(
            "stream_start",
            {"stream_id": self.capture_session_id},
            timeout=10,
        )

    async def _push_audio_to_stt(self, audio: np.ndarray) -> TranscriptSnapshot:
        if not self.capture_session_id:
            return self.transcript_stabilizer.snapshot

        pcm = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype("<i2")
        encoded = base64.b64encode(pcm16.tobytes()).decode("ascii")
        snapshot = self.transcript_stabilizer.snapshot
        async for event in self.workers["stt"].request_stream(
            "stream_chunk",
            {
                "stream_id": self.capture_session_id,
                "pcm_b64": encoded,
                "sample_rate": SAMPLE_RATE,
            },
            timeout=30,
        ):
            if event["event"] != "partial":
                continue
            payload = event["data"]
            delta = str(payload.get("delta", ""))
            text = str(payload.get("text", "")).strip()
            if not text:
                text = merge_transcript(self.current_transcript, delta)
            if text:
                self.current_transcript = text
                snapshot = self.transcript_stabilizer.observe(
                    self.current_transcript,
                    now_ms=self._now_ms(),
                )
            if delta:
                if payload.get("first_partial_ms") is not None:
                    self.current_turn_metrics["stt_first_partial_ms"] = payload.get(
                        "first_partial_ms"
                    )
                if payload.get("first_word_ms") is not None:
                    self.current_turn_metrics["stt_first_word_ms"] = payload.get(
                        "first_word_ms"
                    )
                await self._emit(
                    {
                        "type": "conversation.item.input_audio_transcription.delta",
                        "delta": delta,
                        "text": self.current_transcript,
                        "full_text": self._user_transcript_view(
                            self.current_transcript
                        ),
                    }
                )
        snapshot_data = await self.workers["stt"].request_once(
            "stream_snapshot",
            {"stream_id": self.capture_session_id},
            timeout=30,
        )
        snapshot_text = str(snapshot_data.get("text", "")).strip()
        if snapshot_text:
            self.current_transcript = snapshot_text
            snapshot = self.transcript_stabilizer.observe(
                self.current_transcript,
                now_ms=self._now_ms(),
            )
        return snapshot

    def _record_snapshot(self, snapshot: TranscriptSnapshot) -> None:
        if snapshot.raw_text:
            if self.response_task and not self.response_task.done():
                self.phase = SessionPhase.OVERLAP_LISTENING
            else:
                self.phase = SessionPhase.USER_UTTERANCE_OPEN
            if self.settings.save_stt_outputs:
                self.artifacts.append_jsonl(
                    self.run_handle,
                    "stt/partials.jsonl",
                    {
                        "raw_text": snapshot.raw_text,
                        "stable_text": snapshot.stable_text,
                        "unstable_text": snapshot.unstable_text,
                        "revision": snapshot.revision,
                    },
                )

    def _user_transcript_view(self, current_turn_text: str = "") -> str:
        turns = [
            str(message.get("content", "")).strip()
            for message in self.history
            if str(message.get("role", "")).strip() == "user"
            and str(message.get("content", "")).strip()
        ]
        current = current_turn_text.strip()
        if current:
            turns.append(current)
        return "\n\n".join(turns)

    async def _commit_user_turn(self, commit: UserTurnCommit) -> None:
        if not self.capture_session_id:
            return

        capture_session_id = self.capture_session_id
        self.capture_session_id = None
        self.phase = SessionPhase.COMMITTING_USER_TURN
        transcript = commit.text.strip()

        final_data = await self.workers["stt"].request_once(
            "stream_commit",
            {"stream_id": capture_session_id},
            timeout=90,
        )
        text = str(final_data.get("text", "")).strip()
        if text:
            transcript = text
        self.current_turn_metrics["stt_first_partial_ms"] = final_data.get(
            "first_partial_ms", self.current_turn_metrics["stt_first_partial_ms"]
        )
        self.current_turn_metrics["stt_first_word_ms"] = final_data.get(
            "first_word_ms", self.current_turn_metrics["stt_first_word_ms"]
        )
        self.current_turn_metrics["stt_final_ms"] = final_data.get("processing_ms")
        self.current_turn_metrics["stt_words_per_sec"] = final_data.get("words_per_sec")

        turn_audio = self.current_turn_audio.to_array()
        self.current_transcript = ""
        self.current_turn_audio.clear()
        self.transcript_stabilizer.reset()
        self.phase = SessionPhase.LISTENING

        if not transcript:
            return
        if self.settings.save_full_input_audio:
            self.artifacts.write_wav(
                self.run_handle,
                f"input/{commit.turn_id}.wav",
                turn_audio,
                SAMPLE_RATE,
            )
            self._persist_full_input_audio()
        if self.settings.save_stt_outputs:
            self.artifacts.append_jsonl(
                self.run_handle,
                "stt/commits.jsonl",
                {
                    "turn_id": commit.turn_id,
                    "text": transcript,
                    "started_at_ms": commit.started_at_ms,
                    "committed_at_ms": commit.committed_at_ms,
                },
            )
            self.artifacts.write_text(
                self.run_handle,
                "stt/final_transcript.txt",
                transcript,
            )
        if self.response_task and not self.response_task.done():
            self.pending_transcript = merge_transcript(
                self.pending_transcript, transcript
            )
            self.pending_transcript_committed = True
            return
        self.response_task = asyncio.create_task(
            self._run_responses(transcript, committed=True)
        )

    async def _force_commit_current_turn(self) -> None:
        text = (
            self.transcript_stabilizer.snapshot.raw_text or self.current_transcript
        ).strip()
        if not self.capture_session_id or not text:
            return
        commit = UserTurnCommit(
            turn_id=str(uuid.uuid4()),
            text=text,
            started_at_ms=self.transcript_stabilizer.utterance_opened_ms,
            committed_at_ms=self._now_ms(),
        )
        await self._commit_user_turn(commit)

    async def _run_responses(self, transcript: str, *, committed: bool) -> None:
        next_transcript = transcript
        next_transcript_committed = committed
        while next_transcript and not self.closed:
            await self._run_single_response(
                next_transcript,
                committed=next_transcript_committed,
            )
            next_transcript = self.pending_transcript.strip()
            next_transcript_committed = self.pending_transcript_committed
            self.pending_transcript = ""
            self.pending_transcript_committed = False
        self.response_task = None

    async def _run_single_response(
        self, transcript: str, *, committed: bool = True
    ) -> None:
        self.phase = SessionPhase.ASSISTANT_PLANNING
        llm_history = list(self.history)
        if committed:
            self.history.append({"role": "user", "content": transcript})
            llm_history = list(self.history)
        else:
            llm_history.append({"role": "user", "content": transcript})
        response_id = str(uuid.uuid4())
        self.response_id = response_id
        self.response_tts_queue = asyncio.Queue()
        self.response_tts_enabled = False
        self.response_audio_done_sent = False
        self.response_interruptible = True
        self.current_response_interrupted = False
        self.current_response_input_text = transcript
        self.current_response_audio = []
        self.current_tts_chunk_index = 0
        self.response_started_at_ms = self._now_ms()
        self.current_turn_metrics["llm_ttft_ms"] = None
        self.current_turn_metrics["llm_tokens_per_sec"] = None
        self.current_turn_metrics["tts_first_audio_ms"] = None
        self.current_turn_metrics["tts_realtime_factor"] = None
        self.current_turn_metrics["turn_total_ms"] = None

        await self._emit({"type": "response.created", "response": {"id": response_id}})

        action_name = "speak"
        action_mode = "stream"
        llm_text = ""
        tts_buffer = ""
        current_action: dict[str, Any] = {
            "name": action_name,
            "mode": action_mode,
            "interruptible": True,
        }

        try:
            async for event in self.workers["llm"].request_stream(
                "generate",
                {
                    "session_id": self.session_id,
                    "history": llm_history,
                    "persona_text": self.persona_text,
                },
                timeout=300,
            ):
                if event["event"] == "action":
                    payload = event["data"]
                    action_name = str(payload.get("name", "speak"))
                    action_mode = str(payload.get("mode", "stream"))
                    current_action = dict(payload)
                    self.response_interruptible = bool(
                        payload.get("interruptible", True)
                    )
                    await self._emit(
                        {
                            "type": "response.action",
                            "response_id": response_id,
                            "action": payload,
                        }
                    )
                    if self.settings.save_llm_outputs:
                        self.artifacts.append_jsonl(
                            self.run_handle,
                            "llm/floor_actions.jsonl",
                            {
                                "response_id": response_id,
                                "phase": self.phase,
                                **current_action,
                            },
                        )
                    if (
                        should_start_tts_for_action(current_action)
                        and not self.response_tts_enabled
                    ):
                        self.phase = SessionPhase.ASSISTANT_GENERATING
                        self.response_tts_enabled = True
                        self.response_tts_task = asyncio.create_task(
                            self._tts_consumer(response_id)
                        )
                    continue

                if event["event"] == "metrics":
                    payload = event["data"]
                    if payload.get("generation_tps") is not None:
                        self.current_turn_metrics["llm_tokens_per_sec"] = payload.get(
                            "generation_tps"
                        )
                    continue

                if event["event"] != "token":
                    continue

                text = str(event["data"].get("text", ""))
                if not text:
                    continue
                if not llm_text:
                    text = text.lstrip()
                    if not text:
                        continue
                if (
                    self.current_turn_metrics["llm_ttft_ms"] is None
                    and self.response_started_at_ms is not None
                ):
                    self.current_turn_metrics["llm_ttft_ms"] = round(
                        self._now_ms() - self.response_started_at_ms,
                        1,
                    )
                llm_text += text
                await self._emit(
                    {
                        "type": "response.text.delta",
                        "response_id": response_id,
                        "delta": text,
                    }
                )
                if self.settings.save_llm_outputs:
                    self.artifacts.append_jsonl(
                        self.run_handle,
                        "llm/responder_tokens.jsonl",
                        {"response_id": response_id, "text": text},
                    )

                if (
                    should_start_tts_for_action(current_action)
                    and self.response_tts_enabled
                    and self.response_tts_queue is not None
                ):
                    tts_buffer += text
                    if action_mode == "stream":
                        ready_chunks, tts_buffer = self._drain_tts_chunks(
                            tts_buffer, final=False
                        )
                        for chunk in ready_chunks:
                            await self.response_tts_queue.put(chunk)

            if (
                should_start_tts_for_action(current_action)
                and self.response_tts_enabled
                and self.response_tts_queue is not None
            ):
                ready_chunks, tts_buffer = self._drain_tts_chunks(
                    tts_buffer, final=True
                )
                for chunk in ready_chunks:
                    await self.response_tts_queue.put(chunk)
                await self.response_tts_queue.put(None)
                if self.response_tts_task is not None:
                    with contextlib.suppress(asyncio.CancelledError):
                        await self.response_tts_task
            else:
                await self._emit_audio_done(response_id)
        finally:
            assistant_text = llm_text.strip()
            if assistant_text and committed and not self.current_response_interrupted:
                self.history.append({"role": "assistant", "content": assistant_text})
                if self.settings.save_llm_outputs:
                    self.artifacts.append_jsonl(
                        self.run_handle,
                        "llm/final_turns.jsonl",
                        {"response_id": response_id, "text": assistant_text},
                    )
                    self.artifacts.write_text(
                        self.run_handle,
                        "llm/final_output.txt",
                        assistant_text,
                    )
            elif assistant_text and self.settings.save_llm_outputs:
                self.artifacts.append_jsonl(
                    self.run_handle,
                    "llm/interrupted_turns.jsonl",
                    {
                        "response_id": response_id,
                        "text": assistant_text,
                        "committed": committed,
                        "interrupted": self.current_response_interrupted,
                    },
                )
            if self.settings.save_tts_outputs and self.current_response_audio:
                self.artifacts.write_wav(
                    self.run_handle,
                    "tts/final_output.wav",
                    np.concatenate(self.current_response_audio).astype(np.float32),
                    SAMPLE_RATE,
                )
            self.response_id = None
            self.response_tts_queue = None
            self.response_tts_task = None
            self.response_tts_enabled = False
            self.current_response_interrupted = False
            self.current_response_input_text = ""
            self.phase = SessionPhase.LISTENING

    async def _interrupt_active_response(self, reason: str) -> None:
        if not self.response_task or self.response_task.done():
            return
        self.current_response_interrupted = True
        self.response_tts_enabled = False
        if self.response_tts_task and not self.response_tts_task.done():
            self.response_tts_task.cancel()
        with contextlib.suppress(Exception):
            await self.workers["llm"].request_once(
                "cancel_generate",
                {
                    "session_id": self.session_id,
                    "response_id": self.response_id,
                    "reason": reason,
                },
                timeout=5,
            )

    async def _tts_consumer(self, response_id: str) -> None:
        assert self.response_tts_queue is not None
        audio_out = OpusOutputStream(SAMPLE_RATE)
        self.phase = SessionPhase.ASSISTANT_SPEAKING
        try:
            while True:
                text = await self.response_tts_queue.get()
                if text is None or not self.response_tts_enabled:
                    break
                async for event in self.workers["tts"].request_stream(
                    "synthesize",
                    {
                        "text": text,
                        "voice": self.voice,
                        "instruct": self.instruct,
                        "stream": True,
                        "streaming_interval": self.settings.tts_streaming_interval,
                    },
                    timeout=300,
                ):
                    if not self.response_tts_enabled:
                        break
                    if event["event"] == "metrics":
                        payload = event["data"]
                        if payload.get("real_time_factor") is not None:
                            self.current_turn_metrics["tts_realtime_factor"] = (
                                payload.get("real_time_factor")
                            )
                        continue
                    if event["event"] != "audio":
                        continue
                    payload = event["data"]
                    pcm = pcm16_b64_to_float32_array(str(payload["pcm16_b64"]))
                    if self.settings.save_tts_outputs and pcm.size:
                        self.current_tts_chunk_index += 1
                        self.current_response_audio.append(pcm)
                        self.artifacts.write_wav(
                            self.run_handle,
                            f"tts/chunks/{response_id}-{self.current_tts_chunk_index:06d}.wav",
                            pcm,
                            SAMPLE_RATE,
                        )
                    if (
                        self.current_turn_metrics["tts_first_audio_ms"] is None
                        and self.response_started_at_ms is not None
                    ):
                        self.current_turn_metrics["tts_first_audio_ms"] = round(
                            self._now_ms() - self.response_started_at_ms,
                            1,
                        )
                    if self.output_audio_format == "pcm16":
                        await self._emit(
                            {
                                "type": "response.audio.delta",
                                "response_id": response_id,
                                "delta": float32_to_pcm16_b64(pcm),
                                "sample_rate": SAMPLE_RATE,
                                "encoding": "pcm16",
                            }
                        )
                    else:
                        for packet in audio_out.encode_b64_packets(pcm):
                            await self._emit(
                                {
                                    "type": "response.audio.delta",
                                    "response_id": response_id,
                                    "delta": packet,
                                    "encoding": "opus",
                                }
                            )
        except asyncio.CancelledError:
            raise
        finally:
            await self._emit_audio_done(response_id)

    async def _emit_audio_done(self, response_id: str) -> None:
        if self.response_audio_done_sent or self.closed:
            return
        self.response_audio_done_sent = True
        if self.response_started_at_ms is not None:
            self.current_turn_metrics["turn_total_ms"] = round(
                self._now_ms() - self.response_started_at_ms,
                1,
            )
        await self._emit({"type": "turn.metrics", "metrics": self.current_turn_metrics})
        await self._emit({"type": "response.audio.done", "response_id": response_id})

    def _drain_tts_chunks(self, buffer: str, final: bool) -> tuple[list[str], str]:
        chunks: list[str] = []
        working = buffer
        punctuation = ".!?;:,"
        while True:
            split_idx = -1
            for idx, char in enumerate(working):
                if char in punctuation:
                    split_idx = idx + 1
                    break
            if split_idx <= 0:
                break
            chunk = working[:split_idx].strip()
            if chunk:
                chunks.append(chunk)
            working = working[split_idx:].lstrip()
        if not final and len(working) >= self.settings.tts_chunk_soft_limit:
            split_idx = max(
                working.rfind(" ", 0, self.settings.tts_chunk_soft_limit),
                working.rfind(",", 0, self.settings.tts_chunk_soft_limit),
            )
            if split_idx >= self.settings.tts_chunk_min_split:
                chunks.append(working[:split_idx].strip())
                working = working[split_idx + 1 :]
        if final and working.strip():
            chunks.append(working.strip())
            working = ""
        return chunks, working

    def _now_ms(self) -> float:
        return round(time.monotonic() * 1000, 1)

    def _persist_full_input_audio(self) -> None:
        if not self.settings.save_full_input_audio:
            return
        full_audio = self.full_input_audio.to_array()
        if full_audio.size == 0:
            return
        self.artifacts.write_wav(
            self.run_handle,
            "input/full_input.wav",
            full_audio,
            SAMPLE_RATE,
        )

    async def _emit(self, message: dict[str, Any]) -> None:
        if self.closed:
            return
        self.artifacts.append_event(
            self.run_handle, str(message.get("type", "event")), message
        )
        async with self.send_lock:
            try:
                await self.websocket.send_text(json.dumps(message))
            except RuntimeError:
                self.closed = True
