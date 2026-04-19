from __future__ import annotations

import asyncio
import base64
import json
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import WebSocket

from .config import Settings
from .protocol import DEFAULT_METRICS, TurnState, server_event
from .worker_client import WorkerLike


PUNCTUATION_RE = re.compile(r"(.+?[.!?;:](?:\s+|$))")


class Session:
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
        self.state = TurnState.IDLE
        self.mode = "mic"
        self.voice = settings.default_tts_voice
        self.instruct = settings.tts_instruct
        self.history: list[dict[str, str]] = []
        self.send_lock = asyncio.Lock()
        self.file_bytes = bytearray()
        self.file_name = "input.wav"
        self.file_mime_type = "audio/wav"
        self.turn_started_at: float | None = None
        self.metrics: dict[str, Any] = dict(DEFAULT_METRICS)
        self.current_task: asyncio.Task[None] | None = None
        self.stt_session_open = False
        self.closed = False

    async def handle(self, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if message_type == "session.start":
            self.mode = message.get("mode", "mic")
            self.voice = message.get("voice", self.voice)
            self.instruct = message.get("instruct", self.instruct)
            await self._emit_state(TurnState.IDLE)
            return
        if message_type == "session.cancel":
            await self.cancel()
            return
        if message_type == "session.close":
            await self.close()
            return
        if message_type == "file.input.start":
            self.file_bytes = bytearray()
            self.file_name = message.get("file_name", "input.wav")
            self.file_mime_type = message.get("mime_type", "application/octet-stream")
            await self._emit(server_event("file.status", {"status": "receiving"}))
            return
        if message_type == "file.input.chunk":
            self.file_bytes.extend(base64.b64decode(message["chunk_b64"]))
            return
        if message_type == "file.input.end":
            if self.current_task and not self.current_task.done():
                return
            self.current_task = asyncio.create_task(self._run_file_turn())
            return
        if message_type == "audio.input.chunk":
            await self._handle_audio_chunk(message)
            return
        if message_type == "audio.input.end":
            if self.current_task and not self.current_task.done():
                return
            self.current_task = asyncio.create_task(self._finish_audio_turn())
            return

    async def cancel(self, emit_state: bool = True) -> None:
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
        if self.stt_session_open:
            try:
                await self.workers["stt"].request_once(
                    "session_discard",
                    {"session_id": self.session_id},
                    timeout=5,
                )
            except Exception:
                pass
            self.stt_session_open = False
        if emit_state:
            await self._emit_state(TurnState.IDLE)

    async def close(self) -> None:
        self.closed = True
        await self.cancel(emit_state=False)
        try:
            await self.workers["llm"].request_once(
                "session_reset",
                {"session_id": self.session_id},
                timeout=5,
            )
        except Exception:
            pass

    async def _handle_audio_chunk(self, message: dict[str, Any]) -> None:
        if not self.stt_session_open:
            await self.workers["stt"].request_once(
                "session_start",
                {"session_id": self.session_id},
                timeout=10,
            )
            self.stt_session_open = True
            self.turn_started_at = time.perf_counter()
            self.metrics = dict(DEFAULT_METRICS)
            await self._emit_state(TurnState.LISTENING)
        async for event in self.workers["stt"].request_stream(
            "session_chunk",
            {
                "session_id": self.session_id,
                "pcm_b64": message["chunk_b64"],
                "sample_rate": message["sample_rate"],
            },
            timeout=30,
        ):
            if event["event"] == "partial":
                payload = event["data"]
                await self._record_stt_metrics(payload)
                await self._emit(server_event("stt.partial", payload))

    async def _finish_audio_turn(self) -> None:
        if not self.stt_session_open:
            return
        transcript = ""
        async for event in self.workers["stt"].request_stream(
            "session_end",
            {"session_id": self.session_id},
            timeout=60,
        ):
            if event["event"] == "partial":
                payload = event["data"]
                await self._record_stt_metrics(payload)
                await self._emit(server_event("stt.partial", payload))
            elif event["event"] == "final":
                payload = event["data"]
                transcript = payload["text"]
                await self._record_stt_metrics(payload, final=True)
                await self._emit(server_event("stt.final", payload))
        self.stt_session_open = False
        await self._run_llm_and_tts(transcript)

    async def _run_file_turn(self) -> None:
        suffix = Path(self.file_name).suffix or ".wav"
        with tempfile.NamedTemporaryFile(
            dir=self.settings.uploads_root,
            suffix=suffix,
            delete=False,
        ) as handle:
            handle.write(self.file_bytes)
            file_path = Path(handle.name)
        self.turn_started_at = time.perf_counter()
        self.metrics = dict(DEFAULT_METRICS)
        await self._emit_state(TurnState.THINKING)
        transcript = ""
        async for event in self.workers["stt"].request_stream(
            "transcribe_file",
            {"path": str(file_path)},
            timeout=300,
        ):
            if event["event"] == "partial":
                payload = event["data"]
                await self._record_stt_metrics(payload)
                await self._emit(server_event("stt.partial", payload))
            elif event["event"] == "final":
                payload = event["data"]
                transcript = payload["text"]
                await self._record_stt_metrics(payload, final=True)
                await self._emit(server_event("stt.final", payload))
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass
        await self._run_llm_and_tts(transcript)

    async def _run_llm_and_tts(self, transcript: str) -> None:
        transcript = transcript.strip()
        if not transcript:
            await self._emit(
                server_event("error", {"message": "No transcript produced for turn."})
            )
            await self._finish_turn("")
            return

        self.history.append({"role": "user", "content": transcript})
        await self._emit_state(TurnState.THINKING)
        llm_started_at = time.perf_counter()
        llm_text = ""
        action_mode = "stream"
        action_name = "speak"
        tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
        tts_task = asyncio.create_task(self._tts_consumer(tts_queue))
        tts_buffer = ""

        async for event in self.workers["llm"].request_stream(
            "generate",
            {"session_id": self.session_id, "history": self.history},
            timeout=300,
        ):
            if event["event"] == "action":
                payload = event["data"]
                action_name = payload.get("name", action_name)
                action_mode = payload.get("mode", action_mode)
                await self._emit(server_event("llm.action", payload))
                continue
            if event["event"] == "token":
                payload = event["data"]
                if self.metrics["llm_ttft_ms"] is None:
                    self.metrics["llm_ttft_ms"] = round(
                        (time.perf_counter() - llm_started_at) * 1000, 1
                    )
                text = payload["text"]
                llm_text += text
                await self._emit(server_event("llm.token", payload))
                if action_name == "speak" and action_mode == "stream":
                    tts_buffer += text
                    ready_chunks, tts_buffer = self._drain_tts_chunks(
                        tts_buffer, final=False
                    )
                    for chunk in ready_chunks:
                        await tts_queue.put(chunk)
                continue
            if event["event"] == "metrics":
                payload = event["data"]
                self.metrics["llm_tokens_per_sec"] = payload.get("generation_tps")
                continue

        if action_name == "speak":
            final_chunks, tts_buffer = self._drain_tts_chunks(tts_buffer, final=True)
            for chunk in final_chunks:
                await tts_queue.put(chunk)
        await tts_queue.put(None)
        await tts_task
        await self._finish_turn(llm_text)

    async def _tts_consumer(self, queue: asyncio.Queue[str | None]) -> None:
        speaking_started = False
        while True:
            text = await queue.get()
            if text is None:
                break
            if not speaking_started:
                speaking_started = True
                await self._emit_state(TurnState.SPEAKING)
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
                if event["event"] == "audio":
                    payload = event["data"]
                    if (
                        self.metrics["tts_first_audio_ms"] is None
                        and self.turn_started_at
                    ):
                        self.metrics["tts_first_audio_ms"] = round(
                            (time.perf_counter() - self.turn_started_at) * 1000, 1
                        )
                    if payload.get("real_time_factor") is not None:
                        self.metrics["tts_realtime_factor"] = payload[
                            "real_time_factor"
                        ]
                    await self._emit(server_event("tts.audio.chunk", payload))
                elif event["event"] == "metrics":
                    payload = event["data"]
                    if payload.get("real_time_factor") is not None:
                        self.metrics["tts_realtime_factor"] = payload[
                            "real_time_factor"
                        ]
        await self._emit(server_event("tts.done", {}))

    async def _finish_turn(self, llm_text: str) -> None:
        assistant_text = llm_text.strip()
        if assistant_text:
            self.history.append({"role": "assistant", "content": assistant_text})
        if self.turn_started_at:
            self.metrics["turn_total_ms"] = round(
                (time.perf_counter() - self.turn_started_at) * 1000, 1
            )
        await self._emit(server_event("turn.summary", {"metrics": self.metrics}))
        await self._emit(server_event("turn.done", {"assistant_text": assistant_text}))
        await self._emit_state(TurnState.IDLE)

    def _drain_tts_chunks(self, buffer: str, final: bool) -> tuple[list[str], str]:
        chunks: list[str] = []
        while True:
            match = PUNCTUATION_RE.match(buffer)
            if not match:
                break
            chunk = match.group(1).strip()
            if chunk:
                chunks.append(chunk)
            buffer = buffer[match.end() :]
        if not final and len(buffer) >= self.settings.tts_chunk_soft_limit:
            split_idx = max(
                buffer.rfind(" ", 0, self.settings.tts_chunk_soft_limit),
                buffer.rfind(",", 0, self.settings.tts_chunk_soft_limit),
            )
            if split_idx >= self.settings.tts_chunk_min_split:
                chunks.append(buffer[:split_idx].strip())
                buffer = buffer[split_idx + 1 :]
        if final and buffer.strip():
            chunks.append(buffer.strip())
            buffer = ""
        return chunks, buffer

    async def _record_stt_metrics(
        self,
        payload: dict[str, Any],
        final: bool = False,
    ) -> None:
        if payload.get("first_partial_ms") is not None:
            self.metrics["stt_first_partial_ms"] = payload["first_partial_ms"]
        if payload.get("first_word_ms") is not None:
            self.metrics["stt_first_word_ms"] = payload["first_word_ms"]
        if final:
            if payload.get("processing_ms") is not None:
                self.metrics["stt_final_ms"] = payload["processing_ms"]
            if payload.get("words_per_sec") is not None:
                self.metrics["stt_words_per_sec"] = payload["words_per_sec"]

    async def _emit_state(self, state: TurnState) -> None:
        self.state = state
        await self._emit(server_event("state.changed", {"state": state}))

    async def _emit(self, message: dict[str, Any]) -> None:
        if self.closed:
            return
        async with self.send_lock:
            try:
                await self.websocket.send_text(json.dumps(message))
            except RuntimeError:
                self.closed = True
