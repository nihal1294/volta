from __future__ import annotations
import base64
import io
import json
import math
import re
import time
import wave
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
import numpy as np

from .audio_streams import SAMPLE_RATE, float32_to_pcm16_b64, resample_linear
from .config import Settings

LLM_ALLOWED_ACTIONS = [
    "wait",
    "yield_to_user",
    "continue_speaking",
    "speak",
    "hold_silence",
    "end_turn",
]

PLANNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "enum": LLM_ALLOWED_ACTIONS},
        "mode": {"type": "string", "enum": ["stream", "turn"]},
        "interruptible": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["name", "mode", "interruptible", "reason"],
}

PLANNER_RUNTIME_PROMPT = """You are the floor-control planner inside a realtime voice conversation.
Choose whether the assistant should speak right now or stay silent and wait for more user input.

Rules:
- Return exactly one action.
- Use `wait` when the user thought is still forming or more transcript would help.
- Use `yield_to_user` when the user is clearly still taking the floor.
- Use `hold_silence` when input is empty, noise-like, or a reply would be unhelpful.
- Use `speak` when the assistant should respond now.
- Use `continue_speaking` only when the current conversational direction clearly merits an immediate follow-up.
- `mode` should usually be `stream`.
- `interruptible` should usually be true unless the reply must finish cleanly.
"""

RESPONDER_RUNTIME_PROMPT = """You are the speaking voice inside a realtime STT -> LLM -> TTS loop.
Produce only the spoken assistant reply text.

Rules:
- Do not emit markdown.
- Do not emit tool syntax or JSON.
- Keep the reply concise unless the user explicitly asked for detail.
- Speak naturally and directly.
"""


class OpenAIProviderError(RuntimeError):
    pass


def _merge_delta(previous: str, current: str) -> str:
    if not current:
        return ""
    if not previous:
        return current
    if current.startswith(previous):
        suffix = current[len(previous) :]
        if suffix.startswith(" "):
            return suffix
        return suffix.lstrip()
    return current


def _merge_window_transcript(previous: str, current_window: str) -> str:
    previous = previous.strip()
    current_window = current_window.strip()
    if not current_window:
        return previous
    if not previous:
        return current_window
    if current_window.startswith(previous):
        return current_window
    if previous.startswith(current_window):
        return previous

    previous_words = previous.split()
    current_words = current_window.split()
    previous_norm = [_normalize_word(word) for word in previous_words]
    current_norm = [_normalize_word(word) for word in current_words]
    if previous_norm == current_norm:
        return previous

    max_overlap = min(len(previous_words), len(current_words), 32)
    for overlap in range(max_overlap, 0, -1):
        if previous_norm[-overlap:] == current_norm[:overlap]:
            return " ".join(previous_words + current_words[overlap:])
    return previous


_WORD_EDGE_RE = re.compile(r"(^[^\w']+)|([^\w']+$)")


def _normalize_word(word: str) -> str:
    return _WORD_EDGE_RE.sub("", word).lower()


def _extract_output_text(payload: dict[str, Any]) -> str:
    output = payload.get("output", [])
    for item in output:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    return text
    if isinstance(payload.get("output_text"), str):
        return str(payload["output_text"])
    raise OpenAIProviderError("OpenAI response did not contain output_text.")


def _pcm_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    clipped = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def _wav_bytes_to_float32(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        raw = wav_file.readframes(frames)
    if sample_width != 2:
        raise OpenAIProviderError(f"Unsupported WAV sample width: {sample_width}")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm.astype(np.float32), sample_rate


def _chunk_text(text: str, chunk_size: int = 80) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    chunks: list[str] = []
    remaining = normalized
    while remaining:
        if len(remaining) <= chunk_size:
            chunks.append(remaining)
            break
        split_idx = remaining.rfind(" ", 0, chunk_size)
        if split_idx < max(12, chunk_size // 3):
            split_idx = chunk_size
        chunks.append(remaining[:split_idx].rstrip())
        remaining = remaining[split_idx:].lstrip()
    return chunks


def _history_to_prompt(history: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in history:
        role = str(message.get("role", "user")).strip().upper()
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


@dataclass
class SttStreamState:
    opened_at_ms: float
    audio_chunks: list[np.ndarray] = field(default_factory=list)
    transcript: str = ""
    last_transcribed_samples: int = 0
    first_partial_ms: float | None = None
    first_word_ms: float | None = None

    @property
    def total_samples(self) -> int:
        return sum(chunk.size for chunk in self.audio_chunks)

    def append(self, audio: np.ndarray) -> None:
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        if chunk.size:
            self.audio_chunks.append(chunk)

    def to_array(self) -> np.ndarray:
        if not self.audio_chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.audio_chunks).astype(np.float32)

    def recent_array(self, sample_count: int) -> np.ndarray:
        audio = self.to_array()
        if audio.size <= sample_count:
            return audio
        return audio[-sample_count:].astype(np.float32)


class OpenAIServiceBase:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            base_url=self.settings.openai_base_url.rstrip("/"),
            headers=self._headers(),
            timeout=httpx.Timeout(300.0),
        )

    async def stop(self) -> None:
        if self._client is None:
            return
        await self._client.aclose()
        self._client = None

    async def health(self) -> dict[str, Any]:
        return {
            "ok": bool(self.settings.openai_api_key),
            "provider": "openai",
        }

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.settings.openai_api_key:
            headers["Authorization"] = f"Bearer {self.settings.openai_api_key}"
        return headers

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            await self.start()
        assert self._client is not None
        return self._client

    def _require_api_key(self) -> None:
        if not self.settings.openai_api_key:
            raise OpenAIProviderError("OpenAI API key is not configured.")


class OpenAISttService(OpenAIServiceBase):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._streams: dict[str, SttStreamState] = {}
        self._min_transcribe_samples = max(
            1,
            int(SAMPLE_RATE * self.settings.openai_transcription_min_interval_sec),
        )
        self._partial_window_samples = max(
            self._min_transcribe_samples,
            int(SAMPLE_RATE * self.settings.openai_transcription_partial_window_sec),
        )

    async def health(self) -> dict[str, Any]:
        payload = await super().health()
        payload["model"] = self.settings.openai_transcription_model
        return payload

    async def request_once(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del timeout
        if action == "health":
            return await self.health()
        if action == "stream_start":
            stream_id = str(payload["stream_id"])
            self._streams[stream_id] = SttStreamState(opened_at_ms=self._now_ms())
            return {"ok": True, "stream_id": stream_id}
        if action == "stream_snapshot":
            state = self._require_stream(str(payload["stream_id"]))
            return {"ok": True, "text": state.transcript}
        if action == "stream_commit":
            stream_id = str(payload["stream_id"])
            state = self._require_stream(stream_id)
            started = self._now_ms()
            audio = state.to_array()
            transcript = await self._transcribe(
                audio,
                prompt=self._build_prompt(),
            )
            state.transcript = transcript
            duration_sec = max(audio.size / SAMPLE_RATE, 1e-6)
            words = len(transcript.split())
            self._streams.pop(stream_id, None)
            return {
                "ok": True,
                "text": transcript,
                "first_partial_ms": state.first_partial_ms,
                "first_word_ms": state.first_word_ms,
                "processing_ms": round(self._now_ms() - started, 1),
                "words_per_sec": round(words / duration_sec, 3) if words else 0.0,
            }
        if action == "stream_reset":
            self._streams.pop(str(payload["stream_id"]), None)
            return {"ok": True}
        raise OpenAIProviderError(f"Unsupported STT action: {action}")

    async def request_stream(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        del timeout
        if action != "stream_chunk":
            raise OpenAIProviderError(f"Unsupported STT stream action: {action}")
        state = self._require_stream(str(payload["stream_id"]))
        pcm_b64 = str(payload["pcm_b64"])
        sample_rate = int(payload.get("sample_rate", SAMPLE_RATE))
        chunk = (
            np.frombuffer(
                base64.b64decode(pcm_b64),
                dtype="<i2",
            ).astype(np.float32)
            / 32768.0
        )
        if sample_rate != SAMPLE_RATE:
            chunk = resample_linear(chunk, sample_rate, SAMPLE_RATE)
        state.append(chunk)

        if (
            state.total_samples - state.last_transcribed_samples
            < self._min_transcribe_samples
        ):
            return

        partial_audio = state.to_array()
        transcript = await self._transcribe(
            partial_audio,
            prompt=self._build_prompt(),
        )
        state.last_transcribed_samples = state.total_samples
        transcript = transcript.strip()
        merged_transcript = _merge_window_transcript(state.transcript, transcript)
        if not merged_transcript or merged_transcript == state.transcript:
            return

        delta = _merge_delta(state.transcript, merged_transcript)
        state.transcript = merged_transcript
        elapsed_ms = round(self._now_ms() - state.opened_at_ms, 1)
        if state.first_partial_ms is None:
            state.first_partial_ms = elapsed_ms
        if state.first_word_ms is None and merged_transcript.split():
            state.first_word_ms = elapsed_ms

        yield {
            "event": "partial",
            "data": {
                "text": merged_transcript,
                "delta": delta or merged_transcript,
                "first_partial_ms": state.first_partial_ms,
                "first_word_ms": state.first_word_ms,
            },
        }

    async def _transcribe(self, audio: np.ndarray, prompt: str | None = None) -> str:
        self._require_api_key()
        if audio.size == 0:
            return ""
        client = await self._ensure_client()
        wav_bytes = _pcm_to_wav_bytes(audio, SAMPLE_RATE)
        data = {
            "model": self.settings.openai_transcription_model,
            "language": self.settings.openai_transcription_language,
        }
        if prompt:
            data["prompt"] = prompt
        response = await client.post(
            "/audio/transcriptions",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            data=data,
            files={"file": ("utterance.wav", wav_bytes, "audio/wav")},
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("text", "")).strip()

    def _build_prompt(self, context_text: str | None = None) -> str | None:
        parts: list[str] = []
        base_prompt = (self.settings.openai_transcription_prompt or "").strip()
        if base_prompt:
            parts.append(base_prompt)
        context = (context_text or "").strip()
        if context:
            parts.append(
                context[-self.settings.openai_transcription_context_chars :].strip()
            )
        combined = "\n\n".join(part for part in parts if part)
        return combined or None

    def _require_stream(self, stream_id: str) -> SttStreamState:
        state = self._streams.get(stream_id)
        if state is None:
            raise OpenAIProviderError(f"Unknown STT stream: {stream_id}")
        return state

    def _now_ms(self) -> float:
        return round(time.monotonic() * 1000, 1)


class OpenAILlmService(OpenAIServiceBase):
    async def health(self) -> dict[str, Any]:
        payload = await super().health()
        payload["model"] = self.settings.openai_llm_model
        return payload

    async def request_once(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del payload, timeout
        if action == "health":
            return await self.health()
        if action in {"session_reset", "cancel_generate"}:
            return {"ok": True}
        raise OpenAIProviderError(f"Unsupported LLM action: {action}")

    async def request_stream(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        del timeout
        if action != "generate":
            raise OpenAIProviderError(f"Unsupported LLM stream action: {action}")

        history = payload.get("history")
        if not isinstance(history, list):
            raise OpenAIProviderError("LLM history payload must be a list.")
        persona_text = str(payload.get("persona_text", "")).strip()
        conversation = _history_to_prompt(history)

        planned_action = await self._plan_action(persona_text, conversation)
        yield {"event": "action", "data": planned_action}

        if planned_action["name"] not in {"speak", "continue_speaking"}:
            yield {"event": "metrics", "data": {"generation_tps": None}}
            return

        started = time.monotonic()
        response_text = await self._generate_reply(persona_text, conversation)
        chunks = _chunk_text(response_text)
        for chunk in chunks:
            yield {"event": "token", "data": {"text": chunk}}
        elapsed = max(time.monotonic() - started, 1e-6)
        pseudo_tps = (
            round(len(response_text.split()) / elapsed, 3)
            if response_text.strip()
            else 0.0
        )
        yield {"event": "metrics", "data": {"generation_tps": pseudo_tps}}

    async def _plan_action(
        self, persona_text: str, conversation: str
    ) -> dict[str, Any]:
        payload = {
            "model": self.settings.openai_llm_model,
            "instructions": "\n\n".join(
                segment
                for segment in [persona_text.strip(), PLANNER_RUNTIME_PROMPT]
                if segment.strip()
            ),
            "input": conversation or "No conversation history yet.",
            "store": False,
            "max_output_tokens": 200,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "floor_action",
                    "strict": True,
                    "schema": PLANNER_SCHEMA,
                }
            },
        }
        self._apply_reasoning(payload)
        response = await self._post_json("/responses", payload)
        content = _extract_output_text(response)
        action = json.loads(content)
        action_name = str(action.get("name", "wait"))
        return {
            "name": action_name if action_name in LLM_ALLOWED_ACTIONS else "wait",
            "mode": str(action.get("mode", "stream")),
            "interruptible": bool(action.get("interruptible", True)),
            "reason": str(action.get("reason", "")),
        }

    async def _generate_reply(self, persona_text: str, conversation: str) -> str:
        payload = {
            "model": self.settings.openai_llm_model,
            "instructions": "\n\n".join(
                segment
                for segment in [persona_text.strip(), RESPONDER_RUNTIME_PROMPT]
                if segment.strip()
            ),
            "input": conversation or "No conversation history yet.",
            "store": False,
            "max_output_tokens": self.settings.llm_max_tokens,
        }
        self._apply_reasoning(payload)
        response = await self._post_json("/responses", payload)
        return _extract_output_text(response).strip()

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._require_api_key()
        client = await self._ensure_client()
        response = await client.post(path, json=payload)
        response.raise_for_status()
        return response.json()

    def _apply_reasoning(self, payload: dict[str, Any]) -> None:
        effort = (self.settings.openai_llm_reasoning or "").strip().lower()
        if not effort or effort == "none":
            return
        payload["reasoning"] = {"effort": effort}


class OpenAITtsService(OpenAIServiceBase):
    async def health(self) -> dict[str, Any]:
        payload = await super().health()
        payload["model"] = self.settings.openai_tts_model
        payload["voice"] = self.settings.openai_tts_voice
        return payload

    async def request_once(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del payload, timeout
        if action == "health":
            return await self.health()
        raise OpenAIProviderError(f"Unsupported TTS action: {action}")

    async def request_stream(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        del timeout
        if action != "synthesize":
            raise OpenAIProviderError(f"Unsupported TTS stream action: {action}")

        text = str(payload.get("text", "")).strip()
        if not text:
            yield {"event": "metrics", "data": {"real_time_factor": 0.0}}
            return

        voice = str(payload.get("voice") or self.settings.openai_tts_voice)
        instruct = str(payload.get("instruct") or "").strip() or None
        started = time.monotonic()
        audio_bytes = await self._synthesize(text=text, voice=voice, instruct=instruct)
        pcm, sample_rate = _wav_bytes_to_float32(audio_bytes)
        pcm = resample_linear(pcm, sample_rate, SAMPLE_RATE)

        interval = float(
            payload.get("streaming_interval", self.settings.tts_streaming_interval)
        )
        chunk_samples = max(1, int(round(SAMPLE_RATE * interval)))
        total_chunks = max(1, math.ceil(pcm.size / chunk_samples))

        for index in range(total_chunks):
            start = index * chunk_samples
            end = min(pcm.size, start + chunk_samples)
            chunk = pcm[start:end]
            if chunk.size == 0:
                continue
            yield {
                "event": "audio",
                "data": {
                    "pcm16_b64": float32_to_pcm16_b64(chunk),
                    "sample_rate": SAMPLE_RATE,
                    "samples": int(chunk.size),
                    "segment_idx": index,
                    "is_final_chunk": index == total_chunks - 1,
                },
            }

        duration_sec = max(pcm.size / SAMPLE_RATE, 1e-6)
        elapsed = max(time.monotonic() - started, 1e-6)
        yield {
            "event": "metrics",
            "data": {
                "real_time_factor": round(elapsed / duration_sec, 3),
            },
        }

    async def _synthesize(
        self, *, text: str, voice: str, instruct: str | None = None
    ) -> bytes:
        self._require_api_key()
        client = await self._ensure_client()
        payload = {
            "model": self.settings.openai_tts_model,
            "voice": voice,
            "input": text,
            "response_format": self.settings.openai_tts_format,
        }
        if instruct and self.settings.openai_tts_model == "gpt-4o-mini-tts":
            payload["instructions"] = instruct
        response = await client.post("/audio/speech", json=payload)
        if response.status_code == 400 and voice != self.settings.openai_tts_voice:
            fallback_payload = dict(payload)
            fallback_payload["voice"] = self.settings.openai_tts_voice
            response = await client.post("/audio/speech", json=fallback_payload)
        response.raise_for_status()
        return response.content
