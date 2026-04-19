from __future__ import annotations

import asyncio
import base64
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from volta_api.app import build_workers
from volta_api.config import Settings, settings
from volta_api.openai_services import (
    OpenAILlmService,
    OpenAISttService,
    OpenAITtsService,
    _merge_delta,
    _merge_window_transcript,
    _pcm_to_wav_bytes,
)
from volta_api.worker_client import WorkerClient


def _pcm16_b64(audio: np.ndarray) -> str:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    return base64.b64encode(pcm16.tobytes()).decode("ascii")


def test_openai_settings_defaults_follow_bargeai(monkeypatch) -> None:
    for name in [
        "PROVIDER",
        "PIPELINE_PROVIDER",
        "VOLTA_PIPELINE_PROVIDER",
        "OPENAI_API_KEY",
        "VOLTA_OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "VOLTA_OPENAI_BASE_URL",
        "OPENAI_LLM_MODEL",
        "VOLTA_OPENAI_LLM_MODEL",
        "OPENAI_TRANSCRIPTION_MODEL",
        "VOLTA_OPENAI_TRANSCRIPTION_MODEL",
        "OPENAI_TTS_MODEL",
        "VOLTA_OPENAI_TTS_MODEL",
        "OPENAI_TTS_VOICE",
        "VOLTA_OPENAI_TTS_VOICE",
    ]:
        monkeypatch.delenv(name, raising=False)

    loaded = Settings.from_env(env={}, dotenv_path=Path("/tmp/volta-missing.env"))

    assert loaded.pipeline_provider == "openai"
    assert loaded.openai_base_url == "https://api.openai.com/v1"
    assert loaded.openai_llm_model == "gpt-5.4-mini"
    assert loaded.openai_transcription_model == "gpt-4o-transcribe"
    assert loaded.openai_transcription_language == "en"
    assert loaded.openai_transcription_min_interval_sec == 1.6
    assert loaded.openai_transcription_partial_window_sec == 6.0
    assert loaded.openai_transcription_context_chars == 220
    assert loaded.openai_commit_stability_window_ms == 5000
    assert loaded.openai_max_open_utterance_ms == 30000
    assert loaded.openai_tts_model == "gpt-4o-mini-tts"
    assert loaded.openai_tts_voice == "alloy"


def test_settings_load_plain_keys_from_repo_dotenv(tmp_path: Path) -> None:
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "\n".join(
            [
                "PROVIDER=openai",
                "OPENAI_API_KEY=sk-test",
                "OPENAI_BASE_URL=https://api.openai.com/v1",
                "OPENAI_LLM_MODEL=gpt-5.4-mini",
                "LLM_REASONING=none",
                "OPENAI_TRANSCRIPTION_MODEL=gpt-4o-transcribe",
                "OPENAI_TRANSCRIPTION_LANGUAGE=en",
                "OPENAI_TRANSCRIPTION_PROMPT=Investor pitch vocabulary.",
                "OPENAI_TTS_MODEL=gpt-4o-mini-tts",
                "OPENAI_TTS_VOICE=alloy",
            ]
        ),
        encoding="utf-8",
    )

    loaded = Settings.from_env(env={}, dotenv_path=dotenv_path)

    assert loaded.pipeline_provider == "openai"
    assert loaded.openai_api_key == "sk-test"
    assert loaded.openai_tts_voice == "alloy"
    assert loaded.llm_reasoning == "none"
    assert loaded.openai_llm_reasoning == "none"
    assert loaded.openai_transcription_prompt == "Investor pitch vocabulary."
    assert loaded.openai_transcription_language == "en"


def test_build_workers_selects_provider() -> None:
    openai_workers = build_workers(
        replace(
            settings,
            pipeline_provider="openai",
            openai_api_key="sk-test",
        )
    )
    assert isinstance(openai_workers["stt"], OpenAISttService)
    assert isinstance(openai_workers["llm"], OpenAILlmService)
    assert isinstance(openai_workers["tts"], OpenAITtsService)

    local_workers = build_workers(
        replace(
            settings,
            pipeline_provider="local",
            llm_reasoning="none",
            stt_python=Path(sys.executable),
            llm_python=Path(sys.executable),
            tts_python=Path(sys.executable),
            llm_model_path=str(settings.workspace_root),
        )
    )
    assert isinstance(local_workers["stt"], WorkerClient)
    assert isinstance(local_workers["llm"], WorkerClient)
    assert isinstance(local_workers["tts"], WorkerClient)
    assert "--reasoning" in local_workers["llm"].args
    assert (
        local_workers["llm"].args[local_workers["llm"].args.index("--reasoning") + 1]
        == "none"
    )


def test_openai_llm_service_emits_action_tokens_and_metrics() -> None:
    async def scenario() -> None:
        service = OpenAILlmService(replace(settings, openai_api_key="sk-test"))

        async def fake_plan_action(
            persona_text: str, conversation: str
        ) -> dict[str, Any]:
            assert "investor" in persona_text.lower() or persona_text
            assert "USER: give me the numbers" in conversation
            return {
                "name": "speak",
                "mode": "stream",
                "interruptible": True,
                "reason": "enough context",
            }

        async def fake_generate_reply(persona_text: str, conversation: str) -> str:
            del persona_text, conversation
            return "Give me margin, retention, and CAC."

        service._plan_action = fake_plan_action  # type: ignore[method-assign]
        service._generate_reply = fake_generate_reply  # type: ignore[method-assign]

        events = [
            event
            async for event in service.request_stream(
                "generate",
                {
                    "session_id": "session-1",
                    "history": [{"role": "user", "content": "give me the numbers"}],
                    "persona_text": "You are a hard-edged investor.",
                },
            )
        ]

        assert events[0]["event"] == "action"
        assert events[0]["data"]["name"] == "speak"
        assert any(event["event"] == "token" for event in events)
        assert events[-1]["event"] == "metrics"

    asyncio.run(scenario())


def test_openai_stt_service_streams_and_commits() -> None:
    async def scenario() -> None:
        service = OpenAISttService(
            replace(
                settings,
                openai_api_key="sk-test",
                openai_transcription_min_interval_sec=0.5,
            )
        )
        transcripts = iter(["hello there", "hello there founder"])

        async def fake_transcribe(audio: np.ndarray, prompt: str | None = None) -> str:
            del prompt
            assert audio.size >= 19_200
            return next(transcripts)

        service._transcribe = fake_transcribe  # type: ignore[method-assign]

        await service.request_once("stream_start", {"stream_id": "stream-1"})
        audio = np.ones(20_000, dtype=np.float32) * 0.05
        partials = [
            event
            async for event in service.request_stream(
                "stream_chunk",
                {
                    "stream_id": "stream-1",
                    "pcm_b64": _pcm16_b64(audio),
                    "sample_rate": 24_000,
                },
            )
        ]
        snapshot = await service.request_once(
            "stream_snapshot", {"stream_id": "stream-1"}
        )
        commit = await service.request_once("stream_commit", {"stream_id": "stream-1"})

        assert partials[0]["event"] == "partial"
        assert partials[0]["data"]["text"] == "hello there"
        assert snapshot["text"] == "hello there"
        assert commit["text"] == "hello there founder"
        assert commit["first_partial_ms"] is not None
        assert commit["words_per_sec"] > 0

    asyncio.run(scenario())


def test_openai_stt_service_sends_default_english_language() -> None:
    async def scenario() -> None:
        service = OpenAISttService(
            replace(
                settings,
                openai_api_key="sk-test",
                openai_transcription_language="en",
            )
        )

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict[str, str]:
                return {"text": "hello founder"}

        captured: dict[str, Any] = {}

        class FakeClient:
            async def post(
                self,
                path: str,
                headers: dict[str, str],
                data: dict[str, str],
                files: dict[str, tuple[str, bytes, str]],
            ):
                captured["path"] = path
                captured["headers"] = headers
                captured["data"] = data
                captured["files"] = files
                return FakeResponse()

        service._client = FakeClient()  # type: ignore[assignment]
        transcript = await service._transcribe(np.ones(24_000, dtype=np.float32) * 0.05)

        assert transcript == "hello founder"
        assert captured["path"] == "/audio/transcriptions"
        assert captured["data"]["model"] == service.settings.openai_transcription_model
        assert captured["data"]["language"] == "en"

    asyncio.run(scenario())


def test_merge_window_transcript_uses_word_overlap() -> None:
    merged = _merge_window_transcript(
        "Charles, we are from Chorkoi and parents are tired",
        "parents are tired so we fixed it with Chorkoi",
    )
    assert (
        merged
        == "Charles, we are from Chorkoi and parents are tired so we fixed it with Chorkoi"
    )


def test_merge_window_transcript_ignores_non_overlapping_partial_branch() -> None:
    merged = _merge_window_transcript(
        "Hi sharks, I'm the founder of ShoreCoin.",
        "So it's ethical.",
    )
    assert merged == "Hi sharks, I'm the founder of ShoreCoin."


def test_merge_window_transcript_dedupes_normalized_repeat() -> None:
    merged = _merge_window_transcript(
        "No one is coordinating this inefficiency.",
        "coordinating this inefficiency.",
    )
    assert merged == "No one is coordinating this inefficiency."


def test_merge_window_transcript_extends_with_normalized_overlap() -> None:
    merged = _merge_window_transcript(
        "Here's the problem.",
        "here's the problem. No one is coordinating this inefficiency.",
    )
    assert merged == "Here's the problem. No one is coordinating this inefficiency."


def test_merge_delta_preserves_word_boundary_space() -> None:
    assert _merge_delta("the founder", "the founder of Shorecoin") == " of Shorecoin"


def test_openai_stt_service_partial_uses_full_accumulated_audio_snapshots() -> None:
    async def scenario() -> None:
        service = OpenAISttService(
            replace(
                settings,
                openai_api_key="sk-test",
                openai_transcription_prompt="Investor pitch vocabulary.",
                openai_transcription_min_interval_sec=0.5,
            )
        )

        prompts: list[str | None] = []
        audio_sizes: list[int] = []
        transcripts = iter(
            [
                "Charles, we are from Chorkoi",
                "from Chorkoi and parents are tired",
            ]
        )

        async def fake_transcribe(audio: np.ndarray, prompt: str | None = None) -> str:
            audio_sizes.append(int(audio.size))
            prompts.append(prompt)
            return next(transcripts)

        service._transcribe = fake_transcribe  # type: ignore[method-assign]

        await service.request_once("stream_start", {"stream_id": "stream-1"})
        audio = np.ones(200_000, dtype=np.float32) * 0.05

        first = [
            event
            async for event in service.request_stream(
                "stream_chunk",
                {
                    "stream_id": "stream-1",
                    "pcm_b64": _pcm16_b64(audio[:100_000]),
                    "sample_rate": 24_000,
                },
            )
        ]
        second = [
            event
            async for event in service.request_stream(
                "stream_chunk",
                {
                    "stream_id": "stream-1",
                    "pcm_b64": _pcm16_b64(audio[100_000:]),
                    "sample_rate": 24_000,
                },
            )
        ]

        assert audio_sizes == [100_000, 200_000]
        assert first[0]["data"]["text"] == "Charles, we are from Chorkoi"
        assert (
            second[0]["data"]["text"]
            == "Charles, we are from Chorkoi and parents are tired"
        )
        assert second[0]["data"]["delta"] == " and parents are tired"
        assert prompts[0] == "Investor pitch vocabulary."
        assert prompts[1] == "Investor pitch vocabulary."

    asyncio.run(scenario())


def test_openai_tts_service_chunks_wav_audio() -> None:
    async def scenario() -> None:
        service = OpenAITtsService(replace(settings, openai_api_key="sk-test"))

        async def fake_synthesize(
            *, text: str, voice: str, instruct: str | None = None
        ) -> bytes:
            assert text == "Tear down the claim."
            assert voice == "alloy"
            assert instruct is None
            audio = np.linspace(-0.15, 0.15, num=24_000, dtype=np.float32)
            return _pcm_to_wav_bytes(audio)

        service._synthesize = fake_synthesize  # type: ignore[method-assign]

        events = [
            event
            async for event in service.request_stream(
                "synthesize",
                {
                    "text": "Tear down the claim.",
                    "voice": "alloy",
                    "streaming_interval": 0.2,
                },
            )
        ]

        audio_events = [event for event in events if event["event"] == "audio"]
        assert audio_events
        assert events[-1]["event"] == "metrics"
        assert events[-1]["data"]["real_time_factor"] >= 0

    asyncio.run(scenario())


def test_openai_llm_reasoning_none_omits_reasoning_payload() -> None:
    service = OpenAILlmService(
        replace(settings, openai_api_key="sk-test", openai_llm_reasoning="none")
    )
    payload = {"model": "gpt-5.4-mini"}
    service._apply_reasoning(payload)
    assert "reasoning" not in payload


def test_openai_llm_reasoning_low_sets_reasoning_payload() -> None:
    service = OpenAILlmService(
        replace(settings, openai_api_key="sk-test", openai_llm_reasoning="low")
    )
    payload = {"model": "gpt-5.4-mini"}
    service._apply_reasoning(payload)
    assert payload["reasoning"] == {"effort": "low"}
