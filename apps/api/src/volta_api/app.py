from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
import re

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import Settings, settings
from .openai_services import OpenAILlmService, OpenAISttService, OpenAITtsService
from .public_session import PublicRealtimeSession
from .session import Session
from .worker_client import WorkerClient, WorkerLike
from .artifacts import ArtifactStore


def _require_existing_path(path: Path, *, env_name: str, label: str) -> Path:
    if path.exists():
        return path
    raise RuntimeError(
        f"Local provider requires {label} to exist. "
        f"Set {env_name} in the repo root .env. Current value: {path}"
    )


def build_workers(runtime_settings: Settings | None = None) -> dict[str, WorkerLike]:
    active_settings = runtime_settings or settings
    if active_settings.use_openai_provider:
        return {
            "stt": OpenAISttService(active_settings),
            "llm": OpenAILlmService(active_settings),
            "tts": OpenAITtsService(active_settings),
        }

    scripts_root = active_settings.scripts_root
    stt_python = _require_existing_path(
        active_settings.stt_python,
        env_name="STT_PYTHON",
        label="the STT Python runtime",
    )
    llm_python = _require_existing_path(
        active_settings.llm_python,
        env_name="LLM_PYTHON",
        label="the LLM Python runtime",
    )
    tts_python = _require_existing_path(
        active_settings.tts_python,
        env_name="TTS_PYTHON",
        label="the TTS Python runtime",
    )
    llm_model_path = _require_existing_path(
        Path(active_settings.llm_model_path),
        env_name="LLM_MODEL_PATH",
        label="the local LLM model path",
    )
    _require_existing_path(
        active_settings.llm_system_prompt_file,
        env_name="LLM_SYSTEM_PROMPT_FILE",
        label="the LLM system prompt file",
    )
    return {
        "stt": WorkerClient(
            "stt",
            stt_python,
            scripts_root / "stt_worker.py",
            ["--hf-repo", active_settings.stt_model_repo],
        ),
        "llm": WorkerClient(
            "llm",
            llm_python,
            scripts_root / "llm_worker.py",
            [
                "--model-path",
                str(llm_model_path),
                "--system-prompt-file",
                str(active_settings.llm_system_prompt_file),
                "--max-tokens",
                str(active_settings.llm_max_tokens),
                "--prompt-cache",
                "true" if active_settings.llm_prompt_cache_enabled else "false",
                "--reasoning",
                active_settings.llm_reasoning or "none",
            ],
        ),
        "tts": WorkerClient(
            "tts",
            tts_python,
            scripts_root / "tts_worker.py",
            ["--model-path", active_settings.tts_model_path],
        ),
    }


workers = build_workers()
artifact_store = ArtifactStore(settings.artifact_root)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.gather(*(worker.start() for worker in workers.values()))
    await asyncio.gather(*(worker.health() for worker in workers.values()))
    yield
    await asyncio.gather(*(worker.stop() for worker in workers.values()))


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:4173",
        "http://127.0.0.1:4174",
        "http://localhost:4173",
        "http://localhost:4174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(
    "/v1/artifacts", StaticFiles(directory=settings.artifact_root), name="artifacts"
)


@app.get("/health")
async def health() -> dict[str, object]:
    statuses = {}
    for name, worker in workers.items():
        try:
            statuses[name] = await worker.health()
        except Exception as exc:
            statuses[name] = {"ok": False, "error": str(exc)}
    return {"ok": True, "workers": statuses}


@app.get("/v1/health")
async def health_v1() -> dict[str, object]:
    return await health()


@app.get("/v1/voices")
async def voices_v1() -> list[dict[str, object]]:
    return [
        {
            "id": settings.default_tts_voice.lower(),
            "name": settings.default_tts_voice,
            "source": {"path_on_server": settings.default_tts_voice},
        }
    ]


def _read_persona_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _persona_display_lines(text: str) -> list[str]:
    section_headings = {
        "persona",
        "core role",
        "behavior",
        "listening and reaction rules",
        "investor lens",
        "style",
        "game framing",
        "conversation flow",
        "output constraints",
    }
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[-*+]\s*", "", line)
        line = re.sub(r"\[(.*?)\]\([^)]+\)", r"\1", line)
        line = re.sub(r"`([^`]*)`", r"\1", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"__(.*?)__", r"\1", line)
        line = re.sub(r"\s+", " ", line).strip(" -")
        if line and line.lower() not in section_headings:
            lines.append(line)
    return lines


@app.get("/v1/persona")
async def persona_v1() -> dict[str, object]:
    text = _read_persona_prompt(settings.llm_system_prompt_file)
    excerpt_lines = _persona_display_lines(text)[:6]
    return {
        "name": "Volta Investor Judge",
        "source_label": "Bundled default persona",
        "text": text,
        "excerpt_lines": excerpt_lines,
    }


@app.get("/v1/runs")
async def runs_v1(limit: int = 24) -> dict[str, object]:
    limit = max(1, min(limit, 100))
    runs = artifact_store.list_runs(limit=limit)
    return {
        "runs": [
            {
                "id": run.session_id,
                "updated_at_ms": run.updated_at_ms,
                "transcript": run.transcript,
                "llm_output": run.llm_output,
                "latest_action": run.latest_action,
                "input_audio_url": f"/v1/artifacts/{run.session_id}/{run.input_audio_path}"
                if run.input_audio_path
                else None,
                "tts_audio_url": f"/v1/artifacts/{run.session_id}/{run.tts_audio_path}"
                if run.tts_audio_path
                else None,
            }
            for run in runs
        ]
    }


@app.get("/v1/runs/{session_id}")
async def run_detail_v1(session_id: str) -> dict[str, object]:
    run = artifact_store.read_run_detail(session_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {
        "id": run.session_id,
        "updated_at_ms": run.updated_at_ms,
        "transcript": run.transcript,
        "llm_output": run.llm_output,
        "latest_action": run.latest_action,
        "input_audio_url": f"/v1/artifacts/{run.session_id}/{run.input_audio_path}"
        if run.input_audio_path
        else None,
        "tts_audio_url": f"/v1/artifacts/{run.session_id}/{run.tts_audio_path}"
        if run.tts_audio_path
        else None,
        "metrics": run.metrics,
        "turns": [
            {
                "turn_id": turn.turn_id,
                "text": turn.text,
                "started_at_ms": turn.started_at_ms,
                "committed_at_ms": turn.committed_at_ms,
                "audio_url": f"/v1/artifacts/{run.session_id}/{turn.audio_path}"
                if turn.audio_path
                else None,
            }
            for turn in run.turns
        ],
        "timeline": [
            {
                "recorded_at_ms": event.recorded_at_ms,
                "type": event.type,
                "detail": event.detail,
            }
            for event in run.timeline
        ],
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session = Session(websocket, settings, workers)
    try:
        while True:
            try:
                message = await websocket.receive_json()
            except RuntimeError as exc:
                if "disconnect message has been received" in str(exc):
                    break
                raise
            await session.handle(message)
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()


@app.websocket("/v1/realtime")
async def realtime_endpoint(websocket: WebSocket) -> None:
    subprotocols = websocket.scope.get("subprotocols", [])
    accepted_subprotocol = "realtime" if "realtime" in subprotocols else None
    await websocket.accept(subprotocol=accepted_subprotocol)
    session = PublicRealtimeSession(websocket, settings, workers)
    try:
        while True:
            try:
                envelope = await websocket.receive()
            except RuntimeError as exc:
                if "disconnect message has been received" in str(exc):
                    break
                raise
            if envelope.get("type") == "websocket.disconnect":
                break
            if envelope.get("bytes") is not None:
                await session.handle_binary_audio(envelope["bytes"])
                continue
            text = envelope.get("text")
            if text is None:
                continue
            await session.handle(json.loads(text))
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()


def main() -> None:
    uvicorn.run(
        "volta_api.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
