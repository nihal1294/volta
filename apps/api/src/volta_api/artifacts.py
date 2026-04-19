from __future__ import annotations

import json
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RunHandle:
    root: Path
    session_id: str


@dataclass(frozen=True, slots=True)
class RunSummary:
    session_id: str
    updated_at_ms: int
    transcript: str
    llm_output: str
    latest_action: str
    input_audio_path: str | None
    tts_audio_path: str | None


@dataclass(frozen=True, slots=True)
class RunEvent:
    recorded_at_ms: int
    type: str
    detail: str


@dataclass(frozen=True, slots=True)
class RunTurn:
    turn_id: str
    text: str
    started_at_ms: float | None
    committed_at_ms: float | None
    audio_path: str | None


@dataclass(frozen=True, slots=True)
class RunDetail:
    session_id: str
    updated_at_ms: int
    transcript: str
    llm_output: str
    latest_action: str
    input_audio_path: str | None
    tts_audio_path: str | None
    metrics: dict[str, Any]
    turns: list[RunTurn]
    timeline: list[RunEvent]


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def open_run(self, session_id: str) -> RunHandle:
        run_root = self.root / session_id
        for child in ("input", "stt", "llm", "tts", "tts/chunks"):
            (run_root / child).mkdir(parents=True, exist_ok=True)
        return RunHandle(root=run_root, session_id=session_id)

    def append_event(
        self,
        run: RunHandle,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        self.append_jsonl(
            run,
            "events.jsonl",
            {
                "type": event_type,
                "recorded_at_ms": int(time.time() * 1000),
                **payload,
            },
        )

    def append_jsonl(
        self,
        run: RunHandle,
        relative_path: str | Path,
        payload: dict[str, Any],
    ) -> None:
        path = run.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def write_json(
        self,
        run: RunHandle,
        relative_path: str | Path,
        payload: dict[str, Any],
    ) -> Path:
        path = run.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return path

    def write_text(
        self,
        run: RunHandle,
        relative_path: str | Path,
        text: str,
    ) -> Path:
        path = run.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def write_wav(
        self,
        run: RunHandle,
        relative_path: str | Path,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Path:
        path = run.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype("<i2")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm16.tobytes())
        return path

    def list_runs(self, limit: int = 50) -> list[RunSummary]:
        if not self.root.exists():
            return []
        run_dirs = [path for path in self.root.iterdir() if path.is_dir()]
        run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        run_dirs = [path for path in run_dirs if self._is_meaningful_run(path)]
        return [self._summarize_run(path) for path in run_dirs[:limit]]

    def read_run(self, session_id: str) -> RunSummary | None:
        run_root = self.root / session_id
        if not run_root.is_dir():
            return None
        return self._summarize_run(run_root)

    def read_run_detail(self, session_id: str) -> RunDetail | None:
        run_root = self.root / session_id
        if not run_root.is_dir():
            return None
        summary = self._summarize_run(run_root)
        return RunDetail(
            session_id=summary.session_id,
            updated_at_ms=summary.updated_at_ms,
            transcript=summary.transcript,
            llm_output=summary.llm_output,
            latest_action=summary.latest_action,
            input_audio_path=summary.input_audio_path,
            tts_audio_path=summary.tts_audio_path,
            metrics=self._read_latest_metrics(run_root),
            turns=self._read_turns(run_root),
            timeline=self._read_timeline(run_root, updated_at_ms=summary.updated_at_ms),
        )

    def _summarize_run(self, run_root: Path) -> RunSummary:
        session_id = run_root.name
        transcript = self._read_committed_transcript(run_root)
        llm_output = self._read_text(run_root / "llm" / "final_output.txt")
        latest_action = self._read_latest_action(
            run_root / "llm" / "floor_actions.jsonl"
        )
        input_audio_path = self._preferred_input_audio_path(run_root)
        tts_audio_path = (
            "tts/final_output.wav"
            if (run_root / "tts" / "final_output.wav").is_file()
            else None
        )
        updated_at_ms = int(run_root.stat().st_mtime * 1000)
        return RunSummary(
            session_id=session_id,
            updated_at_ms=updated_at_ms,
            transcript=transcript,
            llm_output=llm_output,
            latest_action=latest_action,
            input_audio_path=input_audio_path,
            tts_audio_path=tts_audio_path,
        )

    def _read_committed_transcript(self, run_root: Path) -> str:
        commit_texts = [
            str(payload.get("text", "")).strip()
            for payload in self._read_jsonl(run_root / "stt" / "commits.jsonl")
            if str(payload.get("text", "")).strip()
        ]
        if commit_texts:
            return "\n".join(commit_texts).strip()
        return self._read_text(run_root / "stt" / "final_transcript.txt")

    def _is_meaningful_run(self, run_root: Path) -> bool:
        summary = self._summarize_run(run_root)
        if summary.transcript or summary.llm_output or summary.latest_action:
            return True
        if summary.input_audio_path or summary.tts_audio_path:
            return True
        return len(self._read_jsonl(run_root / "events.jsonl")) > 1

    def _preferred_input_audio_path(self, run_root: Path) -> str | None:
        full_input = run_root / "input" / "full_input.wav"
        if full_input.is_file():
            return "input/full_input.wav"
        return self._latest_relative_file(run_root / "input", "*.wav")

    def _latest_relative_file(self, directory: Path, pattern: str) -> str | None:
        if not directory.is_dir():
            return None
        files = sorted(
            directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True
        )
        if not files:
            return None
        return str(files[0].relative_to(directory.parent))

    def _read_text(self, path: Path) -> str:
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def _read_latest_action(self, path: Path) -> str:
        if not path.is_file():
            return ""
        latest = ""
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                latest = str(payload.get("name", "")).strip()
        return latest

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        return rows

    def _read_latest_metrics(self, run_root: Path) -> dict[str, Any]:
        latest: dict[str, Any] = {}
        for payload in self._read_jsonl(run_root / "events.jsonl"):
            if payload.get("type") == "turn.metrics":
                latest = payload.get("metrics", {}) or {}
        return latest

    def _read_turns(self, run_root: Path) -> list[RunTurn]:
        turns: list[RunTurn] = []
        for payload in self._read_jsonl(run_root / "stt" / "commits.jsonl"):
            turn_id = str(payload.get("turn_id", "")).strip()
            audio_path = None
            if turn_id and (run_root / "input" / f"{turn_id}.wav").is_file():
                audio_path = f"input/{turn_id}.wav"
            turns.append(
                RunTurn(
                    turn_id=turn_id,
                    text=str(payload.get("text", "")).strip(),
                    started_at_ms=self._coerce_float(payload.get("started_at_ms")),
                    committed_at_ms=self._coerce_float(payload.get("committed_at_ms")),
                    audio_path=audio_path,
                )
            )
        return turns

    def _read_timeline(self, run_root: Path, *, updated_at_ms: int) -> list[RunEvent]:
        timeline: list[RunEvent] = []
        for payload in self._read_jsonl(run_root / "events.jsonl"):
            event_type = str(payload.get("type", "")).strip()
            if event_type in {
                "conversation.item.input_audio_transcription.delta",
                "response.text.delta",
                "response.audio.delta",
            }:
                continue
            recorded_at_ms = int(payload.get("recorded_at_ms") or updated_at_ms)
            detail = self._summarize_event(payload)
            timeline.append(
                RunEvent(
                    recorded_at_ms=recorded_at_ms,
                    type=event_type or "event",
                    detail=detail,
                )
            )
        return timeline

    def _summarize_event(self, payload: dict[str, Any]) -> str:
        event_type = str(payload.get("type", "")).strip()
        if event_type == "session.started":
            return "Session started."
        if event_type == "session.updated":
            session = payload.get("session", {}) or {}
            voice = str(session.get("voice", "")).strip()
            audio_format = str(session.get("output_audio_format", "")).strip()
            parts = [
                part
                for part in (
                    voice and f"Voice {voice}",
                    audio_format and f"Output {audio_format}",
                )
                if part
            ]
            return " · ".join(parts) or "Session updated."
        if event_type == "response.created":
            return f"Response {str((payload.get('response') or {}).get('id', '')).strip()[:8]} created."
        if event_type == "response.action":
            action = payload.get("action", {}) or {}
            name = str(action.get("name", "")).strip() or "unknown"
            mode = str(action.get("mode", "")).strip()
            return f"LLM chose {name}{f' ({mode})' if mode else ''}."
        if event_type == "response.audio.done":
            return "Speech output finished."
        if event_type == "turn.metrics":
            metrics = payload.get("metrics", {}) or {}
            parts: list[str] = []
            for label, key, unit in (
                ("STT final", "stt_final_ms", "ms"),
                ("LLM TTFT", "llm_ttft_ms", "ms"),
                ("TTS first audio", "tts_first_audio_ms", "ms"),
                ("Turn", "turn_total_ms", "ms"),
            ):
                value = metrics.get(key)
                if value is None:
                    continue
                parts.append(f"{label} {value}{unit}")
            return " · ".join(parts) or "Turn metrics captured."
        if event_type == "error":
            return (
                str(payload.get("message", "Pipeline error")).strip()
                or "Pipeline error."
            )
        return event_type.replace(".", " ").strip() or "Event"

    def _coerce_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except TypeError, ValueError:
            return None
