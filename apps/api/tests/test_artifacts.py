from __future__ import annotations

import json
from pathlib import Path

from volta_api.artifacts import ArtifactStore


def test_artifact_store_creates_expected_paths(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    run = store.open_run(session_id="session-123")

    assert run.root == tmp_path / "session-123"
    assert (run.root / "input").is_dir()
    assert (run.root / "stt").is_dir()
    assert (run.root / "llm").is_dir()
    assert (run.root / "tts").is_dir()
    assert (run.root / "tts" / "chunks").is_dir()


def test_artifact_store_appends_jsonl_events(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    run = store.open_run(session_id="session-123")
    store.append_event(run, "session.started", {"seq": 1})
    store.append_event(run, "session.started", {"seq": 2})

    lines = (run.root / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["type"] == "session.started"
    assert second["type"] == "session.started"
    assert first["seq"] == 1
    assert second["seq"] == 2
    assert isinstance(first["recorded_at_ms"], int)
    assert isinstance(second["recorded_at_ms"], int)


def test_artifact_store_lists_run_summaries(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    run = store.open_run(session_id="session-123")
    store.append_jsonl(
        run, "stt/commits.jsonl", {"turn_id": "turn-1", "text": "tell me"}
    )
    store.append_jsonl(
        run, "stt/commits.jsonl", {"turn_id": "turn-2", "text": "the numbers"}
    )
    store.write_text(run, "llm/final_output.txt", "Your unit economics are weak.")
    store.append_jsonl(run, "llm/floor_actions.jsonl", {"name": "speak"})
    store.write_text(run, "input/source.wav", "not-a-real-wav")

    summaries = store.list_runs()

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.session_id == "session-123"
    assert summary.transcript == "tell me\nthe numbers"
    assert summary.llm_output == "Your unit economics are weak."
    assert summary.latest_action == "speak"
    assert summary.input_audio_path == "input/source.wav"


def test_artifact_store_filters_empty_runs(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    empty_run = store.open_run(session_id="empty-run")
    store.append_event(empty_run, "session.started", {"session_id": "empty-run"})

    full_run = store.open_run(session_id="full-run")
    store.append_event(full_run, "session.started", {"session_id": "full-run"})
    store.write_text(full_run, "stt/final_transcript.txt", "show me the revenue")

    summaries = store.list_runs()

    assert [summary.session_id for summary in summaries] == ["full-run"]


def test_artifact_store_reads_run_detail(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    run = store.open_run(session_id="session-123")
    store.append_event(run, "session.started", {"session_id": "session-123"})
    store.append_event(
        run,
        "turn.metrics",
        {
            "metrics": {
                "stt_final_ms": 1200.0,
                "llm_ttft_ms": 800.0,
            }
        },
    )
    store.append_jsonl(
        run,
        "stt/commits.jsonl",
        {
            "turn_id": "turn-1",
            "text": "show me the numbers",
            "started_at_ms": 10.0,
            "committed_at_ms": 40.0,
        },
    )
    store.write_text(run, "llm/final_output.txt", "Those numbers do not work.")
    store.append_jsonl(run, "llm/floor_actions.jsonl", {"name": "speak"})
    store.write_text(run, "input/full_input.wav", "not-a-real-wav")
    store.write_text(run, "input/turn-1.wav", "not-a-real-wav")
    store.write_text(run, "tts/final_output.wav", "not-a-real-wav")

    detail = store.read_run_detail("session-123")

    assert detail is not None
    assert detail.input_audio_path == "input/full_input.wav"
    assert detail.transcript == "show me the numbers"
    assert detail.metrics == {"stt_final_ms": 1200.0, "llm_ttft_ms": 800.0}
    assert detail.turns[0].turn_id == "turn-1"
    assert detail.turns[0].audio_path == "input/turn-1.wav"
    assert detail.timeline[0].type == "session.started"
    assert detail.timeline[1].type == "turn.metrics"
