from __future__ import annotations

from volta_api.transcript_stabilizer import TranscriptStabilizer


def test_stabilizer_extracts_stable_prefix() -> None:
    stabilizer = TranscriptStabilizer(stability_window_ms=400)
    stabilizer.observe("hello there", now_ms=0)
    stabilizer.observe("hello there investor", now_ms=250)
    snapshot = stabilizer.observe("hello there investor", now_ms=700)

    assert snapshot.stable_text == "hello there investor"
    assert snapshot.unstable_text == ""


def test_stabilizer_commits_after_stable_pause() -> None:
    stabilizer = TranscriptStabilizer(stability_window_ms=400)
    stabilizer.observe("give me the numbers", now_ms=0)
    stabilizer.observe("give me the numbers", now_ms=500)

    commit = stabilizer.maybe_commit(now_ms=700)
    assert commit is not None
    assert commit.text == "give me the numbers"


def test_stabilizer_does_not_commit_tiny_fragment_too_early() -> None:
    stabilizer = TranscriptStabilizer(stability_window_ms=400)
    stabilizer.observe("no", now_ms=0)
    stabilizer.observe("no", now_ms=500)

    assert stabilizer.maybe_commit(now_ms=700) is None


def test_stabilizer_eventually_commits_tiny_fragment_after_long_pause() -> None:
    stabilizer = TranscriptStabilizer(stability_window_ms=400)
    stabilizer.observe("no", now_ms=0)
    stabilizer.observe("no", now_ms=500)

    commit = stabilizer.maybe_commit(now_ms=1600)
    assert commit is not None
    assert commit.text == "no"


def test_stabilizer_commits_short_punctuated_phrase() -> None:
    stabilizer = TranscriptStabilizer(stability_window_ms=400)
    stabilizer.observe("no.", now_ms=0)
    stabilizer.observe("no.", now_ms=500)

    commit = stabilizer.maybe_commit(now_ms=700)
    assert commit is not None
    assert commit.text == "no."
