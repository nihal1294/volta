from __future__ import annotations

import uuid

from .runtime_types import TranscriptSnapshot, UserTurnCommit


def common_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    prefix = left[:index].rstrip()
    if not prefix:
        return ""
    if index < len(left) and not left[index - 1].isspace():
        split = prefix.rfind(" ")
        if split > 0:
            return prefix[:split].rstrip()
    return prefix


class TranscriptStabilizer:
    def __init__(self, stability_window_ms: int, max_open_ms: int = 8000) -> None:
        self.snapshot = TranscriptSnapshot()
        self.last_change_ms = 0.0
        self.utterance_opened_ms = 0.0
        self.stability_window_ms = stability_window_ms
        self.max_open_ms = max_open_ms
        self.short_utterance_window_ms = max(stability_window_ms * 3, 1200)

    def observe(self, text: str, now_ms: float) -> TranscriptSnapshot:
        normalized = text.strip()
        previous = self.snapshot.raw_text
        if normalized != previous:
            self.last_change_ms = now_ms
            if not previous:
                self.utterance_opened_ms = now_ms

        stable_prefix = common_prefix(previous, normalized)
        unstable_suffix = (
            normalized[len(stable_prefix) :].strip() if stable_prefix else normalized
        )
        if (
            normalized
            and normalized == previous
            and now_ms - self.last_change_ms >= self.stability_window_ms
        ):
            stable_prefix = normalized
            unstable_suffix = ""

        self.snapshot = TranscriptSnapshot(
            raw_text=normalized,
            stable_text=stable_prefix,
            unstable_text=unstable_suffix,
            revision=self.snapshot.revision + 1,
        )
        return self.snapshot

    def maybe_commit(self, now_ms: float) -> UserTurnCommit | None:
        if not self.snapshot.raw_text:
            return None
        stable_for_ms = now_ms - self.last_change_ms
        open_for_ms = now_ms - self.utterance_opened_ms
        if stable_for_ms < self.stability_window_ms and open_for_ms < self.max_open_ms:
            return None

        committed_text = self.snapshot.stable_text or self.snapshot.raw_text
        if (
            open_for_ms < self.max_open_ms
            and stable_for_ms < self.short_utterance_window_ms
            and not self._is_commit_ready(committed_text)
        ):
            return None

        commit = UserTurnCommit(
            turn_id=str(uuid.uuid4()),
            text=committed_text.strip(),
            started_at_ms=self.utterance_opened_ms,
            committed_at_ms=now_ms,
        )
        self.reset()
        return commit

    def reset(self) -> None:
        self.snapshot = TranscriptSnapshot()
        self.last_change_ms = 0.0
        self.utterance_opened_ms = 0.0

    def _is_commit_ready(self, text: str) -> bool:
        normalized = text.strip()
        if not normalized:
            return False
        if normalized[-1] in ".!?;:":
            return True
        if len(normalized) >= 12:
            return True
        if len(normalized.split()) >= 3:
            return True
        return False
