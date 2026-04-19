from __future__ import annotations

import base64
import json
import sys
from typing import Any

import numpy as np


def write_message(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def write_ready(worker: str) -> None:
    write_message({"type": "ready", "worker": worker})


def write_error(request_id: str, error: str) -> None:
    write_message({"request_id": request_id, "event": "error", "error": error})


def write_done(request_id: str, data: dict[str, Any] | None = None) -> None:
    write_message({"request_id": request_id, "event": "done", "data": data or {}})


def write_event(request_id: str, event: str, data: dict[str, Any]) -> None:
    write_message({"request_id": request_id, "event": event, "data": data})


def pcm16_b64_to_float32_array(data: str) -> np.ndarray:
    pcm = np.frombuffer(base64.b64decode(data), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def float32_to_pcm16_b64(audio: np.ndarray) -> str:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio
    duration = audio.shape[0] / src_rate
    dst_samples = max(1, int(round(duration * dst_rate)))
    src_positions = np.linspace(
        0, audio.shape[0] - 1, num=audio.shape[0], dtype=np.float32
    )
    dst_positions = np.linspace(
        0, audio.shape[0] - 1, num=dst_samples, dtype=np.float32
    )
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)
