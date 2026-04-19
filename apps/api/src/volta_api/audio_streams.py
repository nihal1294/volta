from __future__ import annotations

import base64

import numpy as np
import sphn


SAMPLE_RATE = 24_000


def pcm16_b64_to_float32_array(data: str) -> np.ndarray:
    pcm = np.frombuffer(base64.b64decode(data), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


def float32_to_pcm16_b64(audio: np.ndarray) -> str:
    clipped = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return np.asarray(audio, dtype=np.float32)
    duration = audio.shape[0] / src_rate
    dst_samples = max(1, int(round(duration * dst_rate)))
    src_positions = np.linspace(
        0, audio.shape[0] - 1, num=audio.shape[0], dtype=np.float32
    )
    dst_positions = np.linspace(
        0, audio.shape[0] - 1, num=dst_samples, dtype=np.float32
    )
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)


class OpusInputStream:
    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._reader = sphn.OpusStreamReader(sample_rate)

    def append_b64(self, data: str) -> np.ndarray:
        decoded = self._reader.append_bytes(base64.b64decode(data))
        return np.asarray(decoded, dtype=np.float32).reshape(-1)


class OpusOutputStream:
    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._writer = sphn.OpusStreamWriter(sample_rate)

    def encode_b64_packets(self, audio: np.ndarray) -> list[str]:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return []
        packet = self._writer.append_pcm(samples)
        if not packet:
            return []
        return [base64.b64encode(packet).decode("utf-8")]
