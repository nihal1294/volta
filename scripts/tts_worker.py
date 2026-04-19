#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time

import numpy as np
from mlx_audio.tts.utils import load_model

from worker_common import (
    float32_to_pcm16_b64,
    write_done,
    write_error,
    write_event,
    write_ready,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        model = load_model(model_path=args.model_path)
    write_ready("tts")

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        request = json.loads(raw_line)
        request_id = request["request_id"]
        action = request["action"]
        payload = request.get("payload", {})
        try:
            if action == "health":
                write_done(request_id, {"ok": True, "sample_rate": model.sample_rate})
                continue

            if action == "synthesize":
                started = time.perf_counter()
                total_samples = 0
                last_rtf = None
                for result in model.generate(
                    text=payload["text"],
                    voice=payload.get("voice"),
                    instruct=payload.get("instruct"),
                    stream=bool(payload.get("stream", True)),
                    streaming_interval=float(payload.get("streaming_interval", 0.5)),
                    verbose=False,
                ):
                    audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                    total_samples += audio.shape[0]
                    last_rtf = result.real_time_factor
                    write_event(
                        request_id,
                        "audio",
                        {
                            "pcm16_b64": float32_to_pcm16_b64(audio),
                            "sample_rate": result.sample_rate,
                            "samples": int(audio.shape[0]),
                            "segment_idx": result.segment_idx,
                            "real_time_factor": result.real_time_factor,
                            "is_final_chunk": result.is_final_chunk,
                        },
                    )
                elapsed = time.perf_counter() - started
                duration = (
                    total_samples / model.sample_rate if model.sample_rate else 0.0
                )
                write_event(
                    request_id,
                    "metrics",
                    {
                        "processing_seconds": round(elapsed, 3),
                        "audio_seconds": round(duration, 3),
                        "real_time_factor": last_rtf if last_rtf is not None else None,
                    },
                )
                write_done(request_id, {"ok": True})
                continue

            write_error(request_id, f"Unknown action: {action}")
        except Exception as exc:  # noqa: BLE001
            write_error(request_id, str(exc))


if __name__ == "__main__":
    main()
