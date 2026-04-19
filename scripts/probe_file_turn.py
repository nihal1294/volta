#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
from typing import Any

import websockets


async def run_probe(
    audio_path: Path,
    ws_url: str,
    voice: str,
    instruct: str,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    final_transcript: str | None = None
    llm_action: str | None = None
    llm_text = ""
    first_tts_chunk: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None

    async with websockets.connect(ws_url, max_size=10_000_000) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.start",
                    "mode": "file",
                    "voice": voice,
                    "instruct": instruct,
                }
            )
        )
        data = audio_path.read_bytes()
        await ws.send(
            json.dumps(
                {
                    "type": "file.input.start",
                    "file_name": audio_path.name,
                    "mime_type": "audio/mpeg",
                }
            )
        )
        chunk_size = 64 * 1024
        for cursor in range(0, len(data), chunk_size):
            chunk_b64 = base64.b64encode(data[cursor : cursor + chunk_size]).decode(
                "ascii"
            )
            await ws.send(
                json.dumps({"type": "file.input.chunk", "chunk_b64": chunk_b64})
            )
        await ws.send(json.dumps({"type": "file.input.end"}))

        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=300))
            event_type = message["type"]
            counts[event_type] = counts.get(event_type, 0) + 1

            if event_type == "stt.final":
                final_transcript = message["text"]
            elif event_type == "llm.action":
                llm_action = message["name"]
            elif event_type == "llm.token":
                llm_text += message["text"]
            elif event_type == "tts.audio.chunk" and first_tts_chunk is None:
                first_tts_chunk = {
                    key: message.get(key)
                    for key in (
                        "sample_rate",
                        "samples",
                        "segment_idx",
                        "real_time_factor",
                        "is_final_chunk",
                    )
                }
            elif event_type == "turn.summary":
                metrics = message["metrics"]
            elif event_type == "turn.done":
                break

    return {
        "audio_path": str(audio_path),
        "counts": counts,
        "stt_final": final_transcript,
        "llm_action": llm_action,
        "llm_text": llm_text,
        "first_tts_chunk": first_tts_chunk,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8765/v1/realtime")
    parser.add_argument("--voice", default="Ryan")
    parser.add_argument(
        "--instruct",
        default="Very angry, aggressive, intense, raised voice, confrontational delivery, emotionally expressive delivery.",
    )
    args = parser.parse_args()

    result = asyncio.run(
        run_probe(
            audio_path=args.audio_path.expanduser().resolve(),
            ws_url=args.ws_url,
            voice=args.voice,
            instruct=args.instruct,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
