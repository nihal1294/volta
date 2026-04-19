#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import rustymimi
import sentencepiece
import sphn
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils
from stt_stream_protocol import StreamSessionProtocol

from worker_common import (
    pcm16_b64_to_float32_array,
    resample_linear,
    write_done,
    write_error,
    write_event,
    write_ready,
)


BLOCK_SIZE = 1920
FLUSH_SAMPLES = 48000


@dataclass
class SessionState:
    gen: models.LmGen
    audio_tokenizer: rustymimi.Tokenizer
    text_tokenizer: sentencepiece.SentencePieceProcessor
    other_codebooks: int
    remainder: np.ndarray
    transcript: str
    start_time: float
    first_partial_ms: float | None = None
    first_word_ms: float | None = None


def create_generator(model: models.Lm) -> models.LmGen:
    return models.LmGen(
        model=model,
        max_steps=4096,
        text_sampler=utils.Sampler(top_k=25, temp=0),
        audio_sampler=utils.Sampler(top_k=250, temp=0.8),
        check=False,
    )


def decode_block(
    state: SessionState,
    block: np.ndarray,
) -> str:
    block_arr = block.reshape(1, 1, -1).astype(np.float32)
    other_audio_tokens = state.audio_tokenizer.encode_step(block_arr)
    other_audio_tokens = mx.array(other_audio_tokens).transpose(0, 2, 1)[
        :, :, : state.other_codebooks
    ]
    text_token = state.gen.step(other_audio_tokens[0])[0].item()
    if text_token in (0, 3):
        return ""
    piece = state.text_tokenizer.id_to_piece(text_token)
    return piece.replace("▁", " ")


def process_audio_chunk(state: SessionState, samples: np.ndarray) -> str:
    if samples.size == 0:
        return ""
    merged = np.concatenate([state.remainder, samples]).astype(np.float32)
    outputs: list[str] = []
    usable = (merged.shape[0] // BLOCK_SIZE) * BLOCK_SIZE
    for start_idx in range(0, usable, BLOCK_SIZE):
        piece = decode_block(state, merged[start_idx : start_idx + BLOCK_SIZE])
        if piece:
            outputs.append(piece)
            if state.first_partial_ms is None:
                state.first_partial_ms = round(
                    (time.perf_counter() - state.start_time) * 1000, 1
                )
            if state.first_word_ms is None and piece.strip():
                state.first_word_ms = round(
                    (time.perf_counter() - state.start_time) * 1000, 1
                )
    state.remainder = merged[usable:]
    chunk_text = "".join(outputs)
    if chunk_text:
        state.transcript += chunk_text
    return chunk_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="kyutai/stt-1b-en_fr-mlx")
    args = parser.parse_args()

    config_path = hf_hub_download(args.hf_repo, "config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = json.load(handle)

    mimi_weights = hf_hub_download(args.hf_repo, raw_config["mimi_name"])
    moshi_name = raw_config.get("moshi_name", "model.safetensors")
    moshi_weights = hf_hub_download(args.hf_repo, moshi_name)
    tokenizer_path = hf_hub_download(args.hf_repo, raw_config["tokenizer_name"])

    lm_config = models.LmConfig.from_config_dict(raw_config)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if moshi_weights.endswith(".q4.safetensors"):
        nn.quantize(model, bits=4, group_size=32)
    elif moshi_weights.endswith(".q8.safetensors"):
        nn.quantize(model, bits=8, group_size=64)
    model.load_weights(moshi_weights, strict=True)
    model.warmup()

    generated_codebooks = lm_config.generated_codebooks
    other_codebooks = lm_config.other_codebooks
    mimi_codebooks = max(generated_codebooks, other_codebooks)
    audio_tokenizer = rustymimi.Tokenizer(mimi_weights, num_codebooks=mimi_codebooks)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    protocol = StreamSessionProtocol()
    write_ready("stt")

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
                write_done(request_id, {"ok": True})
                continue

            if action in {"stream_start", "session_start"}:
                stream_id = payload.get("stream_id") or payload.get("session_id")
                if not stream_id:
                    write_done(request_id, {"ok": False, "ignored": True})
                    continue
                protocol.start(
                    stream_id,
                    SessionState(
                        gen=create_generator(model),
                        audio_tokenizer=audio_tokenizer,
                        text_tokenizer=text_tokenizer,
                        other_codebooks=other_codebooks,
                        remainder=np.array([], dtype=np.float32),
                        transcript="",
                        start_time=time.perf_counter(),
                    ),
                )
                write_done(request_id, {"ok": True})
                continue

            if action in {"stream_chunk", "session_chunk"}:
                stream_id = payload.get("stream_id") or payload.get("session_id")
                session = protocol.sessions.get(stream_id)
                if session is None:
                    write_done(request_id, {"ok": False, "ignored": True})
                    continue
                audio = pcm16_b64_to_float32_array(payload["pcm_b64"])
                audio = resample_linear(audio, int(payload["sample_rate"]), 24000)
                text = process_audio_chunk(session, audio)
                if text:
                    write_event(
                        request_id,
                        "partial",
                        {
                            "text": session.transcript,
                            "delta": text,
                            "first_partial_ms": session.first_partial_ms,
                            "first_word_ms": session.first_word_ms,
                        },
                    )
                write_done(request_id, {"ok": True})
                continue

            if action == "stream_snapshot":
                stream_id = payload.get("stream_id") or payload.get("session_id")
                write_done(request_id, protocol.snapshot(stream_id))
                continue

            if action in {"stream_commit", "session_end"}:
                stream_id = payload.get("stream_id") or payload.get("session_id")
                session = protocol.sessions.get(stream_id)
                if session is None:
                    write_done(request_id, {"ok": False, "ignored": True, "text": ""})
                    continue
                flush = np.zeros(FLUSH_SAMPLES, dtype=np.float32)
                process_audio_chunk(session, flush)
                write_done(request_id, protocol.commit(stream_id))
                continue

            if action in {"stream_reset", "session_discard"}:
                stream_id = payload.get("stream_id") or payload.get("session_id")
                write_done(request_id, protocol.reset(stream_id))
                continue

            if action == "transcribe_file":
                audio, _ = sphn.read(payload["path"], sample_rate=24000)
                session = SessionState(
                    gen=create_generator(model),
                    audio_tokenizer=audio_tokenizer,
                    text_tokenizer=text_tokenizer,
                    other_codebooks=other_codebooks,
                    remainder=np.array([], dtype=np.float32),
                    transcript="",
                    start_time=time.perf_counter(),
                )
                samples = np.array(audio).reshape(-1).astype(np.float32)
                cursor = 0
                while cursor < samples.shape[0]:
                    chunk_text = process_audio_chunk(
                        session, samples[cursor : cursor + BLOCK_SIZE * 4]
                    )
                    if chunk_text:
                        write_event(
                            request_id,
                            "partial",
                            {
                                "text": session.transcript,
                                "delta": chunk_text,
                                "first_partial_ms": session.first_partial_ms,
                                "first_word_ms": session.first_word_ms,
                            },
                        )
                    cursor += BLOCK_SIZE * 4
                process_audio_chunk(session, np.zeros(FLUSH_SAMPLES, dtype=np.float32))
                elapsed = time.perf_counter() - session.start_time
                words = len(session.transcript.split())
                write_event(
                    request_id,
                    "final",
                    {
                        "text": session.transcript.strip(),
                        "first_partial_ms": session.first_partial_ms,
                        "first_word_ms": session.first_word_ms,
                        "processing_ms": round(elapsed * 1000, 1),
                        "words_per_sec": round(words / elapsed, 3)
                        if elapsed > 0
                        else 0.0,
                    },
                )
                write_done(request_id, {"ok": True})
                continue

            write_error(request_id, f"Unknown action: {action}")
        except Exception as exc:  # noqa: BLE001
            write_error(request_id, str(exc))


if __name__ == "__main__":
    main()
