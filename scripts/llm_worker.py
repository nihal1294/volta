#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from worker_common import write_message


PROTOCOL_PROMPT = """You are a realtime voice assistant inside a local STT -> LLM -> TTS pipeline.
You must choose exactly one primary action and emit it on the first line in this exact format:
ACTION {"name":"<action>","mode":"stream","interruptible":true}

Allowed actions:
1. wait
   - Use when you want more stable user transcript before taking the floor.
   - Do not include spoken text after the action line.
2. yield_to_user
   - Use when the user is clearly still taking the floor or barging in.
   - Do not include spoken text after the action line.
3. continue_speaking
   - Use when the assistant should keep talking in the current floor-taking direction.
   - Include `mode` as either `stream` or `turn`.
   - Include `interruptible` as true or false.
   - Spoken text must start on the next line after the ACTION line.
4. speak
   - Use when you want the system to speak to the user.
   - Include `mode` as either `stream` or `turn`.
   - Include `interruptible` as true or false.
   - Spoken text must start on the next line after the ACTION line.
5. hold_silence
   - Use when the input is empty, just noise, or a reply is inappropriate.
   - Do not include spoken text after the action line.
6. end_turn
   - Use only as a non-speaking terminator when needed.

Additional rules:
- Do not use markdown.
- Do not emit transport or device commands.
- Keep replies concise unless the user explicitly asked for detail.
- `wait`, `yield_to_user`, `hold_silence`, and `end_turn` must not emit spoken text.
- `speak` and `continue_speaking` must emit spoken text on following lines.
"""

ACTION_LINE_RE = re.compile(r"^\s*ACTION\s+(\{.*\})\s*(?:\n|$)", re.DOTALL)


@dataclass
class SessionCacheState:
    prompt_cache: list[object]
    cached_tokens: list[int]


@dataclass
class ActiveGeneration:
    request_id: str
    session_id: str
    cancel_event: threading.Event
    thread: threading.Thread


OUTPUT_LOCK = threading.Lock()


def safe_write_message(message: dict[str, object]) -> None:
    with OUTPUT_LOCK:
        write_message(message)


def safe_write_ready(worker: str) -> None:
    safe_write_message({"type": "ready", "worker": worker})


def safe_write_error(request_id: str, error: str) -> None:
    safe_write_message({"request_id": request_id, "event": "error", "error": error})


def safe_write_done(request_id: str, data: dict[str, object] | None = None) -> None:
    safe_write_message({"request_id": request_id, "event": "done", "data": data or {}})


def safe_write_event(request_id: str, event: str, data: dict[str, object]) -> None:
    safe_write_message({"request_id": request_id, "event": event, "data": data})


def is_speaking_action(name: str) -> bool:
    return name in {"speak", "continue_speaking"}


def str2bool(string: str) -> bool:
    return string.strip().lower() not in {"false", "f", "0", "no", "off"}


def reasoning_enables_thinking(reasoning: str) -> bool:
    return reasoning.strip().lower() not in {"", "none", "false", "f", "0", "no", "off"}


def load_system_prompt(prompt_file: Path) -> str:
    persona_prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not persona_prompt:
        raise ValueError(f"System prompt file is empty: {prompt_file}")
    return compose_system_prompt(persona_prompt)


def compose_system_prompt(persona_prompt: str) -> str:
    return f"{persona_prompt.strip()}\n\n## Runtime protocol\n{PROTOCOL_PROMPT}"


def extract_action(buffer: str) -> tuple[dict[str, object] | None, str, bool]:
    match = ACTION_LINE_RE.match(buffer)
    if match:
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None, buffer, False
        rest = buffer[match.end() :].lstrip("\r\n")
        return payload, rest, True
    if buffer.lstrip().startswith("ACTION ") and "}" in buffer:
        candidate = buffer.split("}", 1)[0] + "}"
        try:
            payload = json.loads(candidate.split("ACTION", 1)[1].strip())
        except json.JSONDecodeError:
            return None, buffer, False
        rest = buffer[buffer.index("}") + 1 :].lstrip("\r\n")
        return payload, rest, True
    return None, buffer, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--system-prompt-file", required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--prompt-cache", type=str2bool, default=True)
    parser.add_argument("--reasoning", default="none")
    args = parser.parse_args()

    model, tokenizer = load(args.model_path)
    default_system_prompt = load_system_prompt(Path(args.system_prompt_file))
    session_states: dict[str, SessionCacheState] = {}
    state_lock = threading.Lock()
    active_generation_lock = threading.Lock()
    active_generation: ActiveGeneration | None = None
    enable_thinking = reasoning_enables_thinking(args.reasoning)

    def clear_active_generation(request_id: str) -> None:
        nonlocal active_generation
        with active_generation_lock:
            if active_generation and active_generation.request_id == request_id:
                active_generation = None

    def run_generate(
        request_id: str,
        payload: dict[str, object],
        cancel_event: threading.Event,
    ) -> None:
        session_id = str(payload.get("session_id", "default"))
        try:
            history = payload["history"]
            if not isinstance(history, list):
                raise ValueError("history must be a list")
            persona_text = str(payload.get("persona_text", "")).strip()
            system_prompt = (
                compose_system_prompt(persona_text)
                if persona_text
                else default_system_prompt
            )
            messages = [{"role": "system", "content": system_prompt}, *history]
            prompt_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            active_prompt_cache = make_prompt_cache(model)
            prompt = prompt_tokens
            reused_prefix_tokens = 0
            if args.prompt_cache:
                with state_lock:
                    session_state = session_states.get(session_id)
                if (
                    session_state
                    and prompt_tokens[: len(session_state.cached_tokens)]
                    == session_state.cached_tokens
                ):
                    reused_prefix_tokens = len(session_state.cached_tokens)
                    prompt = prompt_tokens[reused_prefix_tokens:]
                    active_prompt_cache = session_state.prompt_cache

            raw_buffer = ""
            action_sent = False
            speaking = False
            token_count = 0
            last_metrics = None
            assistant_segments: list[str] = []

            for response in stream_generate(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                sampler=make_sampler(temp=0.0),
                prompt_cache=active_prompt_cache,
            ):
                if cancel_event.is_set():
                    break
                last_metrics = {
                    "prompt_tokens": response.prompt_tokens,
                    "prompt_tps": response.prompt_tps,
                    "generation_tokens": response.generation_tokens,
                    "generation_tps": response.generation_tps,
                    "peak_memory_gb": response.peak_memory,
                    "prompt_cache_hit": reused_prefix_tokens > 0,
                    "reused_prefix_tokens": reused_prefix_tokens,
                }
                token_count = response.generation_tokens
                if not response.text:
                    continue
                raw_buffer += response.text
                if not action_sent:
                    parsed_action, remainder, complete = extract_action(raw_buffer)
                    if complete and parsed_action is not None:
                        current_action = {
                            "name": str(parsed_action.get("name", "speak")),
                            "mode": str(parsed_action.get("mode", "stream")),
                            "interruptible": bool(
                                parsed_action.get("interruptible", True)
                            ),
                        }
                        if "reason" in parsed_action:
                            current_action["reason"] = str(
                                parsed_action.get("reason", "")
                            )
                        action_sent = True
                        speaking = is_speaking_action(str(current_action["name"]))
                        safe_write_event(request_id, "action", current_action)
                        raw_buffer = remainder
                    elif len(raw_buffer) > 96 and not raw_buffer.lstrip().startswith(
                        "ACTION "
                    ):
                        action_sent = True
                        speaking = True
                        safe_write_event(
                            request_id,
                            "action",
                            {
                                "name": "speak",
                                "mode": "stream",
                                "interruptible": True,
                            },
                        )
                if cancel_event.is_set():
                    break
                if action_sent and speaking and raw_buffer:
                    assistant_segments.append(raw_buffer)
                    safe_write_event(request_id, "token", {"text": raw_buffer})
                    raw_buffer = ""

            if not cancel_event.is_set():
                if not action_sent:
                    stripped = raw_buffer.strip()
                    if not stripped:
                        safe_write_event(request_id, "action", {"name": "hold_silence"})
                    else:
                        safe_write_event(
                            request_id,
                            "action",
                            {
                                "name": "speak",
                                "mode": "stream",
                                "interruptible": True,
                            },
                        )
                        assistant_segments.append(stripped)
                        safe_write_event(request_id, "token", {"text": stripped})
                        speaking = True
                if last_metrics is not None:
                    safe_write_event(request_id, "metrics", last_metrics)
                if args.prompt_cache:
                    assistant_text = "".join(assistant_segments).strip()
                    with state_lock:
                        if speaking and assistant_text:
                            cached_messages = [
                                *messages,
                                {"role": "assistant", "content": assistant_text},
                            ]
                            cached_tokens = tokenizer.apply_chat_template(
                                cached_messages,
                                tokenize=True,
                                add_generation_prompt=False,
                                enable_thinking=enable_thinking,
                            )
                            session_states[session_id] = SessionCacheState(
                                prompt_cache=active_prompt_cache,
                                cached_tokens=list(cached_tokens),
                            )
                        elif session_id not in session_states:
                            pass
                safe_write_done(
                    request_id,
                    {"ok": True, "generation_tokens": token_count, "cancelled": False},
                )
            else:
                safe_write_done(
                    request_id,
                    {"ok": True, "generation_tokens": token_count, "cancelled": True},
                )
        except Exception as exc:  # noqa: BLE001
            safe_write_error(request_id, str(exc))
        finally:
            clear_active_generation(request_id)

    safe_write_ready("llm")

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
                safe_write_done(request_id, {"ok": True})
                continue

            if action == "session_reset":
                session_id = payload.get("session_id")
                if session_id:
                    with state_lock:
                        session_states.pop(str(session_id), None)
                safe_write_done(request_id, {"ok": True})
                continue

            if action == "cancel_generate":
                session_id = str(payload.get("session_id", ""))
                target_request_id = str(payload.get("target_request_id", ""))
                cancelled = False
                with active_generation_lock:
                    active = active_generation
                    if active and (
                        (session_id and active.session_id == session_id)
                        or (
                            target_request_id and active.request_id == target_request_id
                        )
                    ):
                        active.cancel_event.set()
                        cancelled = True
                safe_write_done(request_id, {"ok": True, "cancelled": cancelled})
                continue

            if action == "generate":
                with active_generation_lock:
                    if active_generation is not None:
                        safe_write_error(
                            request_id,
                            "llm worker error: generation already in progress",
                        )
                        continue
                    cancel_event = threading.Event()
                    generation_thread = threading.Thread(
                        target=run_generate,
                        args=(request_id, payload, cancel_event),
                        daemon=True,
                    )
                    active_generation = ActiveGeneration(
                        request_id=request_id,
                        session_id=str(payload.get("session_id", "default")),
                        cancel_event=cancel_event,
                        thread=generation_thread,
                    )
                    generation_thread.start()
                continue

            safe_write_error(request_id, f"Unknown action: {action}")
        except Exception as exc:  # noqa: BLE001
            safe_write_error(request_id, str(exc))


if __name__ == "__main__":
    main()
