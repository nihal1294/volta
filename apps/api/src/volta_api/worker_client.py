from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Protocol


class WorkerLike(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def health(self) -> dict[str, Any]: ...

    async def request_once(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]: ...

    async def request_stream(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]: ...


class WorkerClient:
    def __init__(
        self, name: str, python_path: Path, script_path: Path, args: list[str]
    ):
        self.name = name
        self.python_path = python_path
        self.script_path = script_path
        self.args = args
        self.process: asyncio.subprocess.Process | None = None
        self._ready = asyncio.Event()
        self._pending: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()

    async def start(self) -> None:
        if self.process is not None:
            return
        self.process = await asyncio.create_subprocess_exec(
            str(self.python_path),
            str(self.script_path),
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stdout_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())
        await asyncio.wait_for(self._ready.wait(), timeout=180)

    async def stop(self) -> None:
        if self.process is None:
            return
        try:
            self.process.terminate()
        except ProcessLookupError:
            self.process = None
            return
        await self.process.wait()
        self.process = None

    async def health(self) -> dict[str, Any]:
        return await self.request_once("health", {})

    async def request_once(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError(f"Worker {self.name} is not started.")
        request_id = str(uuid.uuid4())
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending[request_id] = queue
        request = {"request_id": request_id, "action": action, "payload": payload}
        async with self._write_lock:
            self.process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
            await self.process.stdin.drain()

        try:
            while True:
                if timeout is None:
                    message = await queue.get()
                else:
                    message = await asyncio.wait_for(queue.get(), timeout=timeout)
                if message.get("event") == "done":
                    return message.get("data", {})
                if message.get("event") == "error":
                    raise RuntimeError(
                        f"{self.name} worker error: {message.get('error', 'unknown')}"
                    )
        finally:
            self._pending.pop(request_id, None)

    async def request_stream(
        self,
        action: str,
        payload: dict[str, Any],
        timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError(f"Worker {self.name} is not started.")
        request_id = str(uuid.uuid4())
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending[request_id] = queue
        request = {"request_id": request_id, "action": action, "payload": payload}
        async with self._write_lock:
            self.process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
            await self.process.stdin.drain()

        try:
            while True:
                if timeout is None:
                    message = await queue.get()
                else:
                    message = await asyncio.wait_for(queue.get(), timeout=timeout)
                if message.get("event") == "done":
                    break
                if message.get("event") == "error":
                    raise RuntimeError(
                        f"{self.name} worker error: {message.get('error', 'unknown')}"
                    )
                yield message
        finally:
            self._pending.pop(request_id, None)

    async def _read_stdout(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        while line := await self.process.stdout.readline():
            text = line.decode("utf-8").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError:
                sys.stderr.write(f"[{self.name}:stdout] {text}\n")
                continue
            request_id = message.get("request_id")
            if message.get("type") == "ready":
                self._ready.set()
                continue
            if request_id and request_id in self._pending:
                await self._pending[request_id].put(message)

    async def _read_stderr(self) -> None:
        assert self.process is not None
        assert self.process.stderr is not None
        while line := await self.process.stderr.readline():
            sys.stderr.write(f"[{self.name}] {line.decode('utf-8')}")
