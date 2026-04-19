#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def lookup(
    env: dict[str, str], dotenv: dict[str, str], *names: str, default: str | None = None
) -> str | None:
    for name in names:
        value = env.get(name)
        if value:
            return value
    for name in names:
        value = dotenv.get(name)
        if value:
            return value
    return default


def resolve_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def main() -> int:
    env = dict(os.environ)
    dotenv = load_dotenv(ROOT / ".env")
    failures: list[str] = []

    def ok(message: str) -> None:
        print(f"[ok] {message}")

    def fail(message: str) -> None:
        failures.append(message)
        print(f"[fail] {message}")

    def check(condition: bool, success: str, failure: str) -> None:
        if condition:
            ok(success)
        else:
            fail(failure)

    for tool in ("python3", "uv", "node", "npm"):
        tool_path = shutil.which(tool)
        check(
            tool_path is not None,
            f"{tool} found at {tool_path}",
            f"{tool} is required on PATH",
        )
    python314 = shutil.which("python3.14")
    if python314:
        ok(f"python3.14 found at {python314}")
    else:
        print(
            "[warn] python3.14 is not on PATH; uv will need to supply Python 3.14 for the API project."
        )

    just_path = shutil.which("just")
    if just_path:
        ok(f"just found at {just_path}")
    else:
        print("[warn] just is not on PATH; you can still run the scripts directly.")

    check(
        (ROOT / "apps/api/pyproject.toml").is_file(),
        "apps/api present",
        "apps/api/pyproject.toml is missing",
    )
    check(
        (ROOT / "apps/web/package.json").is_file(),
        "apps/web present",
        "apps/web/package.json is missing",
    )
    check(
        (ROOT / "prompts/investor-judge.md").is_file(),
        "default persona prompt present",
        "prompts/investor-judge.md is missing",
    )
    check(
        (ROOT / "samples/sample_input_synthetic.wav").is_file(),
        "public synthetic sample present",
        "samples/sample_input_synthetic.wav is missing",
    )
    check(
        (ROOT / "apps/web/node_modules").is_dir(),
        "frontend dependencies installed",
        "apps/web/node_modules is missing; run just setup",
    )

    provider = (
        (
            lookup(
                env,
                dotenv,
                "PROVIDER",
                "PIPELINE_PROVIDER",
                "VOLTA_PIPELINE_PROVIDER",
                default="openai",
            )
            or "openai"
        )
        .strip()
        .lower()
    )
    ok(f"provider set to {provider}")

    if provider == "openai":
        api_key = lookup(env, dotenv, "OPENAI_API_KEY", "VOLTA_OPENAI_API_KEY")
        check(
            bool(api_key),
            "OPENAI_API_KEY is configured",
            "Set OPENAI_API_KEY in .env for PROVIDER=openai",
        )
        model = lookup(
            env,
            dotenv,
            "OPENAI_LLM_MODEL",
            "VOLTA_OPENAI_LLM_MODEL",
            default="gpt-5.4-mini",
        )
        ok(f"OpenAI LLM model: {model}")
    elif provider == "local":
        runtime_checks = [
            ("STT_PYTHON", "VOLTA_STT_PYTHON", True, "local STT runtime"),
            ("LLM_PYTHON", "VOLTA_LLM_PYTHON", True, "local LLM runtime"),
            ("TTS_PYTHON", "VOLTA_TTS_PYTHON", True, "local TTS runtime"),
            ("LLM_MODEL_PATH", "VOLTA_LLM_MODEL_PATH", False, "local LLM model path"),
        ]
        for plain, prefixed, executable, label in runtime_checks:
            resolved = resolve_path(lookup(env, dotenv, plain, prefixed))
            if resolved is None:
                fail(f"Set {plain} in .env for PROVIDER=local ({label})")
                continue
            if executable:
                check(
                    resolved.is_file() and os.access(resolved, os.X_OK),
                    f"{label} ready at {resolved}",
                    f"{label} is not executable: {resolved}",
                )
            else:
                check(
                    resolved.exists(),
                    f"{label} ready at {resolved}",
                    f"{label} does not exist: {resolved}",
                )
    else:
        fail(f"Unsupported PROVIDER value: {provider}")

    if failures:
        print("\nDoctor failed. Fix the items above and rerun `just doctor`.")
        return 1

    print("\nDoctor passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
