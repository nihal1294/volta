from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _load_dotenv(path: Path) -> dict[str, str]:
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
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def _config_get(
    env: Mapping[str, str],
    dotenv: Mapping[str, str],
    *names: str,
    default: str | None = None,
) -> str | None:
    for name in names:
        value = env.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    for name in names:
        value = dotenv.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def _config_bool(
    env: Mapping[str, str],
    dotenv: Mapping[str, str],
    *names: str,
    default: bool,
) -> bool:
    value = _config_get(env, dotenv, *names)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    workspace_root: Path
    api_root: Path
    scripts_root: Path
    prompts_root: Path
    uploads_root: Path
    artifact_root: Path
    fixture_root: Path
    save_full_input_audio: bool
    save_stt_outputs: bool
    save_llm_outputs: bool
    save_tts_outputs: bool
    pipeline_provider: str
    commit_stability_window_ms: int
    max_open_utterance_ms: int
    llm_reasoning: str | None
    openai_api_key: str | None
    openai_base_url: str
    openai_llm_model: str
    openai_llm_reasoning: str | None
    openai_transcription_model: str
    openai_transcription_language: str
    openai_transcription_prompt: str | None
    openai_transcription_min_interval_sec: float
    openai_transcription_partial_window_sec: float
    openai_transcription_context_chars: int
    openai_commit_stability_window_ms: int
    openai_max_open_utterance_ms: int
    openai_tts_model: str
    openai_tts_voice: str
    openai_tts_format: str
    stt_python: Path
    llm_python: Path
    tts_python: Path
    stt_model_repo: str
    llm_model_path: str
    llm_system_prompt_file: Path
    llm_max_tokens: int
    llm_prompt_cache_enabled: bool
    tts_model_path: str
    tts_voice: str
    tts_instruct: str
    tts_streaming_interval: float
    tts_chunk_soft_limit: int
    tts_chunk_min_split: int
    host: str
    port: int

    @property
    def use_openai_provider(self) -> bool:
        return self.pipeline_provider == "openai"

    @property
    def default_tts_voice(self) -> str:
        if self.use_openai_provider:
            return self.openai_tts_voice
        return self.tts_voice

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        dotenv_path: Path | None = None,
    ) -> "Settings":
        api_root = Path(__file__).resolve().parents[2]
        workspace_root = api_root.parents[1]
        scripts_root = workspace_root / "scripts"
        prompts_root = workspace_root / "prompts"
        uploads_root = workspace_root / ".runtime" / "uploads"
        artifact_root = workspace_root / ".runtime" / "runs"
        fixture_root = workspace_root / ".runtime" / "test-fixtures"
        local_runtime_root = workspace_root / ".local" / "providers" / "local"
        env_values = (
            dict(os.environ) if env is None else {k: str(v) for k, v in env.items()}
        )
        dotenv_values = _load_dotenv(dotenv_path or (workspace_root / ".env"))
        prompts_root.mkdir(parents=True, exist_ok=True)
        uploads_root.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
        fixture_root.mkdir(parents=True, exist_ok=True)
        return cls(
            workspace_root=workspace_root,
            api_root=api_root,
            scripts_root=scripts_root,
            prompts_root=prompts_root,
            uploads_root=uploads_root,
            artifact_root=artifact_root,
            fixture_root=fixture_root,
            save_full_input_audio=_config_bool(
                env_values,
                dotenv_values,
                "SAVE_FULL_INPUT_AUDIO",
                "VOLTA_SAVE_FULL_INPUT_AUDIO",
                default=True,
            ),
            save_stt_outputs=_config_bool(
                env_values,
                dotenv_values,
                "SAVE_STT_OUTPUTS",
                "VOLTA_SAVE_STT_OUTPUTS",
                default=True,
            ),
            save_llm_outputs=_config_bool(
                env_values,
                dotenv_values,
                "SAVE_LLM_OUTPUTS",
                "VOLTA_SAVE_LLM_OUTPUTS",
                default=True,
            ),
            save_tts_outputs=_config_bool(
                env_values,
                dotenv_values,
                "SAVE_TTS_OUTPUTS",
                "VOLTA_SAVE_TTS_OUTPUTS",
                default=True,
            ),
            pipeline_provider=_config_get(
                env_values,
                dotenv_values,
                "PROVIDER",
                "PIPELINE_PROVIDER",
                "VOLTA_PIPELINE_PROVIDER",
                default="openai",
            )
            .strip()
            .lower(),
            commit_stability_window_ms=int(
                _config_get(
                    env_values,
                    dotenv_values,
                    "COMMIT_STABILITY_WINDOW_MS",
                    "VOLTA_COMMIT_STABILITY_WINDOW_MS",
                    default="900",
                )
            ),
            max_open_utterance_ms=int(
                _config_get(
                    env_values,
                    dotenv_values,
                    "MAX_OPEN_UTTERANCE_MS",
                    "VOLTA_MAX_OPEN_UTTERANCE_MS",
                    default="8000",
                )
            ),
            llm_reasoning=_config_get(
                env_values,
                dotenv_values,
                "LLM_REASONING",
                "LOCAL_LLM_REASONING",
                "VOLTA_LLM_REASONING",
            ),
            openai_api_key=_config_get(
                env_values,
                dotenv_values,
                "OPENAI_API_KEY",
                "VOLTA_OPENAI_API_KEY",
            ),
            openai_base_url=_config_get(
                env_values,
                dotenv_values,
                "OPENAI_BASE_URL",
                "VOLTA_OPENAI_BASE_URL",
                default="https://api.openai.com/v1",
            ),
            openai_llm_model=_config_get(
                env_values,
                dotenv_values,
                "OPENAI_LLM_MODEL",
                "VOLTA_OPENAI_LLM_MODEL",
                default="gpt-5.4-mini",
            ),
            openai_llm_reasoning=_config_get(
                env_values,
                dotenv_values,
                "LLM_REASONING",
                "OPENAI_LLM_REASONING",
                "VOLTA_OPENAI_LLM_REASONING",
            ),
            openai_transcription_model=_config_get(
                env_values,
                dotenv_values,
                "OPENAI_TRANSCRIPTION_MODEL",
                "VOLTA_OPENAI_TRANSCRIPTION_MODEL",
                default="gpt-4o-transcribe",
            ),
            openai_transcription_language=_config_get(
                env_values,
                dotenv_values,
                "OPENAI_TRANSCRIPTION_LANGUAGE",
                "VOLTA_OPENAI_TRANSCRIPTION_LANGUAGE",
                default="en",
            ),
            openai_transcription_prompt=_config_get(
                env_values,
                dotenv_values,
                "OPENAI_TRANSCRIPTION_PROMPT",
                "VOLTA_OPENAI_TRANSCRIPTION_PROMPT",
            ),
            openai_transcription_min_interval_sec=float(
                _config_get(
                    env_values,
                    dotenv_values,
                    "OPENAI_TRANSCRIPTION_MIN_INTERVAL_SEC",
                    "VOLTA_OPENAI_TRANSCRIPTION_MIN_INTERVAL_SEC",
                    default="1.6",
                )
            ),
            openai_transcription_partial_window_sec=float(
                _config_get(
                    env_values,
                    dotenv_values,
                    "OPENAI_TRANSCRIPTION_PARTIAL_WINDOW_SEC",
                    "VOLTA_OPENAI_TRANSCRIPTION_PARTIAL_WINDOW_SEC",
                    default="6.0",
                )
            ),
            openai_transcription_context_chars=int(
                _config_get(
                    env_values,
                    dotenv_values,
                    "OPENAI_TRANSCRIPTION_CONTEXT_CHARS",
                    "VOLTA_OPENAI_TRANSCRIPTION_CONTEXT_CHARS",
                    default="220",
                )
            ),
            openai_commit_stability_window_ms=int(
                _config_get(
                    env_values,
                    dotenv_values,
                    "OPENAI_COMMIT_STABILITY_WINDOW_MS",
                    "VOLTA_OPENAI_COMMIT_STABILITY_WINDOW_MS",
                    default="5000",
                )
            ),
            openai_max_open_utterance_ms=int(
                _config_get(
                    env_values,
                    dotenv_values,
                    "OPENAI_MAX_OPEN_UTTERANCE_MS",
                    "VOLTA_OPENAI_MAX_OPEN_UTTERANCE_MS",
                    default="30000",
                )
            ),
            openai_tts_model=_config_get(
                env_values,
                dotenv_values,
                "OPENAI_TTS_MODEL",
                "VOLTA_OPENAI_TTS_MODEL",
                default="gpt-4o-mini-tts",
            ),
            openai_tts_voice=_config_get(
                env_values,
                dotenv_values,
                "OPENAI_TTS_VOICE",
                "VOLTA_OPENAI_TTS_VOICE",
                default="alloy",
            ),
            openai_tts_format=_config_get(
                env_values,
                dotenv_values,
                "OPENAI_TTS_FORMAT",
                "VOLTA_OPENAI_TTS_FORMAT",
                default="wav",
            ),
            stt_python=Path(
                _config_get(
                    env_values,
                    dotenv_values,
                    "STT_PYTHON",
                    "VOLTA_STT_PYTHON",
                    default=str(local_runtime_root / "stt-python"),
                )
            ),
            llm_python=Path(
                _config_get(
                    env_values,
                    dotenv_values,
                    "LLM_PYTHON",
                    "VOLTA_LLM_PYTHON",
                    default=str(local_runtime_root / "llm-python"),
                )
            ),
            tts_python=Path(
                _config_get(
                    env_values,
                    dotenv_values,
                    "TTS_PYTHON",
                    "VOLTA_TTS_PYTHON",
                    default=str(local_runtime_root / "tts-python"),
                )
            ),
            stt_model_repo=_config_get(
                env_values,
                dotenv_values,
                "STT_MODEL_REPO",
                "VOLTA_STT_MODEL_REPO",
                default="kyutai/stt-1b-en_fr-mlx",
            ),
            llm_model_path=_config_get(
                env_values,
                dotenv_values,
                "LLM_MODEL_PATH",
                "VOLTA_LLM_MODEL_PATH",
                default=str(local_runtime_root / "llm-model"),
            ),
            llm_system_prompt_file=Path(
                _config_get(
                    env_values,
                    dotenv_values,
                    "LLM_SYSTEM_PROMPT_FILE",
                    "VOLTA_LLM_SYSTEM_PROMPT_FILE",
                    default=str(prompts_root / "investor-judge.md"),
                )
            ),
            llm_max_tokens=int(
                _config_get(
                    env_values,
                    dotenv_values,
                    "LLM_MAX_TOKENS",
                    "VOLTA_LLM_MAX_TOKENS",
                    default="2048",
                )
            ),
            llm_prompt_cache_enabled=_config_bool(
                env_values,
                dotenv_values,
                "LLM_PROMPT_CACHE",
                "VOLTA_LLM_PROMPT_CACHE",
                default=True,
            ),
            tts_model_path=_config_get(
                env_values,
                dotenv_values,
                "TTS_MODEL_PATH",
                "VOLTA_TTS_MODEL_PATH",
                default="mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
            ),
            tts_voice=_config_get(
                env_values,
                dotenv_values,
                "TTS_VOICE",
                "VOLTA_TTS_VOICE",
                default="Ryan",
            ),
            tts_instruct=_config_get(
                env_values,
                dotenv_values,
                "TTS_INSTRUCT",
                "VOLTA_TTS_INSTRUCT",
                default="Very angry, aggressive, intense, raised voice, confrontational delivery, emotionally expressive delivery.",
            ),
            tts_streaming_interval=float(
                _config_get(
                    env_values,
                    dotenv_values,
                    "TTS_STREAMING_INTERVAL",
                    "VOLTA_TTS_STREAMING_INTERVAL",
                    default="0.15",
                )
            ),
            tts_chunk_soft_limit=int(
                _config_get(
                    env_values,
                    dotenv_values,
                    "TTS_CHUNK_SOFT_LIMIT",
                    "VOLTA_TTS_CHUNK_SOFT_LIMIT",
                    default="48",
                )
            ),
            tts_chunk_min_split=int(
                _config_get(
                    env_values,
                    dotenv_values,
                    "TTS_CHUNK_MIN_SPLIT",
                    "VOLTA_TTS_CHUNK_MIN_SPLIT",
                    default="24",
                )
            ),
            host=_config_get(
                env_values,
                dotenv_values,
                "HOST",
                "VOLTA_HOST",
                default="127.0.0.1",
            ),
            port=int(
                _config_get(
                    env_values,
                    dotenv_values,
                    "PORT",
                    "VOLTA_PORT",
                    default="8765",
                )
            ),
        )


settings = Settings.from_env()
