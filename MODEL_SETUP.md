# Model Setup

Volta supports two backend providers:

- `openai` - the default; no local model runtimes required
- `local` - uses your own Apple Silicon STT, LLM, and TTS runtimes

Use the repo-root `.env` file to select the provider and point Volta at the
correct runtimes.

## OpenAI Provider

The OpenAI path does not need any local model checkouts.

1. Copy the example env file.
2. Set `PROVIDER=openai`.
3. Set `OPENAI_API_KEY`.

```bash
cp .env.example .env
```

Recommended OpenAI defaults:

```dotenv
PROVIDER=openai
OPENAI_API_KEY=your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_LLM_MODEL=gpt-5.4-mini
OPENAI_TRANSCRIPTION_MODEL=gpt-4o-transcribe
OPENAI_TTS_MODEL=gpt-4o-mini-tts
OPENAI_TTS_VOICE=alloy
OPENAI_TRANSCRIPTION_LANGUAGE=en
LLM_REASONING=none
```

## Local Provider

The local path is intended for Apple Silicon machines that already have working
MLX-compatible STT, LLM, and TTS runtimes.

Volta expects three runnable Python environments and one local Gemma model
directory:

- STT runtime Python
- LLM runtime Python
- TTS runtime Python
- local Gemma model path

You can keep those checkouts anywhere on disk. The repo does not require a
fixed folder layout.

### 1. Clone or place the local runtimes

Create a stable directory for the runtimes and models you want to use. Example:

```bash
mkdir -p ~/src/volta-models
cd ~/src/volta-models
git clone https://github.com/kyutai-labs/delayed-streams-modeling.git delayed-streams-modeling
git clone https://github.com/Blaizzy/mlx-audio.git mlx-audio
git clone https://huggingface.co/Jiunsong/supergemma4-e4b-abliterated-mlx supergemma-mlx
```

If you already have working local checkouts, reuse them instead of recloning.

### 2. Create the runtime environments

Each runtime needs its own Python environment with the packages that runtime
expects.

Typical pattern:

```bash
cd ~/src/volta-models/delayed-streams-modeling
python3.14 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
deactivate
```

Repeat the same pattern for your TTS runtime and for the Python environment you
use with the local Gemma MLX stack.

### 3. Configure Volta for local mode

Update `.env` with your actual local paths:

```dotenv
PROVIDER=local

STT_PYTHON=/absolute/path/to/delayed-streams-modeling/.venv/bin/python
LLM_PYTHON=/absolute/path/to/your-mlx-llm-runtime/.venv/bin/python
TTS_PYTHON=/absolute/path/to/mlx-audio/.venv/bin/python

STT_MODEL_REPO=kyutai/stt-1b-en_fr-mlx
LLM_MODEL_PATH=/absolute/path/to/your-supergemma-mlx-model
LLM_SYSTEM_PROMPT_FILE=prompts/investor-judge.md
LLM_MAX_TOKENS=2048
LLM_PROMPT_CACHE=true

TTS_MODEL_PATH=mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit
TTS_VOICE=Ryan
TTS_INSTRUCT=Very angry, aggressive, intense, raised voice, confrontational delivery, emotionally expressive delivery.
```

Provider-specific voice defaults:

- `openai` uses `alloy`
- `local` uses `Ryan`

### 4. Verify the configuration

Run:

```bash
just doctor
```

If any runtime path is missing or not executable, `just doctor` will tell you
which env var to fix.

## Notes

- Only the synthetic sample under `samples/` is shipped publicly.
- Saved run artifacts are written to `.runtime/`, which is local-only and
  gitignored.
- If you switch providers often, keep `.env.example` unchanged and only edit the
  real `.env`.
