---
title: Buzzwords & Misdemeanors
emoji: ⚖️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# ⚖️ Buzzwords & Misdemeanors

You wake up in a courtroom. A judge, a prosecutor and a defense counsel argue your case
— burying you in dense, barely-comprehensible **jargon you picked yourself**. That jargon
is a *smokescreen*: it has nothing to do with what you actually did. See through it, then
guess your **real profession and the charge against you**. A model scores you 0–100% and
reveals the hidden truth.

Built for the Hugging Face **Build Small** hackathon — small models only, fully off-grid.

## How it works

A **Game Master** directs a live courtroom debate; **actors** improvise in your chosen
jargon. All text runs through the **llama.cpp** runtime on **CPU** (compiled with AVX2),
so the whole game runs on a free **cpu-basic** Space — no GPU.

- **Game Master** — *MiniCPM5-1B + a distilled **director** LoRA*, emits GBNF-constrained
  JSON beats (who speaks next, intensity, wrap-up). Writes a hidden **Case File** (profession +
  fault + facts) — unrelated to the jargon — and directs the turn loop, doubling as the
  scoring judge. Same 1B base as the actors; only the adapter differs.
- **Actors** — *MiniCPM5-1B Q4_K_M* + **one LoRA per jargon style** (corporate, aviation, …).
  Three roles (judge / prosecutor / defense) = three system prompts on the same adapter.
  Each beat is generated *directly* in jargon from the GM's stage direction.
- **Closure** — no deterministic latch in v1: the GM is trusted, nudged toward a verdict
  by prompt-injected **turn pressure** and its own `wrap_up` flag.
- **TTS** — *VoxCPM2* voices each character (optional, GPU; falls back to text-only).
- **UI** — a `gr.Walkthrough` (Gradio 6) steps you through the four phases:
  *Charges → The hearing → Your plea → The verdict*.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design.
The LoRA adapters are trained offline on Modal — see [`training/`](training/README.md).

## Run locally

CPU-only — no GPU required (llama.cpp). For the fast AVX2 build, compile llama-cpp-python
from source (as the Dockerfile does); a plain `pip install` grabs an un-vectorized wheel.

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
CMAKE_ARGS="-DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON" pip install --no-binary llama-cpp-python -r requirements.txt
python app.py
```

The UI launches even with no weights — clicking **Start** then tells you exactly which
GGUFs are missing. To actually play, drop them into `models/`:

- `MiniCPM5-1B-Q4_K_M.gguf` — the shared base (Game Master **and** actors)
- `director.lora.gguf` — the Game Master adapter (`DIRECTOR_LORA`)
- `style-<style>.lora.gguf` — *optional* per-style actor adapters from [`training/`](training/README.md);
  until trained, actors run on the vanilla base.

Paths live in `buzzwords/config.py`. On HF Spaces, set `BW_FETCH_WEIGHTS=1` to auto-pull
the GGUFs from the Hub at startup instead — see [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Project layout

```
app.py                  # HF Space / Gradio entrypoint
requirements.txt        # runtime deps (gradio, llama-cpp-python, …)
buzzwords/              # the app package
  config.py             # paths, jargon styles, turn budget, required-weights list
  models.py             # CaseFile / GMDecision / Line / Case / GameSession
  text_engine.py        # llama.cpp: Game Master (Qwen3.5-4B) + actors (MiniCPM5-1B + LoRA), GBNF
  pipeline.py           # case file → turn loop → scoring (+ preflight checks)
  tts_engine.py         # optional VoxCPM2 (text-only fallback)
  scene.py / ui.py / theme.py / static/   # gr.Walkthrough UI + HTML/CSS
training/               # offline Modal pipeline (teacher data-gen → LoRA → GGUF)
assets/                 # maps/<court>/variant_NN.png (backdrops), voices/ (TTS refs)
models/                 # GGUF weights — git-ignored, you add these
docs/ARCHITECTURE.md    # full design
```
