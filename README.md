---
title: Buzzwords & Misdemeanors
emoji: ⚖️
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 6.0.1
app_file: app.py
python_version: 3.12.12
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
jargon. All text runs through the **llama.cpp** runtime (CPU).

- **Game Master** — *Nemotron 3 Nano 4B*, vanilla, emits GBNF-constrained JSON beats
  (who speaks next, intensity, wrap-up). Writes a hidden **Case File** (profession +
  fault + facts) — unrelated to the jargon — and directs the turn loop, doubling as the
  scoring judge.
- **Actors** — *MiniCPM5-1B* + **one LoRA per jargon style** (corporate, aviation, …).
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

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
python app.py
```

The UI launches even with no weights — clicking **Start** then tells you exactly which
GGUFs are missing. To actually play, drop them into `models/`:

- `nemotron-nano-4b.Q4_K_M.gguf` — the Game Master (`GM_MODEL`)
- `minicpm5-1b.Q4_K_M.gguf` — the actor base (`JARGON_BASE_MODEL`)
- `style-<style>.lora.gguf` — *optional* per-style adapters from [`training/`](training/README.md);
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
  text_engine.py        # llama.cpp: Game Master (Nemotron) + actors (MiniCPM + LoRA), GBNF
  pipeline.py           # case file → turn loop → scoring (+ preflight checks)
  tts_engine.py         # optional VoxCPM2 (text-only fallback)
  scene.py / ui.py / theme.py / static/   # gr.Walkthrough UI + HTML/CSS
training/               # offline Modal pipeline (teacher data-gen → LoRA → GGUF)
assets/                 # maps/<court>/variant_NN.png (backdrops), voices/ (TTS refs)
models/                 # GGUF weights — git-ignored, you add these
docs/ARCHITECTURE.md    # full design
```
