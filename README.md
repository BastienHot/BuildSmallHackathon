---
title: Buzzwords & Misdemeanors
emoji: ⚖️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: A courtroom deduction game on one 1B model, fully local on CPU
tags:
  - thousand-token-wood
  - local-first
  - fine-tuned
  - custom-ui
  - llama-cpp
  - open-trace
  - tentative
---

# ⚖️ Buzzwords & Misdemeanors

You wake up in a courtroom. A judge, a prosecutor and a defense counsel argue your case
— burying you in dense, barely-comprehensible **jargon you picked yourself**. That jargon
is a *smokescreen*: it has nothing to do with what you actually did. See through it, then
guess your **real profession and the charge against you**. A model scores you 0–100% and
reveals the hidden truth.

Built for the Hugging Face **Build Small** hackathon — small models only, fully off-grid.

## What this is (hackathon submission)

**Track — 🍄 Thousand Token Wood.** A delightful, AI-native game: a whole courtroom
hearing is improvised by one ~1B model, and the fun *is* the model doing the work — it
directs the trial, acts every role in dense jargon, drops the clues, and grades your plea.

**The idea.** You pick a jargon (aviation, medical, corporate…) that becomes a
*smokescreen*; the court buries your real (unrelated) profession and charge under it.
You read past the buzzwords and the evidence board, then plead in plain English. A model
scores you and the truth is revealed.

**The tech.** One **MiniCPM5-1B** base in 4-bit GGUF, served by a single
**`llama-server`** (llama.cpp, AVX2) on **pure CPU** — a free `cpu-basic` Space, no GPU.
It wears small LoRA adapters we fine-tuned and published: one distilled *director* and
one per jargon *style*. Code (GBNF grammars + deterministic guards in
`buzzwords/contracts.py`) enforces the rules; the models provide the flavor.

**Merit badges we're going for:**

| Badge | Why we qualify |
|---|---|
| 🔌 **Local-first** | No cloud APIs — the whole game runs on CPU in the Space. |
| 🎯 **Fine-tuned** | Our own LoRAs, published on the Hub (director + 8 styles). |
| 🎨 **Custom UI** | A hand-built courtroom front-end, well past the default Gradio look. |
| 🦙 **llama.cpp** | Every token runs through a `llama-server` (llama.cpp) runtime. |
| 📡 **Open trace** | 64 full agent traces published as a dataset. |
| 📓 **Field Notes** | A complete build log: [`docs/FIELD_NOTES.md`](docs/FIELD_NOTES.md). |

**Links** — Blog: [`docs/FIELD_NOTES.md`](docs/FIELD_NOTES.md) ·
Adapters: [director](https://huggingface.co/BastienHot/buzzwords-director-lora),
[styles](https://huggingface.co/BastienHot/buzzwords-style-loras) ·
Traces: [dataset](https://huggingface.co/datasets/BastienHot/buzzwords-agent-trace) ·
Demo video: _<add link>_ · Social post: _<add link>_

## How it works

The whole game is **one ~1B model** (MiniCPM5-1B) wearing small LoRA adapters, served by
a single **llama-server** (llama.cpp, AVX2) on **pure CPU** — it runs on a free
`cpu-basic` Space, no GPU anywhere.

- **The truth is sampled in code** from a curated pool (`buzzwords/pools.py`): a
  profession + a domain-matched fault, with the jargon's own domain excluded — the
  smokescreen holds by construction.
- **Game Master** — the base + a distilled **director** LoRA. It writes the oblique
  clue facts, then per beat emits a GBNF-constrained JSON decision (who speaks, beat
  type, which clue to surface, intensity, stage direction, wrap-up) and doubles as the
  scoring judge.
- **Deterministic guards** (`buzzwords/contracts.py`) enforce the courtroom invariants
  the model only *mostly* learned: beat/speaker compatibility (only the defense
  pleads), never the same speaker three times running, the defense heard by mid-game,
  and every clue fact forced out before the hearing ends.
- **Actors** — the same base + **one LoRA per jargon style**. Three roles = three
  system prompts. Actors see the last two public lines, the stage direction, and the
  clue to weave in — **never the truth**, so they cannot leak it.
- **No waiting room** — the hearing starts on the first generated beat; the rest is
  generated in the background while you read.

Every prompt, grammar, and rule lives in **`buzzwords/contracts.py`** — the single
source of truth shared verbatim with the training pipeline, so the training
distribution *is* the inference distribution.

The LoRAs are trained offline on Modal — see [`training/`](training/README.md).

## Run locally

CPU-only. You need a **llama-server** binary built with AVX2 (the Dockerfile does this
for the Space; locally, grab a [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases)
or build from source) — point `BW_LLAMA_SERVER` at it if it's not on PATH.

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
BW_FETCH_WEIGHTS=1 python app.py    # auto-pulls the GGUFs from the Hub into models/
```

The UI launches even with no weights — clicking **Start** then tells you exactly what
is missing. Required in `models/`: `MiniCPM5-1B-Q4_K_M.gguf` (the shared base) and
`director.lora.gguf` (the Game Master adapter); the `style-<style>.lora.gguf` actor
adapters are optional until trained.

## Project layout

```
app.py                  # HF Space / Gradio entrypoint
Dockerfile              # builds llama-server with AVX2 (the whole point of Docker here)
buzzwords/              # the app package
  contracts.py          # SINGLE SOURCE OF TRUTH: grammars, prompts, guards, budgets
  pools.py              # professions (domain-tagged) + matched faults + fallbacks
  config.py             # paths, server settings, style->LoRA map
  models.py             # CaseFile / GMDecision / Line / Case / GameSession
  engine.py             # managed llama-server subprocess + typed game calls
  pipeline.py           # sampled case -> guarded turn loop -> scoring (+ preflight)
  scene.py / ui.py / theme.py / static/   # gr.Walkthrough UI
training/               # offline Modal pipeline (teacher datagen -> LoRA -> GGUF -> gates)
tests/                  # model-free tests for contracts/pools/guards/pipeline
assets/                 # maps/<court>/variant_NN.png backdrops
models/                 # GGUF weights — git-ignored / pulled from the Hub
```
