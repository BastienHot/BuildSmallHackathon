# Training pipeline (Modal GPU, offline)

Produces the actor LoRA adapters. The base MiniCPM5-1B stays **vanilla** — we ship
adapters, never a merged base. Everything here runs on Modal, not the player's machine.

## One-time setup

```bash
pip install -r training/requirements.txt
modal token new                      # interactive auth (opens a browser)
modal secret create huggingface HF_TOKEN=hf_xxx   # for gated model downloads
```

## Run order

```bash
# 1. Teacher (Gemma 4 31B-it, FP8 on L40S) generates ShareGPT data -> buzzwords-data volume
#    Prompts are SEEDED, so a tiny run is reproducible. ALWAYS smoke-test small and
#    eyeball the printed sample before scaling (--base_seed re-rolls the variety):
modal run training/teacher_datagen.py --style aviation --n 5
modal run training/teacher_datagen.py            # then the full run: legal_generic + every style

# 2. Two-stage curriculum LoRA on MiniCPM5-1B -> buzzwords-adapters volume
modal run training/finetune.py                   # legal_base, then actor_<style> forks

# 3. Convert PEFT adapters -> GGUF (no merge) -> buzzwords-gguf volume
modal run training/convert_gguf.py

# 4. Pull the adapters into the app's (git-ignored) models/ dir
modal volume get buzzwords-gguf / ./models
```

The app then loads `models/style-<style>.lora.gguf` on the vanilla MiniCPM base
(see `buzzwords/config.py:STYLE_LORAS`).

## Design notes
- **Quantized teacher:** Gemma 4 31B-it runs as **FP8** (`RedHatAI/gemma-4-31B-it-FP8-block`,
  ~29 GB vs ~62 GB bf16), auto-detected by vLLM from the checkpoint config. Default GPU is
  **L40S** (native FP8, 48 GB). Override with `BW_TEACHER_MODEL` / `BW_TEACHER_QUANT` /
  `BW_TEACHER_GPU` — e.g. an AWQ repo (`BW_TEACHER_QUANT=awq`, `BW_TEACHER_GPU=A100-40GB`)
  or unsloth bnb-4bit (`BW_TEACHER_QUANT=bitsandbytes`).
- **Seeded variety:** each sample is built from `random.Random(base_seed + i)` over
  profession × fault × tone × disposition × severity × turn-count. Same seed → same
  data (reproducible smoke tests); change `--base_seed` for a fresh roll.
- **Sampling:** temp 1.05, top_p 0.95, repetition_penalty 1.08, `max_tokens` scaled to
  turn count, per-sample `seed`. Tune in `Teacher.generate`.
- **Smokescreen jargon:** each `style_<style>.jsonl` covers *varied* hidden professions
  so the adapter learns the jargon *style*, not one case.
- **Example shape:** one `system → user → assistant` example per line, matching the
  runtime call (role + style + hidden brief in the system message).
- **Checkpoint-fork (not resume):** stage 2 loads `legal_base` as *initialization* and
  starts a fresh run — never `resume_from_checkpoint`. See `finetune.py`.
- Model IDs (`openbmb/MiniCPM5-1B`, the AWQ teacher) are constants at the top of each
  script — adjust if names differ at run time.
