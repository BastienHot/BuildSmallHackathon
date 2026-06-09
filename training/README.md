# Training pipeline (Modal GPU, offline)

Produces every adapter the game uses on the vanilla MiniCPM5-1B base — the eight per-style
**actor** LoRAs and the single multitask **director** (Game Master) LoRA. We ship adapters,
never a merged base. Everything here runs on Modal, not the player's machine.

## One-time setup

```bash
pip install -r training/requirements.txt
modal token new                      # interactive auth (opens a browser)
modal secret create huggingface HF_TOKEN=hf_xxx   # for gated model downloads
```

## Run order

Two parallel distillations off the same Gemma teacher: the per-style **actors** and the single
**director** (Game Master). ALWAYS smoke-test small first and eyeball the printed sample before
scaling (`--base_seed` / `--base-seed` re-rolls the variety).

```bash
# --- Actors (8 per-style LoRAs) -> buzzwords-data + buzzwords-adapters ---
modal run training/teacher_datagen.py --style aviation --n 5   # smoke test
modal run training/teacher_datagen.py                          # legal_generic + every style
modal run training/finetune.py                                 # legal_base, then actor_<style> forks

# --- Director / Game Master (1 multitask LoRA: casefile + decide + score) ---
modal run training/director_datagen.py --task game --n 4       # smoke test (eyeball the trace)
modal run training/director_datagen.py --n 300 --casefile-extra 600   # full mixed director.jsonl
modal run training/director_finetune.py                        # -> "director" adapter

# --- Convert all adapters -> GGUF (no merge) and pull into models/ ---
modal run training/convert_gguf.py                             # style-*.lora.gguf + director.lora.gguf
modal volume get buzzwords-gguf / ./models
```

Gate the director before shipping it:
```bash
# generate a disjoint HELD-OUT set first, then benchmark base-vs-LoRA on its real contexts:
modal run training/director_datagen.py --task game --n 40 --base-seed 900000
modal run training/director_evaluate.py   # speaker balance / transitions / teacher-agreement / calibration
modal run training/director_gate.py       # full grammar-constrained game via llama.cpp
```

The app then loads `models/MiniCPM5-1B-Q4_K_M.gguf` with `director.lora.gguf` for the GM and
`style-<style>.lora.gguf` for the actors (see `buzzwords/config.py`).

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
- **Example shape = runtime shape:** every example matches the exact call the app makes —
  actors get `role+style` system / `"Stage direction: …"` user (NO hidden brief; the runtime
  actor never sees the truth), and the director gets the casefile / decide / score shapes from
  `buzzwords/text_engine.py`. Train/runtime mismatch is a silent quality killer.
- **Smokescreen by construction (director):** the profession is drawn from a pool with the
  jargon's own domain excluded, so every casefile target is provably profession-⟂-jargon;
  few-shot sampling + a banned-word list + a corrective retry keep the leak rate near 0.
- **Checkpoint-fork (not resume):** stage 2 loads `legal_base` as *initialization* and
  starts a fresh run — never `resume_from_checkpoint`. See `finetune.py`.
- Model IDs (`openbmb/MiniCPM5-1B`, the AWQ teacher) are constants at the top of each
  script — adjust if names differ at run time.
