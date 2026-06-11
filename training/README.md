# Training pipeline (Modal GPU, offline)

Produces every adapter the game uses on the vanilla MiniCPM5-1B base — the eight
per-style **actor** LoRAs and the single multitask **director** (Game Master) LoRA.
We ship adapters, never a merged base. Everything here runs on Modal, never on the
player's machine.

**Single source of truth:** every prompt, grammar, enum, and budget comes from
`buzzwords/contracts.py` (+ the truth pools in `buzzwords/pools.py`), imported by both
the runtime and every script in this directory. Datagen stamps
`contracts.SHAPE_VERSION` into a `<name>.manifest.json` next to each dataset, and
training **refuses** a dataset whose shape version doesn't match the code.
See `docs/REBUILD_REVIEW.md` for the full design.

## One-time setup

```bash
pip install -r training/requirements.txt
modal token new                      # interactive auth (opens a browser)
modal secret create huggingface HF_TOKEN=hf_xxx   # for gated model downloads
```

## Run order

ALWAYS smoke-test small first and eyeball the printed sample before scaling
(`--base-seed` re-rolls the variety).

```bash
# --- Actors (8 per-style LoRAs) ---
modal run training/teacher_datagen.py --style aviation --n 5    # smoke test
modal run training/teacher_datagen.py                           # legal_generic + every style
modal run training/finetune.py                                  # legal_base, then actor_<style> forks
modal run training/evaluate.py                                  # style lift + fluency + engagement

# Optional one-off ablation (§6.5): does stage 1 actually help?
modal run training/finetune.py --style corporate --skip-stage1 --no-fork

# --- Director / Game Master (1 multitask LoRA: facts + decide + score) ---
modal run training/director_datagen.py --task game --n 4        # smoke test (eyeball the trace)
modal run training/director_datagen.py --n 300 --facts-extra 600   # full mixed director.jsonl
modal run training/director_finetune.py                         # -> "director" adapter

# --- Held-out benchmark (teacher-forced regime) ---
modal run training/director_datagen.py --task game --n 40 --base-seed 900000 --out director_game_heldout
modal run training/director_datagen.py --task label --source director_game_heldout.jsonl   # self-agreement baseline
modal run training/director_evaluate.py

# --- Convert all adapters -> GGUF (no merge) ---
modal run training/convert_gguf.py
modal volume get buzzwords-gguf / ./models

# --- THE ship/no-ship gate: N full self-rollout games, real actors, + solvability ---
modal run training/e2e_gate.py --n 12
```

If the e2e gate fails on sequencing, iterate in this order (REBUILD_REVIEW.md §13.6):
the deterministic guards already ship; then regenerate/retrain; and only then run the
DAgger pass — the gate dumps `director_contexts_selfplay.jsonl`, so:

```bash
modal run training/director_datagen.py --task label --source director_contexts_selfplay.jsonl --out director_dagger
# merge director_dagger.jsonl into director.jsonl and re-run director_finetune.py
```

Finally, publish the trace pool (runs locally, needs models/ + llama-server):
```bash
python training/share_trace.py --n 64
```

## Design notes
- **Truth is sampled in code, not generated.** Profession + domain-matched fault come
  from `buzzwords/pools.py` with the jargon's domain excluded — the smokescreen and
  profession/fault coherence hold by construction; the director model only writes the
  oblique facts (REBUILD_REVIEW.md §13.4).
- **Example shape = runtime shape**, enforced by the shared contracts module: actors
  get the last 1-2 public lines + optional fact + stage direction (never the truth);
  the director gets the stable-prefix gm prompt with the fact_index clue channel.
- **bf16 LoRA, not QLoRA** (§6.4): zero training-side quant noise; the e2e llama.cpp
  gate validates the serving-side Q4_K_M.
- **Group-wise split + completion-only loss + fixed seed** (§6.1-§6.5), all in
  `train_common.py`; `train_metrics.json` records configs + manifest lineage.
- **Checkpoint-fork (not resume):** stage 2 loads `legal_base` as *initialization*
  and starts a fresh run — never `resume_from_checkpoint`.
- **Quantized teacher:** Gemma 4 31B-it FP8 (`RedHatAI/gemma-4-31B-it-FP8-block`) on
  L40S; override with `BW_TEACHER_MODEL` / `BW_TEACHER_QUANT` / `BW_TEACHER_GPU`.
- **Single teacher / single prompt family is accepted hackathon scope** — note it in
  the final report (§5.4).
