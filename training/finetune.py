"""Two-stage curriculum LoRA for the ACTORS on Modal GPU (checkpoint-fork, NOT merge).

Student: MiniCPM5-1B (Llama archi), bf16 LoRA (not QLoRA):

  Stage 1  -- ONE LoRA (config C) on legal_generic.jsonl -> `legal_base`.
  Stage 2  -- per style: INITIALIZE from `legal_base` (same config C), FRESH run
              (new optimizer + warmup) on style_<style>.jsonl -> `actor_<style>`.

All Phase-2 mechanics (group split, completion-only loss, seed, manifests,
SHAPE_VERSION check) live in training/train_common.py.

  modal run training/finetune.py --style corporate            # stage 1 + one fork
  modal run training/finetune.py --style aviation --skip-stage1
  modal run training/finetune.py                              # stage 1 + ALL styles
  modal run training/finetune.py --only-stage1
Then benchmark with training/evaluate.py before converting to GGUF.

NOTE: the stage-1-helps ablation is one command:
  modal run training/finetune.py --style corporate --skip-stage1 --no-fork
compares against the forked run's train_metrics.json + evaluate.py results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root -> buzzwords pkg

from buzzwords import contracts

BASE_MODEL = "openbmb/MiniCPM5-1B"

app = modal.App("buzzwords-finetune")
# Let unsloth pull its own compatible torch + transformers.
image = (modal.Image.debian_slim()
         .pip_install("unsloth", "trl", "peft", "datasets")
         .add_local_python_source("train_common")
         .add_local_python_source("buzzwords"))
data_vol = modal.Volume.from_name("buzzwords-data", create_if_missing=True)
adapters_vol = modal.Volume.from_name("buzzwords-adapters", create_if_missing=True)
DATA, OUT = "/data", "/adapters"


@app.function(image=image, gpu="A100-40GB", timeout=4 * 60 * 60,
              volumes={DATA: data_vol, OUT: adapters_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def train(dataset: str, out_name: str, init_adapter: str | None = None):
    import train_common
    metrics = train_common.run_training(
        base_model=BASE_MODEL, dataset=dataset, out_name=out_name,
        data_dir=DATA, out_dir=OUT, shape_version=contracts.SHAPE_VERSION,
        init_adapter=init_adapter)
    adapters_vol.commit()
    return {k: metrics.get(k) for k in ("eval_loss", "perplexity")}


@app.local_entrypoint()
def main(style: str = "", only_stage1: bool = False, skip_stage1: bool = False,
         no_fork: bool = False):
    if not skip_stage1:
        train.remote("legal_generic.jsonl", "legal_base")
    if only_stage1:
        return
    styles = [s.strip() for s in style.split(",")] if style else contracts.STYLES
    init = None if no_fork else "legal_base" # --no-fork = the ablation arm
    for s in styles:
        out = f"actor_{s}" + ("_nofork" if no_fork else "")
        train.remote(f"style_{s}.jsonl", out, init_adapter=init)
