"""Train the single multitask DIRECTOR LoRA on Modal GPU.

One single-stage bf16 LoRA run on the mixed director.jsonl (facts + decide + score),
sharing config C and all Phase-2 mechanics with the actors via train_common.py.
(The optional legal_base warm-start path was removed — it was never used.)

  modal run training/director_finetune.py
Gate it with training/director_evaluate.py + training/e2e_gate.py, then convert
with training/convert_gguf.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root -> buzzwords pkg

from buzzwords import contracts

BASE_MODEL = "openbmb/MiniCPM5-1B"

app = modal.App("buzzwords-director-finetune")
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
def train(dataset: str = "director.jsonl", out_name: str = "director"):
    import train_common
    metrics = train_common.run_training(
        base_model=BASE_MODEL, dataset=dataset, out_name=out_name,
        data_dir=DATA, out_dir=OUT, shape_version=contracts.SHAPE_VERSION)
    adapters_vol.commit()
    return {k: metrics.get(k) for k in ("eval_loss", "perplexity")}


@app.local_entrypoint()
def main(dataset: str = "director.jsonl", out_name: str = "director"):
    train.remote(dataset, out_name)
