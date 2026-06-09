"""Train the single multitask DIRECTOR LoRA on Modal GPU.

Mirrors training/finetune.py (unsloth + QLoRA, LoRA config C, same TRAIN recipe) but it is
a SINGLE-stage run: one adapter on the mixed director.jsonl (casefile + decide + score), so
one MiniCPM5-1B can play the Game Master. The actors keep their own per-style adapters; this
is the director's companion adapter (selected per-request at runtime via llama-server, with
`decide` pinned to its own slot to preserve the transcript KV-cache).

  modal run training/director_finetune.py                         # vanilla base + director LoRA
  modal run training/director_finetune.py --init-adapter legal_base   # warm-start on the judicial register

Why optionally fork from legal_base: that stage-1 actor adapter already encodes the generic
courtroom register, which the director also benefits from -- this is the SAME checkpoint-fork
trick as finetune.py stage 2 (load adapter as INITIALIZATION, fresh optimizer; NOT resume).
The adapter + train_metrics.json land in "buzzwords-adapters"; gate it with the bench trace
test, then convert with training/convert_gguf.py (add "director": "director.lora.gguf").
"""

from __future__ import annotations

import modal

BASE_MODEL = "openbmb/MiniCPM5-1B"

# LoRA config C -- identical to finetune.py (so a legal_base fork is valid).
LORA = dict(
    r=32, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
TRAIN = dict(learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
             per_device_train_batch_size=8, gradient_accumulation_steps=2,
             num_train_epochs=2, optim="adamw_8bit", logging_steps=10, bf16=True)

app = modal.App("buzzwords-director-finetune")
image = modal.Image.debian_slim().pip_install("unsloth", "trl", "peft", "datasets")
data_vol = modal.Volume.from_name("buzzwords-data", create_if_missing=True)
adapters_vol = modal.Volume.from_name("buzzwords-adapters", create_if_missing=True)
DATA, OUT = "/data", "/adapters"


@app.function(image=image, gpu="A100-40GB", timeout=4 * 60 * 60,
              volumes={DATA: data_vol, OUT: adapters_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def train(dataset: str = "director.jsonl", out_name: str = "director",
          init_adapter: str | None = None):
    import json
    import math

    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    if init_adapter:   # warm-start fork from an existing adapter (e.g. legal_base) — NOT resume
        model, tok = FastLanguageModel.from_pretrained(
            f"{OUT}/{init_adapter}", max_seq_length=4096, load_in_4bit=True)
    else:
        model, tok = FastLanguageModel.from_pretrained(
            BASE_MODEL, max_seq_length=4096, load_in_4bit=True)
        model = FastLanguageModel.get_peft_model(model, **LORA)

    ds = load_dataset("json", data_files=f"{DATA}/{dataset}", split="train")

    def fmt(ex):
        return {"text": tok.apply_chat_template(ex["conversations"], tokenize=False)}
    ds = ds.map(fmt)
    split = ds.train_test_split(test_size=0.05, seed=42)

    trainer = SFTTrainer(
        model=model, tokenizer=tok,
        train_dataset=split["train"], eval_dataset=split["test"],
        args=SFTConfig(output_dir=f"/tmp/{out_name}", dataset_text_field="text",
                       eval_strategy="steps", eval_steps=50,
                       per_device_eval_batch_size=8, **TRAIN),
    )
    trainer.train()

    metrics = trainer.evaluate()
    metrics["perplexity"] = math.exp(metrics["eval_loss"]) if metrics.get("eval_loss") else None
    model.save_pretrained(f"{OUT}/{out_name}")     # adapter only, never merged
    tok.save_pretrained(f"{OUT}/{out_name}")
    with open(f"{OUT}/{out_name}/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    adapters_vol.commit()
    print(f"saved adapter -> {out_name} | eval_loss={metrics.get('eval_loss')} "
          f"perplexity={metrics.get('perplexity')}")
    return metrics


@app.local_entrypoint()
def main(dataset: str = "director.jsonl", out_name: str = "director",
         init_adapter: str = ""):
    train.remote(dataset, out_name, init_adapter or None)
