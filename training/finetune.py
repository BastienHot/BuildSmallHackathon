"""Two-stage curriculum LoRA on Modal GPU (checkpoint-fork, NOT merge).

Student: MiniCPM5-1B (Llama archi). Per docs/ARCHITECTURE.md §7.3:

  Stage 1  -- train ONE LoRA (config C) on legal_generic.jsonl -> `legal_base`.
  Stage 2  -- for each jargon style: INITIALIZE a LoRA from `legal_base` weights
              (same config C) and run a FRESH run (new optimizer + warmup) on
              style_<style>.jsonl -> `actor_<style>`.

Critical: stage 2 loads the adapter as *initialization*, it is NOT
`resume_from_checkpoint` (that would reuse the old scheduler -- it is for
preemption recovery, a different thing). The base stays vanilla; we ship N
adapters, never a merged base.

Run:
  modal run training/finetune.py --style corporate            # stage 1 + the corporate fork
  modal run training/finetune.py --style aviation --skip-stage1  # reuse existing legal_base
  modal run training/finetune.py                              # stage 1 + fork ALL styles
  modal run training/finetune.py --only-stage1                # just the legal base
Adapters + a per-adapter train_metrics.json land in the "buzzwords-adapters" Volume.
Then benchmark with training/evaluate.py before converting to GGUF.
"""

from __future__ import annotations

import modal

BASE_MODEL = "openbmb/MiniCPM5-1B"
STYLES = ["corporate", "aviation", "ai", "politics", "medical", "gaming", "sports", "scifi"]

# LoRA config C -- IDENTICAL across stage 1 and stage 2 (required for the fork).
LORA = dict(
    r=32, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
TRAIN = dict(learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
             per_device_train_batch_size=8, gradient_accumulation_steps=2,
             num_train_epochs=2, optim="adamw_8bit", logging_steps=10, bf16=True)

app = modal.App("buzzwords-finetune")
# Let unsloth pull its own compatible torch + transformers; don't list them (or pin an
# old version) separately, or you get torch/transformers version skew at import.
image = modal.Image.debian_slim().pip_install("unsloth", "trl", "peft", "datasets")
data_vol = modal.Volume.from_name("buzzwords-data", create_if_missing=True)
adapters_vol = modal.Volume.from_name("buzzwords-adapters", create_if_missing=True)
DATA, OUT = "/data", "/adapters"


@app.function(image=image, gpu="A100-40GB", timeout=4 * 60 * 60,
              volumes={DATA: data_vol, OUT: adapters_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def train(dataset: str, out_name: str, init_adapter: str | None = None):
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    if init_adapter:
        # Stage-2 FORK: load the saved adapter dir straight as the starting model. Unsloth
        # returns a TRAINABLE PeftModel (grads + checkpointing wired up) and RESETS the
        # optimizer -> a fresh run initialized from legal_base's weights (the curriculum
        # fork). NOT resume_from_checkpoint. (Bolting PeftModel.from_pretrained onto an
        # unsloth model skips the grad setup -> "element 0 ... does not require grad".)
        model, tok = FastLanguageModel.from_pretrained(
            f"{OUT}/{init_adapter}", max_seq_length=4096, load_in_4bit=True)
    else:
        # Stage-1: vanilla base + a brand-new LoRA (config C).
        model, tok = FastLanguageModel.from_pretrained(
            BASE_MODEL, max_seq_length=4096, load_in_4bit=True)  # QLoRA
        model = FastLanguageModel.get_peft_model(model, **LORA)

    import json
    import math

    ds = load_dataset("json", data_files=f"{DATA}/{dataset}", split="train")

    def fmt(ex):
        return {"text": tok.apply_chat_template(ex["conversations"], tokenize=False)}
    ds = ds.map(fmt)
    split = ds.train_test_split(test_size=0.05, seed=42)  # held-out val -> loss/perplexity

    trainer = SFTTrainer(
        model=model, tokenizer=tok,
        train_dataset=split["train"], eval_dataset=split["test"],
        args=SFTConfig(output_dir=f"/tmp/{out_name}", dataset_text_field="text",
                       eval_strategy="steps", eval_steps=50,
                       per_device_eval_batch_size=8, **TRAIN),
    )
    trainer.train()                       # NOTE: no resume_from_checkpoint (fresh run)

    metrics = trainer.evaluate()          # final held-out loss; lower = better fit
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
def main(style: str = "", only_stage1: bool = False, skip_stage1: bool = False):
    # Stage 1 once -> legal_base. Skip it when forking more styles onto an existing base.
    if not skip_stage1:
        train.remote("legal_generic.jsonl", "legal_base")
    if only_stage1:
        return
    styles = [s.strip() for s in style.split(",")] if style else STYLES  # "a,b,c" or all
    for s in styles:                                          # stage 2 fork(s)
        train.remote(f"style_{s}.jsonl", f"actor_{s}", init_adapter="legal_base")
