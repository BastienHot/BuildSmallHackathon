"""Shared training core for the actor and director LoRAs (runs INSIDE Modal containers).

Implements the REBUILD_REVIEW.md Phase-2 decisions in one place:
  * bf16 LoRA (NOT QLoRA) -- a 1B trains comfortably in bf16; zero training-side
    quantization noise; the e2e llama.cpp gate validates the serving-side quant (§6.4).
  * GROUP-wise train/val split on group_id -- sibling examples from one transcript/game
    never straddle the split, so eval_loss/perplexity mean what they claim (§6.1).
  * Completion-only loss -- the assistant response is the only supervised span; the
    repetitive prompt boilerplate contributes nothing to the gradient (§6.2).
  * enable_thinking=False at templating time + a rendered-example audit (§6.3).
  * Fixed seed + enriched train_metrics.json (config, manifest ref, git-able) (§6.5).
  * Manifest SHAPE_VERSION check -- training refuses data generated under a different
    prompt-shape contract (§5.5).
"""

from __future__ import annotations

SEED = 3407

# LoRA config C -- IDENTICAL across stage 1 and stage 2 (required for the fork).
LORA = dict(
    r=32, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
TRAIN = dict(learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.05,
             per_device_train_batch_size=8, gradient_accumulation_steps=2,
             num_train_epochs=2, optim="adamw_8bit", logging_steps=10, bf16=True,
             seed=SEED)

_THINK_MARKERS = ("<think", "<|thought_begin|>")


def check_manifest(data_dir: str, dataset: str, shape_version: str) -> dict | None:
    """Refuse data whose prompt-shape contract differs from the code's (§5.5)."""
    import json
    import os
    path = os.path.join(data_dir, dataset.replace(".jsonl", ".manifest.json"))
    if not os.path.exists(path):
        # No manifest = unverifiable provenance = very likely a stale pre-rebuild file
        # still sitting on the volume. Refuse — this exact hole shipped mismatch-trained
        # actors once already (REBUILD_REVIEW.md §5.1/§5.5).
        raise RuntimeError(f"no manifest at {path} — refusing to train on unverifiable data; "
                           f"regenerate {dataset} with the current datagen")
    manifest = json.load(open(path, encoding="utf-8"))
    got = manifest.get("shape_version")
    if got != shape_version:
        raise RuntimeError(f"SHAPE_VERSION mismatch: dataset={got!r} code={shape_version!r} "
                           f"— regenerate {dataset} before training")
    return manifest


def group_split(ds, test_frac: float = 0.05, seed: int = SEED):
    """Split by unique group_id so sibling rows never leak across the split (§6.1)."""
    import random
    groups = sorted(set(ds["group_id"]))
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * test_frac))
    val_groups = set(groups[:n_val])
    train = ds.filter(lambda ex: ex["group_id"] not in val_groups)
    val = ds.filter(lambda ex: ex["group_id"] in val_groups)
    print(f"group split: {len(groups)} groups -> train {len(train)} rows / "
          f"val {len(val)} rows ({n_val} val groups)")
    return train, val


def render_prompt_completion(tok, conversations) -> tuple[str, str]:
    """Render EXACTLY what inference produces, split at the supervised boundary:
    prompt = the generation prompt (which, with enable_thinking=False, ends in an empty
    <think></think> block on MiniCPM5); completion = the assistant answer + the turn
    terminator. Plain full-conversation templating omits the empty think block and would
    train the model one prefix away from the prompt it always sees at inference
    (verified by probing the tokenizer — §6.3)."""
    prompt = tok.apply_chat_template(conversations[:-1], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
    return prompt, conversations[-1]["content"] + _assistant_end(tok)


def mask_parts(tok) -> tuple[str, str]:
    """(instruction_part, response_part) for unsloth's train_on_responses_only, derived
    by probing the template. response_part is the FULL generation prefix after the user
    content (assistant header + empty think block), so the supervised span starts at
    exactly the first answer token — same boundary as render_prompt_completion (§6.2)."""
    gen = tok.apply_chat_template(
        [{"role": "system", "content": "@@S@@"}, {"role": "user", "content": "@@U@@"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    instruction = gen.split("@@S@@", 1)[1].split("@@U@@", 1)[0]   # user-turn header
    response = gen.split("@@U@@", 1)[1]                           # asst header + think block
    print(f"mask parts: instruction={instruction!r} response={response!r}")
    return instruction, response


def _assistant_end(tok) -> str:
    """The turn terminator after assistant content (e.g. '<|im_end|>\\n'), by probe."""
    full = tok.apply_chat_template(
        [{"role": "user", "content": "U"}, {"role": "assistant", "content": "@@A@@"}],
        tokenize=False, enable_thinking=False)
    return full.split("@@A@@", 1)[1]


def audit_example(tok, text: str) -> None:
    """Print one fully rendered training example; assert the think channel is CLOSED
    (the empty <think></think> prefix is expected; any think CONTENT is not) (§6.3)."""
    print("--- rendered training example (audit) ---")
    print(text[:600])
    stripped = text.replace("<think>\n\n</think>", "").lower()
    for m in _THINK_MARKERS:
        if m in stripped:
            raise RuntimeError(f"Thought-channel content {m!r} in rendered training text — "
                               "the chat template is not honoring enable_thinking=False")


def run_training(*, base_model: str, dataset: str, out_name: str, data_dir: str,
                 out_dir: str, shape_version: str, init_adapter: str | None = None):
    """One SFT run (stage-1, stage-2 fork, or director). Returns metrics dict."""
    import json
    import math

    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    manifest = check_manifest(data_dir, dataset, shape_version)
    torch.manual_seed(SEED)

    if init_adapter:
        # Stage-2 FORK: load the saved adapter dir as the starting model (fresh optimizer,
        # fresh schedule) — NOT resume_from_checkpoint (§3.5).
        model, tok = FastLanguageModel.from_pretrained(
            f"{out_dir}/{init_adapter}", max_seq_length=4096,
            load_in_4bit=False, dtype=torch.bfloat16)
    else:
        model, tok = FastLanguageModel.from_pretrained(
            base_model, max_seq_length=4096, load_in_4bit=False, dtype=torch.bfloat16)
        model = FastLanguageModel.get_peft_model(model, **LORA)

    ds = load_dataset("json", data_files=f"{data_dir}/{dataset}", split="train")

    def fmt(ex):
        prompt, completion = render_prompt_completion(tok, ex["conversations"])
        return {"text": prompt + completion}
    ds = ds.map(fmt, remove_columns=[c for c in ds.column_names if c != "group_id"])
    audit_example(tok, ds[0]["text"])
    train_ds, val_ds = group_split(ds)

    trainer = SFTTrainer(
        model=model, tokenizer=tok,
        train_dataset=train_ds, eval_dataset=val_ds,
        args=SFTConfig(output_dir=f"/tmp/{out_name}", dataset_text_field="text",
                       eval_strategy="steps", eval_steps=50,
                       per_device_eval_batch_size=8, **TRAIN),
    )
    # unsloth-native completion-only loss (§6.2): mask everything before the derived
    # response marker (assistant header + empty think block) — exact answer boundary.
    from unsloth.chat_templates import train_on_responses_only
    instruction_part, response_part = mask_parts(tok)
    trainer = train_on_responses_only(trainer, instruction_part=instruction_part,
                                      response_part=response_part)
    trainer.train()                       # fresh run; never resume_from_checkpoint

    metrics = trainer.evaluate()
    metrics["perplexity"] = math.exp(metrics["eval_loss"]) if metrics.get("eval_loss") else None
    metrics.update({"lora": LORA, "train": {k: v for k, v in TRAIN.items()},
                    "dataset": dataset, "init_adapter": init_adapter,
                    "shape_version": shape_version,
                    "dataset_manifest": manifest, "completion_only": True,
                    "precision": "bf16-lora"})
    model.save_pretrained(f"{out_dir}/{out_name}")     # adapter only, never merged
    tok.save_pretrained(f"{out_dir}/{out_name}")
    with open(f"{out_dir}/{out_name}/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"saved adapter -> {out_name} | eval_loss={metrics.get('eval_loss')} "
          f"perplexity={metrics.get('perplexity')}")
    return metrics
