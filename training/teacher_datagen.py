"""Synthetic data generation on Modal GPU (the OFFLINE teacher).

A big teacher (Gemma 4 31B-it, FP8) distills courtroom transcripts into
ShareGPT training data for the small actors. Two datasets, per docs/ARCHITECTURE.md §7.1:

  1. legal_generic.jsonl  -- courtroom register, NO domain jargon (stage-1 base).
  2. style_<style>.jsonl   -- transcripts saturated with one jargon STYLE, over
     VARIED hidden professions/faults (so the LoRA learns the style, not one case).

The jargon style is a smokescreen: it is unrelated to the hidden profession.

Diversity comes from per-sample SEEDED parameters (profession, fault, tone, severity,
defendant disposition, turn count), so a small run is reproducible -- always smoke-test
tiny first and eyeball the printed sample before scaling up:

  modal run training/teacher_datagen.py --style aviation --n 5
  modal run training/teacher_datagen.py                      # all styles + generic
Outputs land in the "buzzwords-data" Modal Volume (download with `modal volume get`).
"""

from __future__ import annotations

import json
import os
import random

import modal

from jargon_banks import JARGON  # curated per-style vocabulary (anchoring, §7.1)

TERMS_PER_PROMPT = 10  # how many bank terms to seed into each prompt (drives variety)

# FP8 checkpoint of Gemma 4 31B-it (~29 GB vs ~62 GB bf16). FP8 is native on L40S/H100
# (Ada/Hopper); on A100 it runs via Marlin, slower. vLLM reads the quant from the
# checkpoint's compressed-tensors config, so we leave QUANTIZATION empty (auto-detect).
# Override either with BW_TEACHER_MODEL / BW_TEACHER_QUANT (e.g. an AWQ or bnb-4bit repo).
TEACHER_MODEL = os.getenv("BW_TEACHER_MODEL", "RedHatAI/gemma-4-31B-it-FP8-block")
QUANTIZATION = os.getenv("BW_TEACHER_QUANT", "")  # "" = auto-detect; or awq | bitsandbytes
TEACHER_GPU = os.getenv("BW_TEACHER_GPU", "L40S")  # native FP8; use H100 for more headroom

STYLES = ["aviation", "corporate", "ai", "politics", "medical", "gaming", "sports", "scifi"]
PROFESSIONS = [
    "airline pilot", "pastry chef", "marine biologist", "tattoo artist",
    "city bus driver", "wedding photographer", "high-school chemistry teacher",
    "plumber", "air-traffic controller", "museum curator", "florist", "locksmith",
]
# A profession in the jargon's OWN domain would break the smokescreen (e.g. an
# "airline pilot" tried in aviation jargon is a giveaway). Exclude those per style.
STYLE_EXCLUDE = {
    "aviation": {"airline pilot", "air-traffic controller"},
    "corporate": {"management consultant"},  # not in the pool today; future-proofing
}
# Diversity axes -> seeded combinations keep cases varied (not one memorized case).
FAULT_ARCHETYPES = [
    "negligently skipped a mandatory safety check",
    "falsified an official record to hide a mistake",
    "took a payment to look the other way",
    "let an uncertified person do a job that required a licence",
    "ignored a warning any competent professional would have acted on",
    "cut corners to hit a deadline and put others at risk",
    "passed off someone else's work as their own",
    "destroyed evidence of an earlier error",
]
TONES = ["combative and theatrical", "dry and procedural", "indignant and moralizing",
         "coldly clinical", "exasperated and impatient"]
DISPOSITIONS = ["defiant", "remorseful", "oblivious and confused", "smug and evasive"]

TEACHER_SYS = ("You are a comedy-legal scriptwriter producing VARIED courtroom training "
               "transcripts. Output strictly valid JSON and nothing else.")

# Actor system prompt — MUST match buzzwords/text_engine.py:ROLE_SYS + _ACTOR_RULES so the
# training examples have the SAME shape the actor sees at runtime (act(): role+style system,
# "Stage direction: …\nIntensity: …/5" user). Earlier the example put the hidden brief in the
# system prompt and the previous line as the user turn — neither matches runtime.
ROLE_SYS = {
    "judge": "You are the JUDGE. Speak with calm authority in dense {style} jargon.",
    "prosecutor": "You are the PROSECUTOR. Press the case hard in dense {style} jargon.",
    "defense": "You are the DEFENSE counsel. Deflect and reframe in dense {style} jargon.",
}
_GENERIC_SYS = {  # stage-1 (legal_generic, style="") keeps the runtime FORMAT, no jargon
    "judge": "You are the JUDGE. Speak with calm authority in plain professional English.",
    "prosecutor": "You are the PROSECUTOR. Press the case hard in plain professional English.",
    "defense": "You are the DEFENSE counsel. Deflect and reframe in plain professional English.",
}
_ACTOR_RULES = (" English, 1-2 sentences. Follow the stage direction. Never name the "
                "defendant's profession or state the charge in plain words.")


def _actor_system(role: str, style: str) -> str:
    return (ROLE_SYS[role].format(style=style) if style else _GENERIC_SYS[role]) + _ACTOR_RULES

app = modal.App("buzzwords-teacher")
# Install vllm UNPINNED so it pulls a mutually-compatible torch + transformers.
# (Pinning an old vllm but a new transformers breaks on torch.float8_e8m0fnu, and an
# old vllm wouldn't support Gemma 4 / FP8 anyway.) Don't list transformers separately.
# VLLM_USE_FLASHINFER_SAMPLER=0: the slim image ships the CUDA runtime but no nvcc, so
# FlashInfer's JIT-compiled sampler can't build -- fall back to vLLM's native sampler.
# (Attention is already auto-selected as Triton for Gemma 4, which needs no JIT.)
image = (modal.Image.debian_slim()
         .pip_install("vllm")
         .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
         .add_local_python_source("jargon_banks"))  # Modal 1.x: mount the sibling module
vol = modal.Volume.from_name("buzzwords-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)  # persist the ~29 GB weights
DATA = "/data"
HF_CACHE = "/root/.cache/huggingface"


# ------------------------------------------------------------------- prompt spec
def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _make_spec(rng: random.Random, professions: list[str], terms_pool: list[str]) -> dict:
    n_turns = rng.randint(6, 10)
    k = min(TERMS_PER_PROMPT, len(terms_pool))
    return {
        "profession": rng.choice(professions),
        "fault": rng.choice(FAULT_ARCHETYPES),
        "tone": rng.choice(TONES),
        "disposition": rng.choice(DISPOSITIONS),
        "severity": rng.randint(2, 5),
        "n_turns": n_turns,
        "max_tokens": 160 * n_turns + 200,
        "terms": rng.sample(terms_pool, k) if k else [],  # seeded subset of the bank
    }


def _build_prompt(style: str, spec: dict) -> str:
    register = (f"saturated with dense {style} jargon mixed with judicial register; the "
                f"speakers talk almost entirely in {style} buzzwords"
                if style else "in plain professional English with no domain slang")
    unrelated = f" (this is UNRELATED to the {style} jargon they speak)" if style else ""
    return (
        f"Write a {spec['n_turns']}-turn courtroom exchange {register}. "
        f"Roles cycle through judge, prosecutor and defense. "
        f"Overall tone: {spec['tone']}. The defendant comes across as {spec['disposition']}. "
        f"Severity of the matter: {spec['severity']}/5.\n"
        f"HIDDEN TRUTH, never stated in plain words: the defendant is "
        f"{_article(spec['profession'])} {spec['profession']} who {spec['fault']}{unrelated}.\n"
        + (f"Weave several of these {style} terms in naturally (vary which you use; do NOT "
           f"use all of them, do NOT just list them): {', '.join(spec['terms'])}.\n"
           if spec.get("terms") else "")
        + f"Argue entirely AROUND the charge -- allude to it, never name the profession or "
        f"the fault outright. For EACH line also give the oblique stage_direction a director "
        f"would have handed the actor (a short cue"
        + (f" in {style} terms" if style else "")
        + f" that hints at the beat but NEVER names the job or charge) and an intensity 1-5.\n"
        f'Return ONLY a JSON list of {{"role": "judge|prosecutor|defense", "intensity": 3, '
        f'"stage_direction": "<oblique cue>", "text": "<the in-character line>"}}.'
    )


# --------------------------------------------------------------------- teacher
@app.cls(image=image, gpu=TEACHER_GPU, volumes={DATA: vol, HF_CACHE: hf_cache},
         timeout=60 * 60, secrets=[modal.Secret.from_name("huggingface")])
class Teacher:
    @modal.enter()
    def load(self):
        from vllm import LLM
        # max_model_len=4096 comfortably covers our prompt (~400 tok) + max output
        # (160*10+200 = 1800 tok). 8192 left too little KV cache after the 31.7 GiB FP8
        # weights on the 48 GiB L40S; 4096 fixes that and lets vLLM batch more requests.
        self.llm = LLM(model=TEACHER_MODEL, quantization=(QUANTIZATION or None),
                       dtype="auto", max_model_len=4096, gpu_memory_utilization=0.92)

    @modal.method()
    def generate(self, prompts: list[str], metas: list[dict]) -> tuple[list[str], float, int]:
        import time
        from vllm import SamplingParams
        sps = [SamplingParams(temperature=1.05, top_p=0.95, repetition_penalty=1.08,
                              max_tokens=m["max_tokens"], seed=m["seed"]) for m in metas]
        msgs = [[{"role": "system", "content": TEACHER_SYS},
                 {"role": "user", "content": p}] for p in prompts]
        t0 = time.time()
        outs = self.llm.chat(msgs, sps)
        elapsed = time.time() - t0
        out_toks = sum(len(o.outputs[0].token_ids) for o in outs)
        return [o.outputs[0].text for o in outs], elapsed, out_toks


# --------------------------------------------------------------------- shaping
def _parse(text: str) -> list | None:
    """Extract the JSON list, tolerating code fences / preamble around it."""
    a, b = text.find("["), text.rfind("]")
    if a == -1 or b <= a:
        return None
    try:
        data = json.loads(text[a:b + 1])
    except Exception:
        return None
    return data if isinstance(data, list) else None


def _to_examples(role_lines: list[dict], style: str, spec: dict) -> list[dict]:
    """One training example per line, in the EXACT runtime call shape (text_engine.act):
    system = role + style rules (NO hidden brief), user = "Stage direction: …\nIntensity: …/5",
    assistant = the line. The actor never sees the profession/fault at runtime, so it must not
    see them in training either."""
    examples = []
    for ln in role_lines:
        role = (ln.get("role") or "judge").lower()
        line = (ln.get("text") or "").strip()
        if not line or role not in ("judge", "prosecutor", "defense"):
            continue
        sd = str(ln.get("stage_direction", "")).strip() or "Open the hearing."
        intensity = ln.get("intensity") if ln.get("intensity") in (1, 2, 3, 4, 5) else 3
        examples.append({"conversations": [
            {"role": "system", "content": _actor_system(role, style)},
            {"role": "user", "content": f"Stage direction: {sd}\nIntensity: {intensity}/5."},
            {"role": "assistant", "content": line},
        ]})
    return examples


def _reject_reason(role_lines, spec: dict) -> str | None:
    """Why a transcript is rejected (format / length / profession spoiler), or None to keep.

    We only guard the *profession* -- the thing the player must guess. We deliberately do
    NOT reject on words from the fault phrasing: those are generic courtroom/English terms
    ("evidence", "warning", "mandatory", ...) the model uses naturally without revealing
    the charge, so matching them over-rejects good transcripts (§7.1).
    """
    if not isinstance(role_lines, list):
        return "not a list"
    if not (4 <= len(role_lines) <= 14):
        return f"turn count {len(role_lines)} outside 4..14"
    # include stage_directions in the leak check (the actor now sees them at runtime)
    blob = " ".join((l.get("stage_direction", "") + " " + l.get("text", ""))
                    for l in role_lines if isinstance(l, dict)).lower()
    if len(blob) < 200:
        return f"too short ({len(blob)} chars)"
    if spec["profession"].lower() in blob:
        return f"profession leaked '{spec['profession']}'"
    return None


# --------------------------------------------------------------- orchestration
def _build_batch(style: str, n: int, base_seed: int):
    excluded = STYLE_EXCLUDE.get(style, set())
    professions = [p for p in PROFESSIONS if p not in excluded]  # keep jargon ⟂ profession
    terms_pool = JARGON.get(style, [])                           # [] for legal_generic
    specs, prompts, metas = [], [], []
    for i in range(n):
        spec = _make_spec(random.Random(base_seed + i), professions, terms_pool)
        specs.append(spec)
        prompts.append(_build_prompt(style, spec))
        metas.append({"max_tokens": spec["max_tokens"], "seed": base_seed + i})
    return specs, prompts, metas


@app.function(image=image, volumes={DATA: vol}, timeout=60 * 60)
def make_dataset(style: str | None, n: int, base_seed: int = 0):
    from collections import Counter
    specs, prompts, metas = _build_batch(style or "", n, base_seed)
    raw, gen_s, out_toks = Teacher().generate.remote(prompts, metas)

    rows, kept, reasons, shown = [], 0, Counter(), 0
    for spec, text in zip(specs, raw):
        role_lines = _parse(text)
        reason = "unparseable JSON" if role_lines is None else _reject_reason(role_lines, spec)
        if reason is None:
            rows.extend(_to_examples(role_lines, style or "", spec))
            kept += 1
            continue
        reasons[reason.split(" ")[0] + (" " + reason.split(" ")[1] if "leaked" in reason else "")] += 1
        if shown < 4:  # show a few rejects in full so we can judge if the filter over-rejects
            shown += 1
            body = text if role_lines is None else " | ".join(l.get("text", "") for l in role_lines)
            print(f"\n--- REJECTED [{reason}] :: hidden = {spec['profession']} / {spec['fault']} ---")
            print(body[:700])

    name = f"style_{style}.jsonl" if style else "legal_generic.jsonl"
    with open(f"{DATA}/{name}", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    vol.commit()
    print(f"\n{name}: kept {kept}/{n} transcripts -> {len(rows)} examples")
    print(f"reject reasons: {dict(reasons)}")
    print(f"generation: {gen_s:.1f}s for {n} prompts "
          f"({n / gen_s * 60:.1f} transcripts/min, {out_toks / gen_s:.0f} tok/s, "
          f"{out_toks / n:.0f} tok/transcript)")
    if rows:  # eyeball quality before scaling up
        print("--- sample KEPT example ---")
        for m in rows[0]["conversations"]:
            print(f"[{m['role']}] {m['content'][:200]}")
    return len(rows)


@app.local_entrypoint()
def main(style: str = "", n: int = 200, base_seed: int = 0):
    if style:
        make_dataset.remote(style, n, base_seed)
    else:
        make_dataset.remote(None, n, base_seed)              # generic legal base
        for s in STYLES:
            make_dataset.remote(s, n, base_seed)
