"""Synthetic data generation for the ACTOR LoRAs, on Modal GPU (offline teacher).

A big teacher (Gemma 4 31B-it, FP8) writes courtroom transcripts that are exploded into
ShareGPT examples in the EXACT runtime call shape from buzzwords.contracts (the single
source of truth — train shape == runtime shape by construction). Two datasets:

  1. legal_generic.jsonl -- courtroom register, NO domain jargon (stage-1 base).
  2. style_<style>.jsonl  -- transcripts saturated with one jargon STYLE over varied
     hidden professions/faults SAMPLED FROM buzzwords.pools (smokescreen by construction).

Each actor example carries: role+style system, and a user turn with the last 1-2 public
transcript lines, an optional oblique fact to weave in, and the stage direction —
exactly what buzzwords.engine.act sends (REBUILD_REVIEW.md §10.2).

Every run writes <name>.manifest.json (seed, counts, rejects, teacher, SHAPE_VERSION)
and stamps group_id (one per transcript) on every example for leak-free splits.

  modal run training/teacher_datagen.py --style aviation --n 5   # smoke test, eyeball it
  modal run training/teacher_datagen.py                          # legal_generic + all styles
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root -> buzzwords pkg

from buzzwords import contracts, pools
from jargon_banks import JARGON

TERMS_PER_PROMPT = 10  # how many bank terms to seed into each prompt (drives variety)

TEACHER_MODEL = os.getenv("BW_TEACHER_MODEL", "RedHatAI/gemma-4-31B-it-FP8-block")
QUANTIZATION = os.getenv("BW_TEACHER_QUANT", "")  # "" = auto-detect; or awq | bitsandbytes
TEACHER_GPU = os.getenv("BW_TEACHER_GPU", "L40S")  # native FP8

TEACHER_SYS = ("You are a comedy-legal scriptwriter producing VARIED courtroom training "
               "transcripts. Output strictly valid JSON and nothing else.")

app = modal.App("buzzwords-teacher")
image = (modal.Image.debian_slim()
         .pip_install("vllm")
         .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
         .add_local_python_source("jargon_banks")
         .add_local_python_source("buzzwords"))   # contracts + pools (single source of truth)
vol = modal.Volume.from_name("buzzwords-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
DATA = "/data"
HF_CACHE = "/root/.cache/huggingface"


# ------------------------------------------------------------------- prompt spec
def _make_spec(rng: random.Random, style: str, terms_pool: list[str]) -> dict:
    profession, fault = pools.sample_case(rng, style or "corporate")
    n_turns = rng.randint(6, 10)
    k = min(TERMS_PER_PROMPT, len(terms_pool))
    return {
        "style": style, "profession": profession, "fault": fault,
        "tone": rng.choice(pools.TONES), "disposition": rng.choice(pools.DISPOSITIONS),
        "n_turns": n_turns, "max_tokens": 180 * n_turns + 350,
        "terms": rng.sample(terms_pool, k) if k else [],
    }


def _build_prompt(spec: dict) -> str:
    style = spec["style"]
    register = (f"saturated with dense {style} jargon mixed with judicial register; the "
                f"speakers talk almost entirely in {style} buzzwords"
                if style else "in plain professional English with no domain slang")
    unrelated = f" (UNRELATED to the {style} jargon they speak)" if style else ""
    terms = (f"Weave several of these {style} terms in naturally (vary which you use; do "
             f"NOT use all, do NOT just list them): {', '.join(spec['terms'])}.\n"
             if spec.get("terms") else "")
    return (
        f"Write a {spec['n_turns']}-turn courtroom exchange {register}. "
        f"Roles cycle through judge, prosecutor and defense; consecutive lines REACT to "
        f"each other (rebut, concede, redirect) without repeating phrasing. "
        f"Overall tone: {spec['tone']}. The defendant comes across as {spec['disposition']}.\n"
        f"HIDDEN TRUTH, never stated in plain words: the defendant is "
        f"{contracts.article(spec['profession'])} {spec['profession']} who {spec['fault']}{unrelated}.\n"
        f"These word(s) must NOT appear anywhere: "
        f"{', '.join(repr(w) for w in pools.banned_words(spec['profession']))}.\n"
        f"First invent 3-5 short oblique FACTS (concrete clues — logs, records, witnesses — "
        f"that hint at the real job and act without naming them). Then, for EACH line, give "
        f"the oblique stage_direction a director would hand the actor, an intensity 1-5, and "
        f"fact_index: the index of the fact that line weaves in, or null.\n"
        f"Spread every fact across the transcript at least once.\n"
        f'Return ONLY JSON: {{"facts": ["..."], "turns": [{{"role": "judge|prosecutor|defense", '
        f'"intensity": 3, "stage_direction": "<oblique cue>", "fact_index": 0, '
        f'"text": "<the in-character line>"}}]}}.'
    )


# --------------------------------------------------------------------- teacher
@app.cls(image=image, gpu=TEACHER_GPU, volumes={DATA: vol, HF_CACHE: hf_cache},
         timeout=60 * 60, secrets=[modal.Secret.from_name("huggingface")])
class Teacher:
    @modal.enter()
    def load(self):
        from vllm import LLM
        # max_model_len=4096: prompt (~500 tok) + max output fits; 8192 leaves too little
        # KV cache after the ~31.7 GiB FP8 weights on the 48 GiB L40S.
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
def _parse_obj(text: str):
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        d = json.loads(text[a:b + 1])
    except Exception:  # noqa: BLE001
        return None
    return d if isinstance(d, dict) else None


def _validate(spec: dict, text: str):
    """-> (facts, turns, reject_reason|None). Leak check covers facts, stage directions
    AND lines (everything a model will see or say)."""
    obj = _parse_obj(text)
    if obj is None:
        return None, None, "unparseable"
    facts, turns = obj.get("facts"), obj.get("turns")
    if not isinstance(facts, list) or not (contracts.MIN_FACTS <= len(facts) <= contracts.MAX_FACTS):
        return None, None, "bad facts"
    if not isinstance(turns, list) or not (4 <= len(turns) <= 14):
        return None, None, f"turn count {len(turns) if isinstance(turns, list) else '?'}"
    blob = " ".join(str(f) for f in facts)
    for t in turns:
        if not isinstance(t, dict):
            return None, None, "bad turn shape"
        role = (t.get("role") or "").lower()
        if role not in contracts.SPEAKERS or not str(t.get("text", "")).strip():
            return None, None, f"bad role/text ({role})"
        blob += " " + str(t.get("stage_direction", "")) + " " + str(t.get("text", ""))
    if contracts.leaks(blob, spec["profession"]):
        return None, None, "profession leaked"
    return [str(f).strip() for f in facts], turns, None


def _to_examples(facts: list[str], turns: list[dict], spec: dict, group_id: str) -> list[dict]:
    """One example per line, in the EXACT runtime shape (engine.act): role+style system,
    actor_user(stage_direction, intensity, fact, last-2-lines context). The actor never
    sees the hidden truth — in training or at runtime."""
    out, transcript = [], []
    for t in turns:
        role = t["role"].lower()
        sd = str(t.get("stage_direction", "")).strip() or "Open the hearing."
        intensity = t.get("intensity") if t.get("intensity") in (1, 2, 3, 4, 5) else 3
        fi = t.get("fact_index")
        fact = facts[fi] if isinstance(fi, int) and 0 <= fi < len(facts) else None
        out.append({"group_id": group_id, "conversations": [
            {"role": "system", "content": contracts.actor_system(role, spec["style"] or None)},
            {"role": "user", "content": contracts.actor_user(sd, intensity, fact, transcript[-2:])},
            {"role": "assistant", "content": str(t["text"]).strip()},
        ]})
        transcript.append((role, str(t["text"]).strip()))
    return out


def _corrective(prompt: str, profession: str, reason: str) -> str:
    return (f"A previous attempt was REJECTED ({reason}). NEVER write any of these words "
            f"anywhere: {', '.join(repr(w) for w in pools.banned_words(profession))}. "
            f"Redo it correctly, keeping every clue oblique.\n\n" + prompt)


# --------------------------------------------------------------- orchestration
@app.function(image=image, volumes={DATA: vol}, timeout=2 * 60 * 60)
def make_dataset(style: str | None, n: int, base_seed: int = 0):
    from collections import Counter
    style = style or ""
    terms_pool = JARGON.get(style, [])
    specs = [_make_spec(random.Random(base_seed + i), style, terms_pool) for i in range(n)]
    prompts = [_build_prompt(s) for s in specs]
    metas = [{"max_tokens": s["max_tokens"], "seed": base_seed + i} for i, s in enumerate(specs)]

    rows, reasons, kept, pending = [], Counter(), 0, list(range(n))
    for attempt in range(2):   # one corrective retry round on the rejects
        if not pending:
            break
        raw, gen_s, toks = Teacher().generate.remote([prompts[i] for i in pending],
                                                     [metas[i] for i in pending])
        still = []
        for idx, text in zip(pending, raw):
            facts, turns, reason = _validate(specs[idx], text)
            if reason is None:
                rows.extend(_to_examples(facts, turns, specs[idx],
                                         group_id=f"{style or 'generic'}-{base_seed + idx}"))
                kept += 1
            else:
                still.append((idx, reason))
        if attempt == 0:
            for idx, reason in still:
                prompts[idx] = _corrective(prompts[idx], specs[idx]["profession"], reason)
                metas[idx] = {**metas[idx], "seed": metas[idx]["seed"] + 500_000}
            pending = [idx for idx, _ in still]
        else:
            reasons.update(r for _, r in still)
            pending = []
        print(f"attempt {attempt + 1}: kept {kept}/{n} ({gen_s:.0f}s, {toks / gen_s:.0f} tok/s)")

    name = f"style_{style}" if style else "legal_generic"
    with open(f"{DATA}/{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"dataset": name, "shape_version": contracts.SHAPE_VERSION,
                "teacher": TEACHER_MODEL, "base_seed": base_seed, "requested": n,
                "kept_transcripts": kept, "examples": len(rows),
                "reject_reasons": dict(reasons)}
    with open(f"{DATA}/{name}.manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    vol.commit()
    print(f"\n{name}.jsonl: kept {kept}/{n} transcripts -> {len(rows)} examples; "
          f"rejects: {dict(reasons)}")
    if rows:   # eyeball quality before scaling up
        print("--- sample KEPT example ---")
        for m in rows[min(2, len(rows) - 1)]["conversations"]:
            print(f"[{m['role']}] {m['content'][:240]}")
    return len(rows)


@app.local_entrypoint()
def main(style: str = "", n: int = 200, base_seed: int = 0):
    if style:
        make_dataset.remote(style, n, base_seed)
    else:
        make_dataset.remote(None, n, base_seed)              # generic legal base
        for s in contracts.STYLES:
            make_dataset.remote(s, n, base_seed)
