"""Synthetic data generation for the DIRECTOR (Game Master) LoRA, on Modal GPU.

Same Gemma teacher/Modal setup as teacher_datagen.py; produces ONE multitask dataset in
the EXACT runtime call shapes from buzzwords.contracts (the single source of truth):

  * facts  -- the director's only authoring job now: the truth (profession + matched
              fault) is SAMPLED IN CODE from buzzwords.pools, both here and at runtime
              (smokescreen + coherence by construction, REBUILD_REVIEW.md §13.4).
  * decide -- per-beat decisions over the FULL append-only transcript, including the
              fact_index clue channel (§13.5). When the runtime forced-fact rule would
              fire, the training target's fact_index is rewritten to comply — the model
              learns to obey the force nudge it will see at inference.
  * score  -- grade seeded-quality guesses (0-100 + rationale).
  * label  -- teacher labels decisions for EXISTING decide contexts from a jsonl: used
              for the teacher self-agreement baseline (§7.4) and, pointed at student
              self-rollout contexts, as the DAgger pass (§13.6).

Every run writes a manifest (seed, counts, rejects, SHAPE_VERSION) and stamps group_id.

  modal run training/director_datagen.py --task game --n 4      # smoke test, eyeball it
  modal run training/director_datagen.py --n 300 --facts-extra 600   # full director.jsonl
  modal run training/director_datagen.py --task game --n 40 --base-seed 900000 \
      --out director_game_heldout                               # disjoint held-out slice
  modal run training/director_datagen.py --task label --source director_game_heldout.jsonl
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

TEACHER_MODEL = os.getenv("BW_TEACHER_MODEL", "RedHatAI/gemma-4-31B-it-FP8-block")
QUANTIZATION = os.getenv("BW_TEACHER_QUANT", "")
TEACHER_GPU = os.getenv("BW_TEACHER_GPU", "L40S")

TEACHER_SYS = ("You are the show-runner and head writer of a comedy-legal courtroom guessing "
               "game: you design oblique clues, direct the beats, and grade guesses. Output "
               "strictly valid JSON and nothing else.")

# How a guess relates to the truth -> spreads the scorer's calibration across the range.
GUESS_BUCKETS = ["spot_on", "close_paraphrase", "right_job_wrong_charge",
                 "right_charge_wrong_job", "plausible_but_wrong", "vague", "totally_unrelated"]

# Hand-written GOLD facts exemplars, sampled per prompt (pattern, not template).
# Calibrated to the clue band: one inference step from each — never naming the job,
# never so generic any job would fit ("smart, not too obvious, not too hard").
FEWSHOTS = [
    ("marine biologist", "released unverified findings a rival lab later had to retract",
     ["a sample set went missing before peer review",
      "the tide tables in the appendix were back-dated",
      "a junior was credited, then quietly removed"]),
    ("pastry chef", "served a wedding cake he knew had spoiled rather than refund the order",
     ["the cold-storage log had a six-hour gap",
      "two guests filed the same complaint that evening",
      "the disposal bin was emptied hours early"]),
    ("locksmith", "kept copies of a client's keys and let himself in uninvited",
     ["a spare blank was cut after hours",
      "entry used the correct code, never forced",
      "nothing was taken, but a private drawer had been rifled"]),
    ("wedding photographer", "deleted the only copies of a ceremony she was paid to cover and blamed the gear",
     ["the backup drive was reformatted the next morning",
      "the contract had promised redundant storage",
      "the client was told it was a hardware fault"]),
]

app = modal.App("buzzwords-director-teacher")
image = (modal.Image.debian_slim()
         .pip_install("vllm")
         .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
         .add_local_python_source("buzzwords"))
vol = modal.Volume.from_name("buzzwords-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
DATA = "/data"
HF_CACHE = "/root/.cache/huggingface"


# --------------------------------------------------------------- prompt helpers
def _fewshot_block(rng: random.Random, k: int = 2) -> str:
    lines = []
    for prof, fault, facts in rng.sample(FEWSHOTS, k):
        lines.append(f'  (hidden: a {prof} who {fault}): '
                     + json.dumps({"facts": facts}, ensure_ascii=False))
    return "\n".join(lines)


def _banned(profession: str) -> str:
    return ", ".join(repr(w) for w in pools.banned_words(profession))


def _spec(rng: random.Random, style: str) -> dict:
    profession, fault = pools.sample_case(rng, style)
    return {"style": style, "profession": profession, "fault": fault,
            "tone": rng.choice(pools.TONES), "disposition": rng.choice(pools.DISPOSITIONS),
            "budget": contracts.TURN_BUDGET}


_CLUE_BAND = (
    "Calibrate each clue to ONE inference step: a sharp player should be able to NAME "
    "the job and the act from the three together — never from any single clue, and never "
    "have a clue so generic it could fit any job. At least one clue evokes the "
    "distinctive tools, materials or workplace of the job; never restate the charge's "
    "own words.")


def _facts_prompt(spec: dict, rng: random.Random) -> str:
    return (
        "Examples of well-formed oblique FACTS for hidden cases (each fact is a concrete "
        f"clue that hints at the real job and act without naming them):\n{_fewshot_block(rng)}\n\n"
        f"{_CLUE_BAND}\n"
        f"Now write the facts for a NEW case (do NOT reuse the examples). Hidden truth: the "
        f"defendant is {contracts.article(spec['profession'])} {spec['profession']} who "
        f"{spec['fault']}.\n"
        f"These word(s) must NOT appear anywhere: {_banned(spec['profession'])}.\n"
        f'Return ONLY JSON {{"facts": ["<exactly {contracts.N_FACTS} oblique clues>"]}}.')


def _game_prompt(spec: dict, rng: random.Random) -> str:
    # SHAPE 3.0: the whole hearing is written in PLAIN professional English — no jargon
    # anywhere. The jargon is a separate translation layer the director never sees.
    return (
        "Examples of well-formed oblique FACTS — note how each clue hints at the real job "
        f"without ever naming it:\n{_fewshot_block(rng)}\n\n"
        f"Now design and DIRECT a NEW case, entirely in PLAIN professional courtroom "
        f"English. FIXED hidden truth (do NOT change it; NEVER write it in plain words "
        f"anywhere): the defendant is {contracts.article(spec['profession'])} "
        f"{spec['profession']} who {spec['fault']}.\n"
        f"These word(s) must NOT appear anywhere: {_banned(spec['profession'])}.\n"
        f"Tone: {spec['tone']}; the defendant comes across as {spec['disposition']}.\n"
        f"First write exactly {contracts.N_FACTS} oblique facts. {_CLUE_BAND}\n"
        f"Then run EXACTLY {spec['budget']} beats: open, build evidence, escalate, "
        f"CONVERGE to a verdict — set wrap_up=true only on the final 1-2 beats. Beats "
        f"must alternate speakers naturally (NEVER the same speaker 3 times in a row; "
        f"the defense answers the prosecution). beat_type MUST fit the speaker: "
        f"opening=judge|prosecutor; charge=prosecutor; plea=defense; "
        f"objection=defense|prosecutor; escalate=prosecutor|judge; "
        f"evidence/cross_examine=prosecutor|defense; closing/exchange=anyone. "
        f"Surface EVERY fact at least once via fact_index.\n"
        f"Each beat: next_speaker (judge|prosecutor|defense), beat_type (one of "
        f"{', '.join(contracts.BEATS)}), fact_index (int or null), intensity 1-5, the "
        f"spoken line — 1-2 sentences of plain, concrete courtroom English that argues "
        f"the case and reacts to the previous lines; when fact_index is set, the line "
        f"MUST convey that fact's content — and wrap_up.\n"
        f'Return ONLY JSON: {{"facts": ["..."], "turns": [{{"next_speaker": "...", '
        f'"beat_type": "...", "fact_index": null, "intensity": 3, '
        f'"line": "...", "wrap_up": false}}]}}.')


def _guess(rng: random.Random, spec: dict, bucket: str) -> str:
    others = [p for p in pools.PROFESSIONS if p != spec["profession"]]
    op = rng.choice(others)
    of = rng.choice(pools.PROFESSIONS[op][1])
    own_other = [f for f in pools.PROFESSIONS[spec["profession"]][1] if f != spec["fault"]]
    p, f = spec["profession"], spec["fault"]
    a = contracts.article
    return {
        "spot_on": f"{a(p)} {p} who {f}",
        "close_paraphrase": f"I think they work as {a(p)} {p} and {f}",
        "right_job_wrong_charge": f"{a(p)} {p} who {rng.choice(own_other) if own_other else of}",
        "right_charge_wrong_job": f"{a(op)} {op} who {f}",
        "plausible_but_wrong": f"{a(op)} {op} who {of}",
        "vague": "a professional who broke the rules of their job somehow",
        "totally_unrelated": f"{a(op)} {op}",
    }[bucket]


# --------------------------------------------------------------------- teacher
@app.cls(image=image, gpu=TEACHER_GPU, volumes={DATA: vol, HF_CACHE: hf_cache},
         timeout=60 * 60, secrets=[modal.Secret.from_name("huggingface")])
class Teacher:
    @modal.enter()
    def load(self):
        from vllm import LLM
        self.llm = LLM(model=TEACHER_MODEL, quantization=(QUANTIZATION or None),
                       dtype="auto", max_model_len=4096, gpu_memory_utilization=0.92)

    @modal.method()
    def generate(self, prompts: list[str], metas: list[dict]) -> tuple[list[str], float]:
        import time
        from vllm import SamplingParams
        sps = [SamplingParams(temperature=m.get("temp", 1.0), top_p=0.95,
                              repetition_penalty=1.05, max_tokens=m["max_tokens"],
                              seed=m["seed"]) for m in metas]
        msgs = [[{"role": "system", "content": TEACHER_SYS},
                 {"role": "user", "content": p}] for p in prompts]
        t0 = time.time()
        outs = self.llm.chat(msgs, sps)
        return [o.outputs[0].text for o in outs], time.time() - t0


# --------------------------------------------------------------------- shaping
def _parse_obj(text: str):
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        d = json.loads(text[a:b + 1])
        return d if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _ex(system: str, user: str, assistant_obj: dict, group_id: str) -> dict:
    """One ShareGPT example; the assistant target is compact JSON (grammar-valid)."""
    return {"group_id": group_id, "conversations": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False)}]}


def _too_obvious(fact: str, fault: str) -> bool:
    """Clue-band guard ('not too obvious'): a clue that restates most of the charge's
    own content words gives the act away with zero inference."""
    content = {w for w in fault.lower().split() if len(w) >= 5}
    if not content:
        return False
    hits = sum(1 for w in content if w in fact.lower())
    return hits / len(content) > 0.5


def _clean_facts(obj: dict, spec: dict):
    facts = obj.get("facts")
    if not isinstance(facts, list) or len(facts) != contracts.N_FACTS:
        return None, "bad facts shape"
    facts = [str(f).strip() for f in facts]
    if any(contracts.leaks(f, spec["profession"]) for f in facts):
        return None, "profession leaked in facts"
    if any(_too_obvious(f, spec["fault"]) for f in facts):
        return None, "fact restates the charge (too obvious)"
    if any(not (4 <= len(f.split()) <= 22) for f in facts):
        return None, "fact length out of band"
    return facts, None


def _facts_validate(spec: dict, text: str):
    obj = _parse_obj(text)
    if obj is None:
        return [], "unparseable"
    facts, reason = _clean_facts(obj, spec)
    if reason:
        return [], reason
    ex = _ex(contracts.FACTS_SYS,
             contracts.facts_user(spec["profession"], spec["fault"]),
             {"facts": facts}, group_id=spec["group_id"])
    return [ex], None


def _game_validate(spec: dict, text: str):
    """Validate a full directed trace; explode into 1 facts + N decide examples."""
    obj = _parse_obj(text)
    if obj is None:
        return [], "unparseable"
    facts, reason = _clean_facts(obj, spec)
    if reason:
        return [], reason
    turns, budget = obj.get("turns"), spec["budget"]
    if not isinstance(turns, list) or not (max(4, budget // 2) <= len(turns) <= budget + 1):
        return [], f"turn count {len(turns) if isinstance(turns, list) else '?'}"

    out = _facts_validate(spec, json.dumps({"facts": facts}))[0]
    transcript: list[tuple[str, str]] = []
    released: set[int] = set()
    for i, t in enumerate(turns):
        spk, beat = t.get("next_speaker"), t.get("beat_type")
        line = str(t.get("line", "")).strip()
        intensity, fi = t.get("intensity"), t.get("fact_index")
        if spk not in contracts.SPEAKERS or beat not in contracts.BEATS \
                or intensity not in (1, 2, 3, 4, 5):
            return [], f"bad turn {i} ({spk}/{beat}/{intensity})"
        # Targets must respect the SAME invariants the runtime guard enforces — otherwise
        # we train the model into the seatbelt (the smoke test caught judge/objection).
        if spk not in contracts.BEAT_SPEAKERS[beat]:
            return [], f"beat/speaker mismatch turn {i} ({spk}/{beat})"
        if i >= 2 and transcript[-1][0] == transcript[-2][0] == spk:
            return [], f"same speaker 3x at turn {i}"
        if len(line.split()) < 5:
            return [], f"terse line turn {i} ({line!r})"
        if contracts.leaks(line, spec["profession"]):
            return [], f"profession leaked in turn {i}"
        if not (isinstance(fi, int) and 0 <= fi < len(facts)):
            fi = None
        # SHAPE 3.0: the line IS the content — when a fact is claimed, the line must
        # actually carry it (>=1 shared content word). Reject empty claims.
        if fi is not None:
            anchors = {w for w in facts[fi].lower().split() if len(w) >= 4}
            if anchors and not any(w in line.lower() for w in anchors):
                return [], f"line does not carry claimed fact (turn {i})"
        forced = contracts.forced_fact(len(facts), released, i, budget)
        if forced is not None and fi != forced:
            # We cannot rewrite fi post-hoc anymore: the line was not written to carry
            # the forced fact. Skip the decide example, keep the transcript flowing.
            if fi is not None:
                released.add(fi)
            transcript.append((spk, line))
            continue
        decision = {"next_speaker": spk, "beat_type": beat, "fact_index": fi,
                    "intensity": int(intensity), "line": line,
                    "wrap_up": bool(t.get("wrap_up", False))}
        user = contracts.gm_user(spec["profession"], spec["fault"], facts,
                                 transcript, i, budget, forced)
        out.append(_ex(contracts.GM_SYS, user, decision, group_id=spec["group_id"]))
        if fi is not None:
            released.add(fi)
        transcript.append((spk, line))
    return out, None


def _corrective(prompt: str, profession: str, reason: str) -> str:
    return (f"A previous attempt was REJECTED ({reason}). NEVER write any of these words "
            f"anywhere: {_banned(profession)}. Redo it correctly, keeping every clue "
            f"oblique.\n\n" + prompt)


def _gen_validated(specs, rngs, build_prompt, max_tokens_of, validate, base_seed, label):
    """Generate -> validate -> ONE corrective retry on the rejects. -> (rows, reasons)."""
    from collections import Counter
    prompts = [build_prompt(specs[i], rngs[i]) for i in range(len(specs))]
    metas = [{"max_tokens": max_tokens_of(specs[i]), "seed": base_seed + i}
             for i in range(len(specs))]
    rows, reasons, pending, kept = [], Counter(), list(range(len(specs))), 0
    for attempt in range(2):
        if not pending:
            break
        raw, gen_s = Teacher().generate.remote([prompts[i] for i in pending],
                                               [metas[i] for i in pending])
        still = []
        for idx, text in zip(pending, raw):
            exs, reason = validate(specs[idx], text)
            if reason is None:
                rows.extend(exs)
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
        print(f"{label} attempt {attempt + 1}: kept {kept}/{len(specs)} ({gen_s:.0f}s)")
    return rows, reasons, kept


# --------------------------------------------------------------- orchestration
@app.function(image=image, volumes={DATA: vol}, timeout=3 * 60 * 60)
def make_dataset(task: str, n: int, base_seed: int = 0, facts_extra: int = 0,
                 out: str = "", source: str = ""):
    from collections import Counter
    rows, reasons = [], Counter()
    kept = {}

    if task == "label":   # teacher labels EXISTING decide contexts (self-agreement / DAgger)
        contexts = []
        for line in open(f"{DATA}/{source}", encoding="utf-8"):
            ex = json.loads(line)
            if ex["conversations"][0]["content"] == contracts.GM_SYS:
                contexts.append(ex["conversations"][1]["content"])
        prompts = [(u + "\n\nDecide the next beat. Answer with ONLY this JSON — every "
                    "field MUST use the allowed values: "
                    '{"next_speaker": "judge"|"prosecutor"|"defense", '
                    f'"beat_type": one of {"|".join(contracts.BEATS)}, '
                    '"fact_index": <index of the fact to surface, or null>, '
                    '"intensity": <integer 1-5>, '
                    '"line": "<the spoken line, 1-2 sentences of plain courtroom English>", '
                    '"wrap_up": true|false}')
                   for u in contexts]
        metas = [{"max_tokens": 160, "seed": base_seed + i, "temp": 0.7}
                 for i in range(len(prompts))]
        raw, _ = Teacher().generate.remote(prompts, metas)
        name = out or "director_labels"
        kept_n, shown = 0, 0
        with open(f"{DATA}/{name}.jsonl", "w", encoding="utf-8") as fh:
            for user, text in zip(contexts, raw):
                d = _parse_obj(text)
                if isinstance(d, dict):   # normalize the teacher's cosmetic deviations
                    if isinstance(d.get("next_speaker"), str):
                        d["next_speaker"] = d["next_speaker"].lower()
                    if isinstance(d.get("beat_type"), str):
                        d["beat_type"] = d["beat_type"].lower()
                    if isinstance(d.get("intensity"), (int, float)):
                        d["intensity"] = min(5, max(1, int(d["intensity"])))
                if d is not None and contracts.valid_decision(d):
                    fh.write(json.dumps({"user": user, "teacher_decision": d},
                                        ensure_ascii=False) + "\n")
                    kept_n += 1
                elif shown < 3:   # show raw rejects so a systematic failure is visible
                    shown += 1
                    print(f"--- label REJECT (parsed={d!r}) raw: {text[:300]!r}")
        vol.commit()
        print(f"{name}.jsonl: kept {kept_n}/{len(contexts)} labels")
        return

    if task in ("facts", "all"):
        cf_n = n if task == "facts" else facts_extra
        if cf_n > 0:
            rngs = [random.Random(base_seed + 20_000 + i) for i in range(cf_n)]
            specs = []
            for i in range(cf_n):
                s = _spec(rngs[i], contracts.STYLES[i % len(contracts.STYLES)])
                s["group_id"] = f"facts-{base_seed + 20_000 + i}"
                specs.append(s)
            r, rj, k = _gen_validated(specs, rngs, _facts_prompt, lambda s: 320,
                                      _facts_validate, base_seed + 20_000, "facts")
            rows += r; reasons.update(rj); kept["facts"] = k

    if task in ("game", "all"):
        rngs = [random.Random(base_seed + i) for i in range(n)]
        specs = []
        for i in range(n):
            s = _spec(rngs[i], contracts.STYLES[i % len(contracts.STYLES)])
            s["group_id"] = f"game-{base_seed + i}"
            specs.append(s)
        r, rj, k = _gen_validated(specs, rngs, _game_prompt,
                                  lambda s: 170 * s["budget"] + 400,
                                  _game_validate, base_seed, "game")
        rows += r; reasons.update(rj); kept["game"] = k

    if task in ("score", "all"):
        sspecs, prompts, metas, buckets = [], [], [], []
        for i in range(n):
            rng = random.Random(base_seed + 10_000 + i)
            spec = _spec(rng, contracts.STYLES[i % len(contracts.STYLES)])
            bucket = GUESS_BUCKETS[i % len(GUESS_BUCKETS)]
            guess = _guess(rng, spec, bucket)
            spec["group_id"] = f"score-{base_seed + 10_000 + i}"
            sspecs.append((spec, guess)); buckets.append(bucket)
            prompts.append(
                f"Hidden truth: the defendant is {contracts.article(spec['profession'])} "
                f"{spec['profession']} who {spec['fault']}.\nA player guessed: \"{guess}\" "
                f"(intended quality: {bucket}).\nGrade 0-100 (90-100 essentially correct on "
                f"BOTH job and charge; 40-70 partial; 0-20 wrong/unrelated). Return ONLY "
                f'JSON {{"score": <int>, "rationale": "<one sentence>"}}.')
            metas.append({"max_tokens": 160, "seed": base_seed + 10_000 + i})
        raw, _ = Teacher().generate.remote(prompts, metas)
        sk = 0
        for (spec, guess), text in zip(sspecs, raw):
            obj = _parse_obj(text)
            s_val = obj.get("score") if obj else None
            rat = obj.get("rationale") if obj else None
            if isinstance(s_val, (int, float)) and 0 <= int(s_val) <= 100 \
                    and isinstance(rat, str) and rat.strip():
                rows.append(_ex(contracts.SCORE_SYS,
                                contracts.score_user(spec["profession"], spec["fault"], guess),
                                {"score": int(s_val), "rationale": rat.strip()},
                                group_id=spec["group_id"]))
                sk += 1
            else:
                reasons["bad score"] += 1
        kept["score"] = sk

    random.Random(base_seed).shuffle(rows)
    name = out or ("director" if task == "all" else f"director_{task}")
    with open(f"{DATA}/{name}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {"dataset": name, "shape_version": contracts.SHAPE_VERSION,
                "teacher": TEACHER_MODEL, "task": task, "base_seed": base_seed,
                "requested": n, "facts_extra": facts_extra, "kept": kept,
                "examples": len(rows), "reject_reasons": dict(reasons)}
    with open(f"{DATA}/{name}.manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    vol.commit()
    print(f"\n{name}.jsonl: {len(rows)} examples; kept={kept}; rejects: {dict(reasons)}")
    for label, sysp in (("facts", contracts.FACTS_SYS), ("decide", contracts.GM_SYS),
                        ("score", contracts.SCORE_SYS)):
        same = [r for r in rows if r["conversations"][0]["content"] == sysp]
        print(f"\n===== {label}: {len(same)} examples — showing 1 =====")
        for r in same[:1]:
            print(f"  USER: {r['conversations'][1]['content'][:260]}")
            print(f"  ASST: {r['conversations'][2]['content'][:300]}")
    return len(rows)


@app.local_entrypoint()
def main(task: str = "all", n: int = 300, base_seed: int = 0, facts_extra: int = 0,
         out: str = "", source: str = ""):
    make_dataset.remote(task, n, base_seed, facts_extra, out, source)
