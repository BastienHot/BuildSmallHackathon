"""Held-out, in-distribution benchmark for the director LoRA (Modal GPU, transformers).

Same recipe as before (REBUILD_REVIEW.md §3.4) on the new contracts shapes: base vs
LoRA over a disjoint-seed held-out slice of teacher games, the teacher's own choices
as reference labels, no grammar (measures learned behavior; runtime GBNF guarantees
JSON, so json_valid is an informational floor only).

New in the rebuild:
  * teacher SELF-agreement baseline (§7.4): if director_labels.jsonl exists (made by
    `director_datagen.py --task label --source director_game_heldout.jsonl`), the
    student's agreement is reported NEXT TO how often the resampled teacher agrees
    with itself — the number that makes 0.8 interpretable.
  * facts task replaces casefile (the truth is sampled in code now; the model only
    writes oblique facts) and fact_index validity is checked on decide.

Prereq: a held-out slice, e.g.
  modal run training/director_datagen.py --task game --n 40 --base-seed 900000 --out director_game_heldout
Then:
  modal run training/director_evaluate.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root -> buzzwords pkg

from buzzwords import contracts, pools

BASE_MODEL = "openbmb/MiniCPM5-1B"
HELDOUT = "director_game_heldout.jsonl"
LABELS = "director_labels.jsonl"          # optional: teacher resampled on the same contexts

GUESS_BUCKETS = ["spot_on", "right_job_wrong_charge", "right_charge_wrong_job", "totally_unrelated"]

GATE = {
    # one line of rationale each — regression tripwires, not publishable claims
    # facts leak gate is 0.10 at the MODEL level: the runtime filters every fact through
    # the same contracts.leaks() check with retry + fallback (pipeline.new_case), so any
    # token-level leak counted here is unshippable by construction; the e2e gate measures
    # the actual shipped surface and requires 0 there.
    "facts": lambda m: m["leak_rate"] <= 0.10 and m["json_valid"] >= 0.6,
    "decide": lambda m: (m["distinct_speakers"] == 3 and m["defense_share"] >= 0.15
                         and m["prosecutor_to_defense"] >= 0.4 and m["leak_rate"] <= 0.1
                         and m["fact_index_valid"] >= 0.9 and m["json_valid"] >= 0.6),
    "score": lambda m: (m["spot_on_mean"] >= 75 and m["unrelated_mean"] <= 30
                        and m["separation"] >= 45 and m["json_valid"] >= 0.6),
}

app = modal.App("buzzwords-director-evaluate")
image = (modal.Image.debian_slim()
         .pip_install("unsloth", "peft", "transformers", "datasets")
         .add_local_python_source("buzzwords"))
adapters_vol = modal.Volume.from_name("buzzwords-adapters", create_if_missing=True)
data_vol = modal.Volume.from_name("buzzwords-data", create_if_missing=True)
OUT, DATA = "/adapters", "/data"


def _json(text):
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        d = json.loads(text[a:b + 1])
        return d if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _prev_speaker(user):
    last = None
    for line in user.split("Transcript so far:")[-1].splitlines():
        m = re.match(r"(judge|prosecutor|defense):", line.strip())
        if m:
            last = m.group(1)
    return last


def _profession(user):
    m = re.search(r"the defendant is an? (.+?) who ", user)
    return m.group(1) if m else ""


def _eval_guess(prof, fault, bucket):
    a = contracts.article
    others = [f for f in pools.PROFESSIONS[prof][1] if f != fault]
    return {"spot_on": f"{a(prof)} {prof} who {fault}",
            "right_job_wrong_charge": f"{a(prof)} {prof} who "
                                      + (others[0] if others else "took a bribe"),
            "right_charge_wrong_job": f"a night-shift clerk who {fault}",
            "totally_unrelated": "a beekeeper who imported pollen without a permit"}[bucket]


# ------------------------------------------------------------------- metrics
def _facts_metrics(raw, metas):
    n, valid, leak = len(raw), 0, 0
    for text, m in zip(raw, metas):
        o = _json(text)
        facts = o.get("facts") if o else None
        if not (isinstance(facts, list)
                and contracts.MIN_FACTS <= len(facts) <= contracts.MAX_FACTS):
            continue
        valid += 1
        if any(contracts.leaks(str(f), m["prof"]) for f in facts):
            leak += 1
    v = max(valid, 1)
    return {"n": n, "json_valid": valid / n, "leak_rate": leak / v}


def _decide_metrics(raw, metas):
    from collections import Counter
    n, valid, leak, agree, fi_valid = len(raw), 0, 0, 0, 0
    spk, ints = Counter(), set()
    p_ctx, p2d = 0, 0
    for text, m in zip(raw, metas):
        o = _json(text)
        if not (o and contracts.valid_decision(o)):
            continue
        valid += 1
        nx = o["next_speaker"]
        spk[nx] += 1
        ints.add(o["intensity"])
        fi = o.get("fact_index")
        fi_valid += (fi is None or fi < m["n_facts"])
        if m["prev"] == "prosecutor":
            p_ctx += 1
            p2d += (nx == "defense")
        agree += (nx == m["teacher"])
        if m["prof"] and contracts.leaks(o["stage_direction"], m["prof"]):
            leak += 1
    v = max(valid, 1)
    return {"n": n, "json_valid": valid / n, "distinct_speakers": len(spk),
            "judge_share": spk["judge"] / v, "prosecutor_share": spk["prosecutor"] / v,
            "defense_share": spk["defense"] / v, "prosecutor_to_defense": p2d / max(p_ctx, 1),
            "agreement_with_teacher": agree / v, "fact_index_valid": fi_valid / v,
            "distinct_intensity": len(ints), "leak_rate": leak / v}


def _score_metrics(raw, metas):
    from statistics import mean
    from collections import defaultdict
    n, valid, by = len(raw), 0, defaultdict(list)
    for text, m in zip(raw, metas):
        o = _json(text)
        s = o.get("score") if o else None
        if isinstance(s, (int, float)) and 0 <= s <= 100:
            valid += 1
            by[m["bucket"]].append(int(s))
    spot = mean(by["spot_on"]) if by["spot_on"] else 0.0
    unrel = mean(by["totally_unrelated"]) if by["totally_unrelated"] else 100.0
    return {"n": n, "json_valid": valid / n, "spot_on_mean": spot,
            "unrelated_mean": unrel, "separation": spot - unrel}


@app.function(image=image, gpu="A10G", timeout=60 * 60,
              volumes={OUT: adapters_vol, DATA: data_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def evaluate(adapter: str = "director", n_decide: int = 180):
    import os
    import random
    import torch
    from collections import Counter
    from unsloth import FastLanguageModel
    from peft import PeftModel

    # ---- held-out decide contexts (real teacher transcripts + reference label) ----
    dec_sysu, dec_meta = [], []
    teacher_targets = {}     # user -> teacher's original next_speaker (for self-agreement)
    path = f"{DATA}/{HELDOUT}"
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            ex = json.loads(line)
            if ex["conversations"][0]["content"] != contracts.GM_SYS:
                continue
            user = ex["conversations"][1]["content"]
            try:
                target = json.loads(ex["conversations"][2]["content"])
            except Exception:  # noqa: BLE001
                continue
            n_facts = len(re.findall(r"^\s+\d+: ", user.split("Transcript so far:")[0],
                                     flags=re.M))
            dec_sysu.append((contracts.GM_SYS, user))
            dec_meta.append({"prof": _profession(user), "prev": _prev_speaker(user),
                             "teacher": target.get("next_speaker"),
                             "n_facts": max(n_facts, contracts.MIN_FACTS)})
            teacher_targets[user] = target.get("next_speaker")
    dec_sysu, dec_meta = dec_sysu[:n_decide], dec_meta[:n_decide]
    ref = Counter(m["teacher"] for m in dec_meta)
    tot = sum(ref.values()) or 1
    print(f"held-out decide contexts: {len(dec_sysu)}  teacher speaker ref: "
          f"{ {k: round(v / tot, 2) for k, v in ref.items()} }")

    # ---- teacher SELF-agreement baseline (§7.4), if the labels file exists ----
    self_agree = None
    lpath = f"{DATA}/{LABELS}"
    if os.path.exists(lpath):
        match, count = 0, 0
        for line in open(lpath, encoding="utf-8"):
            row = json.loads(line)
            orig = teacher_targets.get(row["user"])
            if orig:
                count += 1
                match += (row["teacher_decision"].get("next_speaker") == orig)
        self_agree = match / count if count else None
        print(f"teacher self-agreement (resampled vs original, n={count}): {self_agree}")

    # ---- facts + score prompt sets (controlled, from pools/contracts) ----
    rng = random.Random(0)
    cf_sysu, cf_meta = [], []
    for i, style in enumerate(contracts.STYLES * 3):
        prof, fault = pools.sample_case(random.Random(1000 + i), style)
        cf_sysu.append((contracts.FACTS_SYS, contracts.facts_user(style, prof, fault)))
        cf_meta.append({"prof": prof})
    sc_sysu, sc_meta = [], []
    for i in range(8):
        prof, fault = pools.sample_case(random.Random(2000 + i), rng.choice(contracts.STYLES))
        for b in GUESS_BUCKETS:
            sc_sysu.append((contracts.SCORE_SYS,
                            contracts.score_user(prof, fault, _eval_guess(prof, fault, b))))
            sc_meta.append({"bucket": b})

    model, tok = FastLanguageModel.from_pretrained(BASE_MODEL, max_seq_length=4096,
                                                   load_in_4bit=False,
                                                   dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, f"{OUT}/{adapter}")

    def run(sysu, max_new, temp):
        outs = []
        for system, user in sysu:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            ids = tok(text, return_tensors="pt").to(model.device)
            out = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                                 temperature=temp, top_p=0.95)
            outs.append(tok.decode(out[0][ids.input_ids.shape[1]:],
                                   skip_special_tokens=True).strip())
        return outs

    def eval_all():
        return {"facts": _facts_metrics(run(cf_sysu, 280, 0.8), cf_meta),
                "decide": _decide_metrics(run(dec_sysu, 140, 0.7), dec_meta),
                "score": _score_metrics(run(sc_sysu, 100, 0.3), sc_meta)}

    torch.manual_seed(0); lora = eval_all()
    with model.disable_adapter():
        torch.manual_seed(0); base = eval_all()

    print("\n================ DIRECTOR BENCHMARK (base -> LoRA) ================")
    overall = True
    for task in ("facts", "decide", "score"):
        b, l, ok = base[task], lora[task], GATE[task](lora[task])
        overall = overall and ok
        print(f"\n[{task}]  {'PASS' if ok else 'FAIL'}")
        for k in l:
            if k != "n":
                print(f"   {k:<24} base={b[k]:>6.2f}   LoRA={l[k]:>6.2f}")
    if self_agree is not None:
        print(f"\nagreement ceiling: student={lora['decide']['agreement_with_teacher']:.2f}"
              f" vs teacher-self={self_agree:.2f}")
    print(f"\nGATE: {'PASS' if overall else 'FAIL'}  "
          "(json_valid is informational — runtime grammar guarantees JSON)")
    return {"pass": overall}


@app.local_entrypoint()
def main(adapter: str = "director"):
    evaluate.remote(adapter)
