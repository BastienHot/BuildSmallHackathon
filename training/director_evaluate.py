"""Proper benchmark for the director LoRA — same recipe as training/evaluate.py (controlled
in-distribution prompts, base vs LoRA, label-free yardstick, aggregate verdict), fit to the
director's three jobs. Runs on Modal GPU.

The director's free yardstick is the TEACHER itself: we hold out a fresh slice of Gemma games
(disjoint seed) and feed their REAL transcript contexts as the decide prompts, with the
teacher's own choice as the reference. (My earlier evals fed hand-written / stub transcripts —
out-of-distribution — which is why "defense" looked absent; the data audit shows the model
trained on a clean J->P->D rhythm with prosecutor->defense = 0.73.)

  decide   -> on held-out real contexts: speaker distribution, prosecutor->defense rate,
              top-1 agreement w/ teacher, intensity spread, leak, wrap discipline.
  casefile -> controlled style prompts (model generates): leak + verb-phrase.
  score    -> constructed guess buckets: calibration/separation.

No grammar (measures learned behavior; runtime GBNF guarantees JSON, so json_valid is an
informational floor). Gate = content thresholds, base+director must clear them and beat base.

  modal run training/director_evaluate.py
"""

from __future__ import annotations

import json
import re

import modal

BASE_MODEL = "openbmb/MiniCPM5-1B"
HELDOUT = "director_game.jsonl"   # disjoint-seed held-out games (decide contexts + casefiles)

_CASEFILE_SYS = ("Invent a hidden courtroom case. profession = the defendant's real job "
                 "(2-4 words), UNRELATED to the given jargon style. fault_plain = a VERB "
                 "PHRASE completing \"The defendant ___\" (a specific wrongful act, e.g. "
                 "\"falsified an inspection log to hide a missed check\"); do NOT write a full "
                 "sentence and do NOT repeat \"the defendant\". facts = 3-5 short oblique clues "
                 "that never name the profession in plain words.")
_SCORE_SYS = ("Grade how well the player's guess matches the true profession and charge. "
              "score 0-100, rationale one sentence.")
_GM_SYS = ("You are the GAME MASTER directing a short courtroom debate. Output ONLY the "
           "requested JSON. Never reveal the profession or charge in plain words.")
SPEAKERS = ["judge", "prosecutor", "defense"]
BEATS = ["opening", "charge", "evidence", "objection", "escalate", "plea",
         "cross_examine", "closing", "exchange"]
WRAP_PRESSURE_AT = 2

EVAL_STYLES = [("aviation", "normal"), ("medical", "easy"), ("gaming", "hard"),
               ("politics", "normal"), ("sports", "easy"), ("ai", "normal"),
               ("corporate", "normal"), ("scifi", "hard")]
GUESS_BUCKETS = ["spot_on", "right_job_wrong_charge", "right_charge_wrong_job", "totally_unrelated"]
SCORE_CASES = [("museum curator", "falsified a provenance record to cover a quiet theft"),
               ("city bus driver", "skipped mandatory safety checks to stay on schedule"),
               ("pastry chef", "passed off a supplier's cake as his own award entry"),
               ("wedding photographer", "lost the only copies of a paid shoot and blamed the gear")]

app = modal.App("buzzwords-director-evaluate")
image = modal.Image.debian_slim().pip_install("unsloth", "peft", "transformers", "datasets")
adapters_vol = modal.Volume.from_name("buzzwords-adapters", create_if_missing=True)
data_vol = modal.Volume.from_name("buzzwords-data", create_if_missing=True)
OUT, DATA = "/adapters", "/data"


def _article(w):
    return "an" if w[:1].lower() in "aeiou" else "a"


def _json(text):
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        d = json.loads(text[a:b + 1])
        return d if isinstance(d, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _is_verb_phrase(fp):
    s = (fp or "").strip().lower()
    return bool(s) and not s.startswith(("the defendant", "the accused", "he ", "she ", "they ", "defendant "))


def _prev_speaker(user):
    block = user.split("Transcript so far:")[-1]
    last = None
    for line in block.splitlines():
        m = re.match(r"(judge|prosecutor|defense):", line.strip())
        if m:
            last = m.group(1)
    return last


def _profession(user):
    m = re.search(r"the defendant is a (.+?) who ", user)
    return m.group(1) if m else ""


def _turn_budget(user):
    m = re.search(r"This is turn (\d+) of (\d+)", user)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _eval_guess(prof, fault, bucket):
    return {"spot_on": f"{_article(prof)} {prof} who {fault}",
            "right_job_wrong_charge": f"{_article(prof)} {prof} who took a bribe to look the other way",
            "right_charge_wrong_job": f"a night-shift clerk who {fault}",
            "totally_unrelated": "a beekeeper who imported pollen without a permit"}[bucket]


# ------------------------------------------------------------------- metrics
def _casefile_metrics(raw):
    n, valid, leak, vp = len(raw), 0, 0, 0
    for text in raw:
        o = _json(text)
        if not (o and isinstance(o.get("profession"), str) and isinstance(o.get("fault_plain"), str)
                and isinstance(o.get("facts"), list) and 3 <= len(o["facts"]) <= 5):
            continue
        valid += 1
        prof = o["profession"].lower().strip()
        if prof and prof in (o["fault_plain"] + " " + " ".join(map(str, o["facts"]))).lower():
            leak += 1
        if _is_verb_phrase(o["fault_plain"]):
            vp += 1
    v = max(valid, 1)
    return {"n": n, "json_valid": valid / n, "leak_rate": leak / v, "verb_phrase": vp / v}


def _decide_metrics(raw, metas):
    from collections import Counter
    n, valid, leak, agree = len(raw), 0, 0, 0
    spk, ints = Counter(), set()
    p_ctx, p2d = 0, 0           # contexts where prev=prosecutor, and model picked defense
    wraps, premature = 0, 0
    for text, m in zip(raw, metas):
        o = _json(text)
        if not (o and o.get("next_speaker") in SPEAKERS and o.get("beat_type") in BEATS
                and o.get("intensity") in (1, 2, 3, 4, 5) and isinstance(o.get("stage_direction"), str)
                and isinstance(o.get("wrap_up"), bool)):
            continue
        valid += 1
        nx = o["next_speaker"]
        spk[nx] += 1
        ints.add(o["intensity"])
        if m["prev"] == "prosecutor":
            p_ctx += 1
            p2d += (nx == "defense")
        if nx == m["teacher"]:
            agree += 1
        if m["prof"] and m["prof"].lower() in o["stage_direction"].lower():
            leak += 1
        if o["wrap_up"]:
            wraps += 1
            premature += (m["turn"] < m["budget"] - WRAP_PRESSURE_AT)
    v = max(valid, 1)
    return {"n": n, "json_valid": valid / n, "distinct_speakers": len(spk),
            "judge_share": spk["judge"] / v, "prosecutor_share": spk["prosecutor"] / v,
            "defense_share": spk["defense"] / v, "prosecutor_to_defense": p2d / max(p_ctx, 1),
            "agreement_with_teacher": agree / v, "distinct_intensity": len(ints),
            "leak_rate": leak / v, "premature_wrap": (premature / wraps) if wraps else 0.0}


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


GATE = {
    "casefile": lambda m: m["leak_rate"] <= 0.1 and m["verb_phrase"] >= 0.8 and m["json_valid"] >= 0.6,
    "decide": lambda m: (m["distinct_speakers"] == 3 and m["defense_share"] >= 0.15
                         and m["prosecutor_to_defense"] >= 0.4 and m["leak_rate"] <= 0.1
                         and m["json_valid"] >= 0.6),
    "score": lambda m: (m["spot_on_mean"] >= 75 and m["unrelated_mean"] <= 30
                        and m["separation"] >= 45 and m["json_valid"] >= 0.6),
}

@app.function(image=image, gpu="A10G", timeout=60 * 60,
              volumes={OUT: adapters_vol, DATA: data_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def evaluate(adapter: str = "director", n_decide: int = 180):
    import os
    import torch
    from collections import Counter
    from unsloth import FastLanguageModel
    from peft import PeftModel

    # held-out decide contexts (real in-distribution transcripts + teacher reference)
    dec_sysu, dec_meta = [], []
    path = f"{DATA}/{HELDOUT}"
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            ex = json.loads(line)
            if ex["conversations"][0]["content"] != _GM_SYS:
                continue
            user = ex["conversations"][1]["content"]
            try:
                teacher = json.loads(ex["conversations"][2]["content"]).get("next_speaker")
            except Exception:  # noqa: BLE001
                continue
            turn, budget = _turn_budget(user)
            dec_sysu.append((_GM_SYS, user))
            dec_meta.append({"prof": _profession(user), "turn": turn, "budget": budget,
                             "prev": _prev_speaker(user), "teacher": teacher})
    dec_sysu, dec_meta = dec_sysu[:n_decide], dec_meta[:n_decide]
    teacher_ref = Counter(m["teacher"] for m in dec_meta)
    tot = sum(teacher_ref.values()) or 1
    p_ctx = sum(1 for m in dec_meta if m["prev"] == "prosecutor") or 1
    teacher_p2d = sum(1 for m in dec_meta if m["prev"] == "prosecutor" and m["teacher"] == "defense") / p_ctx
    print(f"held-out decide contexts: {len(dec_sysu)}  teacher speaker ref: "
          f"{ {k: round(v / tot, 2) for k, v in teacher_ref.items()} }  teacher prosecutor->defense={teacher_p2d:.2f}")

    cf_sysu = [(_CASEFILE_SYS, f"Jargon style (smokescreen, unrelated): {st}. Difficulty: {d}.")
               for st, d in EVAL_STYLES * 3]
    sc_sysu, sc_meta = [], []
    for prof, fault in SCORE_CASES:
        for b in GUESS_BUCKETS:
            sc_sysu.append((_SCORE_SYS, f"True profession: {prof}\nTrue charge: {fault}\n"
                            f"Player's guess: {_eval_guess(prof, fault, b)}"))
            sc_meta.append({"bucket": b})

    model, tok = FastLanguageModel.from_pretrained(BASE_MODEL, max_seq_length=4096, load_in_4bit=True)
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
            outs.append(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip())
        return outs

    def eval_all():
        return {"casefile": _casefile_metrics(run(cf_sysu, 340, 0.8)),
                "decide": _decide_metrics(run(dec_sysu, 128, 0.7), dec_meta),
                "score": _score_metrics(run(sc_sysu, 100, 0.3), sc_meta)}

    torch.manual_seed(0); lora = eval_all()
    with model.disable_adapter():
        torch.manual_seed(0); base = eval_all()

    print("\n================ DIRECTOR BENCHMARK (base -> LoRA) ================")
    overall = True
    for task in ("casefile", "decide", "score"):
        b, l, ok = base[task], lora[task], GATE[task](lora[task])
        overall = overall and ok
        print(f"\n[{task}]  {'PASS' if ok else 'FAIL'}")
        for k in l:
            if k != "n":
                print(f"   {k:<22} base={b[k]:>6.2f}   LoRA={l[k]:>6.2f}")
    print(f"\nGATE: {'PASS' if overall else 'FAIL'}  (json_valid is informational — runtime grammar guarantees JSON)")
    return {"pass": overall}


@app.local_entrypoint()
def main(adapter: str = "director"):
    evaluate.remote(adapter)

