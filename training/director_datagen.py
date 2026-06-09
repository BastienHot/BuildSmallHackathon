"""Synthetic data generation for the DIRECTOR (Game Master) LoRA, on Modal GPU.

Mirrors training/teacher_datagen.py (same Gemma 4 31B-it FP8 teacher, same Modal/vLLM
setup, same per-sample SEEDED diversity, same "buzzwords-data" Volume), but produces ONE
multitask dataset for the single director adapter -- the GM owns one shared "case
world-model" (casefile creates the truth, decide uses it obliquely, score judges it), so
the three responsibilities live in one LoRA (split later only if the gate shows a weak
sub-task). Examples are emitted in the EXACT runtime call shapes (buzzwords/text_engine.py):

  * casefile -- invent the hidden case; profession UNRELATED to the jargon (the smokescreen).
  * decide   -- one beat decision over the transcript so far (next_speaker / beat_type /
                intensity / oblique stage_direction / wrap_up).
  * score    -- grade a player's guess against the hidden truth (0-100 + rationale).

SMOKESCREEN BY CONSTRUCTION (the fix for the base-1B failure): exactly as the actor
pipeline injects the hidden truth from the seed, the profession is drawn from a pool with
the jargon's OWN domain excluded (STYLE_EXCLUDE), so every casefile target is guaranteed
profession ⟂ style. The teacher only writes the oblique phrasing + the directing.

Smoke-test tiny and eyeball the printed samples before scaling up:

  modal run training/director_datagen.py --task casefile --n 4
  modal run training/director_datagen.py --task game --n 4      # casefile + per-turn decide
  modal run training/director_datagen.py --task score --n 4
  modal run training/director_datagen.py                        # full mixed director.jsonl
Outputs land in the "buzzwords-data" Volume next to the actor data.
"""

from __future__ import annotations

import json
import os
import random

import modal

# --- diversity axes: mirror training/teacher_datagen.py (kept in sync by hand) ---
STYLES = ["aviation", "corporate", "ai", "politics", "medical", "gaming", "sports", "scifi"]
DIFFICULTIES = {"easy": 8, "normal": 12, "hard": 16}   # mirrors config.TURN_BUDGET_BY_DIFFICULTY
PROFESSIONS = [
    "airline pilot", "pastry chef", "marine biologist", "tattoo artist",
    "city bus driver", "wedding photographer", "high-school chemistry teacher",
    "plumber", "air-traffic controller", "museum curator", "florist", "locksmith",
]
STYLE_EXCLUDE = {  # keep the profession out of the jargon's OWN domain (smokescreen)
    "aviation": {"airline pilot", "air-traffic controller"},
    "corporate": {"management consultant"},
}
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
SPEAKERS = ["judge", "prosecutor", "defense"]
BEATS = ["opening", "charge", "evidence", "objection", "escalate", "plea",
         "cross_examine", "closing", "exchange"]
# how a guess relates to the truth -> spreads the scorer's calibration across the range
GUESS_BUCKETS = ["spot_on", "close_paraphrase", "right_job_wrong_charge",
                 "right_charge_wrong_job", "plausible_but_wrong", "vague", "totally_unrelated"]
WRAP_PRESSURE_AT = 2   # mirrors config.WRAP_PRESSURE_AT

# --- runtime call shapes: these strings MUST match buzzwords/text_engine.py exactly ---
_CASEFILE_SYS = ("Invent a hidden courtroom case. profession = the defendant's real job "
                 "(2-4 words), UNRELATED to the given jargon style. fault_plain = a VERB "
                 "PHRASE completing \"The defendant ___\" (a specific wrongful act, e.g. "
                 "\"falsified an inspection log to hide a missed check\"); do NOT write a full "
                 "sentence and do NOT repeat \"the defendant\". facts = 3-5 short oblique clues "
                 "that never name the profession in plain words.")
_GM_SYS = ("You are the GAME MASTER directing a short courtroom debate. Output ONLY the "
           "requested JSON. Never reveal the profession or charge in plain words.")
_SCORE_SYS = ("Grade how well the player's guess matches the true profession and charge. "
              "score 0-100, rationale one sentence.")

TEACHER_SYS = ("You are the show-runner and head writer of a comedy-legal courtroom guessing "
               "game: you design hidden cases, direct the beats, and grade guesses. Output "
               "strictly valid JSON and nothing else.")

# Same teacher + Modal setup as teacher_datagen.py.
TEACHER_MODEL = os.getenv("BW_TEACHER_MODEL", "RedHatAI/gemma-4-31B-it-FP8-block")
QUANTIZATION = os.getenv("BW_TEACHER_QUANT", "")
TEACHER_GPU = os.getenv("BW_TEACHER_GPU", "L40S")

app = modal.App("buzzwords-director-teacher")
image = (modal.Image.debian_slim()
         .pip_install("vllm")
         .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"}))
vol = modal.Volume.from_name("buzzwords-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
DATA = "/data"
HF_CACHE = "/root/.cache/huggingface"


# --------------------------------------------------------------- prompt helpers
def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


# Hand-written GOLD case files. We sample a couple per prompt (not all, and seeded) so the
# teacher learns the PATTERN — facts hint the job but never name it; fault_plain is a verb
# phrase — without copying a fixed exemplar (sampling => diversity + a leak-resistance net).
FEWSHOTS = [
    {"style": "corporate", "profession": "marine biologist",
     "fault_plain": "released unverified findings a rival later had to retract",
     "facts": ["a sample set went missing before peer review",
               "the tide tables in the appendix were back-dated",
               "a junior was credited, then quietly removed"]},
    {"style": "aviation", "profession": "pastry chef",
     "fault_plain": "served a batch he knew had spoiled rather than miss a large order",
     "facts": ["the cold-storage log had a six-hour gap",
               "two regulars filed the same complaint that evening",
               "the disposal bin was emptied hours early"]},
    {"style": "gaming", "profession": "locksmith",
     "fault_plain": "kept copies of a client's keys and let himself in uninvited",
     "facts": ["a spare blank was cut after hours",
               "entry used the correct code, never forced",
               "nothing was taken, but a private drawer had been rifled"]},
    {"style": "medical", "profession": "wedding photographer",
     "fault_plain": "deleted the only copies of an event she was paid to cover",
     "facts": ["the backup drive was reformatted the next morning",
               "the contract had promised redundant storage",
               "the client was told it was a hardware fault"]},
    {"style": "politics", "profession": "plumber",
     "fault_plain": "signed off on work an unlicensed trainee had actually done",
     "facts": ["the sign-off sheet shows one name in two hands",
               "the permit number belongs to a different address",
               "a follow-up visit was billed but never made"]},
    {"style": "scifi", "profession": "florist",
     "fault_plain": "swapped in cheaper stock and pocketed the difference on a big order",
     "facts": ["the manifest lists varieties that were out of season",
               "a supplier invoice was altered by hand",
               "a regular noticed the arrangements had quietly changed"]},
]


def _banned(profession: str) -> str:
    """The profession words the teacher must never write into the case file."""
    toks, seen, out = [profession] + [w for w in profession.split() if len(w) > 3], set(), []
    for t in toks:
        if t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
    return ", ".join(f'"{t}"' for t in out)


def _fewshot_block(rng: random.Random, k: int = 2) -> str:
    lines = []
    for ex in rng.sample(FEWSHOTS, min(k, len(FEWSHOTS))):
        cf = {"profession": ex["profession"], "fault_plain": ex["fault_plain"], "facts": ex["facts"]}
        lines.append(f'  ({ex["style"]} jargon, hidden job "{ex["profession"]}"): '
                     + json.dumps(cf, ensure_ascii=False))
    return "\n".join(lines)


def _corrective(base_prompt: str, profession: str, reason: str) -> str:
    return (f"A previous attempt was REJECTED ({reason}). In particular, NEVER write any of "
            f"these words in fault_plain or facts: {_banned(profession)}. Redo it correctly, "
            f"keeping every clue oblique.\n\n" + base_prompt)


def _gm_prompt(cf: dict, transcript: list[tuple[str, str]], turn: int, budget: int) -> str:
    """Copy of buzzwords/text_engine.py:_gm_prompt -- MUST stay in sync (it is the exact
    user message the director sees at runtime, so training must match it token-for-token)."""
    if turn >= budget - 1:
        pressure = "This MUST be the closing beat: set wrap_up=true."
    elif turn >= budget - WRAP_PRESSURE_AT:
        pressure = "Begin converging; set wrap_up=true once a verdict is natural."
    else:
        pressure = ""
    recent = "\n".join(f"{a}: {t}" for a, t in transcript[-6:]) or "(no lines yet)"
    return (f"Hidden brief (keep oblique): the defendant is a {cf['profession']} who "
            f"{cf['fault_plain']}. Facts: {'; '.join(cf['facts'])}\n"
            f"Transcript so far:\n{recent}\n"
            f"This is turn {turn + 1} of {budget}. {pressure}")


def _game_spec(rng: random.Random, style: str) -> dict:
    professions = [p for p in PROFESSIONS if p not in STYLE_EXCLUDE.get(style, set())]
    diff = rng.choice(list(DIFFICULTIES))
    return {"style": style, "difficulty": diff, "budget": DIFFICULTIES[diff],
            "profession": rng.choice(professions), "fault": rng.choice(FAULT_ARCHETYPES),
            "tone": rng.choice(TONES), "disposition": rng.choice(DISPOSITIONS),
            "severity": rng.randint(2, 5)}


def _game_prompt(spec: dict, rng: random.Random) -> str:
    """Ask the teacher to DESIGN + DIRECT a full case as one JSON trace. Profession is FIXED
    from the seed (smokescreen by construction); the teacher writes the verb-phrase fault,
    oblique facts, and the turn-by-turn {decision, line}. Sampled few-shots demonstrate it."""
    return (
        "Examples of well-formed hidden CASE FILES — note how the facts HINT at the real job "
        f"without ever naming it, and fault_plain is a verb phrase:\n{_fewshot_block(rng)}\n\n"
        f"Now design and DIRECT a NEW case (do NOT reuse the examples).\n"
        f"FIXED hidden profession (do NOT change it; NEVER write it in plain words anywhere): "
        f"{_article(spec['profession'])} {spec['profession']}. The defendant {spec['fault']}.\n"
        f"These word(s) must NOT appear in fault_plain, facts, or any stage_direction: "
        f"{_banned(spec['profession'])}.\n"
        f"The courtroom speaks dense {spec['style']} jargon — a SMOKESCREEN unrelated to the "
        f"profession. Tone: {spec['tone']}; the defendant comes across as {spec['disposition']}; "
        f"severity {spec['severity']}/5.\n"
        f"Run EXACTLY {spec['budget']} turns: open, build evidence, escalate, then CONVERGE to a "
        f"verdict — set wrap_up=true only on the final 1-2 turns.\n"
        f"Each turn: pick next_speaker (judge|prosecutor|defense), beat_type (one of "
        f"{', '.join(BEATS)}), intensity 1-5, an oblique stage_direction (hints at the charge in "
        f"{spec['style']} terms but NEVER names the profession or the plain charge), wrap_up, and "
        f"the resulting in-jargon line.\n"
        f'Return ONLY JSON: {{"fault_plain": "<verb phrase completing \'The defendant ___\'>", '
        f'"facts": ["<3-5 oblique clues, none naming the profession>"], "turns": [{{"next_speaker": '
        f'"...", "beat_type": "...", "intensity": 3, "stage_direction": "...", "wrap_up": false, '
        f'"line": "..."}}]}}.'
    )


def _casefile_prompt(spec: dict, rng: random.Random) -> str:
    """Standalone case-file generation (no full trace) — cheap way to UPSAMPLE the
    make-or-break smokescreen skill. Profession is fixed from the seed."""
    return (
        "Examples of well-formed hidden CASE FILES (facts hint the job, never name it; "
        f"fault_plain is a verb phrase completing \"The defendant ___\"):\n{_fewshot_block(rng)}\n\n"
        f"Now invent ONE NEW case (do NOT reuse the examples). FIXED hidden profession (NEVER "
        f"write it in plain words): {_article(spec['profession'])} {spec['profession']}. "
        f"The defendant {spec['fault']}. The jargon style {spec['style']} is a SMOKESCREEN "
        f"unrelated to the profession.\n"
        f"These word(s) must NOT appear in fault_plain or facts: {_banned(spec['profession'])}.\n"
        f'Return ONLY JSON {{"fault_plain": "<verb phrase completing \'The defendant ___\'>", '
        f'"facts": ["<3-5 oblique clues, none naming the profession>"]}}.')


def _make_guess(rng: random.Random, spec: dict, bucket: str) -> str:
    """Synthesize a player guess of a seeded QUALITY, so the scorer sees the full range."""
    others_p = [p for p in PROFESSIONS if p != spec["profession"]]
    others_f = [f for f in FAULT_ARCHETYPES if f != spec["fault"]]
    p, f = spec["profession"], spec["fault"]
    op, of = rng.choice(others_p), rng.choice(others_f)
    return {
        "spot_on": f"{_article(p)} {p} who {f}",
        "close_paraphrase": f"I think they work as {_article(p)} {p} and {f}",
        "right_job_wrong_charge": f"{_article(p)} {p} who {of}",
        "right_charge_wrong_job": f"{_article(op)} {op} who {f}",
        "plausible_but_wrong": f"{_article(op)} {op} who {of}",
        "vague": "a professional who broke the rules of their job somehow",
        "totally_unrelated": f"{_article(op)} {op}",
    }[bucket]


# --------------------------------------------------------------------- teacher
@app.cls(image=image, gpu=TEACHER_GPU, volumes={DATA: vol, HF_CACHE: hf_cache},
         timeout=60 * 60, secrets=[modal.Secret.from_name("huggingface")])
class Teacher:
    @modal.enter()
    def load(self):
        from vllm import LLM
        # max_model_len=4096 (NOT 8192): same as teacher_datagen.py — after the ~31.7 GiB
        # FP8 weights + multimodal/CUDA-graph reservations on the 48 GiB L40S, 8192 leaves
        # too little KV cache (needs 6.89 GiB, only ~6.27 available) and the engine fails.
        self.llm = LLM(model=TEACHER_MODEL, quantization=(QUANTIZATION or None),
                       dtype="auto", max_model_len=4096, gpu_memory_utilization=0.92)

    @modal.method()
    def generate(self, prompts: list[str], metas: list[dict]) -> tuple[list[str], float, int]:
        import time
        from vllm import SamplingParams
        sps = [SamplingParams(temperature=1.0, top_p=0.95, repetition_penalty=1.05,
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
    """Extract the outermost JSON object, tolerating code fences / preamble."""
    a, b = text.find("{"), text.rfind("}")
    if a == -1 or b <= a:
        return None
    try:
        d = json.loads(text[a:b + 1])
    except Exception:
        return None
    return d if isinstance(d, dict) else None


def _leaks(text: str, profession: str) -> bool:
    return profession.lower() in (text or "").lower()


def _ex(system: str, user: str, assistant_obj: dict) -> dict:
    """One ShareGPT example; the assistant target is compact JSON (grammar-valid at runtime)."""
    return {"conversations": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": json.dumps(assistant_obj, ensure_ascii=False)}]}


def _game_to_examples(trace: dict, spec: dict):
    """Validate a director trace and explode it into 1 casefile + N decide examples.
    Returns (examples, reject_reason | None)."""
    fault_plain, facts, turns = trace.get("fault_plain"), trace.get("facts"), trace.get("turns")
    if not isinstance(fault_plain, str) or not isinstance(facts, list) or not isinstance(turns, list):
        return [], "bad shape"
    if not (3 <= len(facts) <= 5):
        return [], f"facts count {len(facts)}"
    if not (max(4, spec["budget"] // 2) <= len(turns) <= spec["budget"] + 1):
        return [], f"turn count {len(turns)}"
    prof = spec["profession"]
    if _leaks(fault_plain, prof) or any(_leaks(str(x), prof) for x in facts):
        return [], "profession leaked in casefile"

    cf = {"profession": prof, "fault_plain": fault_plain.strip(), "facts": [str(x).strip() for x in facts]}
    out = [_ex(_CASEFILE_SYS,
               f"Jargon style (smokescreen, unrelated): {spec['style']}. Difficulty: {spec['difficulty']}.",
               cf)]
    transcript: list[tuple[str, str]] = []
    for i, t in enumerate(turns):
        spk, beat = t.get("next_speaker"), t.get("beat_type")
        sd, line = str(t.get("stage_direction", "")), str(t.get("line", ""))
        intensity = t.get("intensity")
        if spk not in SPEAKERS or beat not in BEATS or intensity not in (1, 2, 3, 4, 5):
            return [], f"bad turn {i} ({spk}/{beat}/{intensity})"
        if _leaks(sd, prof):
            return [], f"profession leaked in stage_direction {i}"
        decision = {"next_speaker": spk, "beat_type": beat, "intensity": int(intensity),
                    "stage_direction": sd.strip(), "wrap_up": bool(t.get("wrap_up", False))}
        out.append(_ex(_GM_SYS, _gm_prompt(cf, transcript, i, spec["budget"]), decision))
        transcript.append((spk, line.strip()))
    return out, None


def _casefile_to_example(obj: dict, spec: dict):
    fp, facts = obj.get("fault_plain"), obj.get("facts")
    if not isinstance(fp, str) or not isinstance(facts, list) or not (3 <= len(facts) <= 5):
        return None, "bad casefile shape"
    if _leaks(fp, spec["profession"]) or any(_leaks(str(x), spec["profession"]) for x in facts):
        return None, "profession leaked in casefile"
    cf = {"profession": spec["profession"], "fault_plain": fp.strip(),
          "facts": [str(x).strip() for x in facts]}
    user = f"Jargon style (smokescreen, unrelated): {spec['style']}. Difficulty: {spec['difficulty']}."
    return _ex(_CASEFILE_SYS, user, cf), None


def _score_to_example(spec: dict, guess: str, obj: dict):
    score, rationale = obj.get("score"), obj.get("rationale")
    if not isinstance(score, (int, float)) or not isinstance(rationale, str) or not rationale.strip():
        return None, "bad score shape"
    score = int(score)
    if not (0 <= score <= 100):
        return None, f"score {score} out of range"
    user = f"True profession: {spec['profession']}\nTrue charge: {spec['fault']}\nPlayer's guess: {guess}"
    return _ex(_SCORE_SYS, user, {"score": score, "rationale": rationale.strip()}), None


def _game_validate(spec: dict, text: str):
    trace = _parse_obj(text)
    return ([], "unparseable") if trace is None else _game_to_examples(trace, spec)


def _casefile_validate(spec: dict, text: str):
    obj = _parse_obj(text)
    if obj is None:
        return [], "unparseable"
    ex, reason = _casefile_to_example(obj, spec)
    return ([] if reason else [ex]), reason


def _gen_validated(specs, rngs, build_prompt, build_meta, validate, base_seed, label):
    """Generate -> validate -> ONE corrective retry on the rejects (re-prompt naming the
    banned words). Turns the ~50% one-shot reject rate into near-full yield. -> (rows, reasons)."""
    from collections import Counter
    prompts = [build_prompt(specs[i], rngs[i]) for i in range(len(specs))]
    metas = [build_meta(specs[i], base_seed + i) for i in range(len(specs))]
    rows, reasons, pending, kept, shown = [], Counter(), list(range(len(specs))), 0, 0
    for attempt in range(2):
        if not pending:
            break
        raw, gen_s, _ = Teacher().generate.remote([prompts[i] for i in pending],
                                                  [metas[i] for i in pending])
        still = []
        for idx, text in zip(pending, raw):
            exs, reason = validate(specs[idx], text)
            if reason is None:
                rows.extend(exs); kept += 1
            else:
                still.append((idx, reason))
        if attempt == 0:   # build a corrective retry for the rejects
            for idx, reason in still:
                if shown < 3:
                    shown += 1
                    print(f"--- {label} retry [{reason}] hidden={specs[idx]['profession']} ---")
                prompts[idx] = _corrective(prompts[idx], specs[idx]["profession"], reason)
                metas[idx] = {**metas[idx], "seed": metas[idx]["seed"] + 500_000}
            pending = [idx for idx, _ in still]
        else:
            for _, reason in still:
                reasons[reason] += 1
            pending = []
        print(f"{label} attempt {attempt + 1}: kept {kept}/{len(specs)} ({gen_s:.0f}s)")
    return rows, reasons


# --------------------------------------------------------------- orchestration
@app.function(image=image, volumes={DATA: vol}, timeout=3 * 60 * 60)
def make_dataset(task: str, n: int, base_seed: int = 0, casefile_extra: int = 0):
    from collections import Counter
    rows, reasons, shown = [], Counter(), 0

    # casefile examples: standalone (task="casefile", count=n) OR upsampled inside "all"
    # (count=casefile_extra) — they're the make-or-break smokescreen skill, so we boost them.
    cf_n = n if task == "casefile" else (casefile_extra if task == "all" else 0)
    if cf_n > 0:
        rngs = [random.Random(base_seed + 20_000 + i) for i in range(cf_n)]
        specs = [_game_spec(rngs[i], STYLES[i % len(STYLES)]) for i in range(cf_n)]
        r, rj = _gen_validated(specs, rngs, _casefile_prompt,
                               lambda s, seed: {"max_tokens": 320, "seed": seed},
                               _casefile_validate, base_seed + 20_000, "casefile")
        rows += r
        reasons.update(rj)

    if task in ("game", "all"):
        # ~160 tok/turn keeps prompt+output within max_model_len=4096; few-shot block +
        # banned-word + a corrective retry round push the reject rate down (see _gen_validated).
        rngs = [random.Random(base_seed + i) for i in range(n)]
        specs = [_game_spec(rngs[i], STYLES[i % len(STYLES)]) for i in range(n)]
        r, rj = _gen_validated(specs, rngs, _game_prompt,
                               lambda s, seed: {"max_tokens": 160 * s["budget"] + 300, "seed": seed},
                               _game_validate, base_seed, "game")
        rows += r
        reasons.update(rj)

    if task in ("score", "all"):
        score_rows, kept = [], 0
        sspecs, guesses, prompts, metas = [], [], [], []
        for i in range(n):
            rng = random.Random(base_seed + 10_000 + i)
            spec = _game_spec(rng, STYLES[i % len(STYLES)])
            bucket = GUESS_BUCKETS[i % len(GUESS_BUCKETS)]
            guess = _make_guess(rng, spec, bucket)
            sspecs.append(spec); guesses.append(guess)
            prompts.append(f"Hidden truth: the defendant is {_article(spec['profession'])} "
                           f"{spec['profession']} who {spec['fault']}.\nA player guessed: \"{guess}\" "
                           f"(intended quality: {bucket}).\nGrade 0-100 (90-100 essentially correct on "
                           f"BOTH job and charge; 40-70 partial; 0-20 wrong/unrelated). Return ONLY "
                           f'JSON {{"score": <int>, "rationale": "<one sentence>"}}.')
            metas.append({"max_tokens": 160, "seed": base_seed + 10_000 + i})
        raw, gen_s, toks = Teacher().generate.remote(prompts, metas)
        for spec, guess, text in zip(sspecs, guesses, raw):
            obj = _parse_obj(text)
            ex, reason = (None, "unparseable") if obj is None else _score_to_example(spec, guess, obj)
            if reason is None:
                score_rows.append(ex); kept += 1
            else:
                reasons[reason] += 1
        rows.extend(score_rows)
        print(f"score: kept {kept}/{n} -> {len(score_rows)} examples ({gen_s:.0f}s)")

    random.Random(base_seed).shuffle(rows)
    name = "director.jsonl" if task == "all" else f"director_{task}.jsonl"
    with open(f"{DATA}/{name}", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    vol.commit()
    print(f"\n{name}: {len(rows)} examples total; reject reasons: {dict(reasons)}")
    for label, sysp in (("casefile", _CASEFILE_SYS), ("decide", _GM_SYS), ("score", _SCORE_SYS)):
        same = [r for r in rows if r["conversations"][0]["content"] == sysp]
        print(f"\n===== {label}: {len(same)} examples — showing up to 2 =====")
        for r in same[:2]:
            print(f"  USER: {r['conversations'][1]['content'][:220]}")
            print(f"  ASST: {r['conversations'][2]['content'][:300]}")
    return len(rows)


@app.local_entrypoint()
def main(task: str = "all", n: int = 200, base_seed: int = 0, casefile_extra: int = 0):
    # task "all" -> games + scores + casefile_extra standalone case files, all into director.jsonl.
    # "game"/"score"/"casefile" generate a single type for smoke tests.
    make_dataset.remote(task, n, base_seed, casefile_extra)
