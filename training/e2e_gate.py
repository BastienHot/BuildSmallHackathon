"""END-TO-END gate on SELF-ROLLOUTS: N full games, real llama.cpp + GBNF + REAL actors.

This is the ship/no-ship check (REBUILD_REVIEW.md §10.4). It plays N >= 10 complete
games across styles exactly as the app does — sampled truth from buzzwords.pools, the
director LoRA under grammar, the deterministic guards from buzzwords.contracts, and the
ACTUAL style-LoRA actors (no stubs) — then aggregates SEQUENCE metrics, not single
trajectories (§7.1) and not marginals only (§13.6):

  * raw guard-trigger rates (how often the model needed the seatbelt)
  * post-guard speaker shares, max same-speaker run, transition matrix
  * cross-line repetition (consecutive 4-gram overlap) + distinct-word ratio
  * profession leaks in lines/stage directions (must be 0)
  * wrap-within-budget rate + fact coverage (force rule => must be 1.0)
  * scorer calibration on constructed spot-on / unrelated guesses
  * SOLVABILITY (§7.2, the headline): the Gemma teacher reads each transcript
    (player view only) and guesses; deterministic bucket scoring; the average must
    land in a band — guessable but not given away.

Also dumps every decide context to director_contexts_selfplay.jsonl — point
`director_datagen.py --task label --source` at it for the DAgger pass (§13.6).

  modal run training/e2e_gate.py --n 12          # full gate (games on CPU, solver on GPU)
  modal run training/e2e_gate.py --n 4 --no-solve   # quick smoke, skip the teacher solver
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

BASE_REPO, BASE_FILE = "openbmb/MiniCPM5-1B-GGUF", "MiniCPM5-1B-Q4_K_M.gguf"
TEACHER_MODEL = os.getenv("BW_TEACHER_MODEL", "RedHatAI/gemma-4-31B-it-FP8-block")
TEACHER_GPU = os.getenv("BW_TEACHER_GPU", "L40S")

# Thresholds (regression tripwires, not publishable claims — one line of rationale each):
GATE = {
    "leaks": lambda m: m["leak_lines"] == 0,                  # one leak ends the game's premise
    "defense_share": lambda m: m["defense_share"] >= 0.20,    # guards guarantee presence; share shows health
    "max_run": lambda m: m["max_same_speaker_run"] <= 2,      # the §13.3 invariant, post-guard
    "guard_rate": lambda m: m["guard_trigger_rate"] <= 0.35,  # seatbelt firing >35% = model not learning sequencing
    "repetition": lambda m: m["consec_4gram_overlap"] <= 0.15,  # §13.5 echo failure ("your silence" x3)
    "wrap": lambda m: m["wrapped_rate"] >= 0.8,               # games must end on their own
    "facts": lambda m: m["fact_coverage"] >= 0.99,            # force rule => all clues reach the player
    "score_sep": lambda m: m["score_spot_mean"] - m["score_unrel_mean"] >= 45,  # §7.3 separation
    "solvability": lambda m: m["solver_mean"] is None or 30 <= m["solver_mean"] <= 85,  # §7.2 band
}

app = modal.App("buzzwords-e2e-gate")
cpu_image = (modal.Image.debian_slim()
             .apt_install("build-essential", "cmake", "git")
             .env({"CMAKE_ARGS": "-DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_NATIVE=OFF"})
             .pip_install("llama-cpp-python", "huggingface_hub")
             .add_local_python_source("buzzwords"))
gpu_image = (modal.Image.debian_slim()
             .pip_install("vllm")
             .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
             .add_local_python_source("buzzwords"))
gguf_vol = modal.Volume.from_name("buzzwords-gguf", create_if_missing=True)
data_vol = modal.Volume.from_name("buzzwords-data", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
GGUF, DATA = "/gguf", "/data"


# ----------------------------------------------------------------- game playing
@app.function(image=cpu_image, timeout=3 * 60 * 60,
              volumes={GGUF: gguf_vol, DATA: data_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def play_games(n: int, base_seed: int = 0) -> list[dict]:
    """Play n full self-rollout games (round-robin styles). Returns per-game records
    and dumps the decide contexts for the DAgger/label path."""
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama, LlamaGrammar

    base = hf_hub_download(BASE_REPO, BASE_FILE)
    director = f"{GGUF}/director.lora.gguf"
    if not os.path.exists(director):
        raise RuntimeError(f"{director} not found — run convert_gguf first")
    gm = Llama(model_path=base, lora_path=director, n_ctx=8192, n_gpu_layers=0, verbose=False)
    actor_cache: dict[str, Llama] = {}

    def actor_model(style: str) -> Llama:
        lora = f"{GGUF}/style-{style}.lora.gguf"
        key = style if os.path.exists(lora) else "__base__"
        if key not in actor_cache:
            actor_cache.clear()   # one actor resident at a time
            actor_cache[key] = Llama(model_path=base,
                                     lora_path=lora if key != "__base__" else None,
                                     n_ctx=4096, n_gpu_layers=0, verbose=False)
        return actor_cache[key]

    def jcall(model, system, user, grammar, max_tokens, temp):
        out = model.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            grammar=LlamaGrammar.from_string(grammar), max_tokens=max_tokens,
            temperature=temp)
        return json.loads(out["choices"][0]["message"]["content"])

    def actor_say(style: str, system: str, user: str) -> str:
        """Raw-completion actor call rendering the EXACT training/runtime prompt shape
        (ChatML + the empty think block MiniCPM5 emits under enable_thinking=False).
        The binding's default chat template omits that prefix, which made the trained
        translator produce real <think> rambles in the 2026-06-12 gate run."""
        prompt = (f"<s><|im_start|>system\n{system}<|im_end|>\n"
                  f"<|im_start|>user\n{user}<|im_end|>\n"
                  f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
        out = actor_model(style).create_completion(
            prompt=prompt, max_tokens=120, temperature=0.8, repeat_penalty=1.15,
            stop=["<|im_end|>"])
        return out["choices"][0]["text"].strip()

    games, contexts = [], []
    budget = contracts.TURN_BUDGET
    for g in range(n):
        rng = random.Random(base_seed + g)
        style = contracts.STYLES[g % len(contracts.STYLES)]
        profession, fault = pools.sample_case(rng, style)

        facts = None
        for _ in range(2):   # mirror pipeline.new_case: leak-check + retry + fallback
            try:
                cand = jcall(gm, contracts.FACTS_SYS,
                             contracts.facts_user(profession, fault),
                             contracts.FACTS_GRAMMAR, 350, 0.9)["facts"]
            except Exception:  # noqa: BLE001
                continue
            cand = [str(f).strip() for f in cand
                    if str(f).strip() and not contracts.leaks(str(f), profession)]
            if len(cand) >= contracts.MIN_FACTS:
                facts = cand[:contracts.MAX_FACTS]
                break
        used_fallback = facts is None
        if used_fallback:
            facts = list(pools.FALLBACK_FACTS[:contracts.MAX_FACTS])

        transcript, history, beats = [], [], []
        released, guard_hits, wrapped = set(), 0, False
        for turn in range(budget):
            forced = contracts.forced_fact(len(facts), released, turn, budget)
            user = contracts.gm_user(profession, fault, facts, transcript, turn,
                                     budget, forced)
            contexts.append({"conversations": [
                {"role": "system", "content": contracts.GM_SYS},
                {"role": "user", "content": user}]})
            # Mirror pipeline.next_turn: re-sample the decision if the plain line
            # leaks the profession (the runtime's retry; ~2.5% of raw beats).
            for attempt in range(3):
                d = jcall(gm, contracts.GM_SYS, user, contracts.DECISION_GRAMMAR,
                          200, 0.7 + 0.1 * attempt)
                plain = str(d["line"]).strip()
                if not contracts.leaks(plain, profession):
                    break
            speaker, beat = contracts.guard_speaker(d["next_speaker"], d["beat_type"],
                                                    history, turn, budget)
            guarded = (speaker, beat) != (d["next_speaker"], d["beat_type"])
            guard_hits += guarded
            fi = forced if forced is not None else d.get("fact_index")
            if fi is not None and not (0 <= fi < len(facts)):
                fi = None
            # SHAPE 3.0: the actor TRANSLATES the director's plain line into jargon.
            line = actor_say(style, contracts.actor_system(speaker, style),
                             contracts.actor_user(plain))
            if not line or contracts.leaks(line, profession):  # runtime fallback
                line = plain
            if fi is not None:
                released.add(fi)
            beats.append({"raw": d, "speaker": speaker, "beat": beat, "plain": plain,
                          "fact_index": fi, "guarded": guarded, "line": line})
            transcript.append((speaker, plain))   # the DIRECTOR conditions on plain
            history.append(speaker)
            if d["wrap_up"] and turn + 1 >= max(4, budget // 2):
                wrapped = True
                break
        wrapped = wrapped or len(beats) >= budget

        spot = f"{contracts.article(profession)} {profession} who {fault}"
        s_spot = jcall(gm, contracts.SCORE_SYS,
                       contracts.score_user(profession, fault, spot),
                       contracts.SCORE_GRAMMAR, 96, 0.3)["score"]
        s_unrel = jcall(gm, contracts.SCORE_SYS,
                        contracts.score_user(profession, fault,
                                             "a beekeeper who mislabeled honey jars"),
                        contracts.SCORE_GRAMMAR, 96, 0.3)["score"]
        games.append({"style": style, "profession": profession, "fault": fault,
                      "facts": facts, "used_fallback_facts": used_fallback,
                      "beats": beats, "wrapped": wrapped, "guard_hits": guard_hits,
                      "fact_coverage": len(released) / len(facts),
                      "score_spot": s_spot, "score_unrel": s_unrel})
        print(f"game {g + 1}/{n} [{style}] {profession}: {len(beats)} beats, "
              f"guards={guard_hits}, wrap={wrapped}, spot={s_spot}, unrel={s_unrel}")

    with open(f"{DATA}/director_contexts_selfplay.jsonl", "w", encoding="utf-8") as fh:
        for c in contexts:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    # Persist the games so a dead local client can't lose an hour of CPU rollouts —
    # `--solve-only` resumes from here.
    with open(f"{DATA}/e2e_games.json", "w", encoding="utf-8") as fh:
        json.dump(games, fh, ensure_ascii=False)
    data_vol.commit()
    print(f"dumped {len(contexts)} self-play decide contexts + {len(games)} games")
    return games


@app.function(image=cpu_image, volumes={DATA: data_vol})
def load_games() -> list[dict]:
    return json.load(open(f"{DATA}/e2e_games.json", encoding="utf-8"))


TRACE_CARD = """\
---
license: apache-2.0
tags: [agent-trace, llama-cpp, courtroom-game]
---

# Buzzwords & Misdemeanors — agent traces

Each JSON is one full game of [Buzzwords & Misdemeanors](https://huggingface.co/spaces/build-small-hackathon/BuzzwordsMisdemeanors).
A **Game Master** (MiniCPM5-1B + a distilled *director* LoRA, GBNF-constrained) directs a
hearing beat by beat over a truth sampled in code; deterministic guards enforce courtroom
sequencing invariants; **actors** (the same base + a per-style LoRA) deliver the lines.
Every trace records the GM's raw structured decision per beat (speaker, beat type,
fact_index clue channel, intensity, stage direction, wrap-up), whether a guard remapped
it, the resulting line, the hidden truth, and scorer calibration probes. All inference is
llama.cpp on CPU — one ~1B base + small adapters.
"""


@app.function(image=cpu_image, timeout=30 * 60,
              secrets=[modal.Secret.from_name("huggingface-write")])
def publish_traces(games: list[dict], repo: str):
    import datetime
    import tempfile
    from pathlib import Path
    from huggingface_hub import HfApi

    tmp = Path(tempfile.mkdtemp())
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for i, g in enumerate(games):
        trace = {
            "game": "Buzzwords & Misdemeanors",
            "shape_version": contracts.SHAPE_VERSION,
            "generated_at": now,
            "jargon_style": g["style"],
            "models": {"game_master": f"{BASE_FILE} + director.lora.gguf",
                       "actors": f"{BASE_FILE} + style-{g['style']}.lora.gguf"},
            "hidden_case_file": {"profession": g["profession"],
                                 "fault_plain": g["fault"], "facts": g["facts"]},
            "turns": [{"turn": t + 1, "gm_decision": b["raw"],
                       "guard_remapped": b["guarded"],
                       "speaker": b["speaker"], "beat_type": b["beat"],
                       "fact_index": b["fact_index"], "line": b["line"]}
                      for t, b in enumerate(g["beats"])],
            "scorer_probes": {"spot_on": g["score_spot"], "unrelated": g["score_unrel"]},
        }
        (tmp / f"trace_{i:03d}_{g['style']}.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=TRACE_CARD.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")
    api.upload_folder(folder_path=str(tmp), path_in_repo="traces",
                      repo_id=repo, repo_type="dataset")
    print(f"published {len(games)} traces -> https://huggingface.co/datasets/{repo}")


# ----------------------------------------------------------------- solvability
@app.cls(image=gpu_image, gpu=TEACHER_GPU, timeout=60 * 60,
         volumes={"/root/.cache/huggingface": hf_cache},
         secrets=[modal.Secret.from_name("huggingface")])
class Solver:
    """The teacher reads each transcript (player view ONLY — no case file) and guesses;
    scoring is DETERMINISTIC token overlap, not the learned scorer (§7.2)."""

    @modal.enter()
    def load(self):
        from vllm import LLM
        self.llm = LLM(model=TEACHER_MODEL, dtype="auto", max_model_len=4096,
                       gpu_memory_utilization=0.92)

    @modal.method()
    def solve(self, games: list[dict], use_facts: bool = False) -> list[dict]:
        """use_facts=True is the DIAGNOSTIC baseline: solve from the raw oblique facts
        (no jargon layer). If this also fails, the facts don't carry the signal (data
        problem); if it succeeds while transcripts fail, the actors bury it (§7.2)."""
        from vllm import SamplingParams
        prompts = []
        for g in games:
            docket = "Court record — exhibits entered:\n" + "\n".join(f"- {f}" for f in g["facts"])
            if use_facts:
                body = docket          # diagnostic: facts-only baseline
            else:                      # the PLAYER VIEW: evidence docket + the hearing
                body = (docket + "\n\nThe hearing:\n"
                        + "\n".join(f"{b['speaker']}: {b['line']}" for b in g["beats"]))
            prompts.append(
                "You are a contestant in a guessing game, and you know its rules: the "
                "courtroom speaks in a jargon that is a pure SMOKESCREEN — the "
                "defendant's real profession is NEVER from the jargon's own domain, so "
                "actively exclude it. The exhibits in the court record are the reliable "
                "evidence; decode the hearing through them.\n\n" + body +
                "\n\nReturn ONLY JSON "
                '{"profession": "<your best guess>", "fault": "<what they actually did>"}.')
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        sps = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=120)
        outs = self.llm.chat(msgs, sps)
        results, guesses = [], []
        for g, o in zip(games, outs):
            text = o.outputs[0].text
            a, b = text.find("{"), text.rfind("}")
            try:
                guess = json.loads(text[a:b + 1]) if a != -1 and b > a else {}
            except Exception:  # noqa: BLE001
                guess = {}
            guesses.append(guess)
            results.append(_solver_score(g, guess))

        # LLM grading (the GATE metric): semantic credit for near-misses the token
        # metric zeroes ("janitor who sold master keys" vs locksmith, 2026-06-12).
        # Same rubric as the in-game scorer; token job_hit stays as an informational
        # column. Safe from gaming: it grades the solver's guesses, not a trainee.
        gmsgs = [[{"role": "user", "content":
                   f"True profession: {g['profession']}\nTrue act: {g['fault']}\n"
                   f"A player guessed — profession: {gu.get('profession', '?')}; "
                   f"act: {gu.get('fault', '?')}.\n"
                   "Grade STRICTLY on this scale:\n"
                   "95 = profession AND act both right (exact or true synonym, e.g. "
                   "'attorney' for 'lawyer').\n"
                   "65 = act essentially right, profession an ADJACENT job (e.g. "
                   "'janitor' instead of 'locksmith' for selling copied keys).\n"
                   "45 = act essentially right, profession wrong and not adjacent.\n"
                   "50 = profession right, act wrong.\n"
                   "25 = only the general situation type recognized (e.g. 'some fraud').\n"
                   "5 = unrelated.\n"
                   "Interpolate between anchors; do NOT round up generously. Return "
                   'ONLY JSON {"score": <int>}.'}] for g, gu in zip(games, guesses)]
        gouts = self.llm.chat(gmsgs, SamplingParams(temperature=0.0, max_tokens=40))
        for r, o in zip(results, gouts):
            text = o.outputs[0].text
            a, b = text.find("{"), text.rfind("}")
            try:
                r["llm_score"] = max(0, min(100, int(json.loads(text[a:b + 1])["score"])))
            except Exception:  # noqa: BLE001
                r["llm_score"] = r["score"]   # fall back to the deterministic number
        return results


_STOP = {"the", "a", "an", "to", "of", "and", "his", "her", "their", "who", "that",
         "for", "on", "in", "with", "had", "was", "were", "he", "she", "they", "it"}


def _content(text: str) -> set[str]:
    return {w.strip(".,;:!?\"'").lower() for w in (text or "").split()} - _STOP - {""}


def _solver_score(game: dict, guess: dict) -> dict:
    """Deterministic bucket scoring: job hit = any profession token guessed;
    charge overlap = fraction of the fault's content words recovered."""
    job_hit = bool(_content(str(guess.get("profession", "")))
                   & {t.lower() for t in pools.banned_words(game["profession"])})
    fault_words = _content(game["fault"])
    overlap = (len(_content(str(guess.get("fault", ""))) & fault_words) / len(fault_words)
               if fault_words else 0.0)
    score = 50 * job_hit + 50 * min(1.0, overlap / 0.5)   # half the content words = full credit
    return {"style": game["style"], "job_hit": job_hit, "charge_overlap": round(overlap, 2),
            "score": round(score), "guess": guess}


# ----------------------------------------------------------------- aggregation
def _aggregate(games: list[dict], solver: list[dict] | None) -> dict:
    from collections import Counter
    n_beats = sum(len(g["beats"]) for g in games)
    speakers = Counter(b["speaker"] for g in games for b in g["beats"])
    trans = Counter((g["beats"][i]["speaker"], g["beats"][i + 1]["speaker"])
                    for g in games for i in range(len(g["beats"]) - 1))

    max_run = 0
    for g in games:
        run, prev = 0, None
        for b in g["beats"]:
            run = run + 1 if b["speaker"] == prev else 1
            prev = b["speaker"]
            max_run = max(max_run, run)

    def grams(text, k=4):
        w = text.lower().split()
        return {tuple(w[i:i + k]) for i in range(len(w) - k + 1)}

    overlaps, pairs = 0, 0
    leak_lines = 0
    for g in games:
        for i, b in enumerate(g["beats"]):
            if contracts.leaks(b["line"] + " " + b.get("plain", ""), g["profession"]):
                leak_lines += 1
            if i:
                a, c = grams(g["beats"][i - 1]["line"]), grams(b["line"])
                pairs += 1
                overlaps += bool(a & c)

    from statistics import mean
    m = {
        "games": len(games), "beats": n_beats,
        "defense_share": speakers["defense"] / max(n_beats, 1),
        "judge_share": speakers["judge"] / max(n_beats, 1),
        "prosecutor_share": speakers["prosecutor"] / max(n_beats, 1),
        "max_same_speaker_run": max_run,
        "guard_trigger_rate": sum(g["guard_hits"] for g in games) / max(n_beats, 1),
        "consec_4gram_overlap": overlaps / max(pairs, 1),
        "leak_lines": leak_lines,
        "wrapped_rate": mean(g["wrapped"] for g in games),
        "fact_coverage": mean(g["fact_coverage"] for g in games),
        "fallback_facts_rate": mean(g["used_fallback_facts"] for g in games),
        "score_spot_mean": mean(g["score_spot"] for g in games),
        "score_unrel_mean": mean(g["score_unrel"] for g in games),
        "solver_mean": (mean(r.get("llm_score", r["score"]) for r in solver)
                        if solver else None),   # LLM-graded (gate metric)
        "solver_mean_token": mean(r["score"] for r in solver) if solver else None,
        "solver_job_hit_rate": mean(r["job_hit"] for r in solver) if solver else None,
        "transitions": {f"{a}->{b}": c for (a, b), c in sorted(trans.items())},
    }
    return m


@app.local_entrypoint()
def main(n: int = 12, base_seed: int = 0, no_solve: bool = False, solve_only: bool = False,
         solve_facts: bool = False, traces: int = 0,
         trace_repo: str = "BastienHot/buzzwords-agent-trace"):
    if traces:   # trace-pool mode: 8 parallel containers, then publish to the Hub
        per = max(1, traces // 8)
        batches = play_games.starmap([(per, 5_000 + 1_000 * i) for i in range(8)])
        games = [g for batch in batches for g in batch]
        publish_traces.remote(games, trace_repo)
        return
    games = load_games.remote() if solve_only else play_games.remote(n, base_seed)
    solver = None if no_solve else Solver().solve.remote(games, solve_facts)

    metrics = _aggregate(games, solver)
    print("\n================ E2E GATE (self-rollouts) ================")
    for k, v in metrics.items():
        if k != "transitions":
            print(f"  {k:<24} {v}")
    print(f"  transitions: {metrics['transitions']}")
    if solver:
        for r in solver:
            print(f"  solver [{r['style']}] score={r['score']} job_hit={r['job_hit']} "
                  f"guess={r['guess']}")

    print("\n===== GATE CHECKS =====")
    overall = True
    for name, check in GATE.items():
        ok = check(metrics)
        overall = overall and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    verdict = ("PASS — clear to ship" if overall else
               "FAIL — iterate (REBUILD_REVIEW.md §13.6: guards -> data -> DAgger, in that order)")
    print(f"\nE2E GATE: {verdict}")
