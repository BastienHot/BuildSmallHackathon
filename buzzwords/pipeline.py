"""Game orchestration: sampled truth + Game-Master-directed turn loop + guards.

  1. ``new_case``        -- CODE samples profession + domain-matched fault (pools), the
     director model writes the oblique facts (leak-checked, retried, with a fallback).
  2. ``start_generation`` -- a background worker generates beats while the player reads
     (no pre-generated games; REBUILD_REVIEW.md §8.3).
  3. ``next_turn``       -- one beat: GM decision -> deterministic guards (§13.2-13.3)
     -> fact scheduling (§13.5) -> actor line with dialogue context.
  4. ``score_guess``     -- the GM grades the plain-English guess.

``preflight`` reports anything missing so the UI shows a clear message, not a crash.
"""

from __future__ import annotations

import logging
import os
import random
import threading

from . import config, contracts, pools
from .models import Case, CaseFile, GMDecision, GameSession, Line

log = logging.getLogger(__name__)

_engine = None
_engine_lock = threading.Lock()


def _eng():
    global _engine
    with _engine_lock:
        if _engine is None:
            from .engine import TextEngine
            eng = TextEngine()
            eng.start()
            _engine = eng
    return _engine


def preflight() -> str | None:
    """Return a human-readable problem if the game can't run, else None."""
    problems = []
    if not os.path.exists(config.LLAMA_SERVER_BIN):
        problems.append(f"llama-server binary not found at {config.LLAMA_SERVER_BIN} "
                        "(install llama.cpp or set BW_LLAMA_SERVER)")
    for label, path in config.REQUIRED_MODELS:
        if not os.path.exists(path):
            problems.append(f"{label}: missing GGUF at {path}")
    if not problems:
        return None
    return "The local inference runtime is not ready:\n" + "\n".join(f"• {p}" for p in problems)


# --------------------------------------------------------------- case assembly
def new_case(jargon_style: str) -> Case:
    """Sample the truth in code; the model writes only the oblique facts (§13.4)."""
    rng = random.Random()
    profession, fault = pools.sample_case(rng, jargon_style)
    facts: list[str] | None = None
    for attempt in range(2):
        try:
            cand = _eng().facts(jargon_style, profession, fault)
        except Exception:
            log.exception("facts generation failed (attempt %d)", attempt + 1)
            continue
        cand = [f for f in cand if f and not contracts.leaks(f, profession)]
        if len(cand) >= contracts.MIN_FACTS:
            facts = cand[:contracts.MAX_FACTS]
            break
        log.warning("facts attempt %d: only %d clean fact(s), retrying", attempt + 1, len(cand))
    if facts is None:
        log.warning("Falling back to generic facts for %s", profession)
        facts = list(pools.FALLBACK_FACTS[:contracts.MAX_FACTS])
    cf = CaseFile(profession=profession, fault_plain=fault, facts=facts,
                  turn_budget=contracts.TURN_BUDGET)
    return Case(case_file=cf, jargon_style=jargon_style)


# --------------------------------------------------------------------- turn loop
def next_turn(session: GameSession) -> Line | None:
    """Generate one beat (GM decision + guards + actor line). Returns the new Line,
    or None once generation has reached closure."""
    case = session.case
    if case is None or session.generation_finished:
        return None
    cf, budget = case.case_file, case.case_file.turn_budget
    transcript = [(l.actor, l.text) for l in case.lines]
    history = [l.actor for l in case.lines]
    forced = contracts.forced_fact(len(cf.facts), case.facts_released, session.turn, budget)

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            d = _eng().decide(cf.profession, cf.fault_plain, cf.facts, transcript,
                              session.turn, budget, forced)
            speaker, beat = contracts.guard_speaker(d["next_speaker"], d["beat_type"],
                                                    history, session.turn, budget)
            fi = forced if forced is not None else d.get("fact_index")
            if fi is not None and not (0 <= fi < len(cf.facts)):
                fi = None
            fact_text = cf.facts[fi] if fi is not None else None
            text = _eng().act(speaker, case.jargon_style, d["stage_direction"],
                              d["intensity"], fact_text, transcript[-2:])
            break
        except Exception as e:  # noqa: BLE001 — retry any runtime failure
            last_err = e
            log.exception("Beat failed: turn %d/%d, style=%s, attempt %d/3",
                          session.turn + 1, budget, case.jargon_style, attempt + 1)
    else:
        raise last_err

    if fi is not None:
        case.facts_released.add(fi)
    line = Line(actor=speaker, beat_type=beat, text=text, fact_index=fi)
    case.lines.append(line)
    session.turn += 1
    floor = max(4, budget // 2)   # ignore an over-eager wrap_up; guarantee a real trial
    session.wrapped = session.turn >= budget or (d["wrap_up"] and session.turn >= floor)
    return line


def start_generation(session: GameSession) -> threading.Thread:
    """Generate beats in a background daemon thread while the player reads (§8.3).
    Engine calls are already serialized by the engine's own lock. The worker closes
    the hearing early (rather than aborting) if a beat fails with >= 2 lines played."""
    def worker():
        try:
            while not session.generation_finished:
                if next_turn(session) is None:
                    break
        except Exception as e:  # noqa: BLE001
            if len(session.case.lines) >= 2:
                log.warning("Beat generation gave up at turn %d; closing early with "
                            "%d beat(s): %s", session.turn + 1, len(session.case.lines), e)
                session.wrapped = True
            else:
                session.gen_error = f"{type(e).__name__}: {e}"
        finally:
            session.gen_done = True
            log.info("Hearing generation finished: %d beat(s), error=%r",
                     len(session.case.lines) if session.case else 0, session.gen_error)

    t = threading.Thread(target=worker, daemon=True, name="bw-gen")
    t.start()
    return t


# ----------------------------------------------------------------- scoring
def score_guess(case: Case, guess: str) -> tuple[int, str]:
    cf = case.case_file
    return _eng().score(cf.profession, cf.fault_plain, guess)
