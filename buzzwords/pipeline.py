"""Game orchestration: hidden Case File + Game-Master-directed turn loop.

  1. ``new_case``  -- the GM writes the hidden Case File (profession + fault, unrelated
     to the chosen jargon style) and we start an empty transcript.
  2. ``next_turn`` -- one beat: the GM decides who speaks / how, the actor speaks in
     jargon, the line is appended. Ends when the GM sets ``wrap_up`` or the turn budget
     is spent (no latch -- closure is nudged by turn pressure in the GM prompt).
  3. ``score_guess`` -- the GM grades the plain-English guess against the Case File.

Real models are required. ``preflight`` reports anything missing so the UI can show a
clear message instead of crashing.
"""

from __future__ import annotations

import importlib.util
import logging
import os

from . import config
from .models import Case, CaseFile, Line

log = logging.getLogger(__name__)

_engine = None  # lazily created so the app still launches (to show preflight) with no llama.cpp


def _eng():
    global _engine
    if _engine is None:
        from .text_engine import TextEngine
        _engine = TextEngine()
    return _engine


def preflight() -> str | None:
    """Return a human-readable problem if the game can't run, else None."""
    problems = []
    if importlib.util.find_spec("llama_cpp") is None:
        problems.append("llama-cpp-python is not installed — run: pip install -r requirements.txt")
    for label, path in config.REQUIRED_MODELS:
        if not os.path.exists(path):
            problems.append(f"{label}: missing GGUF at {path}")
    if not problems:
        return None
    return "Local model weights are required to play:\n" + "\n".join(f"• {p}" for p in problems)


# --------------------------------------------------------------- case assembly
def new_case(jargon_style: str, difficulty: str) -> Case:
    data = _eng().casefile(jargon_style, difficulty)
    cf = CaseFile(profession=data["profession"], fault_plain=data["fault_plain"],
                  facts=list(data["facts"]), difficulty=difficulty,
                  turn_budget=config.TURN_BUDGET_BY_DIFFICULTY[difficulty])
    return Case(case_file=cf, jargon_style=jargon_style)


# --------------------------------------------------------------------- turn loop
def next_turn(session) -> Line | None:
    """Generate one beat (GM decision + actor line). Returns the new Line, or None if
    generation has reached closure. A failure propagates to the caller (the UI shows it and
    pipeline logs the full traceback) — no silent retry/fallback masking the real error."""
    case = session.case
    if case is None or session.finished_playback:
        return None
    budget = case.case_file.turn_budget
    try:
        d = _eng().gm_decide(case.case_file, case.lines, session.turn, budget)
        text = _eng().act(d.next_speaker, d.stage_direction, case.jargon_style, d.intensity)
    except Exception:
        # log the FULL traceback (incl. the llama_cpp file/line that raised) for debugging.
        log.exception("Beat generation failed: turn %d/%d, style=%s, lines_so_far=%d",
                      session.turn + 1, budget, case.jargon_style, len(case.lines))
        raise

    line = Line(actor=d.next_speaker, beat_type=d.beat_type, text=text)
    case.lines.append(line)
    session.turn += 1
    session.last_decision = d
    floor = max(4, budget // 2)   # ignore an over-eager wrap_up; guarantee a real trial
    session.wrapped = session.turn >= budget or (d.wrap_up and session.turn >= floor)
    return line


# ----------------------------------------------------------------- scoring
def score_guess(case: Case, guess: str) -> tuple[int, str]:
    cf = case.case_file
    return _eng().score(cf.profession, cf.fault_plain, guess)
