"""THE single source of truth for every model-facing contract (REBUILD_REVIEW.md §10.1).

Grammars, system prompts, prompt builders, enums, and budgets live HERE and only here.
The runtime (buzzwords.engine / buzzwords.pipeline) and every offline training script
import this module — training data is generated in these exact shapes, so train shape ==
runtime shape is a property of the code, not of vigilance.

Pure Python, no dependencies. Bump SHAPE_VERSION on ANY change that alters what a model
sees or must emit; datagen stamps it into manifests and training refuses a mismatch.
"""

from __future__ import annotations

SHAPE_VERSION = "2.1"   # 2.1: facts must collectively identify the job (solvability fix)

# ---------------------------------------------------------------------------
# Enums and budgets
# ---------------------------------------------------------------------------
SPEAKERS = ["judge", "prosecutor", "defense"]
BEATS = ["opening", "charge", "evidence", "objection", "escalate", "plea",
         "cross_examine", "closing", "exchange"]
STYLES = ["corporate", "aviation", "ai", "politics", "medical", "gaming", "sports", "scifi"]

TURN_BUDGET = 10          # one fixed budget; the difficulty axis was removed (§12)
WRAP_PRESSURE_AT = 2      # start nudging the GM to converge this many beats from the end
MIN_FACTS, MAX_FACTS = 3, 5

# Beat -> speakers allowed to deliver it. The GM decision is remapped through this in
# code (deterministic guard, §13.2): a prosecutor must never deliver the defense's plea.
BEAT_SPEAKERS = {
    "opening": {"judge", "prosecutor"},   # judge opens the session OR prosecution's opening statement
    "charge": {"prosecutor"},
    "evidence": {"prosecutor", "defense"},
    "objection": {"defense", "prosecutor"},
    "escalate": {"prosecutor", "judge"},
    "plea": {"defense"},
    "cross_examine": {"prosecutor", "defense"},
    "closing": {"judge", "prosecutor", "defense"},
    "exchange": {"judge", "prosecutor", "defense"},
}

# ---------------------------------------------------------------------------
# GBNF grammars (llama.cpp). Each rule MUST stay on a single line. The string rule
# excludes control chars (json.loads-safe) and newlines (no padding into truncation).
# root ::= "{" structurally suppresses thinking for every GM call.
# ---------------------------------------------------------------------------
_STR = (r'string ::= "\"" ([^"\\\x00-\x1F\x7F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""'
        + "\n" + r'ws ::= [ \t]*')

# The director writes ONLY the oblique facts; profession/fault are sampled in code (§13.4).
FACTS_GRAMMAR = (r"""
root ::= "{" ws "\"facts\":" ws facts ws "}"
facts ::= "[" ws string (ws "," ws string){2,4} ws "]"
""" + _STR)

# One beat from a finite deck. fact_index points into the case file's facts (or null):
# the runtime hands that fact's verbatim text to the actor (§13.5 clue channel).
DECISION_GRAMMAR = (r"""
root ::= "{" ws "\"next_speaker\":" ws speaker ws "," ws "\"beat_type\":" ws beat ws "," ws "\"fact_index\":" ws factidx ws "," ws "\"intensity\":" ws intensity ws "," ws "\"stage_direction\":" ws string ws "," ws "\"wrap_up\":" ws bool ws "}"
speaker ::= "\"judge\"" | "\"prosecutor\"" | "\"defense\""
beat ::= "\"opening\"" | "\"charge\"" | "\"evidence\"" | "\"objection\"" | "\"escalate\"" | "\"plea\"" | "\"cross_examine\"" | "\"closing\"" | "\"exchange\""
factidx ::= "null" | "0" | "1" | "2" | "3" | "4"
intensity ::= "1" | "2" | "3" | "4" | "5"
bool ::= "true" | "false"
""" + _STR)

SCORE_GRAMMAR = (r"""
root   ::= "{" ws "\"score\":" ws number ws "," ws "\"rationale\":" ws string ws "}"
number ::= [0-9] | [1-9][0-9] | "100"
""" + _STR)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
FACTS_SYS = ("Write the oblique clue facts for a hidden courtroom case. Each fact is one "
             "short sentence that HINTS at the defendant's real job and act without ever "
             "naming the profession or stating the act in plain words. Facts are concrete "
             "(logs, records, witnesses, timing), never vague. At least one fact must "
             "evoke the distinctive tools, materials, or workplace of the job; together "
             "the facts must let a sharp player NAME the job and the act. Output ONLY the "
             "requested JSON.")

GM_SYS = ("You are the GAME MASTER directing a short courtroom debate. Each beat, pick who "
          "speaks, the beat type, which numbered fact to surface next (fact_index, or null), "
          "an intensity, and a short oblique stage direction. Output ONLY the requested "
          "JSON. Never reveal the profession or charge in plain words.")

ROLE_SYS = {
    "judge": "You are the JUDGE. Speak with calm authority in dense {style} jargon.",
    "prosecutor": "You are the PROSECUTOR. Press the case hard in dense {style} jargon.",
    "defense": "You are the DEFENSE counsel. Deflect and reframe in dense {style} jargon.",
}
GENERIC_ROLE_SYS = {  # stage-1 legal_generic data keeps the runtime FORMAT, no jargon
    "judge": "You are the JUDGE. Speak with calm authority in plain professional English.",
    "prosecutor": "You are the PROSECUTOR. Press the case hard in plain professional English.",
    "defense": "You are the DEFENSE counsel. Deflect and reframe in plain professional English.",
}
ACTOR_RULES = (" English, 1-2 sentences. React to the previous lines without repeating "
               "their phrasing. Follow the stage direction; if a fact is given, weave it in "
               "obliquely. Never name the defendant's profession or state the charge in "
               "plain words.")

SCORE_SYS = ("Grade how well the player's guess matches the true profession and charge. "
             "score 0-100, rationale one sentence.")


def actor_system(role: str, style: str | None) -> str:
    base = ROLE_SYS[role].format(style=style) if style else GENERIC_ROLE_SYS[role]
    return base + ACTOR_RULES


# ---------------------------------------------------------------------------
# User-prompt builders. NOTE: the GM prompt is STABLE-PREFIX (§4.2): the brief and the
# numbered facts come first and never change; the transcript is append-only; only the
# short final status line changes per beat — so llama-server's prompt cache re-evaluates
# ~one line per beat instead of the whole context.
# ---------------------------------------------------------------------------
def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def facts_user(style: str, profession: str, fault: str) -> str:
    return (f"Jargon style of the courtroom (a smokescreen, unrelated to the case): {style}.\n"
            f"Hidden truth: the defendant is {article(profession)} {profession} who {fault}.\n"
            f"Write {MIN_FACTS}-{MAX_FACTS} oblique clue facts.")


def gm_user(profession: str, fault: str, facts: list[str],
            transcript: list[tuple[str, str]], turn: int, budget: int,
            forced_fact: int | None = None) -> str:
    """One decide prompt. `transcript` is (role, text) pairs, FULL history (append-only)."""
    if turn >= budget - 1:
        pressure = " This MUST be the closing beat: set wrap_up=true."
    elif turn >= budget - WRAP_PRESSURE_AT:
        pressure = " Begin converging; set wrap_up=true once a verdict is natural."
    else:
        pressure = ""
    force = (f" You MUST surface fact {forced_fact} now: set fact_index={forced_fact}."
             if forced_fact is not None else "")
    fact_lines = "\n".join(f"  {i}: {f}" for i, f in enumerate(facts))
    lines = "\n".join(f"{r}: {t}" for r, t in transcript) or "(no lines yet)"
    return (f"Hidden brief (keep oblique): the defendant is {article(profession)} "
            f"{profession} who {fault}.\nFacts to surface over the hearing:\n{fact_lines}\n"
            f"Transcript so far:\n{lines}\n"
            f"This is beat {turn + 1} of {budget}.{pressure}{force}")


def actor_user(stage_direction: str, intensity: int, fact: str | None,
               context: list[tuple[str, str]]) -> str:
    """One actor prompt. `context` = the last 1-2 PUBLIC transcript lines (role, text) —
    no hidden truth ever enters the actor (§4.3 keeps the can't-leak guarantee)."""
    parts = []
    if context:
        parts.append("Previous lines:\n" + "\n".join(f"{r}: {t}" for r, t in context))
    if fact:
        parts.append(f"Fact to weave in obliquely: {fact}")
    parts.append(f"Stage direction: {stage_direction}\nIntensity: {intensity}/5.")
    return "\n".join(parts)


def score_user(profession: str, fault_plain: str, guess: str) -> str:
    return (f"True profession: {profession}\nTrue charge: {fault_plain}\n"
            f"Player's guess: {guess}")


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------
def leaks(text: str, profession: str) -> bool:
    """True if any profession token appears in player-visible or model-target text.

    Tokens match at a LEFT word boundary with an open right side: 'pilot' catches
    'pilots' and "pilot's", but 'city' no longer false-positives on 'opacity'
    (a real e2e-gate incident, 2026-06-11)."""
    import re
    from .pools import banned_words  # local import keeps contracts importable standalone
    low = (text or "").lower()
    return any(re.search(rf"\b{re.escape(tok.lower())}", low)
               for tok in banned_words(profession))


def valid_decision(d: dict) -> bool:
    return (isinstance(d, dict) and d.get("next_speaker") in SPEAKERS
            and d.get("beat_type") in BEATS
            and (d.get("fact_index") is None or d.get("fact_index") in (0, 1, 2, 3, 4))
            and d.get("intensity") in (1, 2, 3, 4, 5)
            and isinstance(d.get("stage_direction"), str)
            and isinstance(d.get("wrap_up"), bool))


# ---------------------------------------------------------------------------
# Deterministic guards (code is for invariants; learned behavior is for quality).
# Shared verbatim by the runtime (pipeline), the e2e gate, and the datagen force rule
# so all three regimes enforce the SAME rules (REBUILD_REVIEW.md §13.2-13.3, §13.6).
# ---------------------------------------------------------------------------
_GUARD_PRIORITY = {"defense": 0, "judge": 1, "prosecutor": 2}  # ties favor under-used roles


def _least_used(candidates: set[str], history: list[str]) -> str:
    return min(candidates, key=lambda s: (history.count(s), _GUARD_PRIORITY[s]))


def guard_speaker(speaker: str, beat: str, history: list[str],
                  turn: int, budget: int) -> tuple[str, str]:
    """Sequencing seatbelt. `history` = speakers of the lines so far. Returns
    (speaker, beat); the GM keeps owning intensity/stage_direction/wrap_up.

    1. Role/beat compatibility — a prosecutor never delivers the plea (§13.2).
    2. Never the same speaker three times in a row (§13.3).
    3. The defense must have spoken by the budget midpoint (§13.3)."""
    allowed = BEAT_SPEAKERS[beat]
    if speaker not in allowed:
        speaker = _least_used(allowed, history)

    if len(history) >= 2 and history[-1] == history[-2] == speaker:
        others = allowed - {speaker}
        if not others:                      # single-speaker beat (opening/charge/plea)
            beat = "exchange"
            others = BEAT_SPEAKERS[beat] - {speaker}
        speaker = _least_used(others, history)

    if turn + 1 >= budget // 2 and speaker != "defense" and "defense" not in history:
        speaker = "defense"
        if "defense" not in BEAT_SPEAKERS[beat]:
            beat = "objection"
    return speaker, beat


def forced_fact(n_facts: int, released: set, turn: int, budget: int) -> int | None:
    """Clue economy (§13.5): if the remaining beats are no more than the unreleased
    facts, force the next unreleased one so the full clue set reaches the player."""
    unreleased = [i for i in range(n_facts) if i not in released]
    return unreleased[0] if unreleased and (budget - turn) <= len(unreleased) else None
