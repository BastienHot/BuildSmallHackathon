"""Unit tests for the model-free game logic: contracts, pools, guards, pipeline.

Every failure mode from the live-transcript post-mortem has a
test here proving the deterministic layer makes it unrepresentable. Run: pytest tests/
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from buzzwords import contracts, pools
from buzzwords.models import Case, CaseFile, GameSession


# --------------------------------------------------------------------- pools
def test_pools_integrity():
    assert len(pools.PROFESSIONS) >= 50
    for prof, (domain, faults) in pools.PROFESSIONS.items():
        assert len(faults) == 3, prof
        for f in faults:
            assert not f.lower().startswith(("the defendant", "he ", "she ")), f
            # faults must not name the profession (they're shown at the reveal)
            assert not contracts.leaks(f, prof), (prof, f)


def test_smokescreen_by_construction():
    for style in contracts.STYLES:
        excluded = pools.STYLE_EXCLUDED_DOMAINS[style]
        for i in range(50):
            prof, fault = pools.sample_case(random.Random(i), style)
            assert pools.PROFESSIONS[prof][0] not in excluded
            assert fault in pools.PROFESSIONS[prof][1] # domain-matched fault


# ----------------------------------------------------------------- contracts
def test_grammar_rules_are_single_line():
    for g in (contracts.FACTS_GRAMMAR, contracts.DECISION_GRAMMAR, contracts.SCORE_GRAMMAR):
        for line in g.strip().splitlines():
            assert "::=" in line, line   # llama.cpp ends a rule at a newline


def test_gm_prompt_is_append_only():
    """Stable-prefix property: turn N's prompt must be a prefix of turn N+1's,
    up to the final status line."""
    facts = ["f0", "f1", "f2"]
    t1 = [("judge", "line one")]
    t2 = t1 + [("prosecutor", "line two")]
    u1 = contracts.gm_user("plumber", "did x", facts, t1, 1, 10)
    u2 = contracts.gm_user("plumber", "did x", facts, t2, 2, 10)
    stable1 = u1[:u1.rindex("This is beat")]
    assert u2.startswith(stable1)


def test_actor_never_sees_truth():
    # SHAPE 3.0: the actor is a translator — it receives ONLY the public plain line.
    u = contracts.actor_user("The records show the controls were held by someone else.")
    assert "plumber" not in u and "ferry" not in u
    assert u == "Line to rewrite: The records show the controls were held by someone else."
    sys_p = contracts.actor_system("defense", "aviation")
    assert "KEEP THE MEANING" in sys_p and "aviation jargon" in sys_p
    assert "plain English" in contracts.actor_system("judge", None)  # stage-1 register


def test_leak_detection():
    assert contracts.leaks("clearly an airline pilot here", "airline pilot")
    assert contracts.leaks("the pilot's seat", "airline pilot")      # token-level
    assert contracts.leaks("two pilots walked in", "airline pilot")  # plural
    assert not contracts.leaks("a cockpit transcript", "airline pilot")
    # left-word-boundary: 'opacity' must NOT trip 'city' (real e2e false positive)
    assert not contracts.leaks("the defendant's opacity is noted", "city council clerk")
    assert contracts.leaks("the city archives", "city council clerk")


# -------------------------------------------------------------------- guards
def test_guard_prosecutor_cannot_plead():
    # the live transcript had the prosecutor delivering the defense's plea
    speaker, beat = contracts.guard_speaker("prosecutor", "plea", ["judge"], 1, 10)
    assert (speaker, beat) == ("defense", "plea")


def test_guard_no_three_in_a_row():
    # prosecutor x3 consecutive in the live transcript
    speaker, beat = contracts.guard_speaker(
        "prosecutor", "evidence", ["judge", "prosecutor", "prosecutor"], 3, 10)
    assert speaker != "prosecutor"
    # single-speaker beat escapes via "exchange"
    speaker, beat = contracts.guard_speaker(
        "prosecutor", "charge", ["judge", "prosecutor", "prosecutor"], 3, 10)
    assert speaker != "prosecutor" and beat == "exchange"


def test_guard_defense_by_midpoint():
    history = ["judge", "prosecutor", "judge", "prosecutor"]   # no defense yet
    speaker, beat = contracts.guard_speaker("prosecutor", "charge", history, 4, 10)
    assert speaker == "defense" and beat == "objection"        # charge disallows defense
    # already satisfied -> untouched
    speaker, beat = contracts.guard_speaker(
        "prosecutor", "evidence", ["judge", "defense", "prosecutor", "judge"], 4, 10)
    assert (speaker, beat) == ("prosecutor", "evidence")


def test_forced_fact_guarantees_full_coverage():
    # 4 facts, 10 beats: simulate a GM that NEVER volunteers a fact — the force rule
    # must still deliver every fact by the end.
    released: set[int] = set()
    for turn in range(10):
        fi = contracts.forced_fact(4, released, turn, 10)
        if fi is not None:
            released.add(fi)
    assert released == {0, 1, 2, 3}
    assert contracts.forced_fact(3, {0, 1, 2}, 9, 10) is None   # nothing left to force


# ------------------------------------------------------------------ pipeline
class FakeEngine:
    """Scripted engine reproducing the degenerate director (prosecutor-only,
    plea-by-prosecutor, never volunteers facts) — the guards must still produce a
    well-formed hearing."""

    def __init__(self):
        self.calls = 0

    def facts(self, profession, fault):
        return ["the log had a gap", "two names in one hand", "the bin was emptied early"]

    def decide(self, profession, fault, facts, transcript, turn, budget, forced):
        self.calls += 1
        return {"next_speaker": "prosecutor", "beat_type": "plea", "fact_index": None,
                "intensity": 1, "line": "The record speaks for itself here.",
                "wrap_up": turn >= budget - 1}

    def act(self, role, style, plain_line):
        return f"[{role}] jargonized: {plain_line}"

    def score(self, profession, fault, guess):
        return 80, "close enough"


def _fake_session(monkeypatch_engine):
    import buzzwords.pipeline as pipeline
    pipeline._engine = monkeypatch_engine
    case = Case(case_file=CaseFile(profession="plumber", fault_plain="did x",
                                   facts=monkeypatch_engine.facts("", ""),
                                   turn_budget=contracts.TURN_BUDGET),
                jargon_style="corporate")
    s = GameSession()
    s.case = case
    return pipeline, s


def test_pipeline_survives_degenerate_director():
    pipeline, s = _fake_session(FakeEngine())
    while not s.generation_finished:
        if pipeline.next_turn(s) is None:
            break
    speakers = [l.actor for l in s.case.lines]
    # plea-by-prosecutor remapped every beat
    assert all(sp == "defense" for l, sp in zip(s.case.lines, speakers)
               if l.beat_type == "plea")
    # no 3-in-a-row, defense present, all facts released despite fact_index=None
    assert all(speakers[i] != speakers[i + 1] or speakers[i + 1] != speakers[i + 2]
               for i in range(len(speakers) - 2))
    assert "defense" in speakers
    assert s.case.facts_released == {0, 1, 2}
    assert s.wrapped


def test_wrap_floor_ignores_early_wrap():
    class EagerWrap(FakeEngine):
        def decide(self, *a, **k):
            d = super().decide(*a, **k)
            d["wrap_up"] = True   # tries to end at beat 1
            return d
    pipeline, s = _fake_session(EagerWrap())
    while not s.generation_finished:
        if pipeline.next_turn(s) is None:
            break
    assert s.turn >= max(4, contracts.TURN_BUDGET // 2)   # the floor held


def test_empty_guess_scores_zero_without_model():
    class NeverCalled(FakeEngine):
        def score(self, *a):
            raise AssertionError("model must not grade an empty plea")
    import buzzwords.pipeline as pipeline
    pipeline._engine = NeverCalled()
    case = Case(case_file=CaseFile(profession="plumber", fault_plain="did x", facts=["f"]))
    for guess in ("", "   ", "??", None):
        score, rationale = pipeline.score_guess(case, guess)
        assert score == 0 and "plea" in rationale.lower()
    # a real guess still reaches the engine
    pipeline._engine = FakeEngine()
    assert pipeline.score_guess(case, "a plumber who did x") == (80, "close enough")


def test_plain_line_leak_falls_back_safely():
    class LeakyActor(FakeEngine):
        def act(self, role, style, plain_line):
            return "obviously a plumber thing"   # jargon layer leaks -> plain fallback
    pipeline, s = _fake_session(LeakyActor())
    pipeline.next_turn(s)
    assert s.case.lines[0].text == s.case.lines[0].plain_text  # fell back to plain
    assert not contracts.leaks(s.case.lines[0].text, "plumber")


def test_new_case_falls_back_on_leaky_facts():
    class Leaky(FakeEngine):
        def facts(self, profession, fault):
            return [f"obviously a {profession} thing", "x", "y"]
    import buzzwords.pipeline as pipeline
    pipeline._engine = Leaky()
    case = pipeline.new_case("corporate")
    assert case.case_file.facts == list(pools.FALLBACK_FACTS[:contracts.MAX_FACTS])
    assert pools.PROFESSIONS[case.case_file.profession][0] not in \
        pools.STYLE_EXCLUDED_DOMAINS["corporate"]
