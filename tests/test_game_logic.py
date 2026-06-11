"""Unit tests for the model-free game logic: contracts, pools, guards, pipeline.

Every failure mode from the live-transcript post-mortem (REBUILD_REVIEW.md §13) has a
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
            assert fault in pools.PROFESSIONS[prof][1]   # domain-matched fault (§13.4)


# ----------------------------------------------------------------- contracts
def test_grammar_rules_are_single_line():
    for g in (contracts.FACTS_GRAMMAR, contracts.DECISION_GRAMMAR, contracts.SCORE_GRAMMAR):
        for line in g.strip().splitlines():
            assert "::=" in line, line   # llama.cpp ends a rule at a newline


def test_gm_prompt_is_append_only():
    """Stable-prefix property (§4.2): turn N's prompt must be a prefix of turn N+1's,
    up to the final status line."""
    facts = ["f0", "f1", "f2"]
    t1 = [("judge", "line one")]
    t2 = t1 + [("prosecutor", "line two")]
    u1 = contracts.gm_user("plumber", "did x", facts, t1, 1, 10)
    u2 = contracts.gm_user("plumber", "did x", facts, t2, 2, 10)
    stable1 = u1[:u1.rindex("This is beat")]
    assert u2.startswith(stable1)


def test_actor_never_sees_truth():
    u = contracts.actor_user("Press on the gap.", 4, "a log had a gap",
                             [("judge", "Order."), ("prosecutor", "Explain.")])
    assert "plumber" not in u
    assert "Stage direction: Press on the gap." in u
    assert "Fact to weave in obliquely: a log had a gap" in u
    assert "judge: Order." in u
    # factless + contextless beat keeps the minimal shape
    assert contracts.actor_user("Open.", 2, None, []) == "Stage direction: Open.\nIntensity: 2/5."


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
    # §13.2: the live transcript had the prosecutor delivering the defense's plea
    speaker, beat = contracts.guard_speaker("prosecutor", "plea", ["judge"], 1, 10)
    assert (speaker, beat) == ("defense", "plea")


def test_guard_no_three_in_a_row():
    # §13.3: prosecutor x3 consecutive in the live transcript
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
    # must still deliver every fact by the end (§13.5).
    released: set[int] = set()
    for turn in range(10):
        fi = contracts.forced_fact(4, released, turn, 10)
        if fi is not None:
            released.add(fi)
    assert released == {0, 1, 2, 3}
    assert contracts.forced_fact(3, {0, 1, 2}, 9, 10) is None   # nothing left to force


# ------------------------------------------------------------------ pipeline
class FakeEngine:
    """Scripted engine reproducing the §13 degenerate director (prosecutor-only,
    plea-by-prosecutor, never volunteers facts) — the guards must still produce a
    well-formed hearing."""

    def __init__(self):
        self.calls = 0

    def facts(self, style, profession, fault):
        return ["the log had a gap", "two names in one hand", "the bin was emptied early"]

    def decide(self, profession, fault, facts, transcript, turn, budget, forced):
        self.calls += 1
        return {"next_speaker": "prosecutor", "beat_type": "plea", "fact_index": None,
                "intensity": 1, "stage_direction": "press on", "wrap_up": turn >= budget - 1}

    def act(self, role, style, sd, intensity, fact, context):
        return f"[{role}] says something oblique"

    def score(self, profession, fault, guess):
        return 80, "close enough"


def _fake_session(monkeypatch_engine):
    import buzzwords.pipeline as pipeline
    pipeline._engine = monkeypatch_engine
    case = Case(case_file=CaseFile(profession="plumber", fault_plain="did x",
                                   facts=monkeypatch_engine.facts("", "", ""),
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


def test_new_case_falls_back_on_leaky_facts():
    class Leaky(FakeEngine):
        def facts(self, style, profession, fault):
            return [f"obviously a {profession} thing", "x", "y"]
    import buzzwords.pipeline as pipeline
    pipeline._engine = Leaky()
    case = pipeline.new_case("corporate")
    assert case.case_file.facts == list(pools.FALLBACK_FACTS[:contracts.MAX_FACTS])
    assert pools.PROFESSIONS[case.case_file.profession][0] not in \
        pools.STYLE_EXCLUDED_DOMAINS["corporate"]
