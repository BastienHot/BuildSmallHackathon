"""Core data model (Game Master + actors design).

The truth of a case lives in a hidden ``CaseFile`` (profession + fault + facts +
clues), *never* shown during play. The courtroom transcript is built up one
``Line`` at a time as the Game Master directs the debate; each line is already
in-character jargon (there is no plain twin per line anymore). At the end the
hidden CaseFile is revealed and compared against the player's guess.

The chosen jargon is a *style* (smokescreen) that is unrelated to the hidden
profession/fault -- the player must see through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CaseFile:
    """Hidden ground truth, written once at game start. Never shown during play."""

    profession: str             # revealed only at the end
    fault_plain: str            # the charge in plain English -> reveal + scoring
    facts: list[str] = field(default_factory=list)
    difficulty: str = "normal"
    turn_budget: int = 12       # how many beats the debate runs for


@dataclass
class GMDecision:
    """One structured beat from the Game Master (GBNF-constrained at runtime)."""

    next_speaker: str           # judge | prosecutor | defense
    beat_type: str = "exchange"
    stage_direction: str = ""   # oblique instruction handed to the actor
    intensity: int = 3          # 1..5
    wrap_up: bool = False       # True -> the debate should head to closure


@dataclass
class Line:
    actor: str                  # role key: judge | prosecutor | defense (matches BUBBLES)
    text: str                   # the in-character jargon line shown/spoken
    beat_type: str = ""         # the GMDecision.beat_type that produced it
    audio_path: Optional[str] = None   # cached TTS of `text`; None = text-only


@dataclass
class Case:
    case_file: CaseFile                      # hidden truth (profession + fault + ...)
    jargon_style: str = "corporate"          # selects background + actor LoRA style
    lines: list[Line] = field(default_factory=list)   # the transcript, built up live


@dataclass
class GameSession:
    """Per-user state held in gr.State (one instance per browser session)."""

    case: Optional[Case] = None
    turn: int = 0                    # beats generated so far
    wrapped: bool = False            # GM signalled closure
    mode: str = "off"                # config.PLAYBACK_OFF | PLAYBACK_ON
    guess: str = ""
    score: Optional[int] = None
    rationale: str = ""
    audio_dir: str = ""              # per-session temp dir for live-synth wavs
    last_decision: Optional[GMDecision] = None   # the GM decision behind the latest line

    # ----- transcript / turn helpers -----
    @property
    def budget(self) -> int:
        return self.case.case_file.turn_budget if self.case else 0

    @property
    def finished_playback(self) -> bool:
        """True once the debate has reached closure (wrap_up or out of budget)."""
        if not self.case:
            return False
        return self.wrapped or self.turn >= self.budget

    def current_line(self) -> Optional[Line]:
        if not self.case or not self.case.lines:
            return None
        return self.case.lines[-1]

    def reset_playback(self) -> None:
        self.turn = 0
        self.wrapped = False
        self.guess = ""
        self.score = None
        self.rationale = ""
