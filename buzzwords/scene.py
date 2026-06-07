"""HTML for the image-based courtroom stage.

The stage is one full pre-generated background image (chosen by the case jargon)
with an overlay speech bubble whose tail points toward the speaker. Character
positions are identical in every background, so config.BUBBLES works for all.
"""

from __future__ import annotations

import hashlib
import html

from . import config
from .models import Case, GameSession, Line

ROLE_NAME = {"judge": "Judge", "prosecutor": "Prosecutor", "defense": "Defense"}


def image_url(jargon: str) -> str:
    """Gradio static-file URL for a jargon's court backdrop (served via allowed_paths).

    Each style picks a fixed variant from its court folder (deterministic by name), so the
    styles sharing the normal court still show distinct backdrops.
    """
    spec = config.JARGONS.get(jargon, config.JARGONS[config.DEFAULT_JARGON])
    n = int(hashlib.md5(jargon.encode()).hexdigest(), 16) % config.COURT_VARIANTS + 1
    path = (config.MAPS_DIR / spec["court"] / f"variant_{n:02d}.png").resolve().as_posix()
    return f"/gradio_api/file={path}"


def _bubble(line: Line) -> str:
    b = config.BUBBLES.get(line.actor, {"x": 50, "y": 42, "tail": "center"})
    name = ROLE_NAME.get(line.actor, line.actor.title())
    return (
        f'<div class="bubble tail-{b["tail"]}" style="left:{b["x"]}%;top:{b["y"]}%">'
        f'<span class="bub-name">{name}</span>'
        f'<span class="bub-text">{html.escape(line.text)}</span></div>'
    )


def render_stage(case: Case, line: Line | None) -> str:
    bubble = _bubble(line) if line else ""
    hint = '<div class="hint">▶ click <b>Continue</b></div>' if line else ""
    return (
        f'<div class="stage" style="background-image:url(\'{image_url(case.jargon_style)}\')">'
        f'<div class="vignette"></div>{bubble}{hint}</div>'
    )


def title_card(bg_url: str) -> str:
    return (
        f'<div class="hero" style="--bg:url(\'{bg_url}\')">'
        '<div class="hero-content">'
        '<div class="hero-emblem">⚖</div>'
        '<h1>Buzzwords<span>&amp;</span>Misdemeanors</h1>'
        '<p>You wake up on trial, and everyone speaks in impenetrable jargon. '
        'Sit through the hearing, then tell the court what you think you actually did.</p>'
        '</div></div>'
    )


def error_card(message: str) -> str:
    """Friendly 'what's missing' card shown instead of crashing on missing weights."""
    body = html.escape(message).replace("\n", "<br>")
    return ('<div class="hero" style="min-height:170px"><div class="hero-content">'
            '<div class="hero-emblem">⚠</div><h1>Not ready</h1>'
            f'<p style="white-space:pre-line;text-align:left">{body}</p></div></div>')


def render_verdict_banner(s: GameSession) -> str:
    score = s.score or 0
    word = ("Case dismissed!" if score >= 80 else "Reasonable doubt" if score >= 50
            else "Guilty as charged" if score >= 20 else "Throw the book at 'em")
    return (
        f'<div class="verdict"><div class="score">{score}%</div>'
        f'<div class="word">{word}</div>'
        f'<div class="why">&ldquo;{html.escape(s.rationale)}&rdquo;</div>'
        f'<div class="charge"><b>The real charge:</b> {html.escape(s.case.case_file.fault_plain)}</div></div>'
    )


def render_reveal(s: GameSession) -> str:
    """Left: the jargon transcript the player heard. Right: the hidden truth."""
    cf = s.case.case_file
    transcript = "".join(
        f'<div class="rl"><b>{ROLE_NAME.get(ln.actor, ln.actor.title())}</b> '
        f'{html.escape(ln.text)}</div>' for ln in s.case.lines)
    truth = (f'<div class="rt"><div><b>You were:</b> {html.escape(cf.profession)}</div>'
             f'<div><b>What you actually did:</b> {html.escape(cf.fault_plain)}</div></div>')
    return ('<div class="files">'
            '<div class="col head jargon">What you heard</div>'
            '<div class="col head plain">The truth</div>'
            f'<div class="col jargon">{transcript}</div>'
            f'<div class="col plain">{truth}</div></div>')
