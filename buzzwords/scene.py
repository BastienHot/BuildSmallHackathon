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


def _sentence(s: str, period: bool = False) -> str:
    """Display-case a pool/model string: capitalize the first letter, optionally
    ensure a closing period. Data stays lowercase; only the rendering changes."""
    s = (s or "").strip()
    s = s[:1].upper() + s[1:]
    if period and s and s[-1] not in ".!?":
        s += "."
    return s


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


def render_stage(case: Case, line: Line | None, evidence: list[str] | None = None) -> str:
    """The stage plus the EVIDENCE DOCKET: oblique facts surface to the player as they
    are woven into beats. The jargon theater is the entertainment layer; the docket is
    the solvable puzzle layer (REBUILD_REVIEW.md solvability fix — transcript-only was
    measured unwinnable even for the 31B teacher)."""
    bubble = _bubble(line) if line else ""
    hint = '<div class="hint">▶ click <b>Continue</b></div>' if line else ""
    docket = ""
    if evidence:
        rows = "".join(
            f'<div class="exhibit"><span class="ex-tag">Exhibit {chr(65 + i)}</span>'
            f'<span class="ex-text">{html.escape(_sentence(f, period=True))}</span></div>'
            for i, f in enumerate(evidence))
        docket = (f'<div class="docket"><div class="docket-title">Court record · '
                  f'exhibits entered</div>{rows}</div>')
    return (
        f'<div class="stage" style="background-image:url(\'{image_url(case.jargon_style)}\')">'
        f'<div class="vignette"></div>{bubble}{hint}</div>{docket}'
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


def loading_card(frac: float, desc: str) -> str:
    """Full-bleed progress card: shown while the case is drafted / the first beat
    generates, and briefly mid-hearing if the player out-reads the background worker."""
    pct = max(0, min(100, round(frac * 100)))
    return (
        '<div class="hero loading" style="min-height:240px"><div class="hero-content">'
        '<div class="hero-emblem spin">⚖</div>'
        '<h1>Preparing your hearing</h1>'
        f'<p class="load-desc">{html.escape(desc)}</p>'
        '<div class="loadbar" role="progressbar" aria-valuemin="0" aria-valuemax="100" '
        f'aria-valuenow="{pct}"><div class="loadbar-fill" style="width:{pct}%"></div></div>'
        f'<div class="loadpct">{pct}%</div>'
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
        f'<div class="charge"><b>The real charge:</b> '
        f'{html.escape(_sentence(s.case.case_file.fault_plain, period=True))}</div></div>'
    )


def render_reveal(s: GameSession) -> str:
    """Left: the jargon transcript the player heard. Right: the hidden truth."""
    cf = s.case.case_file
    transcript = "".join(
        f'<div class="rl"><b>{ROLE_NAME.get(ln.actor, ln.actor.title())}</b> '
        f'{html.escape(ln.text)}</div>' for ln in s.case.lines)
    article = "An" if cf.profession[:1].lower() in "aeiou" else "A"
    truth = (f'<div class="rt"><div><b>You were:</b> '
             f'{html.escape(f"{article} {cf.profession}")}</div>'
             f'<div><b>What you actually did:</b> '
             f'{html.escape(_sentence(cf.fault_plain, period=True))}</div></div>')
    return ('<div class="files">'
            '<div class="col head jargon">What you heard</div>'
            '<div class="col head plain">The truth</div>'
            f'<div class="col jargon">{transcript}</div>'
            f'<div class="col plain">{truth}</div></div>')
