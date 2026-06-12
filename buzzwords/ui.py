"""Gradio UI built as a `gr.Walkthrough`: the game's four phases ARE the steps
(Charges → The hearing → Your plea → The verdict).

No pre-generated games (REBUILD_REVIEW.md §8.3): the hearing starts as soon as the
FIRST beat exists, and a background worker keeps generating while the player reads.
"Continue" is instant when the next beat is ready and briefly shows a deliberation
card when the player out-reads the model.
"""

from __future__ import annotations

import logging
import time

import gradio as gr

from . import config, pipeline, scene
from .models import GameSession

log = logging.getLogger(__name__)

JARGON_CHOICES = [(f"{spec['icon']} {spec['label']}", key)
                  for key, spec in config.JARGONS.items()]

CHARGES, HEARING, PLEA, VERDICT = 1, 2, 3, 4

PLEA_SCREEN = ('<div class="hero" style="min-height:170px"><div class="hero-content">'
               '<div class="hero-emblem">⚖</div><h1>Your defense</h1>'
               '<p>The court turns to you. In your own words — what do you believe you '
               'actually stand accused of?</p></div></div>')

_HIDE = gr.update(visible=False)
_SHOW = gr.update(visible=True)


def _nav(s: GameSession):
    """Return (prev_update, next_update, plea_update, counter_html, progress_frac)."""
    done = s.playback_done
    if s.case and s.case.lines:
        total = len(s.case.lines)
        idx = min(s.playback_idx, total - 1) + 1
        counter = f'<div class="beat-counter">{idx}&thinsp;/&thinsp;{total}</div>'
        progress = idx / total
    else:
        counter, progress = "", 0.0
    return (
        gr.update(visible=s.playback_idx > 0),
        gr.update(visible=not done),
        gr.update(visible=done),
        counter,
        progress,
    )


def _charges_error(s, message):
    """Bounce back to the Charges step with an explanation."""
    return s, gr.Walkthrough(selected=CHARGES), scene.error_card(message), "", _HIDE, _HIDE, _HIDE, ""


def _evidence(s: GameSession) -> list[str]:
    """Facts surfaced up to the beat the player is currently viewing, in court order."""
    seen, out = set(), []
    for ln in s.case.lines[:s.playback_idx + 1]:
        fi = ln.fact_index
        if fi is not None and fi not in seen and fi < len(s.case.case_file.facts):
            seen.add(fi)
            out.append(s.case.case_file.facts[fi])
    return out


def _hearing_view(s: GameSession):
    prev_u, cont_u, plea_u, counter, progress = _nav(s)
    return (s, gr.Walkthrough(selected=HEARING), "",
            scene.render_stage(s.case, s.current_line(), _evidence(s), progress=progress),
            prev_u, cont_u, plea_u, counter)


def _loading_view(s, frac, desc):
    return (s, gr.Walkthrough(selected=HEARING), "", scene.loading_card(frac, desc),
            _HIDE, _HIDE, _HIDE, "")


# ------------------------------------------------------------------ handlers
def start_case(jargon, s):
    """Sample the case and PRE-GENERATE the whole hearing behind the progress bar
    (player decision: the total wait is short, and every Continue must be instant).
    The worker still runs in a background thread; this handler just streams progress
    until it finishes. A generator: each yield repaints the UI."""
    problem = pipeline.preflight()
    if problem:
        yield _charges_error(s, problem)
        return

    s = GameSession()
    yield _loading_view(s, 0.04, "Calling the court to order…")
    try:
        s.case = pipeline.new_case(jargon)
    except Exception as e:  # noqa: BLE001 — engine/startup failure
        log.exception("Could not start the hearing")
        yield _charges_error(s, f"Could not start the hearing:\n• {e}")
        return

    log.info("Case sampled (style=%s); pre-generating the hearing", jargon)
    pipeline.start_generation(s)
    budget, shown = s.case.case_file.turn_budget, -1
    while not s.gen_done:
        if s.turn != shown:   # only repaint when a new beat lands
            shown = s.turn
            frac = min(0.96, 0.10 + 0.86 * (s.turn / max(1, budget)))
            yield _loading_view(s, frac, f"Staging the hearing — beat {s.turn} of {budget}…")
        time.sleep(0.5)
    if not s.case.lines:
        yield _charges_error(s, f"Generation failed:\n• {s.gen_error or 'no beats produced'}")
        return
    s.playback_idx = 0
    yield _hearing_view(s)


def advance(s):
    """Step to the next pre-generated beat — instant. (The deliberation-wait branch
    below is a safety net; with full pre-generation it should never be reached.)"""
    if s.next_line_ready():
        s.playback_idx += 1
        yield _hearing_view(s)
        return
    if s.gen_done:                      # nothing more coming — refresh button state
        yield _hearing_view(s)
        return
    yield _loading_view(s, 0.85, "The court deliberates…")
    while not s.next_line_ready() and not s.gen_done:
        time.sleep(0.5)
    if s.next_line_ready():
        s.playback_idx += 1
    yield _hearing_view(s)


def go_back(s):
    """Step back to the previous beat — always instant."""
    if s.case and s.playback_idx > 0:
        s.playback_idx -= 1
    return _hearing_view(s)


def go_to_plea(s):
    return gr.Walkthrough(selected=PLEA)


def submit_plea(guess, s):
    s.guess = (guess or "").strip()
    try:
        s.score, s.rationale = pipeline.score_guess(s.case, s.guess)
    except Exception as e:  # noqa: BLE001
        log.exception("Scoring failed for guess=%r", s.guess)
        return s, gr.Walkthrough(selected=VERDICT), scene.error_card(f"Scoring failed:\n• {e}")
    return (s, gr.Walkthrough(selected=VERDICT),
            scene.render_verdict_banner(s) + scene.render_reveal(s))


def play_again():
    return GameSession(), gr.Walkthrough(selected=CHARGES), gr.update(value="")


# -------------------------------------------------------------------- layout
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Buzzwords & Misdemeanors", elem_id="bw-root") as demo:
        session = gr.State(GameSession())
        with gr.Walkthrough(selected=CHARGES) as walk:
            with gr.Step("⚖ Charges", id=CHARGES):
                gr.HTML(scene.title_card(scene.image_url(config.DEFAULT_JARGON)))
                jargon = gr.Radio(JARGON_CHOICES, value=config.DEFAULT_JARGON,
                                  label="Jargon", elem_classes="bw-pick")
                start = gr.Button("⚖  Start the hearing", elem_classes="bw-btn")
                charges_status = gr.HTML()   # preflight / startup error messages
            with gr.Step("The hearing", id=HEARING):
                screen = gr.HTML(elem_id="bw-screen")
                with gr.Row(elem_classes="bw-nav"):
                    prev = gr.Button("← Back", elem_classes="bw-btn ghost bw-nav-prev",
                                     visible=False, scale=0, min_width=120)
                    beat_counter = gr.HTML(elem_id="bw-counter", scale=1)
                    cont = gr.Button("Next →", elem_classes="bw-btn bw-nav-next",
                                     scale=0, min_width=120)
                plea = gr.Button("⚖  Enter your plea", elem_classes="bw-btn danger", visible=False)
            with gr.Step("Your plea", id=PLEA):
                gr.HTML(PLEA_SCREEN)
                guess = gr.Textbox(value="", lines=2, show_label=False,
                                   placeholder="What is the charge against you?",
                                   elem_classes="bw-guess")
                submit = gr.Button("Deliver my defense", elem_classes="bw-btn danger")
            with gr.Step("The verdict", id=VERDICT):
                verdict = gr.HTML()
                again = gr.Button("↻  New case", elem_classes="bw-btn ghost")

        hearing_outputs = [session, walk, charges_status, screen, prev, cont, plea, beat_counter]
        start.click(start_case, [jargon, session], hearing_outputs)
        cont.click(advance, [session], hearing_outputs)
        prev.click(go_back, [session], hearing_outputs)
        plea.click(go_to_plea, [session], [walk])
        submit.click(submit_plea, [guess, session], [session, walk, verdict])
        again.click(play_again, None, [session, walk, guess])
    return demo
