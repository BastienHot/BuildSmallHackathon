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

JARGON_CHOICES = [(spec["label"], key) for key, spec in config.JARGONS.items()]

CHARGES, HEARING, PLEA, VERDICT = 1, 2, 3, 4

PLEA_SCREEN = ('<div class="hero" style="min-height:170px"><div class="hero-content">'
               '<div class="hero-emblem">⚖</div><h1>Your defense</h1>'
               '<p>The court turns to you. In your own words — what do you believe you '
               'actually stand accused of?</p></div></div>')

_HIDE = gr.update(visible=False)
_SHOW = gr.update(visible=True)


def _buttons(s: GameSession):
    """Continue until the last generated beat is on screen and no more are coming."""
    done = s.playback_done
    return gr.update(visible=not done), gr.update(visible=done)


def _charges_error(s, message):
    """Bounce back to the Charges step with an explanation."""
    return s, gr.Walkthrough(selected=CHARGES), scene.error_card(message), "", _HIDE, _HIDE


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
    cont, plea = _buttons(s)
    return (s, gr.Walkthrough(selected=HEARING), "",
            scene.render_stage(s.case, s.current_line(), _evidence(s)), cont, plea)


def _loading_view(s, frac, desc):
    return (s, gr.Walkthrough(selected=HEARING), "", scene.loading_card(frac, desc),
            _HIDE, _HIDE)


# ------------------------------------------------------------------ handlers
def start_case(jargon, s):
    """Sample the case, start the background generator, and show beat 1 the moment it
    exists (~case file + one beat, not the whole hearing). A generator: each yield
    repaints the UI."""
    problem = pipeline.preflight()
    if problem:
        yield _charges_error(s, problem)
        return

    s = GameSession()
    yield _loading_view(s, 0.10, "Calling the court to order…")
    try:
        s.case = pipeline.new_case(jargon)
    except Exception as e:  # noqa: BLE001 — engine/startup failure
        log.exception("Could not start the hearing")
        yield _charges_error(s, f"Could not start the hearing:\n• {e}")
        return

    log.info("Case sampled (style=%s); starting background generation", jargon)
    pipeline.start_generation(s)
    yield _loading_view(s, 0.45, "The Game Master is drafting your charges…")

    while not s.case.lines and not s.gen_done:
        time.sleep(0.5)
    if not s.case.lines:
        yield _charges_error(s, f"Generation failed:\n• {s.gen_error or 'no beats produced'}")
        return
    s.playback_idx = 0
    yield _hearing_view(s)


def advance(s):
    """Step to the next beat. Instant when the worker is ahead of the player; shows a
    short deliberation card when it isn't (the player out-read the model)."""
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
                cont = gr.Button("Continue  ▶", elem_classes="bw-btn")
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

        start.click(start_case, [jargon, session],
                    [session, walk, charges_status, screen, cont, plea])
        cont.click(advance, [session], [session, walk, charges_status, screen, cont, plea])
        plea.click(go_to_plea, [session], [walk])
        submit.click(submit_plea, [guess, session], [session, walk, verdict])
        again.click(play_again, None, [session, walk, guess])
    return demo
