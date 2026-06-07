"""Gradio UI built as a `gr.Walkthrough`: the game's four phases ARE the steps
(Charges → The hearing → Your plea → The verdict). Each step owns its widgets, so
there is no manual show/hide juggling — switching step = return gr.Walkthrough(selected=…).
"""

from __future__ import annotations

import tempfile

import gradio as gr

from . import config, pipeline, scene, tts_engine
from .models import GameSession

VOICE_CHOICES = [("Voices", config.PLAYBACK_ON), ("Silent", config.PLAYBACK_OFF)]
JARGON_CHOICES = [(spec["label"], key) for key, spec in config.JARGONS.items()]

CHARGES, HEARING, PLEA, VERDICT = 1, 2, 3, 4

PLEA_SCREEN = ('<div class="hero" style="min-height:170px"><div class="hero-content">'
               '<div class="hero-emblem">⚖</div><h1>Your defense</h1>'
               '<p>The court turns to you. In your own words — what do you believe you '
               'actually stand accused of?</p></div></div>')


def _audio(s: GameSession):
    ln = s.current_line()
    return (gr.update(value=ln.audio_path, autoplay=True) if (ln and ln.audio_path)
            else gr.update(value=None))


def _speak_current(s: GameSession, line) -> None:
    if line is not None:
        line.audio_path = tts_engine.voice_line(line, s.mode, s.audio_dir, s.turn - 1)


def _hearing_buttons(s: GameSession):
    """Show Continue until the debate wraps, then offer the plea button."""
    done = s.finished_playback
    return gr.update(visible=not done), gr.update(visible=done)


_HIDE = gr.update(visible=False)


# ------------------------------------------------------------------ handlers
def start_case(mode, jargon, s):
    problem = pipeline.preflight()
    if not problem:
        try:
            s = GameSession(mode=mode)
            s.audio_dir = tempfile.mkdtemp(prefix="bw_audio_")
            s.case = pipeline.new_case(jargon, config.DEFAULT_DIFFICULTY)
            _speak_current(s, pipeline.next_turn(s))   # generate + voice the opening beat
        except Exception as e:  # model present but failed to load/generate
            problem = f"Could not start the hearing:\n• {e}"
    if problem:
        return s, gr.Walkthrough(selected=CHARGES), scene.error_card(problem), "", \
            gr.update(value=None), _HIDE, _HIDE
    cont, plea = _hearing_buttons(s)
    return (s, gr.Walkthrough(selected=HEARING), "",
            scene.render_stage(s.case, s.current_line()), _audio(s), cont, plea)


def advance(s):
    try:
        _speak_current(s, pipeline.next_turn(s))       # generate + voice the next beat
    except Exception as e:
        return s, scene.error_card(f"Generation failed:\n• {e}"), gr.update(value=None), \
            gr.update(visible=True), _HIDE
    cont, plea = _hearing_buttons(s)
    return s, scene.render_stage(s.case, s.current_line()), _audio(s), cont, plea


def go_to_plea(s):
    return gr.Walkthrough(selected=PLEA)


def submit_plea(guess, s):
    s.guess = (guess or "").strip()
    try:
        s.score, s.rationale = pipeline.score_guess(s.case, s.guess)
    except Exception as e:
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
                voice = gr.Radio(VOICE_CHOICES, value=config.PLAYBACK_OFF,
                                 label="Voice", elem_classes="bw-pick")
                start = gr.Button("⚖  Start the hearing", elem_classes="bw-btn")
                charges_status = gr.HTML()   # preflight / startup error messages
            with gr.Step("The hearing", id=HEARING):
                screen = gr.HTML(elem_id="bw-screen")
                audio = gr.Audio(visible=False, autoplay=True)
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

        start.click(start_case, [voice, jargon, session],
                    [session, walk, charges_status, screen, audio, cont, plea])
        cont.click(advance, [session], [session, screen, audio, cont, plea])
        plea.click(go_to_plea, [session], [walk])
        submit.click(submit_plea, [guess, session], [session, walk, verdict])
        again.click(play_again, None, [session, walk, guess])
    return demo
