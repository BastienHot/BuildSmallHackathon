"""Buzzwords & Misdemeanors - a courtroom deduction game built on small models.

See docs/ARCHITECTURE.md for the design. Package layout:
    config       - settings, model paths, required-weights list
    models       - CaseFile / GMDecision / Line / Case / GameSession dataclasses
    text_engine  - llama.cpp: Game Master (Nemotron) + actors (MiniCPM + style LoRA)
    tts_engine   - optional VoxCPM2 voice cloning (@spaces.GPU; text-only fallback)
    pipeline     - case file + turn loop + scoring (+ preflight checks)
    ui           - the Gradio gr.Walkthrough UI (phases -> steps)
"""

__version__ = "0.0.1"
