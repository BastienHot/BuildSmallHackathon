"""Buzzwords & Misdemeanors - a courtroom deduction game built on small models.

Package layout:
    contracts    - THE single source of truth: grammars, prompts, enums (shared w/ training)
    pools        - curated truths: professions w/ domain tags + matched faults
    config       - settings, model/server paths, required-weights list
    models       - CaseFile / GMDecision / Line / Case / GameSession dataclasses
    engine       - managed llama-server subprocess (1 base + per-request LoRA scales)
    pipeline     - sampled case + guarded turn loop + background generation + scoring
    ui           - the Gradio gr.Walkthrough UI (phases -> steps)
"""

__version__ = "0.1.0"
