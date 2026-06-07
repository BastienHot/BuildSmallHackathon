"""Play one game with the real models, capture the agent trace, and publish it to the Hub.

The "agent trace" is the Game Master's structured decisions (who speaks, which beat, intensity,
stage direction, wrap-up) plus the actors' lines for a full game -- the GM-directed pipeline in
action, alongside the hidden Case File and the final scoring. Needs the GGUFs in models/.

  python training/share_trace.py --style corporate --repo BastienHot/buzzwords-agent-trace
  python training/share_trace.py --style medical --no-upload      # just write locally
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from buzzwords import config, pipeline
from buzzwords.models import GameSession, Line

CARD = """\
---
license: apache-2.0
tags: [agent-trace, llama-cpp, courtroom-game]
---

# Buzzwords & Misdemeanors -- agent traces

Each JSON is one full game of [Buzzwords & Misdemeanors](https://huggingface.co/spaces/build-small-hackathon/BuzzwordsMisdemeanors).
A **Game Master** (Nemotron 3 Nano 4B, GBNF-constrained) writes a hidden Case File and then,
turn by turn, decides who speaks and how; the **actors** (MiniCPM5-1B + a per-style LoRA) deliver
the lines. The trace records the GM's structured decision and the resulting line for every turn,
plus the hidden truth and the final scored guess. All inference runs locally through `llama.cpp`.
"""


def play_traced(style: str, difficulty: str, guess: str) -> dict:
    eng = pipeline._eng()
    case = pipeline.new_case(style, difficulty)
    cf = case.case_file
    session = GameSession(mode="off")
    session.case = case
    budget = cf.turn_budget

    turns = []
    while not session.finished_playback:
        d = eng.gm_decide(cf, case.lines, session.turn, budget)        # the agent's decision
        text = eng.act(d.next_speaker, d.stage_direction, style, d.intensity)
        case.lines.append(Line(actor=d.next_speaker, beat_type=d.beat_type, text=text))
        session.turn += 1
        session.wrapped = d.wrap_up or session.turn >= budget
        turns.append({
            "turn": session.turn,
            "gm_decision": {"next_speaker": d.next_speaker, "beat_type": d.beat_type,
                            "intensity": d.intensity, "stage_direction": d.stage_direction,
                            "wrap_up": d.wrap_up},
            "actor_line": {"actor": d.next_speaker, "text": text},
        })

    score, rationale = pipeline.score_guess(case, guess)
    return {
        "game": "Buzzwords & Misdemeanors",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "jargon_style": style,
        "difficulty": difficulty,
        "models": {"game_master": Path(config.GM_MODEL["path"]).name,
                   "actors": f"{Path(config.JARGON_BASE_MODEL['path']).name} + style-{style} LoRA"},
        "hidden_case_file": {"profession": cf.profession, "fault_plain": cf.fault_plain,
                             "facts": cf.facts},
        "turns": turns,
        "player_guess": guess,
        "score": score,
        "score_rationale": rationale,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="corporate")
    ap.add_argument("--difficulty", default="normal")
    ap.add_argument("--guess", default="a teacher who falsified an official record")
    ap.add_argument("--repo", default="BastienHot/buzzwords-agent-trace")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()

    trace = play_traced(a.style, a.difficulty, a.guess)
    out = f"trace_{a.style}.json"
    Path(out).write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}: {len(trace['turns'])} turns, hidden={trace['hidden_case_file']['profession']!r}, "
          f"score={trace['score']}")

    if not a.no_upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.repo, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                        repo_id=a.repo, repo_type="dataset")
        api.upload_file(path_or_fileobj=out, path_in_repo=out, repo_id=a.repo, repo_type="dataset")
        print(f"uploaded -> https://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()
