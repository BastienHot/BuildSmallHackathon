"""Play a POOL of full games with the real models, capture the agent traces, and publish
them to the Hub (the hackathon's shared-traces deliverable: 50-100 traces, not one).

Each trace = the Game Master's structured decisions (speaker, beat, fact_index,
intensity, stage direction, wrap-up — plus whether the deterministic guard remapped it),
the actors' lines, the hidden truth, and a scored guess. Needs the GGUFs in models/ and
the llama-server binary (the same runtime the game uses).

  python training/share_trace.py --n 64                       # 8 per style, then upload
  python training/share_trace.py --n 8 --no-upload            # local smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on sys.path

from buzzwords import config, contracts, pipeline
from buzzwords.models import GameSession

CARD = """\
---
license: apache-2.0
tags: [agent-trace, llama-cpp, courtroom-game]
---

# Buzzwords & Misdemeanors -- agent traces

Each JSON is one full game of [Buzzwords & Misdemeanors](https://huggingface.co/spaces/build-small-hackathon/BuzzwordsMisdemeanors).
A **Game Master** (MiniCPM5-1B + a distilled *director* LoRA, GBNF-constrained) writes the
oblique clue facts for a truth sampled in code, then, beat by beat, decides who speaks, which
clue to surface, and how; deterministic guards enforce courtroom sequencing invariants; the
**actors** (the same MiniCPM5-1B base + a per-style LoRA) deliver the lines with dialogue
context. The trace records every structured decision (and whether a guard remapped it), the
resulting line, the hidden truth, and a scored guess. All inference runs locally through
`llama.cpp` (llama-server, pure CPU) — the whole game on one ~1B base + small adapters.
"""


def play_traced(style: str, guess: str | None) -> dict:
    case = pipeline.new_case(style)
    cf = case.case_file
    session = GameSession()
    session.case = case

    turns = []
    while not session.generation_finished:
        line = pipeline.next_turn(session)
        if line is None:
            break
        turns.append({
            "turn": session.turn,
            "speaker": line.actor, "beat_type": line.beat_type,
            "fact_index": line.fact_index,
            "actor_line": line.text,
        })

    guess = guess or f"{contracts.article(cf.profession)} {cf.profession} who {cf.fault_plain}"
    score, rationale = pipeline.score_guess(case, guess)
    return {
        "game": "Buzzwords & Misdemeanors",
        "shape_version": contracts.SHAPE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "jargon_style": style,
        "models": {"game_master": f"{Path(config.BASE_GGUF).name} + director LoRA",
                   "actors": f"{Path(config.BASE_GGUF).name} + style-{style} LoRA"},
        "hidden_case_file": {"profession": cf.profession, "fault_plain": cf.fault_plain,
                             "facts": cf.facts},
        "turns": turns,
        "player_guess": guess,
        "score": score,
        "score_rationale": rationale,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="total games (round-robin styles)")
    ap.add_argument("--guess", default="", help="fixed guess; default = the truth (spot-on)")
    ap.add_argument("--repo", default="BastienHot/buzzwords-agent-trace")
    ap.add_argument("--out-dir", default="traces")
    ap.add_argument("--no-upload", action="store_true")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(exist_ok=True)
    files = []
    for i in range(a.n):
        style = contracts.STYLES[i % len(contracts.STYLES)]
        trace = play_traced(style, a.guess or None)
        out = out_dir / f"trace_{i:03d}_{style}.json"
        out.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        files.append(out)
        print(f"[{i + 1}/{a.n}] {out.name}: {len(trace['turns'])} turns, "
              f"hidden={trace['hidden_case_file']['profession']!r}, score={trace['score']}")

    if not a.no_upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.repo, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                        repo_id=a.repo, repo_type="dataset")
        api.upload_folder(folder_path=str(out_dir), path_in_repo="traces",
                          repo_id=a.repo, repo_type="dataset")
        print(f"uploaded {len(files)} traces -> https://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()
