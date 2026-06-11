"""Central configuration.

Everything that decides *where* a model lives or *how* the app behaves is here so the
rest of the code stays runtime-agnostic. The game is 100% CPU through a managed
`llama-server` subprocess (ONE resident MiniCPM5-1B base; the director and style LoRAs
are registered once and switched per request by scale — REBUILD_REVIEW.md §10.5).
If weights or the server binary are missing the app launches and tells the user what
to fix (see pipeline.preflight) rather than crashing.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("BW_MODELS_DIR", ROOT / "models"))

# ---------------------------------------------------------------------------
# llama-server (the only inference runtime)
# ---------------------------------------------------------------------------
# Binary: env override > PATH > the location the Dockerfile installs to.
LLAMA_SERVER_BIN = (os.getenv("BW_LLAMA_SERVER")
                    or shutil.which("llama-server")
                    or "/usr/local/bin/llama-server")
LLAMA_SERVER_PORT = int(os.getenv("BW_LLAMA_PORT", "8089"))
# Pin threads to the vCPUs actually allocated (cpu-basic = 2). Inside a Space container
# os.cpu_count() can report the host's cores; oversubscription SLOWS llama.cpp (§8.4).
N_THREADS = int(os.getenv("BW_THREADS", "2"))
# Two server slots so the GM and the actor each keep their own warm prompt cache:
# slot 0 = Game Master (director LoRA), slot 1 = actor (style LoRA or vanilla base).
GM_SLOT, ACTOR_SLOT = 0, 1
N_PARALLEL = 2
N_CTX = int(os.getenv("BW_N_CTX", "8192"))     # total; split across the 2 slots

BASE_GGUF = str(MODELS_DIR / "MiniCPM5-1B-Q4_K_M.gguf")
DIRECTOR_LORA = str(MODELS_DIR / "director.lora.gguf")

# One LoRA per jargon STYLE (smokescreen), NOT per profession. PEFT adapters converted
# to GGUF (no merge), registered with llama-server at startup. Keys align with JARGONS.
STYLE_LORAS = {
    s: str(MODELS_DIR / f"style-{s}.lora.gguf")
    for s in ["corporate", "aviation", "ai", "politics", "medical", "gaming", "sports", "scifi"]
}

# Weights that MUST be present to play (style LoRAs are optional — actors fall back to
# the vanilla base until trained). Used by pipeline.preflight().
REQUIRED_MODELS = [
    ("Base GGUF (MiniCPM5-1B, GM + actors)", BASE_GGUF),
    ("Director LoRA (Game Master)", DIRECTOR_LORA),
]

# Generation budgets — tight caps bound the worst case at ~12 tok/s decode (§4.7).
MAX_TOKENS = {"facts": 350, "decide": 128, "act": 120, "score": 96}
REPEAT_PENALTY = 1.15   # actors only: fights cross-line phrase recycling (§13.5)

# ---------------------------------------------------------------------------
# Runtime weight download (HF Spaces / fresh machines). buzzwords.weights.ensure_weights()
# pulls these into MODELS_DIR when BW_FETCH_WEIGHTS=1 or on a Space (SPACE_ID set).
# ---------------------------------------------------------------------------
FETCH_WEIGHTS = os.getenv("BW_FETCH_WEIGHTS", "0") == "1" or bool(os.getenv("SPACE_ID"))
HF_BASE_GGUF = ("openbmb/MiniCPM5-1B-GGUF", "MiniCPM5-1B-Q4_K_M.gguf")          # (repo, file)
HF_DIRECTOR_LORA = (os.getenv("BW_DIRECTOR_REPO", "BastienHot/buzzwords-director-lora"),
                    "director.lora.gguf")
HF_LORA_REPO = os.getenv("BW_LORA_REPO", "BastienHot/buzzwords-style-loras")

# ---------------------------------------------------------------------------
# Jargon STYLES -> a smokescreen the actors speak in. Picking a style selects the
# backdrop art and the actor LoRA. It is deliberately UNRELATED to the hidden
# profession/fault (see buzzwords/pools.py for the sampled truths).
# ---------------------------------------------------------------------------
MAPS_DIR = ROOT / "assets" / "maps"
COURT_VARIANTS = 16          # variant_01..16.png live under assets/maps/<court>/
JARGONS = {
    "corporate": {"court": "corporate_court", "label": "Corporate", "blurb": "Boardroom buzzwords"},
    "aviation":  {"court": "aviation_court", "label": "Aviation",  "blurb": "Cockpit & control-tower lingo"},
    "ai":        {"court": "ia_court",       "label": "AI",        "blurb": "Neural-net & model-training jargon"},
    "politics":  {"court": "political_court", "label": "Politics",  "blurb": "Beltway spin & campaign lingo"},
    "medical":   {"court": "normal_court",   "label": "Medical",   "blurb": "Clinical & bedside jargon"},
    "gaming":    {"court": "gaming_court",   "label": "Gaming",    "blurb": "Esports & gamer slang"},
    "sports":    {"court": "sport_court",    "label": "Sports",    "blurb": "Locker-room & play-by-play"},
    "scifi":     {"court": "scifi_court",    "label": "Sci-fi",    "blurb": "Starship-ops technobabble"},
}
DEFAULT_JARGON = "corporate"

# Speech-bubble layout per speaker: bubble bottom-center at (x%, y%) of the square
# stage + which side the tail points from. Identical in every background.
BUBBLES = {
    "judge":      {"x": 50, "y": 49, "tail": "center"},
    "prosecutor": {"x": 64, "y": 58, "tail": "right"},
    "defense":    {"x": 36, "y": 58, "tail": "left"},
}
