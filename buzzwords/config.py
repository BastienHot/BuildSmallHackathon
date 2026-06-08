"""Central configuration.

Everything that decides *where* a model lives or *how* the app behaves is here so
the rest of the code stays runtime-agnostic. The game requires local GGUF weights;
if they are missing the app launches and tells the user what to download
(see pipeline.preflight) rather than crashing.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("BW_MODELS_DIR", ROOT / "models"))
VOICES_DIR = Path(os.getenv("BW_VOICES_DIR", ROOT / "assets" / "voices"))

# ---------------------------------------------------------------------------
# Mode toggles
# ---------------------------------------------------------------------------
# Whether a GPU may be used at all. When False the app is pure CPU/llama.cpp
# (off-grid mode). TTS still works on CPU, just slowly.
ALLOW_GPU = os.getenv("BW_ALLOW_GPU", "1") == "1"


def _detect_gpu_layers() -> int:
    """Return -1 (full GPU offload) if CUDA is available and allowed, else 0 (CPU)."""
    if not ALLOW_GPU:
        return 0
    try:
        import torch
        return -1 if torch.cuda.is_available() else 0
    except ImportError:
        pass
    # Fallback: check for CUDA device count via nvidia-smi env hint
    return -1 if os.getenv("CUDA_VISIBLE_DEVICES", "") not in ("", "-1") else 0


# Computed once at import; used by both model specs below.
N_GPU_LAYERS = _detect_gpu_layers()

# ---------------------------------------------------------------------------
# Text models (all run through the llama.cpp runtime -> Llama Champion badge)
#
# Two resident models:
#   * GM_MODEL (Qwen3.5-4B, pure transformer) = Game Master: writes the hidden
#     Case File, directs the turn loop (GBNF-constrained JSON), and doubles as the
#     scoring judge. Thinking mode disabled via /no_think (GBNF also enforces this).
#   * JARGON_BASE_MODEL (MiniCPM5-1B, Llama archi) = the 3 actors, with ONE
#     per-style LoRA adapter loaded for the whole game (3 roles = 3 system prompts).
#
# GPU priority: local CUDA GPU > HF T4 persistent GPU > ZeroGPU (via @spaces.GPU
# in text_engine.py). N_GPU_LAYERS drives all three — set to -1 when CUDA is
# detected. For ZeroGPU the @spaces.GPU decorator allocates the GPU around each
# inference call so -1 works there too. CUDA 12.1 wheel is in requirements.txt.
# ---------------------------------------------------------------------------
GM_MODEL = {
    "key": "gm-qwen3.5-4b",
    "path": str(MODELS_DIR / "Qwen3.5-4B-Q4_K_M.gguf"),
    "n_ctx": 8192,
    "n_gpu_layers": N_GPU_LAYERS,
}

JARGON_BASE_MODEL = {
    "key": "minicpm-1b",
    "path": str(MODELS_DIR / "MiniCPM5-1B-Q4_K_M.gguf"),
    "n_ctx": 4096,
    "n_gpu_layers": N_GPU_LAYERS,
}

# One LoRA per jargon STYLE (smokescreen), NOT per profession. These are PEFT
# adapters converted to GGUF (no merge) and applied on the vanilla MiniCPM base
# via llama.cpp `lora_path`. Keys align with JARGONS below.
STYLE_LORAS = {
    "corporate": str(MODELS_DIR / "style-corporate.lora.gguf"),
    "aviation": str(MODELS_DIR / "style-aviation.lora.gguf"),
    "ai": str(MODELS_DIR / "style-ai.lora.gguf"),
    "politics": str(MODELS_DIR / "style-politics.lora.gguf"),
    "medical": str(MODELS_DIR / "style-medical.lora.gguf"),
    "gaming": str(MODELS_DIR / "style-gaming.lora.gguf"),
    "sports": str(MODELS_DIR / "style-sports.lora.gguf"),
    "scifi": str(MODELS_DIR / "style-scifi.lora.gguf"),
}

# Weights that MUST be present to play (the style LoRAs are optional -- actors fall
# back to the vanilla base until trained). Used by pipeline.preflight().
REQUIRED_MODELS = [
    ("Game Master (Qwen3.5-4B)", GM_MODEL["path"]),
    ("Actor base (MiniCPM5-1B)", JARGON_BASE_MODEL["path"]),
]

# ---------------------------------------------------------------------------
# Runtime weight download (HF Spaces / fresh machines). buzzwords.weights.ensure_weights()
# pulls these into MODELS_DIR when BW_FETCH_WEIGHTS=1 (set it as a Space variable).
# ---------------------------------------------------------------------------
# Auto-fetch when running on an HF Space (it sets SPACE_ID), or when asked explicitly.
FETCH_WEIGHTS = os.getenv("BW_FETCH_WEIGHTS", "0") == "1" or bool(os.getenv("SPACE_ID"))
HF_BASE_GGUF = ("openbmb/MiniCPM5-1B-GGUF", "MiniCPM5-1B-Q4_K_M.gguf")          # (repo, file)
HF_GM_GGUF = ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf")
HF_LORA_REPO = os.getenv("BW_LORA_REPO", "BastienHot/buzzwords-style-loras")   # the trained style LoRAs

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
TTS_MODEL_ID = os.getenv("BW_TTS_MODEL", "openbmb/VoxCPM2")
# One reference clip per role for voice cloning (the three courtroom speakers).
VOICE_REFS = {
    "judge": str(VOICES_DIR / "judge.wav"),
    "prosecutor": str(VOICES_DIR / "prosecutor.wav"),
    "defense": str(VOICES_DIR / "defense.wav"),
}
TTS_BASE_DURATION = 15   # ZeroGPU budget per synthesized line (seconds)

# ---------------------------------------------------------------------------
# Jargon STYLES -> a smokescreen the actors speak in. Picking a style selects the
# background art and the actor LoRA. It is deliberately UNRELATED to the hidden
# profession/fault the player must guess (see PROFESSIONS).
# ---------------------------------------------------------------------------
MAPS_DIR = ROOT / "assets" / "maps"
COURT_VARIANTS = 16          # variant_01..16.png live under assets/maps/<court>/
# Each style maps to a courtroom backdrop. Only AVIATION has bespoke art so far; the
# rest reuse the normal court as a placeholder (drop a new assets/maps/<court>/ folder
# and point the style at it to give it its own art).
JARGONS = {
    "corporate": {"court": "normal_court",   "label": "Corporate", "blurb": "Boardroom buzzwords"},
    "aviation":  {"court": "aviation_court", "label": "Aviation",  "blurb": "Cockpit & control-tower lingo"},
    "ai":        {"court": "normal_court",   "label": "AI",        "blurb": "Neural-net & model-training jargon"},
    "politics":  {"court": "political_court", "label": "Politics",  "blurb": "Beltway spin & campaign lingo"},
    "medical":   {"court": "normal_court",   "label": "Medical",   "blurb": "Clinical & bedside jargon"},
    "gaming":    {"court": "normal_court",   "label": "Gaming",    "blurb": "Esports & gamer slang"},
    "sports":    {"court": "normal_court",   "label": "Sports",    "blurb": "Locker-room & play-by-play"},
    "scifi":     {"court": "normal_court",   "label": "Sci-fi",    "blurb": "Starship-ops technobabble"},
}
DEFAULT_JARGON = "corporate"

# ---------------------------------------------------------------------------
# Turn budget -> how many beats the debate runs. The GM is nudged toward closure
# by prompt-injected turn pressure (no deterministic latch in v1).
# ---------------------------------------------------------------------------
TURN_BUDGET_BY_DIFFICULTY = {"easy": 8, "normal": 12, "hard": 16}
DEFAULT_DIFFICULTY = "normal"
WRAP_PRESSURE_AT = 2   # start nudging the GM to converge this many turns from the end

# Speech-bubble layout per speaker: bubble bottom-center at (x%, y%) of the
# square stage + which side the tail points from. Fixed for every background
# (the characters sit in the same spots in all of them). No witness for now.
BUBBLES = {
    "judge":      {"x": 50, "y": 49, "tail": "center"},
    "prosecutor": {"x": 64, "y": 58, "tail": "right"},
    "defense":    {"x": 36, "y": 58, "tail": "left"},
}

# ---------------------------------------------------------------------------
# Playback modes
# ---------------------------------------------------------------------------
PLAYBACK_OFF = "off"      # text-only, click-to-advance subtitles
PLAYBACK_ON = "on"        # voiced when a GPU is available, else text-only
