"""Optional TTS via VoxCPM2 -- the only GPU consumer.

VoxCPM2 is not wired yet, so ``voice_line`` returns None and the game plays text-only.
Fill the two TODOs to enable voiced playback (GPU on ZeroGPU via @spaces.GPU).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from . import config

import spaces  # ZeroGPU decorator; no-op on T4 / local GPU

_TTS = None


def _load_tts():
    global _TTS
    if _TTS is None:
        pass  # TODO: _TTS = VoxCPM2.from_pretrained(config.TTS_MODEL_ID).to("cuda")
    return _TTS


@spaces.GPU(duration=config.TTS_BASE_DURATION)
def _synth(text: str, ref_path: str, out_path: str) -> Optional[str]:
    tts = _load_tts()
    if tts is None:
        return None
    # TODO: wav = tts.clone_and_synth(text=text, prompt_wav=ref_path)
    #       soundfile.write(out_path, wav, samplerate); return out_path
    return None


def voice_line(line, mode: str, out_dir: str, idx: int) -> Optional[str]:
    """Synthesize one freshly-generated line, or None to fall back to on-screen text."""
    if mode != config.PLAYBACK_ON or not _gpu_available():
        return None
    ref = config.VOICE_REFS.get(line.actor)
    if not ref:
        return None
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return _synth(line.text, ref, os.path.join(out_dir, f"line_{idx:03d}.wav"))


def _gpu_available() -> bool:
    if not config.ALLOW_GPU:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
