"""Fetch GGUF weights from the Hugging Face Hub into MODELS_DIR.

Used on HF Spaces / fresh machines: set BW_FETCH_WEIGHTS=1 and the app pulls the base
models + the trained style LoRAs on startup, so the repo itself stays weight-free.
Idempotent -- hf_hub_download skips files already present (etag check).
"""

from __future__ import annotations

from pathlib import Path

from . import config


def ensure_weights() -> None:
    from huggingface_hub import hf_hub_download

    targets = [config.HF_BASE_GGUF, config.HF_DIRECTOR_LORA]              # (repo, filename)
    targets += [(config.HF_LORA_REPO, Path(p).name) for p in config.STYLE_LORAS.values()]

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for repo, filename in targets:
        print(f"[weights] {repo}/{filename}")
        hf_hub_download(repo_id=repo, filename=filename, local_dir=str(config.MODELS_DIR))
    print(f"[weights] ready in {config.MODELS_DIR}")
