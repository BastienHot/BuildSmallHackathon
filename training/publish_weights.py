"""Publish the gated GGUF adapters from the Modal volume straight to the HF Hub.

Volume -> Hub, no local disk round-trip. Uses the same "huggingface" Modal secret as
the rest of the pipeline (needs a WRITE token). Run after the e2e gate passes:

  modal run training/publish_weights.py
  modal run training/publish_weights.py --director-repo you/your-fork --style-repo you/fork2
"""

from __future__ import annotations

import modal

DIRECTOR_REPO = "BastienHot/buzzwords-director-lora"
STYLE_REPO = "BastienHot/buzzwords-style-loras"
STYLES = ["corporate", "aviation", "ai", "politics", "medical", "gaming", "sports", "scifi"]

app = modal.App("buzzwords-publish")
image = modal.Image.debian_slim().pip_install("huggingface_hub")
gguf_vol = modal.Volume.from_name("buzzwords-gguf", create_if_missing=True)
GGUF = "/gguf"


@app.function(image=image, volumes={GGUF: gguf_vol}, timeout=30 * 60,
              secrets=[modal.Secret.from_name("huggingface-write")])  # needs repo.write
def publish(director_repo: str, style_repo: str):
    import os
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(director_repo, exist_ok=True)
    api.create_repo(style_repo, exist_ok=True)

    api.upload_file(path_or_fileobj=f"{GGUF}/director.lora.gguf",
                    path_in_repo="director.lora.gguf", repo_id=director_repo)
    print(f"uploaded director.lora.gguf -> {director_repo}")
    for s in STYLES:
        f = f"style-{s}.lora.gguf"
        if not os.path.exists(f"{GGUF}/{f}"):
            print(f"SKIP missing {f}")
            continue
        api.upload_file(path_or_fileobj=f"{GGUF}/{f}", path_in_repo=f, repo_id=style_repo)
        print(f"uploaded {f} -> {style_repo}")


@app.local_entrypoint()
def main(director_repo: str = DIRECTOR_REPO, style_repo: str = STYLE_REPO):
    publish.remote(director_repo, style_repo)
