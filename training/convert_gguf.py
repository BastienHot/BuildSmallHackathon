"""Convert trained PEFT adapters -> GGUF for llama.cpp (no merge), on Modal.

Wraps llama.cpp's `convert_lora_to_gguf.py`. Each adapter in the "buzzwords-adapters"
Volume becomes one mono-file `style-<style>.lora.gguf` in the "buzzwords-gguf" Volume.
Applied at runtime on the vanilla MiniCPM GGUF via `lora_path` -- see
buzzwords/config.py:STYLE_LORAS and TextEngine.act in buzzwords/text_engine.py.

Run:
  modal run training/convert_gguf.py
Then download into the app's (git-ignored) models/ dir:
  modal volume get buzzwords-gguf / ./models
"""

from __future__ import annotations

import modal

BASE_MODEL = "openbmb/MiniCPM5-1B"
LLAMA_CPP = "/opt/llama.cpp"
# adapter_dir_name -> output gguf filename (matches config.STYLE_LORAS)
ADAPTERS = {
    "actor_corporate": "style-corporate.lora.gguf",
    "actor_aviation": "style-aviation.lora.gguf",
    "actor_ai": "style-ai.lora.gguf",
    "actor_politics": "style-politics.lora.gguf",
    "actor_medical": "style-medical.lora.gguf",
    "actor_gaming": "style-gaming.lora.gguf",
    "actor_sports": "style-sports.lora.gguf",
    "actor_scifi": "style-scifi.lora.gguf",
}

app = modal.App("buzzwords-convert")
image = (modal.Image.debian_slim()
         .apt_install("git")
         .pip_install("torch", "transformers", "gguf", "sentencepiece", "numpy",
                      "huggingface_hub", "safetensors")
         .run_commands(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA_CPP}"))
adapters_vol = modal.Volume.from_name("buzzwords-adapters", create_if_missing=True)
gguf_vol = modal.Volume.from_name("buzzwords-gguf", create_if_missing=True)
ADAPTERS_DIR, GGUF_DIR = "/adapters", "/gguf"


@app.function(image=image, timeout=60 * 60,
              volumes={ADAPTERS_DIR: adapters_vol, GGUF_DIR: gguf_vol},
              secrets=[modal.Secret.from_name("huggingface")])
def convert():
    import os
    import subprocess

    for adapter, out_name in ADAPTERS.items():
        src = f"{ADAPTERS_DIR}/{adapter}"
        if not os.path.isdir(src):
            print(f"skip (missing): {adapter}")
            continue
        out = f"{GGUF_DIR}/{out_name}"
        subprocess.run(
            # --base-model-id reads the base config from the HF hub (no local weights needed);
            # --base would instead expect a LOCAL directory.
            ["python", f"{LLAMA_CPP}/convert_lora_to_gguf.py",
             "--base-model-id", BASE_MODEL, "--outfile", out, src],
            check=True,
        )
        print(f"converted {adapter} -> {out_name}")
    gguf_vol.commit()


@app.local_entrypoint()
def main():
    convert.remote()
