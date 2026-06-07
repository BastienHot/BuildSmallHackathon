# Deploying to a Hugging Face Space

The game runs **text on CPU via llama.cpp** (keeps the Llama Champion + Off-the-Grid
angle) and uses **ZeroGPU only for TTS** (when VoxCPM2 is wired; it's a text-only stub
today). Weights are **not** committed — the Space pulls them from the Hub at startup.

## Architecture of the deploy
```
GitHub (main)  --GitHub Action-->  HF Space (git mirror)  --build-->  runs app.py
                                                              |
                                          startup: ensure_weights() pulls GGUFs from the Hub
```
- Base GGUFs: `openbmb/MiniCPM5-1B-GGUF`, `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF`
- Style LoRAs: `BastienHot/buzzwords-style-loras` (8 × `style-*.lora.gguf`)

## One-time setup

### 1. Create the Space
- New Space → **SDK: Gradio**, owner = the hackathon org (required for ZeroGPU).
- **Hardware:** start with **CPU basic** (TTS is stubbed, so no GPU is used yet). Switch
  to **ZeroGPU** once VoxCPM2 is wired, or to a **T4** later for fast GPU llama.cpp
  (needs a CUDA `llama-cpp-python` build + `n_gpu_layers=-1`).
- **Persistent storage:** enable it so the ~3.5 GB of weights survive rebuilds (otherwise
  they re-download on every cold start).

### 2. Space → Settings → Variables and secrets
- Variable **`BW_FETCH_WEIGHTS` = `1`** → triggers the startup download.
- The LoRA repo is public, so no token is needed. (If you make it private, add an
  **`HF_TOKEN`** *secret* on the Space too.)

### 3. GitHub → Settings → Secrets and variables → Actions
- Secret **`HF_TOKEN`** = a write token (https://huggingface.co/settings/tokens).
- Variable **`HF_USERNAME`** = your HF username (e.g. `BastienHot`).
- Variable **`HF_SPACE`** = `<owner>/<space-name>` (e.g. `hackathon-org/buzzwords`).

### 4. Push
```bash
git push origin main      # the Action mirrors to the Space; the Space rebuilds
```
First boot: builds `llama-cpp-python` (slow, compiles from source) + downloads ~3.5 GB of
weights, then `preflight()` passes and the game runs.

## Expectations / gotchas
- **CPU latency:** the 4B GM + 1B actor per turn on CPU is slow (tens of seconds/beat).
  The turn-based UI hides it somewhat ("the court deliberates…"). If too slow: shrink the
  GM, upgrade the CPU tier, or move to a T4 GPU Space (see `docs/ARCHITECTURE.md`).
- **First boot is slow** (build + 3.5 GB download). Subsequent boots are fast *with*
  persistent storage.
- **Weights never enter git** — `models/` is git-ignored; the Action ships code only, so
  no Git LFS is needed.
- **Local dev** doesn't auto-download: leave `BW_FETCH_WEIGHTS` unset and place GGUFs in
  `models/` yourself (see the README), or set it to `1` to fetch.
