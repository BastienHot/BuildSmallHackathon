# Deploying to a Hugging Face Space

The game runs **entirely on CPU through llama.cpp** — no GPU at any point. It ships as a
**Docker Space** (the UI is still Gradio; Docker only exists to compile `llama-cpp-python`
with AVX2, which is what makes the 1B fast on the free 2-vCPU tier). Weights are **not**
committed — the Space pulls them from the Hub at startup.

## Architecture of the deploy
```
GitHub (main) --GitHub Action--> HF Space (git mirror) --docker build--> CMD: python app.py
                                                            |
                              startup: ensure_weights() pulls the base GGUF + LoRAs from the Hub
```
- Base GGUF (Game Master **and** actors): `openbmb/MiniCPM5-1B-GGUF` (`MiniCPM5-1B-Q4_K_M.gguf`)
- Director LoRA (Game Master): `BastienHot/buzzwords-director-lora` (`director.lora.gguf`)
- Style LoRAs (actors): `BastienHot/buzzwords-style-loras` (8 × `style-*.lora.gguf`)

The whole game is **one ~1B base + two ~90 MB adapters** (director for the GM, style for the
actors), swapped by reference at runtime — see `buzzwords/config.py`.

## One-time setup

### 1. The Space
- The Space type is set by `sdk: docker` in `README.md` frontmatter (+ `app_port: 7860`).
  Pushing that + the `Dockerfile` makes an existing Gradio Space **rebuild as a Docker Space**
  in place — no need to recreate it.
- **Hardware: `cpu-basic`** (free, 2 vCPU). No GPU/ZeroGPU.
- Persistent storage is optional — without it the ~0.9 GB of weights re-download on a cold
  start, which is quick on the Hub.

### 2. Weights fetch
- `ensure_weights()` runs when `SPACE_ID` is set (always true on a Space) or when
  `BW_FETCH_WEIGHTS=1` (the Dockerfile sets it). The LoRA repos are public, so no token is
  needed on the Space. Override repos with `BW_DIRECTOR_REPO` / `BW_LORA_REPO` if you fork them.

### 3. GitHub → Settings → Secrets and variables → Actions
- Secret **`HF_TOKEN`** = a write token (https://huggingface.co/settings/tokens).
- Variable **`HF_USERNAME`** = your HF username (e.g. `BastienHot`).
- Variable **`HF_SPACE`** = `<owner>/<space-name>` (e.g. `build-small-hackathon/BuzzwordsMisdemeanors`).

### 4. Push
```bash
git push origin main      # the Action force-mirrors to the Space; the Space rebuilds
```

## Expectations / gotchas
- **First boot is slow:** the Docker build compiles `llama-cpp-python` with AVX2 (a few
  minutes), then the app downloads the base GGUF + the director/style LoRAs before the UI is
  ready. Subsequent deploys are just `git push` to `main`.
- **CPU latency:** with the AVX2 build, the 1B does ~60–80 tok/s prompt-eval and ~12 tok/s
  generation on 2 vCPU; thinking is disabled (GBNF forces it off) and the turn loop uses prompt
  caching, so per-beat latency stays in seconds. The stock prebuilt wheel is ~30× slower on
  prompt-eval — the AVX2 source build in the `Dockerfile` is not optional.
- **Weights never enter git** — `models/` is git-ignored; the Action ships code only, no LFS.
- **Local dev:** `CMAKE_ARGS="-DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON" pip install
  --no-binary llama-cpp-python -r requirements.txt`, then `python app.py` (set
  `BW_FETCH_WEIGHTS=1` to auto-pull weights, or drop the GGUFs into `models/` yourself).
