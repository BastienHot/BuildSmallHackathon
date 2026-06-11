# Deploying to a Hugging Face Space

The game runs **entirely on CPU through llama.cpp** — no GPU at any point. It ships as a
**Docker Space**: the UI is plain Gradio; Docker exists to build the **`llama-server`**
binary from llama.cpp source with **AVX2** (what makes the 1B fast on the free 2-vCPU
tier). The app starts llama-server as a managed subprocess (`buzzwords/engine.py`) with
ONE resident base and every LoRA registered once, switched per request by scale.
Weights are **not** committed — the Space pulls them from the Hub at startup.

## Architecture of the deploy
```
GitHub (main) --GitHub Action--> HF Space (git mirror) --docker build--> CMD: python app.py
                       stage 1: cmake llama.cpp -> llama-server (AVX2, static)
                       startup: ensure_weights() pulls base GGUF + LoRAs from the Hub
                                engine.start() launches llama-server + health check + self-test
```
- Base GGUF (Game Master **and** actors): `openbmb/MiniCPM5-1B-GGUF` (`MiniCPM5-1B-Q4_K_M.gguf`)
- Director LoRA (Game Master): `BastienHot/buzzwords-director-lora` (`director.lora.gguf`)
- Style LoRAs (actors): `BastienHot/buzzwords-style-loras` (8 × `style-*.lora.gguf`)

## One-time setup

### 1. The Space
- Space type comes from `sdk: docker` + `app_port: 7860` in `README.md` frontmatter.
- **Hardware: `cpu-basic`** (free, 2 vCPU). No GPU/ZeroGPU.
- Threads are pinned to 2 (`BW_THREADS` in the Dockerfile) — inside a container
  `os.cpu_count()` can report the host's cores and oversubscription SLOWS llama.cpp.

### 2. Weights fetch
- `ensure_weights()` runs when `SPACE_ID` is set (always true on a Space) or when
  `BW_FETCH_WEIGHTS=1` (the Dockerfile sets it). Public repos, no token needed.
  Override with `BW_DIRECTOR_REPO` / `BW_LORA_REPO` if you fork them.

### 3. GitHub → Settings → Secrets and variables → Actions
- Secret **`HF_TOKEN`** = a write token; variable **`HF_USERNAME`**;
  variable **`HF_SPACE`** = `<owner>/<space-name>`.

### 4. Push
```bash
git push origin main      # the Action force-mirrors to the Space; the Space rebuilds
```

## Expectations / gotchas
- **First boot is slow:** the Docker build compiles llama-server (a few minutes,
  `-j2` + Release; pin `LLAMA_CPP_TAG` via build-arg for reproducible builds), then
  the app downloads ~0.9 GB of weights and waits for the server's `/health`.
- **CPU latency:** AVX2 gives ~60–80 tok/s prompt-eval and ~12 tok/s generation. The
  GM prompt is stable-prefix/append-only and each role keeps its own server slot with
  `cache_prompt`, so per-beat prefill is ~one new line. The hearing starts on beat 1;
  the rest generates in the background while the player reads — no up-front wall.
- **Thinking is off twice:** grammars force `{` as the first GM token, and every call
  sends `chat_template_kwargs: {enable_thinking: false}`; a startup self-test asserts
  the actor path returns clean text.
- **Weights never enter git** — `models/` is git-ignored; the Action ships code only.
- **Local dev:** install a llama.cpp release (or build with
  `-DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON`), set `BW_LLAMA_SERVER` if it's not on
  PATH, `pip install -r requirements.txt`, then `BW_FETCH_WEIGHTS=1 python app.py`.
