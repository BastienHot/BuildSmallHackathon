# Docker Space (sdk: docker). Docker exists to control ONE thing: building llama.cpp's
# `llama-server` from source with AVX2 (prebuilt CPU binaries/wheels are un-vectorized;
# AVX2 is ~30-40x faster prompt eval on the free 2-vCPU tier). The app itself is plain
# Gradio talking HTTP to the managed llama-server subprocess (buzzwords/engine.py).

# ---------- stage 1: build llama-server (AVX2, mostly-static) ----------
FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# Pin a tag for reproducible builds (override at build time: --build-arg LLAMA_CPP_TAG=bXXXX).
ARG LLAMA_CPP_TAG=master
RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} \
        https://github.com/ggml-org/llama.cpp /opt/llama.cpp
# -DGGML_NATIVE=OFF + explicit AVX2/FMA/F16C: vectorized but portable across Space hosts.
# -DBUILD_SHARED_LIBS=OFF: a self-contained binary we can COPY into the runtime stage.
# -DLLAMA_CURL=OFF: the app downloads weights itself; skip the libcurl dependency.
# -j2: both build vCPUs, keeps the compile inside HF's build-time budget.
RUN cmake -S /opt/llama.cpp -B /opt/llama.cpp/build \
        -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
        -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    && cmake --build /opt/llama.cpp/build --target llama-server -j2

# ---------- stage 2: the app ----------
FROM python:3.12-slim
# libgomp for llama.cpp's OpenMP threading at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server

# HF Spaces run the container as uid 1000 — keep everything under that user's home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    BW_FETCH_WEIGHTS=1 \
    BW_THREADS=2
WORKDIR /home/user/app

# deps first (layer cache)
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user:user . .
EXPOSE 7860
CMD ["python", "app.py"]
