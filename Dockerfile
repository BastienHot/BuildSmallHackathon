# Docker Space (sdk: docker) that runs the SAME Gradio app — we use Docker only to control
# the build so llama-cpp-python is compiled with AVX2. The prebuilt CPU wheels are
# un-vectorized (~2 tok/s prefill on cpu-basic); AVX2 gives ~60-80 prefill / ~12 decode,
# which is what makes the all-1B game playable on the free 2-vCPU tier.
FROM python:3.12-slim

# build toolchain for the from-source llama.cpp compile; libgomp for OpenMP at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run the container as uid 1000 — keep everything under that user's home.
RUN useradd -m -u 1000 user
USER user
# CMAKE_ARGS: AVX2 vectorization (the whole point of the source build), plus
# -DLLAMA_BUILD_{TESTS,EXAMPLES,SERVER}=OFF so CMake skips the llama.cpp sub-targets
# that llama-cpp-python never ships -- they're dead weight that pad the compile.
# CMAKE_BUILD_PARALLEL_LEVEL: use BOTH build vCPUs (the source build is single-job by
# default). Together these keep the from-source AVX2 wheel inside HF's build-time budget;
# without them the compile overran and the Space failed with "Job timeout".
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF" \
    CMAKE_BUILD_PARALLEL_LEVEL=2 \
    BW_FETCH_WEIGHTS=1
WORKDIR /home/user/app

# deps first (layer cache). --no-binary forces the AVX2 SOURCE build of llama-cpp-python
# (so CMAKE_ARGS take effect instead of pip grabbing an un-vectorized wheel).
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user --no-binary llama-cpp-python -r requirements.txt

COPY --chown=user:user . .
EXPOSE 7860
CMD ["python", "app.py"]
