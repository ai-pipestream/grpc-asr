# syntax=docker/dockerfile:1.26
# grpc-asr — CUDA image (the default; CPU-only image: Dockerfile.cpu).
#
# The build stage compiles whisper.cpp with the GGML CUDA backend and runs
# the test suite; the tests gate the image. Model weights are never baked
# in — mount them read-only at /models. Tests that need weights or ffmpeg
# skip cleanly (exit 77) when the build context lacks them, so CI contexts
# without models still build an image while local builds (which keep
# models/ in the context, see .dockerignore) assert the real transcription
# path. Tests linked against the CUDA backend need the driver library at
# load time, which a docker build never has; CMake disables those two at
# configure time when libcuda.so.1 is absent (GPU hosts still run them).

ARG GRPC_ASR_RUNTIME_IMAGE=nvidia/cuda:12.9.2-runtime-ubuntu22.04

FROM nvidia/cuda:12.9.2-devel-ubuntu22.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates cmake g++ git make ninja-build pkg-config ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# The cache id encodes every ABI-sensitive dependency; bump it when gRPC,
# whisper.cpp, CUDA, or the toolchain moves.
RUN --mount=type=cache,id=grpc-asr-ubuntu22-cuda124-grpc1.83.0-whisper1.9.2,target=/build \
    cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
        -DGRPC_ASR_WERROR=ON -DGRPC_ASR_CUDA=ON \
    && cmake --build /build --target grpc-asr-server grpc-asr-tests --parallel \
    && ctest --test-dir /build -L asr --output-on-failure \
    && mkdir -p /out && cp /build/grpc-asr-server /out/

FROM ${GRPC_ASR_RUNTIME_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=build /out/grpc-asr-server /usr/local/bin/grpc-asr-server

ENV GRPC_ASR_LISTEN_ADDRESS=0.0.0.0:50055 \
    GRPC_ASR_BACKEND=cuda \
    GRPC_ASR_MODELS_DIR=/models \
    CUDA_CACHE_DISABLE=1

# Diskless contract: run with --read-only and a tmpfs /tmp; media lives in
# memory and in memfds, model weights are a read-only mount:
#   docker run --rm --read-only --tmpfs /tmp --gpus all \
#     -v ./models:/models:ro -p 50055:50055 grpc-asr
USER 65532:65532
EXPOSE 50055
ENTRYPOINT ["/usr/local/bin/grpc-asr-server"]
