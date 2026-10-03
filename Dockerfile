# syntax=docker/dockerfile:1.26
# grpc-asr — CUDA image (the default; CPU-only image: Dockerfile.cpu).
#
# The build stage compiles whisper.cpp with the GGML CUDA backend and runs
# the test suite; the tests gate the image. Model weights are never baked
# in — mount them read-only at /models. Tests that need weights skip
# cleanly (exit 77) when the build context lacks them, so CI contexts
# without models still build an image while local builds (which keep
# models/ in the context, see .dockerignore) assert the real transcription
# path. Tests linked against the CUDA backend need the driver library at
# load time, which a docker build never has; CMake disables those two at
# configure time when libcuda.so.1 is absent (GPU hosts still run them).

ARG GRPC_ASR_RUNTIME_IMAGE=nvidia/cuda:12.9.2-runtime-ubuntu24.04

# ubuntu24.04 base with g++-14: the project builds as C++23 (std::println
# needs libstdc++ 14+) and GCC 14 is nvcc 12.9's host-compiler ceiling, so
# this image cannot follow the CPU/OpenVINO images to GCC 15.
FROM nvidia/cuda:12.9.2-devel-ubuntu24.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates cmake curl gcc-14 g++-14 git make nasm ninja-build pkg-config xz-utils zlib1g-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# The ffmpeg and ffprobe the image ships: an LGPL-only build of a pinned,
# sha256-checked upstream release with just the demuxers and decoders the
# service runs (scripts/build-ffmpeg.sh). Ubuntu's ffmpeg package, which is
# built with --enable-gpl, is installed above for the build stage only: the
# tests author their fixtures with its encoders (libx264, libopus,
# libvorbis), never ship it, and run the code under test against
# /opt/ffmpeg, which comes first on PATH for ctest.
COPY scripts/build-ffmpeg.sh /tmp/build-ffmpeg.sh
RUN CC=gcc-14 /tmp/build-ffmpeg.sh /opt/ffmpeg

WORKDIR /src
COPY . .

# The cache id encodes every ABI-sensitive dependency; bump it when gRPC,
# whisper.cpp, CUDA, or the toolchain moves.
RUN --mount=type=cache,id=grpc-asr-ubuntu24-cuda12.9-gcc14-cxx23-grpc1.83.0-whisper1.9.3,target=/build \
    cmake -S . -B /build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
        -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
        -DGRPC_ASR_WERROR=ON -DGRPC_ASR_CUDA=ON \
    && cmake --build /build --target grpc-asr-server grpc-asr-tests --parallel \
    && PATH=/opt/ffmpeg/bin:$PATH GRPC_ASR_TEST_FIXTURE_FFMPEG=/usr/bin/ffmpeg \
        ctest --test-dir /build -L asr --output-on-failure \
    && mkdir -p /out && cp /build/grpc-asr-server /out/

FROM ${GRPC_ASR_RUNTIME_IMAGE}

# libgomp1: whisper.cpp's ggml CPU backend uses OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

COPY --from=build /out/grpc-asr-server /usr/local/bin/grpc-asr-server

# ffmpeg and ffprobe with their shared libraries, from the build stage:
# LGPL-2.1-or-later, found through an absolute rpath into /opt/ffmpeg/lib.
# /opt/ffmpeg/share/doc/ffmpeg carries the license texts, the configure
# line and the source tarball the build was made from; NOTICE says so, and
# scripts/smoke-test.sh checks the label, NOTICE and that the build is not
# a GPL one.
COPY --from=build /opt/ffmpeg /opt/ffmpeg
COPY NOTICE /usr/share/doc/grpc-asr/NOTICE
LABEL ai.pipestream.ffmpeg.license="LGPL-2.1-or-later" \
      ai.pipestream.ffmpeg.notice="ffmpeg and ffprobe are built from the upstream FFmpeg release without --enable-gpl or --enable-nonfree; source, configure line and license: /opt/ffmpeg/share/doc/ffmpeg. Details: /usr/share/doc/grpc-asr/NOTICE"

# The server links libcuda.so.1, which nvidia-container-toolkit injects on
# GPU hosts; a plain docker run has no driver library, so the loader stops
# before main. The CUDA stub from the build stage, parked OFF the default
# library path, lets scripts/smoke-test.sh boot-proof this image on a
# GPU-less CI runner via LD_LIBRARY_PATH=/opt/cuda-stubs. Nothing loads it
# otherwise: on GPU hosts the injected real driver wins because this
# directory is never searched.
COPY --from=build /usr/local/cuda/lib64/stubs/libcuda.so /opt/cuda-stubs/libcuda.so.1

ENV PATH=/opt/ffmpeg/bin:$PATH \
    GRPC_ASR_FFMPEG=/opt/ffmpeg/bin/ffmpeg \
    GRPC_ASR_FFPROBE=/opt/ffmpeg/bin/ffprobe \
    GRPC_ASR_LISTEN_ADDRESS=0.0.0.0:50055 \
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
