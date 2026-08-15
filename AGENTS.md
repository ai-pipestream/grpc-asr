# AGENTS.md: grpc-asr

You are implementing **grpc-asr** from scratch in this repo. There is no
application code yet. Specs are the source of truth.

## Read this first, in order

1. This file
2. `docs/architecture.md`: fleet boundary, language, what we refuse to own
3. `docs/design.md`: wire API sketch, Document mapping, tests
4. `docs/guidelines.md`: fleet rules (streaming, proto, git, tests)

Do not start coding until those four are in your context. If architecture
and an existing sibling disagree on *process* (diskless, health, buf),
follow the sibling. If they disagree on *product* (live stream, Document
plane), follow architecture.md.

## This service

gRPC ASR collector (whisper.cpp) for audio and video, projecting transcripts into the gRParse Document data plane

- **Language:** C++ (whisper.cpp). ffmpeg/memfd for video demux only.
- **Copy from:** /work/main/grpc-services/gRParse (CMake, health, metrics) and /work/main/grpc-services/grpc-libreoffice (C++ gRPC server shape)
- **Stack:** whisper.cpp with CUDA and OpenVINO builds as first-class. No openai-whisper Python. Fail loud if the backend or model file is missing.
- **Live stream:** MediaInfo, then PartialSegment/FinalSegment as the decoder commits, optional Keyframe, then TranscriptComplete.

## Definition of done (v1)

Transcribe bidi stream, model pool, duration/byte caps, fixture wav test, video demux test, Dockerfile, health+reflection.

Also: README with build/run; proto lint clean; tests that fail if someone
turns the stream back into a batch (assert an event before the input is
fully consumed, or per-item events before Complete).

## Workspace

Checkout path: `/work/main/grpc-services/grpc-asr`.
Git: `origin` = Forgejo (push `main` here). `github` = GitHub mirror.
Never merge GitHub `main`. See `docs/guidelines.md`.

gRParse wiring (`COLLECTOR_*` enum, endpoint env) is a **follow-up**.
Ship a working server in this repo first.
