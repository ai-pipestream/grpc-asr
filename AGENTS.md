# AGENTS.md: grpc-asr

This is the canonical **grpc-asr** implementation. Preserve the existing
AsrService contract while consolidating the native media and speech-worker
interfaces in this repository. `streaming/` came from whisper-grpc commit
b440304b304a016297763138d227b6a2987621be; it is maintained here now.
Do not restore a separate media-server executable or duplicate Whisper model
loading in the production runtime. Streaming Whisper uses the existing pool.

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

- **Language:** C++ (whisper.cpp). ffmpeg/memfd for video containers and Ogg (Vorbis/Opus).
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
