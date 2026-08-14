# grpc-asr

gRPC ASR collector (whisper.cpp) for audio and video, projecting transcripts
into the gRParse Document data plane.

A standalone C++ gRPC server embedding whisper.cpp. Media streams in over a
bidirectional `Transcribe` RPC; timed transcript segments stream back **as
the decoder commits them** — for audio containers, transcription starts
while the upload is still in flight. Video containers are demuxed through
ffmpeg over a memfd (diskless), with optional keyframe stills emitted
alongside the transcript.

## Wire API

`ai.pipestream.asr.v1.AsrService` (see
[`proto/ai/pipestream/asr/v1/asr_service.proto`](proto/ai/pipestream/asr/v1/asr_service.proto)):

- `Transcribe(stream TranscribeRequest) returns (stream TranscribeResponse)`
  — first message carries `TranscribeOptions` (model, language, task,
  word_timestamps, emit_keyframes, keyframe_interval_seconds,
  emit_document), then the encoded media as chunks. Events: `MediaInfo`,
  then `PartialSegment`/`FinalSegment` (finals replace partials by index),
  optional `Keyframe` PNGs, and a `TranscriptComplete` trailer.
- With `emit_document`, one `ai.pipestream.document.v1.Document` event is
  emitted immediately before the trailer: the collector-side fold of the
  transcript (final segments → text items with `TrackSource` time ranges +
  `CollectorSource{collector: "asr", model, confidence: exp(avg_logprob)}`;
  keyframes → picture items whose `ImageRef.uri` is
  `keyframe:<timestamp_ms>` — a pointer back at the typed event, never
  embedded PNG bytes, so the Document stays one bounded message). The
  typed event stream remains the lossless wire. The schema is vendored
  verbatim from gRParse (`proto/ai/pipestream/document/v1/document.proto`);
  do not edit it here.
- `GetServiceInfo` — build, backend, loaded models, caps.
- Standard `grpc.health.v1.Health` and server reflection are registered.

Containers: wav, mp3, flac, ogg (in-process decode); mp4/mov, mkv/webm
(ffmpeg demux; video needs an audio track). Errors: cap overruns are
`RESOURCE_EXHAUSTED`, undecodable media is `INVALID_ARGUMENT`, unknown
containers are `UNIMPLEMENTED`, decoder faults are `INTERNAL`.

## Build and test

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --target grpc-asr-server grpc-asr-tests
ctest --test-dir build -L asr --output-on-failure
```

gRPC and whisper.cpp are FetchContent-pinned; no system gRPC needed. Two
tests need provisioning and skip (exit 77) without it:

- `transcriber-test` / `asr-service-test` need `models/ggml-tiny.en.bin`
  (`curl -L -o models/ggml-tiny.en.bin
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin`).
- video cases need `ffmpeg`/`ffprobe` on PATH.

Backend variants: `-DGRPC_ASR_CUDA=ON` (GGML CUDA), `-DGRPC_ASR_OPENVINO=ON`
(whisper OpenVINO encoder; needs the OpenVINO SDK at build time — no image
ships for it yet).

## Run

```bash
GRPC_ASR_BACKEND=cpu GRPC_ASR_MODELS_DIR=./models ./build/grpc-asr-server
```

The process loads every configured model at startup and **fails loud**: a
missing backend or weight file stops the boot, never the first RPC. Weights
are files named `ggml-<model>.bin` in the models dir; `GRPC_ASR_MODELS`
picks a subset, unset discovers all.

| Env | Default | Meaning |
|---|---|---|
| `GRPC_ASR_LISTEN_ADDRESS` | `0.0.0.0:50055` | listen address |
| `GRPC_ASR_BACKEND` | `cuda` | `cuda` \| `openvino` \| `cpu`; never silently substituted |
| `GRPC_ASR_CUDA_DEVICE` | `0` | CUDA device index |
| `GRPC_ASR_MODELS_DIR` | `/models` | read-only weight mount |
| `GRPC_ASR_MODELS` | discover | comma list of model names to load |
| `GRPC_ASR_CONCURRENCY` | `2` | whisper states per model (concurrent transcriptions) |
| `GRPC_ASR_MAX_MEDIA_BYTES` | `268435456` | upload cap → `RESOURCE_EXHAUSTED` |
| `GRPC_ASR_MAX_DURATION_SECONDS` | `14400` | media duration cap → `RESOURCE_EXHAUSTED` |
| `GRPC_ASR_WINDOW_SECONDS` | `480` | PCM window per `whisper_full`; bounds resident PCM |
| `GRPC_ASR_THREADS` | `min(4,hw)` | whisper decode threads |
| `GRPC_ASR_KEYFRAME_INTERVAL_SECONDS` | `10` | default keyframe spacing |
| `GRPC_ASR_METRICS_INTERVAL_SECONDS` | `60` | stdout metrics line; `0` off |
| `GRPC_ASR_TOOL_INACTIVITY_SECONDS` | `120` | ffmpeg/ffprobe no-output kill timer |
| `GRPC_ASR_FFMPEG` / `GRPC_ASR_FFPROBE` | `ffmpeg`/`ffprobe` | tool paths |

## Docker

```bash
docker build -t grpc-asr .                      # CUDA (default)
docker build -f Dockerfile.cpu -t grpc-asr:cpu .
docker run --rm --read-only --tmpfs /tmp --gpus all \
  -v ./models:/models:ro -p 50055:50055 grpc-asr
```

Tests run inside the image build and gate it. The hot path is diskless:
media lives in memory and memfds; run the container `--read-only`.

## Remotes

- **Forgejo** (`git.rokkon.com/ai-pipestream/grpc-asr`) is the source of
  truth. `main` lives here.
- **GitHub** is a public push-mirror of `main`. Do not merge to GitHub
  `main`; its default branch is `development`.

Push Forgejo first. GitHub `main` updates from the Forgejo push-mirror.

## Docs

- [Architecture](docs/architecture.md) — where this sits in the collector fleet
- [Design](docs/design.md) — wire API, Document mapping, tests
- [Guidelines](docs/guidelines.md) — how to build it so it matches the fleet
