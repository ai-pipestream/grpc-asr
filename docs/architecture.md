# grpc-asr architecture

**Status:** spec (no implementation yet)
**Updated:** 2026-08-13

Implementers start at [`AGENTS.md`](../AGENTS.md), then this file, `design.md`, and `guidelines.md`.

## Where this sits

Docling's `AUDIO` / `VIDEO` formats run Whisper (or Distil-Whisper /
mlx-whisper) **inside** the convert process. That steals the GPU from
layout/OCR and ties ASR to Python. This service is the isolated
collector for the same feature.

```text
wav/mp3/flac  or  mp4/mkv/webm
        │
        ▼
   grpc-asr            whisper.cpp  (+ ffmpeg demux for video)
        │
        ▼
   gRParse coordinator (COLLECTOR_ASR)
        │  optional keyframe stills → CV path
        ▼
   Document (transcript + optional pictures)
```

Keep ASR **out of gRParse**. OCR/layout session pools and Whisper
weights do not share a device arena.

## Live results (vs Docling)

Docling's ASR pipeline transcribes the file and then builds a
document. We emit **segments as the decoder commits them** (and
keyframes as they are extracted) so a UI can show the transcript
growing, including on a two-hour file. Partial segments are allowed;
finals replace them by index. `TranscriptComplete` is a trailer.

## What this process owns

- Audio decode and transcription. Runtime is **whisper.cpp** (GGML /
  CUDA / Metal / Vulkan / OpenVINO builds), not `openai-whisper`.
- Video: demux the audio track; optionally emit keyframe stills as
  `PictureItem`s (or raw PNGs) so gRParse can run layout on slides.
- Timed segments (`start`, `end`, `text`, optional word timestamps)
  projected as `TextItem`s with time provenance, plus a WebVTT-shaped
  sidecar the sink can export as `OutputFormat.VTT`.
- Language detection and the same family of model names Docling
  exposes (tiny … large-v3, distil-*). Weights are files on disk,
  not a pip extra.

## What this process does not own

| Concern | Owner |
|---|---|
| Diarization, speaker names | later; not Docling v2.119 convert |
| Translation as a product feature | whisper.cpp `task=translate` may be an option, not the default |
| Chunking / embeddings | downstream of Document |
| Export to VTT files | protomolt sink, from the timed items |
| Page OCR | gRParse CV |

## Language

**C++**. whisper.cpp is the system of record; wrapping it in Python
would recreate Docling's cost. ffmpeg (or a pure demuxer) is a
deployment dependency for video, not a second ASR stack.

One process, pooled model instances sized by VRAM, bounded concurrent
transcriptions. Fail loud if the requested model file or backend is
missing.

## Hardware

Same doctrine as gRParse: NVIDIA CUDA and Intel OpenVINO are first
class, CPU is explicit. `GRPC_ASR_BACKEND=cuda|openvino|cpu`. Do not
silently fall back. A separate container from `grparse-server` so a
whisper-large load cannot evict RapidOCR sessions.
