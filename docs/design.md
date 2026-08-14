# grpc-asr design

## 1. Goals

- Feature parity with Docling `InputFormat.AUDIO` and `VIDEO` plus the
  ASR pipeline (Whisper family, including Distil-Whisper).
- Stream partial segments as they decode. A two-hour file must not
  buffer the whole transcript until the end. UIs subscribe to the
  event stream; they do not wait for `TranscriptComplete`. That is
  the difference from Docling's batch ASR convert, not a socket trick.
- Bounded memory: encoded bytes in RAM, decode in chunks, drop PCM
  after the window is transcribed.
- Diskless hot path. Model weights are the only files, mounted
  read-only.

## 2. Non-goals (v1)

- Training, fine-tuning, or serving as a general STT platform.
- mlx-whisper (macOS-only). Linux CUDA/OpenVINO/CPU cover the fleet.
- Subtitle burn-in or video transcode.
- Overlapping speech / music / VAD productization beyond what
  whisper.cpp already does.

## 3. Wire API (sketch)

`ai.pipestream.asr.v1.AsrService`

```text
rpc Transcribe(stream TranscribeRequest) returns (stream TranscribeEvent);
rpc GetServiceInfo(GetServiceInfoRequest) returns (ServiceInfo);
```

Options on the first message:

- `model` — `tiny`, `small`, `medium`, `large-v3`, `distil-large-v3`, …
- `language` — empty means detect
- `task` — `transcribe` (default) or `translate`
- `word_timestamps` (bool)
- `emit_keyframes` (bool, video only)
- `keyframe_interval_seconds`

Events:

1. `MediaInfo` — duration, codec, sample rate, video present
2. `PartialSegment` / `FinalSegment` — index, start_ms, end_ms, text,
   optional words, avg logprob
3. `Keyframe` — timestamp + PNG bytes (only if `emit_keyframes`)
4. `TranscriptComplete` — language, duration, token counts

## 4. Mapping to Document (implemented in-repo)

The fold lives here (`src/document/document_fold.{h,cpp}`), the
DoclingMapper precedent: each collector produces its own source-tagged
Document and the merge heuristic stays downstream. Opt in with
`TranscribeOptions.emit_document`; the server folds its own event stream
and emits the Document immediately before the `TranscriptComplete`
trailer.

| ASR | Document |
|---|---|
| final segment | `TextItem` (label TEXT) under `#/body`, in wire order |
| segment timing | `TrackSource{start_time, end_time}` in seconds — media has no pages, so no `ProvenanceItem` is invented |
| words | typed stream only (no docling slot; `Word` events remain the source) |
| keyframe | `PictureItem` + `ImageRef{mimetype, size, uri: "keyframe:<timestamp_ms>"}` — a pointer at the typed event; PNG bytes are never embedded so the Document stays one bounded message |
| MediaInfo / language | `body.meta` (`asr.*` custom fields, `language.code_raw`) |
| VTT | not produced here; sink renders it from segment timestamps |

Every item carries `CollectorSource{collector: "asr", model: <whisper
model id>, version: <server build>, confidence: exp(avg_logprob) when the
segment reports one}` alongside the `TrackSource`. Partial segments are
not folded (their finals supersede them). The schema is vendored verbatim
from gRParse; gRParse `COLLECTOR_ASR` wiring is a follow-up in that repo.

Video without an audio track is `INVALID_ARGUMENT`. Audio-only files
never emit keyframes.

## 5. Runtime

- Load one whisper.cpp context per pooled worker at process start.
- `whisper_full` with a rolling PCM window; emit a segment when the
  decoder commits it.
- ffmpeg via subprocess is acceptable for demux **if** it writes only
  to a pipe/memfd. No temp files on the container root.
- `RESOURCE_EXHAUSTED` above a configured media byte cap and above a
  configured duration cap.

## 5a. Implementation notes (v1, as built)

- **Streaming input**: for the audio families the decoder pulls from the
  growing upload buffer, so `MediaInfo` and the first segments are
  emitted **before the client half-closes**. The e2e test holds back the
  last half second of the fixture until a `FinalSegment` arrives.
- **Windowing**: PCM is decoded into `GRPC_ASR_WINDOW_SECONDS` windows.
  Segments wholly inside a window finalize immediately after it; the
  window's last segment is emitted as a partial and re-decoded from its
  own start in the next window (finals replace partials by index). PCM
  behind a final is dropped, so resident memory never grows with media
  length.
- **Video** is demuxed via ffmpeg reading a sealed memfd through
  `/dev/fd` — classic mp4 puts its moov index at the end, so the input
  must be complete and seekable; nothing touches a filesystem. Keyframes
  stream from a second ffmpeg child concurrently with transcription.
- **Live feeds (follow-up)**: streamable containers (MPEG-TS, fragmented
  MP4/CMAF, mkv/webm) can be demuxed from a pipe as bytes arrive; routing
  those through a pipe-fed ffmpeg while keeping the memfd path for
  classic mp4 would extend transcribe-during-upload to video, and an edge
  relay (`ffmpeg -i rtsp://… -f mpegts -` into a gRPC client) bridges
  RTSP/RTMP sources without this server speaking those protocols.

## 6. Tests

- Fixture wav with known transcript (assert substring + timestamp
  monotonicity).
- Silence → empty transcript, success.
- Truncated mp3 → `INVALID_ARGUMENT`.
- Video fixture: audio segments present; keyframes count matches the
  interval when enabled, zero when not.
- Backend missing → process exits at startup, not on the first RPC.
