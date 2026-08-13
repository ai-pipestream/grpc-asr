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

## 4. Mapping to Document

| ASR | Document |
|---|---|
| full text | concatenated `TextItem`s in time order |
| segment | `TextItem` with a time range in provenance (no page bbox) |
| words | optional child spans / `TextOffset` |
| keyframe | `PictureItem` + `ImageRef`, page_no synthetic from time |
| VTT | not produced here; sink renders it from segment timestamps |

`CollectorSource.collector = "asr"`, `model` = the whisper.cpp model
id, `confidence` from avg logprob when present.

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

## 6. Tests

- Fixture wav with known transcript (assert substring + timestamp
  monotonicity).
- Silence → empty transcript, success.
- Truncated mp3 → `INVALID_ARGUMENT`.
- Video fixture: audio segments present; keyframes count matches the
  interval when enabled, zero when not.
- Backend missing → process exits at startup, not on the first RPC.
