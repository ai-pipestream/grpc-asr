# grpc-asr design

## 1. Goals

Transcribe audio and video containers with the Whisper model family
(including distil variants) behind the fleet's collector contract. Stream
partial segments as they decode: a two-hour file must not buffer the whole
transcript until the end, because UIs subscribe to the event stream rather
than waiting for `TranscriptComplete`. Memory stays bounded: encoded bytes
in RAM, decode in chunks, drop PCM after the window is transcribed. The hot
path is diskless; model weights are the only files, mounted read-only.

## 2. Non-goals (v1)

Training, fine-tuning, or serving as a general STT platform. mlx-whisper,
which is macOS-only; Linux CUDA, OpenVINO, and CPU cover the fleet.
Subtitle burn-in or video transcode. Overlapping speech, music, or VAD
productization beyond what whisper.cpp already does.

## 3. Wire API

`ai.pipestream.asr.v1.AsrService`

```text
rpc Transcribe(stream TranscribeRequest) returns (stream TranscribeResponse);
rpc GetServiceInfo(GetServiceInfoRequest) returns (GetServiceInfoResponse);
```

Options on the first message: `model` (`tiny`, `small`, `medium`,
`large-v3`, `distil-large-v3`, and the rest of the Whisper family),
`language` (empty means detect), `task` (`transcribe` by default, or
`translate`), `word_timestamps`, `emit_keyframes` (video only),
`keyframe_interval_seconds`, `diarize` (speaker turns; a model without
speaker-turn training never predicts one), and `filename` (names the folded
Document; the server derives one from the container otherwise).

Events, in order:

1. `MediaInfo`: duration, codec, sample rate, video present
2. `PartialSegment` / `FinalSegment`: index, start_ms, end_ms, text,
   optional words, avg logprob
3. `Keyframe`: timestamp plus PNG bytes (only if `emit_keyframes`)
4. `TranscriptComplete`: language, duration, token counts

## 4. Mapping to Document (implemented in-repo)

The fold lives here (`src/document/document_fold.{h,cpp}`), following the
fleet precedent: each collector produces its own source-tagged Document and
the merge heuristic stays downstream. Opt in with
`TranscribeOptions.emit_document`; the server folds its own event stream
and emits the Document immediately before the `TranscriptComplete` trailer.

| ASR | Document |
|---|---|
| final segment | `TextItem` (label TEXT) under `#/body`, in wire order |
| segment timing | `prov[0].time = TimeSpan{start_ms, end_ms}`, plus the item's whole-text charspan. Time is a locator, so it belongs in provenance; media has no pages, so `page_no` stays 0 and no box is invented. `TrackSource{start_time, end_time}` in seconds stays alongside for consumers that already read it, and keeps its 1 ms epsilon where the schema requires `end > start` strictly |
| words | one further `ProvenanceItem` each, in reading order: the word's own `TimeSpan` and its charspan (Unicode code points) inside the item text. On by default, `GRPC_ASR_DOCUMENT_WORD_PROVENANCE=0` off. Per-word probabilities stay on the typed stream: the schema has no per-word confidence slot |
| speaker | `TimeSpan.speaker` on the segment and its words, with the distinct labels registered in `Document.media.speakers` in first-appearance order. Only when `diarize` was requested |
| segment confidence | `CollectorSource.confidence` (`exp(avg_logprob)`, a rescaled decoder score) plus `CollectorSource.raw_score` carrying the avg logprob verbatim, with `raw_score_kind = "avg_logprob"` naming it. A segment that scored no token claims none of the three and no sentinel is written for the absence; a score of 0 is a value, not a gap |
| keyframe | `PictureItem` plus `ImageRef{mimetype, size, uri: "keyframe:<timestamp_ms>"}`, a pointer at the typed event. PNG bytes are never embedded, so the Document stays one bounded message. Its `prov[0].time` is the true instant, a zero-length span |
| language | `body.meta.language` (`code` enum, `code_raw`, `created_by`) and `Document.source_meta.language` |
| MediaInfo / trailer | `Document.media{duration_ms, codec}` for the typed facts, `body.meta` `asr.*` custom fields for the rest |
| source identity | `Document.name` and `DocumentOrigin{filename, mimetype, binary_hash}` from `TranscribeOptions.filename` (basename; a name derived from the sniffed container when absent), the sniffed family, and the FNV-1a content hash of the upload |
| VTT | not produced here; the sink renders it from segment timestamps |

Every item carries `CollectorSource{collector: "asr", model: <whisper model
id>, version: <server build>}`. Partial segments are not folded (their
finals supersede them), and segment indexes and the trailer's counts stay on
the typed stream. The document schema is vendored verbatim from gRParse;
gRParse `COLLECTOR_ASR` wiring is a follow-up in that repo.

Video without an audio track is `INVALID_ARGUMENT`. Audio-only files never
emit keyframes.

## 5. Runtime

One whisper.cpp context per pooled worker is loaded at process start.
`whisper_full` runs over a rolling PCM window and emits a segment when the
decoder commits it. ffmpeg via subprocess is acceptable for demux if it
writes only to a pipe or memfd; no temp files on the container root. Media
above a configured byte cap or duration cap fails with
`RESOURCE_EXHAUSTED`.

## 5a. Implementation notes (v1, as built)

**Streaming input.** For the audio families the decoder pulls from the
growing upload buffer, so `MediaInfo` and the first segments are emitted
before the client half-closes. The e2e test holds back the last half second
of the fixture until a `FinalSegment` arrives.

**Windowing.** PCM is decoded into `GRPC_ASR_WINDOW_SECONDS` windows.
Segments wholly inside a window finalize immediately after it; the window's
last segment is emitted as a partial and re-decoded from its own start in
the next window (finals replace partials by index). PCM behind a final is
dropped, so resident memory never grows with media length.

**Video.** Demuxed via ffmpeg reading a sealed memfd through `/dev/fd`:
classic mp4 puts its moov index at the end, so the input must be complete
and seekable; nothing touches a filesystem. Keyframes stream from a second
ffmpeg child concurrently with transcription.

**Live feeds (follow-up).** Streamable containers (MPEG-TS, fragmented
MP4/CMAF, mkv/webm) can be demuxed from a pipe as bytes arrive. Routing
those through a pipe-fed ffmpeg while keeping the memfd path for classic
mp4 would extend transcribe-during-upload to video, and an edge relay
(`ffmpeg -i rtsp://… -f mpegts -` into a gRPC client) bridges RTSP/RTMP
sources without this server speaking those protocols.

## 6. Tests

A fixture wav with a known transcript (assert substring plus timestamp
monotonicity). Silence produces an empty transcript and success. A
truncated mp3 is `INVALID_ARGUMENT`. The video fixture asserts audio
segments are present and the keyframe count matches the interval when
enabled, zero when not. A missing backend makes the process exit at
startup, not on the first RPC.
