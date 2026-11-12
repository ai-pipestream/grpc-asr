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
Subtitle burn-in or video transcode. Separation of overlapping speech and
music. The media API does support Silero VAD for speech-aware windowing.

## 3. Wire API

`ai.pipestream.asr.v1.AsrService`

```text
rpc Transcribe(stream TranscribeRequest) returns (stream TranscribeResponse);
rpc GetServiceInfo(GetServiceInfoRequest) returns (GetServiceInfoResponse);
```

Options on the first message: `model` (`tiny`, `small`, `medium`,
`large-v3`, `distil-large-v3`, and the rest of the Whisper family),
`language` (empty means detect), `task` / `task_raw` (`transcribe` by default, or
`translate`), `word_timestamps`, `emit_keyframes` (video only),
`keyframe_interval_seconds`, `diarize` (speaker turns; a model without
speaker-turn training never predicts one), `filename` (names the folded
Document and provides extension fallback when magic sniffing is inconclusive),
`window_seconds` (clamped into `[min_window_seconds, window_seconds]`),
`decoding` (initial prompt, greedy/beam sampling, hallucination thresholds,
non-speech token suppression), `channel_mode` (`MIXDOWN` or `PER_CHANNEL`),
and `remote` (allowlisted outbound OpenAI-compatible transcription).

Events, in order:

1. `MediaInfo`: duration, codec, sample rate, channels, video present,
   `ContainerFamily`
2. `PartialSegment` / `FinalSegment`: index, start_ms, end_ms, text,
   optional words, avg logprob, `no_speech_prob`, `suppressed`, `channel`,
   `speaker_id`, `language`
3. `Keyframe`: timestamp plus PNG bytes (only if `emit_keyframes`)
4. `Document`: folded `ai.pipestream.document.v1.Document` (only if
   `emit_document`)
5. `TranscriptComplete`: primary `language`, distinct `languages`, duration,
   segment/token/keyframe counts, effective `window_seconds`,
   `segment_count_by_channel`

### 3a. Consolidated media and worker APIs

`ai.pipestream.media.v1.MediaService` and `SpeechWorkerService` are implemented
in the same server as AsrService. Their contract is
`proto/ai/pipestream/media/v1/media.proto`; their implementation lives under
`streaming/` and is registered by `src/service/streaming_runtime.cpp`.

MediaService accepts ordered direct input through `Ingest`, pulls allowlisted
HTTP inputs through `Pull`, and exposes UUID-filtered `Subscribe` with bounded
in-memory hydration. Open, chunk and finish messages make lifecycle accounting
explicit. Source bytes and checksums remain available alongside decoded
packets, audio and optional speech results. Empty subscription filters follow
all streams, including future streams. Restarting the process loses retained
events; the server epoch identifies that boundary.

Native FFmpeg handles streamable containers incrementally without request
files or subprocesses. SpeechWorkerService supports pooled Whisper and native
CPU Parakeet TDT transcription. Whisper supports English translation; Parakeet
does not. Fixed windows or Silero speech-aware windows feed the worker;
uninterrupted speech still reaches a configured maximum window, where the
caller chooses failure or a forced cut with overlap. Raw hypotheses and
reconciled revisions remain separate results.

The default worker connection is an in-process gRPC channel sharing the loaded
Whisper pool. `GRPC_ASR_WORKER_TARGET` selects an external worker. This media
path does not change AsrService's seekable-video compatibility behavior below.

### 3b. Remote transcription engine (`TranscribeOptions.remote`, docling `ApiAsrOptions`)

`TranscribeOptions.remote` (field 10, `RemoteTranscriptionOptions`) is the
typed form of docling's `ApiAsrOptions`: `url`, `headers`
(`map<string, string>`), `model`, `language`, `initial_prompt`,
`word_timestamps`, `params` (`map<string, ScalarValue>`, so an integer stays
an integer) and `timeout_seconds` (unset is docling's 600). The multipart
form fields are built by `engine::remote_form_fields` and match docling-slim
2.140.0's `_ApiAsrModel._form_fields` field for field (`response_format`
pinned to `verbose_json`, empty strings omitted, bools as `true`/`false`).
The one difference: protobuf maps have no order, so `params` go out sorted
by key.

Validation and endpoint authorization run before any media byte is decoded:

- **Option validation (`INVALID_ARGUMENT`):** `url` must be `http` or `https`
  with a non-empty host, an optional port in `1..65535`, no userinfo and no
  whitespace; header names must be HTTP tokens, unique ignoring case, and
  values free of CR/LF; every `params` key must be non-empty with its value
  kind set; and `timeout_seconds` must be in `1..3600`. A `double` param prints
  as Python's `str()` does (shortest round-trip digits, fixed for decimal
  exponents in `[-4, 16)`, otherwise `1e+16` style).
- **Allowlist & SSRF gate (`GRPC_ASR_REMOTE_ALLOWED_HOSTS`):** Outbound remote
  transcription is off by default (`GetServiceInfoResponse.remote_enabled =
  false`). When `GRPC_ASR_REMOTE_ALLOWED_HOSTS` is unset or empty, any request
  carrying `remote` fails `FAILED_PRECONDITION`. When set (comma-separated
  `host`, `host:port`, `*.domain`, or `*`), an unlisted host:port fails
  `PERMISSION_DENIED`. Before opening a socket, DNS resolution pins the vetted
  addresses via `CURLOPT_RESOLVE` (with redirects disabled) and rejects
  loopback (`127.0.0.0/8`, `::1`), RFC 1918 private, link-local, CGNAT, and
  unspecified addresses with `PERMISSION_DENIED` unless the matching allowlist
  entry explicitly names `localhost` or that exact IP literal.
- **Streaming & status mapping:** `MediaInfo` is emitted first from the local
  container sniff, then `src/engine/remote_engine.cpp` streams the `file`
  multipart body directly out of `ByteStream` via `libcurl`'s read callback
  without duplicating the upload in memory, polling caller `stop_token` in
  `CURLOPT_XFERINFOFUNCTION` so client cancellation aborts an in-flight upload
  within one poll interval (`CANCELLED`). Request headers (such as
  `Authorization`) are never logged or echoed in error messages. Upstream
  responses above 16 MiB fail `RESOURCE_EXHAUSTED`, timeouts fail
  `DEADLINE_EXCEEDED`, HTTP 4xx payloads naming an invalid request field fail
  `INVALID_ARGUMENT`, and other non-2xx or transport errors fail `UNAVAILABLE`.
  `verbose_json` segments and words map onto `FinalSegment` and
  `TranscriptComplete`, feeding `AsrDocumentFold` identically to local Whisper.

## 4. Mapping to Document (implemented in-repo)

The fold lives here (`src/document/document_fold.{h,cpp}`), following the
fleet precedent: each collector produces its own source-tagged Document and
the merge heuristic stays downstream. Opt in with
`TranscribeOptions.emit_document`; the server folds its own event stream
and emits the Document immediately before the `TranscriptComplete` trailer.

| ASR | Document |
|---|---|
| final segment | `TextItem` (label TEXT) in `Document.texts` (segments with `suppressed = true` are skipped); `Document.body.children` orders all text and picture refs chronologically by `(prov[0].time.start_ms, picture_before_text, channel)` and `TranscriptComplete` canonicalizes `Document.texts`, `Document.pictures`, and `Document.media.speakers` to that order so concurrent per-channel workers produce byte-identical Documents |
| segment timing | `prov[0].time = TimeSpan{start_ms, end_ms}`, plus the item's whole-text charspan. Time is a locator, so it belongs in provenance; media has no pages, so `page_no` stays 0 and no box is invented. `TrackSource{start_time, end_time}` in seconds stays alongside for consumers that already read it, and keeps its 1 ms epsilon where the schema requires `end > start` strictly |
| words | one further `ProvenanceItem` each, in reading order: the word's own `TimeSpan` and its charspan (Unicode code points) inside the item text. On by default, `GRPC_ASR_DOCUMENT_WORD_PROVENANCE=0` off. Per-word probabilities stay on the typed stream: the schema has no per-word confidence slot |
| speaker | `TimeSpan.speaker` on the segment and its words (`S1..Sn`, `C1..Cn` under `CHANNEL_MODE_PER_CHANNEL`, or `C1-S1..`), with distinct labels for speakers who actually produced spoken text registered in `Document.media.speakers` in first-appearance order |
| segment confidence | `CollectorSource.confidence` (`exp(avg_logprob)`, a rescaled decoder score) plus `CollectorSource.raw_score` carrying the avg logprob verbatim, with `raw_score_kind = "avg_logprob"` naming it. A segment that scored no token claims none of the three and no sentinel is written for the absence; a score of 0 is a value, not a gap |
| keyframe | `PictureItem` plus `ImageRef{mimetype, size, uri: "keyframe:<timestamp_ms>"}`, a pointer at the typed event. PNG bytes are never embedded, so the Document stays one bounded message. Its `prov[0].time` is the true instant, a zero-length span |
| language | `TextItemBase.meta.language` per segment; `body.meta.language` (`code` enum, `code_raw`, `created_by`) and `Document.source_meta.language` when the transcript has at most one language (left unset when `TranscriptComplete.languages` has two or more) |
| MediaInfo / trailer | `Document.media{duration_ms, codec}` for the typed facts, `body.meta` `asr.*` custom fields for the rest |
| source identity | `Document.name` and `DocumentOrigin{filename, mimetype, binary_hash}` from `TranscribeOptions.filename` (basename; a name derived from the sniffed container when absent), the sniffed family, and the incremental FNV-1a content hash of the upload |
| VTT | not produced here; the sink renders it from segment timestamps |

Every item carries `CollectorSource{collector: "asr", model: <whisper model
id>, version: <server build>}`. Partial segments are not folded (their
finals supersede them), and segment indexes and the trailer's counts stay on
the typed stream. The document schema is vendored verbatim from gRParse;
gRParse dials the collector through `GRPARSE_ASR_TARGET`.

Video without an audio track is `INVALID_ARGUMENT`. Audio-only files never
emit keyframes.

## 5. Runtime

One whisper.cpp context per pooled worker is loaded at process start.
`whisper_full` runs over a rolling PCM window and emits a segment when the
decoder commits it. ffmpeg via subprocess is acceptable for demux if it
writes only to a pipe or memfd; no temp files on the container root. Media
above a configured byte cap or duration cap fails with
`RESOURCE_EXHAUSTED`.

## 5a. Collector AsrService implementation notes

These notes describe AsrService, not the consolidated MediaService above.

**Streaming input and pipe demuxing.** WAV, MP3 and FLAC decode in-process from
the growing `ByteStream`. Streamable containers on the ffmpeg path (Ogg Vorbis
and Opus, Matroska/WebM, fast-start MP4/MOV whose `moov` box precedes `mdat`,
AMR-NB/WB, MPEG-PS, MPEG-TS, and raw ADTS AAC) stream live into
`ffmpeg -i pipe:0` via `VideoDemux::open_audio_from_pipe` while the upload is
still in flight, skipping `ffprobe`, extracting `MediaInfo.audio_codec` from
the container's leading header boxes (Ogg BOS, AMR magic, Matroska/WebM EBML
`Tracks`, and ISO BMFF `moov/.../stsd`), and allocating zero memfds so
`MediaInfo` and committed segments arrive before the client half-closes.

**Seekable layouts and single-copy memfd.** Containers that require random
access (trailing-`moov` MP4, AVI, ASF/WMA/WMV), extension-fallback inputs,
`CHANNEL_MODE_PER_CHANNEL` on ffmpeg containers, and `emit_keyframes` switch
`ByteStream` into memfd-backed mode (`ByteStream::switch_to_memfd`). Incoming
gRPC chunks write directly into that anonymous memfd and seal on upload
completion, keeping resident media memory strictly at one copy (`1x` upload
size, never `2x`) while `ffprobe` and concurrent audio/keyframe `ffmpeg`
children read `/dev/fd/3`.

**Stall-aware windowing.** PCM is decoded into rolling windows bounded by
`GRPC_ASR_WINDOW_SECONDS` (or per-request `TranscribeOptions.window_seconds`,
clamped into `[GRPC_ASR_MIN_WINDOW_SECONDS, GRPC_ASR_WINDOW_SECONDS]`). Once at
least `GRPC_ASR_MIN_WINDOW_SECONDS` (default 30 s) of fresh PCM is buffered, if
the upstream source stalls for `GRPC_ASR_WINDOW_STALL_MS` (default 500 ms)
before EOF, the window cuts at the buffered sample count and runs `whisper_full`
immediately instead of waiting for 480 s of audio. Segments wholly inside a
window finalize immediately after it; the window's last segment is emitted as a
partial and re-decoded from its own start in the next window (finals replace
partials by index). PCM behind a final is dropped, so resident memory never
grows with media length.

**Live feeds.** Both `MediaService` (native libavformat) and `AsrService`
(in-process miniaudio and live `pipe:0` ffmpeg demux for Ogg, WebM/MKV,
fast-start MP4, MPEG-TS, AMR, and ADTS AAC) transcribe streamable feeds while
bytes are still arriving. An external relay can turn RTSP/RTMP into a
streamable container supplied through gRPC; the server itself does not
implement those transport protocols.

## 6. Tests

A fixture wav with a known transcript (assert substring plus timestamp
monotonicity). Silence produces an empty transcript and success. A
truncated mp3 is `INVALID_ARGUMENT`. The video fixture asserts audio
segments are present and the keyframe count matches the interval when
enabled, zero when not. A missing backend makes the process exit at
startup, not on the first RPC.
