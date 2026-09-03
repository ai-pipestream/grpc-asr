# grpc-asr design: video container input

**Status:** design implemented as recorded here (`src/media/sniff.*`,
`src/media/video_demux.*`); this document is the decision record for the
container leg and the contract for what may still change.
**Updated:** 2026-09-03

## 1. Motivation

Video transcription is audio transcription with a container problem. The
model consumes 16 kHz mono PCM and nothing else; everything an mp4, mkv,
webm, or mov adds over a wav is demuxing: find the audio track, decode it,
resample it, and keep the clock straight so segment timestamps still mean
what they meant. The fleet's collectors own their formats in one place, so
the demux lives here rather than pushing "extract the audio first" onto
every caller, and gRParse already routes audio and video to this service
on the strength of that.

## 2. Non-goals

- Video transcoding, subtitle burn-in, or any write-back into containers.
- Image-level understanding of the video track. Keyframes are emitted as
  PNGs for the CV path when asked; nothing here interprets them.
- Live protocols (RTSP/RTMP) server-side. Streamable containers from a
  pipe and an edge relay are a follow-up, noted in `design.md` §5a.
- Supporting every container ffmpeg knows. The accepted set is the
  sniffed families below; anything else is `INVALID_ARGUMENT`.

## 3. ffmpeg integration: subprocess, not libav*

Two ways to demux exist:

1. **Link libav* in-process.** One fewer process, no IPC, and seeking a
   memory buffer is direct. The costs land on the service's own address
   space: the libav* API surface is large and churns between releases,
   a decoder crash on hostile media kills the service, the library set
   pulls a long tail of codec dependencies into the link, and an
   in-process decoder shares the whisper worker pool's failure domain.
2. **Spawn ffmpeg/ffprobe children.** The ffmpeg CLI is the stable
   interface the project itself commits to, a crash kills a child and
   fails one RPC, and the parent never links codec code at all. The cost
   is process plumbing: input has to reach the child without a temp file,
   output has to stream back with backpressure, and a hung child needs a
   watchdog.

**Decision: subprocess.** The family's isolation rule decides it: media
from the wild is adversarial input, and "a poisoned file kills this RPC,
not the service" is not negotiable for a shared GPU box. The CLI contract
is also the more honest dependency: ffmpeg's developers keep it stable in
a way they have never kept libavformat's ABI.

The plumbing that makes the subprocess diskless (as built):

- The uploaded bytes are written to a sealed memfd, an anonymous RAM
  file, and every child reads it through `/dev/fd/<n>`. Nothing touches a
  filesystem. Classic mp4 keeps its moov index at the end of the file, so
  the input must be complete and seekable; the memfd is both.
- ffprobe answers stream facts (duration, codecs, whether audio and video
  exist) before any decode starts, and its facts fill `MediaInfo`.
- One ffmpeg child decodes the chosen audio track to mono f32 PCM at the
  model rate on stdout; the parent reads through a pipe, so backpressure
  is the pipe's and memory stays bounded no matter the media length.
- A second, concurrent child extracts keyframe PNGs on a sampling grid
  when `emit_keyframes` is set.
- An inactivity timeout (`GRPC_ASR_TOOL_INACTIVITY_SECONDS`) kills a
  child that stops producing; a missing or crashing ffmpeg is a
  `ToolError` surfaced as `INTERNAL`, while media ffmpeg itself rejects
  is a `DecodeError` surfaced as `INVALID_ARGUMENT`. The distinction
  matters to callers: one is their bytes' fault, one is ours.

## 4. Input contract

**Detection is by container magic, never the advisory content type** (the
family rule, implemented in `sniff.cpp`): 16 bytes suffice to tell the
families apart. ISO BMFF (`ftyp` box) covers mp4, mov, and m4a as one
family; EBML covers mkv and webm as another. The magic cannot tell
audio-only members of those families (m4a, audio-only webm) from video
ones, and the design does not pretend otherwise: the sniffer reports the
family, the demuxer's probe reports whether a video stream exists, and
`MediaInfo.video_present` is the only authoritative answer a client gets.
Families the sniffer does not name are `INVALID_ARGUMENT`, as is anything
that sniffs as one family and fails to parse as it.

**Video-only containers are rejected.** A container whose probe finds no
audio stream is `INVALID_ARGUMENT` with a message naming the container and
its streams: this is a transcription service, and silently returning an
empty transcript for a silent film would manufacture a result, not report
one.

**Multi-audio-track containers.** The default is the container's own
default audio track (ffmpeg's stream selection, which honors the
container's default flag and falls back to the first audio stream). An
explicit choice is an option on the transcribe request, by track index or
by language tag; asking for a track that does not exist is
`INVALID_ARGUMENT` naming the tracks that do. Never mix tracks: two
languages averaged into one PCM stream is a worse transcript than either
alone.

**Time offsets stay sample-accurate.** Segment timestamps are counted from
decoded PCM samples, not read from container PTS values. The decode starts
at container zero and runs continuously, so the n-th sample of the PCM
stream *is* the media clock, immune to edit lists, timestamp offsets, and
B-frame reordering. That is what keeps VTT cues correct: the sink renders
cues from segment `start_ms`/`end_ms`, and a cue is only right if the
transcript's clock and the player's clock agree from zero. A container
whose audio does not start at zero (a real, if rare, edit-list case) is
normalized by decoding from the first audio packet and treating its
leading silence the way ffmpeg reports it, so the PCM clock and the
presentation clock still coincide.

## 5. Container image impact

The runtime image needs ffmpeg and ffprobe binaries; nothing else about
the image changes. Two ways to get them:

1. **Debian/Ubuntu ffmpeg package in the runtime stage.** One `apt-get`
   line, security updates ride the distribution, and the shared-library
   build matches the base image's glibc. The cost is size and surface:
   the package pulls its codec library tail into the final image.
2. **Static ffmpeg built in the builder stage.** Copies in two
   self-contained binaries with no package tail, which is the only shape
   that survives a move to the hardened dhi.io bases, since those carry
   no ffmpeg package and no apt to install one with. The costs are a
   slower build, a manually maintained ffmpeg version pin, and security
   updates that arrive only when the pin moves.

**Current choice: the distribution package.** The C++ services still run
on plain Ubuntu/CUDA bases where the package exists, and distribution
security cadence beats a hand-pinned static build for a codec parser
exposed to hostile input. The static build is the documented seam for the
day the runtime bases go hardened: when that move happens, ffmpeg becomes
a builder-stage product and the version pin joins the image's manifest.
The binary paths are configuration (`GRPC_ASR_FFMPEG` /
`GRPC_ASR_FFPROBE`), so the swap is an image change, not a code change.

Size is accepted either way: the whisper weights already dominate the
deployment, and the demux binaries are tens of megabytes against
gigabytes of model.

## 6. Failure modes

- Corrupt or truncated container: ffprobe or ffmpeg rejects it,
  `INVALID_ARGUMENT`, before or during decode. A failure mid-decode
  still leaves every already-emitted final segment valid; the stream
  never retracts.
- No audio track: `INVALID_ARGUMENT`, per above.
- ffmpeg missing at runtime: the process fails loud at startup, not on
  the first RPC (the v1 definition of done already requires it).
- Child hangs: the inactivity watchdog kills it, the RPC fails
  `INTERNAL`, the next request gets a fresh child.
- Container above the byte or duration cap: `RESOURCE_EXHAUSTED`, same as
  audio.

## 7. Test plan

Synthetic containers are built by ffmpeg in CI (ffmpeg is a build-stage
dependency already, so the generator costs nothing new):

- mp4, mkv, webm, and mov containers around the same known-audio fixture
  must produce the same transcript substring and monotone timestamps, and
  their `MediaInfo` must report the family the sniffer asserts.
- A video-only container (video track, no audio) is `INVALID_ARGUMENT`.
- A two-audio-track container transcribes the default track by default
  and the named track when the option names it; a bogus track name lists
  the real ones in the error.
- Timestamp accuracy: a fixture with a known spoken moment at a known
  second asserts the covering segment's `start_ms`/`end_ms` bracket it,
  on each container family. This is the VTT-correctness test.
- Keyframes: the video fixture's keyframe count matches the requested
  interval, zero when disabled; audio-only uploads never emit keyframes.
- A truncated container fails `INVALID_ARGUMENT`; a killed ffmpeg (the
  watchdog path, triggered by a stubbed child) fails `INTERNAL` and the
  next transcribe succeeds.
- Stream liveness in the family shape: an event arrives before the input
  is fully consumed, and `TranscriptComplete` is a trailer.

## 8. Milestones

- **M1: demux and decode** (done as designed here). Sniffing, memfd
  input, ffprobe facts into `MediaInfo`, PCM pipe with backpressure,
  caps and watchdog, rejection rules.
- **M2: track choice.** The explicit audio-track option (index and
  language) on the request, with the error contract above.
- **M3: live containers.** Pipe-fed demux for streamable containers
  (MPEG-TS, fragmented MP4/CMAF, mkv/webm as they arrive), extending
  transcribe-during-upload to video; the edge relay note in `design.md`
  §5a stands.
