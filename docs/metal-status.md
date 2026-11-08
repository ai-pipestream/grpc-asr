# Metal implementation checkpoint

`GRPC_ASR_METAL=ON` requires native Apple Silicon at configure time.
`GRPC_ASR_BACKEND=metal` selects the Metal registration explicitly;
`GRPC_ASR_METAL_DEVICE` selects its zero-based device index. The selection
is translated into Whisper's combined GPU ordinal. CPU remains explicit.
Each initialized Whisper state must contain that exact device or startup
fails. This is a state initialization check, not per-stage execution proof.

Local Linux checks: config-test passed; model_pool.cpp passed a C++23
syntax check against the pinned Whisper headers; configuring Metal on Linux
failed as required. These do not qualify Metal.

Native Mac RPC tests are pending. The complete server and tests built at
`17b85d25071c6d395ae9cceb8fb062d6c5d42270` with Apple Clang 21, system
gRPC 1.85.1 and Protobuf 36.2. `file` identifies the server as Mach-O arm64.
Eight selected non-model tests passed in 3.05 seconds: sniff, PCM scrub,
byte stream, configuration, audio decoder, document fold, window cut and
native video demux. This is build and media evidence, not Metal inference
qualification. Access was restored through
Kristians-MacBook-Air.local on 2026-11-08: Apple M2, 24 GiB, arm64,
sysctl.proc_translated=0, macOS 27.2 build 26B123. The Tailscale address
continued to time out.

The server's startup pipe now has a Darwin implementation. macOS defaults
to GRPC_ASR_NATIVE_DEMUX, which uses native FFmpeg with seekable in-memory
AVIO rather than Linux memfd or children. It rejects nested file/network
opens and supports PCM resampling and PNG keyframes. The regression passed
on Linux against FFmpeg 62: three resampled WAV runs, known sample count,
cancel/recovery, invalid input, video/audio probing, PNG dimensions and
keyframe sink cancellation. No disk file is used even for fixture generation.

Track selection preserves the first audio/video track, including when a later
track carries the default disposition. A dual-audio regression distinguishes
220 Hz at 16 kHz from the default second 880 Hz/24 kHz track. The native
regression also remuxes a trailing-moov MP4 in RAM and rejects missing and
truncated media. A separate two-second AAC/MPEG4 MP4 comparison against the
unchanged Linux subprocess path produced byte-identical PCM (SHA256
21cef8ef1d271f13c01a108c71aa4f799cd75f53808aa5b5e8c60b816cea378f).

The configured tool inactivity timeout becomes a per-native-operation timeout:
open/probe and each frame decode/conversion check it before and after work,
and FFmpeg's interrupt/AVIO callbacks check it during demux. Cancellation
uses the same cooperative boundaries. A native codec call already executing
cannot be preempted, so this is not a hard wall-clock deadline. Video dimensions
are rejected before probing/open where known; max_pixels is passed to probe
decoders and set before codec open, with a 16-million-pixel frame check too.
The default Linux subprocess path and the separate media HTTP allowlist are
unchanged. macOS uses no request files, child tools, or nested protocol opens.

Remaining work includes per-stage executed
graph inspection; real transcript and timestamp assertions; cancellation/recovery;
and repeated warm requests against the same Mac's CPU baseline. No Metal
performance, full-stage offload or actual-device correctness is claimed.

Run `scripts/build-macos-metal.sh` on the native macOS host, then
`scripts/run-macos-metal.sh /absolute/read-only/models tiny.en`. The latter
requires a non-writable `ggml-tiny.en.bin`, binds gRPC to `127.0.0.1:50055`,
and uses one model state with two compute threads. It runs in the foreground;
no launch daemon or machine configuration change is required. Linux arm64
Docker builds cannot exercise Apple's Metal runtime.

For actual loopback RPC qualification with the pinned JFK fixture, run the
following only in the assigned hardware window, once with `metal` and then
with `cpu`, retaining each complete stdout/stderr log:

```sh
GRPC_ASR_TEST_MODEL=/absolute/read-only/models/ggml-tiny.en.bin \
GRPC_ASR_TEST_SAMPLE=/absolute/fixtures/jfk.wav \
GRPC_ASR_SERVER_BINARY="$PWD/build-metal/grpc-asr-server" \
GRPC_ASR_TEST_BACKEND=metal GRPC_ASR_TRACE_PLACEMENT=1 \
  nice -n 10 ./build-metal/asr-service-test
```

This test asserts transcript content and timestamps, five warm requests,
streaming, cancellation and recovery. Scheduler stage traces must also be
reviewed for CPU or other execution; selecting Metal alone is not proof of
complete offload.
