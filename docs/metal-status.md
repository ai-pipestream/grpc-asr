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

Native Mac build and RPC tests are pending. Access was restored through
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

Remaining work includes native macOS build verification, per-stage executed
graph inspection; real transcript and timestamp assertions; cancellation/recovery;
and repeated warm requests against the same Mac's CPU baseline. No Metal
performance, full-stage offload or actual-device correctness is claimed.
