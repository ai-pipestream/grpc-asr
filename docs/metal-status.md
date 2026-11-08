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

Native Mac build and RPC tests are pending. On 2026-11-08 the configured
Mac DNS name failed resolution and SSH to the Tailscale peer 100.102.192.43
timed out. Chip, RAM and native toolchain have not been verified.

Remaining work includes macOS portability for the server's pipe2 and the
collector video path's memfd_create/close_range; per-stage executed graph
inspection; real transcript and timestamp assertions; cancellation/recovery;
and repeated warm requests against the same Mac's CPU baseline. No Metal
performance, full-stage offload or actual-device correctness is claimed.
