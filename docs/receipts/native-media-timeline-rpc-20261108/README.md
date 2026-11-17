# Native media timeline TCP RPC — 2026-11-08

Follow-up to `efef065e2580c0e74655c95781649a87af956f04`.
CPU-only loopback TCP, real AsrService, pinned Whisper and tiny.en model.
No Mac/GPU execution. Parent probe-root change remains separate.

## Cases

5 requests share 1 model pool, concurrency 1, 2 inference threads:

- JFK WAV with 2 seconds of trailing PCM padding, then equivalent Matroska
  with container origin +10000ms.
- JFK WAV with 500ms leading and 2 seconds trailing PCM padding, then
  equivalent Matroska with origin +10000ms and audio starting at +10500ms.
- Video spanning relative 500–3500ms and audio spanning 0–2000ms.

The JFK pairs require identical final text, segment/word timestamps, and
PCM duration: 13000ms and 13500ms. Container duration matches these cases.
Every expected keyframe timestamp is checked: 0,2000,...,12000ms.

The extended-video case requires MediaInfo duration 3500ms, Complete duration
2000ms, and keyframes 500/1500/2500ms, all before Complete. The PCM trailer
is not an upper bound on video timestamps. No proto changes.

## Additional duration correction

Matroska metadata can expose an absolute endpoint as duration. The final
fixture with the earlier implementation reports 23000ms instead of 13000ms:
`red.log`, exit 1. The fix measures relative packet endpoints for nonzero
origins, in memory under an operation deadline with cancellation checks.
Unknown packet timing yields unknown duration (0), not a fabricated endpoint.
It does not blindly subtract origins from already-relative duration fields.

The extended native unit suite also checks positive/negative-origin packet
endpoints. ASan, UBSan and leak detection pass (`duration-unit.log`).

## Scope of the speech oracle

Unpadded JFK exposed existing predicted endpoint overrun in BOTH WAV and
Matroska: 12400ms for 11000ms PCM with a 10-second window; delayed audio also
ended at 12880ms for 11500ms PCM with a 30-second window. Diagnostics are
retained. No runtime timestamp adjustment was added for that separate issue.
The final authored fixture adds actual trailing PCM samples on both paths;
strict endpoint bounds and exact reference equality remain enabled.
The focused mode uses a 30-second window. Existing suite thresholds and
10-second streaming/window tests remain unchanged.

The cancellation comment now distinguishes watcher stop detection/request
from completion. Worker join waits for cooperative engine/decoder completion.
Busy-graph hard interruption remains unqualified.

## Build and execution

GCC 15.3.0, Linux x86_64; installed native FFmpeg libraries as recorded in the
previous receipt. Rebuilt all 10 service core sources and CPU-only Whisper.
Reused cached generated proto and gRPC libraries, with unchanged proto inputs.
The exact private build project is retained as `cpu-build.cmake.txt`; this was
not a clean rebuild of gRPC or the full product CMake tree.

gRPC source: `c876f4da50f7da2f331888b88b2a7243514139fe` (1.83.0), with existing
OpenSSL compatibility changes in 3 files; hashes retained. Whisper input:
`d1be6fde11ac6e0407606b4e42fe72d34add8037`, controlled generated SHA256
`5bb518beb0defd5098badee6c7c6e576f15e98a1ce7a8e233408f52ac51fccc6`.
CPU depthwise/Parakeet source patches follow the production CMake helpers.

Build: `nice -n 10 cmake --build build-rpc-cpu --target asr-service-test --parallel 2`.
Run with a 6GiB address-space limit, 240s process timeout, 90s RPC deadline,
`GRPC_ASR_TEST_REQUIRE_MODELS=1`, `GRPC_ASR_TEST_BACKEND=cpu`,
`GRPC_ASR_TEST_MEDIA_TIMELINE_ONLY=1`, `GRPC_ASR_TRACE_PLACEMENT=1`,
`OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`, and the model/sample paths
recorded in `identities.sha256`. The normal project registers this mode as
`native-media-timeline-rpc-test` when native demux is enabled.

Independent review of this follow-up remains with Gauss. Earlier acceptance
of efef065 does not cover the new packet-duration correction automatically.
