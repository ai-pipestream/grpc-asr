# Apple M2 native ASR receipt

Tested server/test source: `17b85d25071c6d395ae9cceb8fb062d6c5d42270`.
The explicit fixture helper is commit `7ca9a3e`; it changes fixture encoding,
not service decoding or test assertions. Native FFmpeg lacks libvorbis, so
the helper uses the actual Vorbis encoder with required stereo input. An
earlier full-suite attempt failed at fixture generation; it is not a pass.

Runtime: whisper.cpp `d1be6fde11ac6e0407606b4e42fe72d34add8037`, canonical
hash-guarded generated placement/state patch from `639d741`. Pristine
whisper.cpp SHA256:
`891168a838301bd2baa5e8b87422a854dafeeb7da914d40a2e4c6bf33ff448ed`.

Host: Apple M2, 24 GiB, native arm64, Rosetta disabled, macOS 27.2 (26B123),
Apple Clang 21; gRPC 1.85.1, Protobuf 36.2, FFmpeg 9.0.2_1. Execution device
MTL0 (Apple M2); exact initialized Whisper state device was checked by the
common model pool. Model and fixture are read-only inputs:

- tiny.en: `921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f`
- JFK WAV: `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`

Both complete suites exited zero with `GRPC_ASR_TEST_REQUIRE_MODELS=1`,
`GRPC_ASR_TRACE_PLACEMENT=1`, two model threads and a 300-second process
watchdog. No other model or compile job ran during their measurements.
The production-server TCP receipt used loopback 127.0.0.1:50055, emitted a
Document and completion trailer, and asserted both JFK phrases and positive
segment durations. Its owned server was terminated after the response.

`summary.json` records all five warm samples per backend and observed stage
counts. The gzip files retain complete unedited logs (SHA256 of the original
RPC-suite logs is in the summary). `response.jsonl` and `info.json` are raw
grpcurl outputs. This qualifies this model/fixture and the tested lifecycle
matrix; other model sizes, long-form quality and wider performance workloads
remain unmeasured. Stage traces are native scheduler evidence, not an Xcode
GPU profiler capture.
