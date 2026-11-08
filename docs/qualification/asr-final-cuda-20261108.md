# Combined ASR CUDA qualification

Source 953cbaefac547fef176d00d454ab5346028920a8. All 107 first-party files match parent14718d3 except tests/transcriber_test.cpp, which matches shared e7954537. Runtime bounds7245, admission540, timeline d1 and shared pool/state ownership are included. Generated build commands and dependency archive hashes retained; 66 first-party units rebuilt with two jobs, followed by shared engine test/header consumers. CMake embedded build identity predates the final test/header-only commits; source archive and binary hashes identify the tested artifacts.

RTX4080SUPER, CUDA12.4, driver615.71.09, native Whisper d1be6fde11ac6e0407606b4e42fe72d34add8037, canonical observed source SHA5bb518beb0defd5098badee6c7c6e576f15e98a1ce7a8e233408f52ac51fccc6. Tiny.en model SHA921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f, ftype1. JFK SHA59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e. Release build, GCC15 and nvcc host GCC13, FFmpeg8.1.2. Host desktop/gRParse jobs remained active.

PASS full requested CUDA loopback TCP suite: five repeats, known speech/document, during-upload streaming, cancellation/deadline/queued cancellation and recovery, byte/error cases, video/Ogg, service info and fail-closed missing model startup. Missing-CUDA startup subcase skipped because CUDA exists.

PASS focused CUDA and CPU timeline/cap TCP suite: four unknown-duration 3500ms endpoint versus3000ms cap requests rejected RESOURCE_EXHAUSTED before events; origin10000ms with delays0/500ms matched reference text, segment/word times, media-relative duration, keyframes and trailer; extended video endpoint/keyframes and service info passed. These common timeline fixtures contain authored trailing padding; the separate engine tests below exercise unpadded audio.

PASS direct shared engine test on CUDA and CPU: eight unpadded window10/30 x delay0/500 x words off/on settings, plus existing engine regressions. Native demux fixtures including duration lower bounds PASS. CTest: 13 executed tests passed; transcriber and stream-backend skipped due configured empty model environment. Direct engine execution supplied real model explicitly; stream-backend remains skipped in this batch.

CUDA placement trace validated all records: fullRPC785, focusedRPC133, engine477; conv/encoder/cross/decoder all gpu_ops>0, cpu_ops=other_ops=0. Host decode/preprocessing/sampling excluded. Traces record successful graph scheduler assignments. Timing in these traced logs is diagnostic, not a benchmark. Prior9ce timings remain historical and are not relabeled.

CPU old/new bounds comparison and original failures are in the separate asr-pcm-bounds receipt. CUDA recognition omits the CPU delayed/window10 duplicate 'you'; no transcript editing occurs. Existing non-time metadata remains intact.

Raw logs, exact source archive, binary hashes, compiler/device identities, build commands and parent source hashes retained. No container rebuild or image qualification in this batch; Nano LightOn remains separate ongoing work.
