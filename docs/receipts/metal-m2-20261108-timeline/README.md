# Native M2 timeline RPC evidence

Full-suite runtime: `db91b87d4812c3ae94ee75652f14ccaec9fb8601`.
Focused test: `42474286f21c78dc8420a866c703a502ce3932cb`.
Imports: efef065 → 397e3aa, 17b2389 → f50a7ee, d1d5809 → db91b87.
Independent review of d1d5809 was active when these runs completed.

Apple M2, 24 GiB, native Apple Silicon, Rosetta disabled, macOS 27.2
(26B123), Apple Clang 21. The manifests provide toolchain, model, binary,
original/generated Whisper and 87 source hashes per run, checked with Git.
Whisper pin: d1be6fde11ac6e0407606b4e42fe72d34add8037; model: tiny.en.
Server SHA256: `2b4b1368cdf752d96eae7f375c0ed7733c4249103ea83396ed496fe73813cae7`.

## Results

Native incremental builds: nice 10, 2 jobs. Final video-demux-test passed
in 0.82s: 9 origin/delay cases plus packet-duration and existing media checks.
Initial build/unit logs cover f50a7ee; final-* logs cover db91b87.

Full Metal RPC suite passed in 14.81s: text/timestamps, repeated requests,
streaming, documents, cancellation/deadline/queue recovery, video, Opus/Vorbis,
limits and startup failures. No video skips. All 1718 graph records had GPU
operations and zero CPU/other operations: conv/encoder/cross 151 each,
decoder 1265. Host preprocessing/sampling are excluded from these counts.

The external receipt gate rejected missing focused-timeline markers in that
successful full suite. The new oracle was intentionally CPU-only and separate.
4247428 permits CPU-or-Metal in the focused mode, changing one test guard.
It does not permit CUDA/SYCL yet. No fixtures, thresholds or window sizes
changed. Only the test binary was rebuilt; production hashes match.

Focused Metal RPC passed in 2.03s (2.074s outer timer): 5 requests on one pool,
2 threads, 30-second windows. Shifted Matroska and reference WAV have equal
text and segment/word times; audio durations 13000/13500ms; keyframes
0,2000,...,12000ms. Extended-video case: MediaInfo 3500ms, PCM trailer 2000ms,
keyframes 500/1500/2500ms before Complete. All 133 graph records were GPU-only:
conv/encoder/cross 5 each, decoder 118.

The shared authored fixtures include 2 seconds of trailing PCM padding.
The separate unpadded predicted-timestamp issue is not fixed here. No CPU
benchmark: prior performance applies only to 17b85d25071c6d395ae9cceb8fb062d6c5d42270.
Existing tiny.en cooperative cancellation tests do not prove busy-graph
preemption. Both suites exited; process check found no owned ASR process,
64% memory free. Mac access was released. Raw logs/runners are retained;
.gz files use deterministic gzip headers. Final acceptance awaits parent review.
