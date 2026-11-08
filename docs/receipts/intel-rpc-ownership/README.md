# Actual Intel B70 ASR RPC qualification — PASS

Runtime sources: a1a05e4 (includes the common d393547 state-ownership fix).
Transcript recorder: 3264eef. Host krick-1, Intel Arc Pro B70 Graphics,
SYCL0, xe driver, Level Zero 1.15.39122, oneAPI compiler 2025.3.3,
Ubuntu 26.10. Private FFmpeg 8.1.2 test executable was present; video and
OGG cases were executed. No installed services or other jobs were stopped.

Whisper pin: d1be6fde11ac6e0407606b4e42fe72d34add8037. Canonical generated
source SHA256: 5bb518beb0defd5098badee6c7c6e576f15e98a1ce7a8e233408f52ac51fccc6.
Model: ggml-tiny.en.bin, SHA256
921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f.
The exact binaries, model, generated source and JFK fixture hashes are in
`exactpin-rpc-ownership/identities.sha256`. All tracked runtime/build/test
source bytes in `source.sha256` were compared with the remote build tree.
The remote git index predates these synced sources; source hashes, rather
than that older HEAD, identify what was compiled and tested.

The common actual loopback `asr-service-test` exited zero for both explicit
SYCL and CPU, using the same binary/model and two CPU threads. It covers
warmup plus five measured requests, transcript/timestamps, output during
upload, document references, word timing, silence, disconnected clients,
active speech-free processing cancellation/deadline/recovery, queued
cancellation/timeouts, error cases, byte caps, service info, video, OGG,
video cancellation/recovery and invalid startup configurations. This is
AsrService qualification; it does not claim every MediaService path or
every model/language has been qualified.

Actual scheduler observation passed for all four stages: convolution,
encoder, cross-attention and decoder. SYCL execution had positive GPU
operation counts and zero CPU/other counts; the distinct CPU run had
positive CPU counts and zero GPU/other counts. Host preprocessing and
sampling are outside this graph-placement claim. These are actual native
Whisper executions, with no CPU fallback or substituted inference engine.

Warm RPC median: SYCL 399.103 ms, CPU 877.195 ms, CPU/SYCL 2.198x.
All five samples are in `timings.json`. Warmup is excluded. A separate
fresh live server's first complete JFK RPC took 5.654462 seconds; that
includes cold work and is not a standalone JIT measurement. Shared CI and
training load remained, and CPU/SYCL suites ran sequentially; these are
fixture measurements, not stable general throughput guarantees.

The saved live RPC asserted both JFK phrase anchors, ordered bounded
segment timestamps, Document, Complete segment count and English language:
“ask not what your country can do for you” and “ask what you can do for
your country.” Full text, word timings and assertions remain readable in
`exactpin-rpc-ownership/live-transcript/`. The separate live placement
check also passed all four stages. The temporary server was stopped and
the B70 model lock released to Tesla after this run.

The prior run1/run2 receipts are preserved. Both test processes printed
PASS, but their SSH wrappers failed because the FFmpeg child inherited
heredoc stdin and consumed command characters. Run1 has an empty exit
file; run2 records zero and its placement check was rerun separately.
The final ownership-fixed run closes test stdin and records a clean zero
exit, avoiding that wrapper defect. Earlier old-pin native proof remains
explicitly historical and is not the evidence for this qualification.

Large raw logs are losslessly gzip-compressed with mtime=0.
`compressed-logs.json` preserves original paths, sizes and SHA256 hashes
as well as compressed hashes. Every compressed log was decompressed and
compared byte-for-byte before replacing its text copy. Use `gzip -dc` to
inspect them. Summaries, source manifests and transcript assertions are
readable without decompression; no failed proof was discarded.

Build limitation: damaged private Ninja dependency metadata caused
redundant builds. Completed pinned gRPC/protobuf archives were reused,
then current CMake-generated compile/generate/archive/link commands were
executed explicitly with bounded workers. `asr-rpc-commands.sh` records the
initial commands; `asr-ownership-rebuild.sh` records the four incremental
commands for the final state-ownership fix. This is not a claim of a clean
ordinary Ninja build. The hardware-free ownership test and config test
also passed. Dockerfile.sycl and the bounded image build recipes passed
BuildKit configuration checks; no full container build, device-container
qualification or image publication is claimed.
