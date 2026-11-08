# CUDA qualification, 2026-11-08

Tested source: `9ce6f52bd112f6795312d6f39671af7b51eb6abc`, including
the shared exact-device selection, canonical Whisper observer and `d393547`
state ownership fix. No merge or image publication is implied.

Host: krick, NVIDIA RTX 4080 SUPER 16 GiB, driver 615.71.09, CUDA 12.4.
Release build uses GCC 15 for C++ and GCC 13 for nvcc. Both CPU and CUDA
tests use two threads and the same binary, model and eleven-second JFK WAV.

## Inputs

- Whisper: `d1be6fde11ac6e0407606b4e42fe72d34add8037`, reporting 1.9.5-dev.
  Canonical generated source SHA256:
  `5bb518beb0defd5098badee6c7c6e576f15e98a1ce7a8e233408f52ac51fccc6`.
- gRPC: `c876f4da50f7da2f331888b88b2a7243514139fe` (v1.83.0), with
  the repository OpenSSL compatibility patch; host OpenSSL 4 and FFmpeg 8.
- `ggml-tiny.en.bin` SHA256:
  `921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f`.
- `samples/jfk.wav` SHA256:
  `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`.

## Actual results

Config and state ownership unit tests passed. The complete real TCP RPC
suite passed on explicit CPU and CUDA, including streaming before upload
completion, transcript/timestamps, typed document provenance, silence,
cancellation, deadlines, queued cancellation, recovery, video, Ogg and
missing-model startup failure. The missing-CUDA-build subcase is skipped
because this binary includes CUDA.

The separate traced CUDA suite passed. Scheduler placement reported
8 convolution, 97 encoder, 24 cross-attention and 144 decoder GPU operations
per respective graph, with zero CPU or unknown-device operations. Host media
processing and sampling are not covered by those graph counts.

With tracing **disabled**, five warm RPCs took:

| Backend | Warm milliseconds | Median |
|---|---|---|
| CPU | 687.850, 687.246, 686.266, 680.731, 677.025 | 686.266 |
| CUDA | 46.261, 45.224, 48.812, 50.089, 44.886 | 46.261 |

The median ratio is 14.83x on this narrow fixture. Runs were sequential on
the same shared host, not an exclusive benchmark machine. Tracing adds
substantial overhead; its approximately three-second RPC timings are not
production performance evidence.

Two additional raw-wire requests per backend asserted all three JFK phrases
(including both “what your country can do for you” and “what you can do for
your country”), bounded segment and word timestamps, a document and the
completion trailer. Their window is 30 seconds; the suite above uses ten.

## Receipts and limits

The local handoff bundle is
`/work/main/grpc-services/hardware-receipts/cuda-20261108/asr/`.
It contains raw output, traced and untraced logs, a source archive,
configuration, input hashes and a binary manifest. Frozen tested binaries
are in `/tmp/asr-cuda-qualified-9ce6f52/`.

An incremental reconfigure encountered a damaged Ninja log and began a
redundant dependency rebuild. That rebuild was stopped; the already linked
ownership-fixed binaries were preserved and used for these runs. Source,
object and link timestamps were checked. This is native host qualification;
the CUDA Docker recipe carries the source and bounded build settings, but
no newly built ASR container or Orin ASR runtime is claimed here.
