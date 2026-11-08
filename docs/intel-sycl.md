# Intel SYCL backend

Build with oneAPI's `icx`/`icpx` after sourcing `setvars.sh` and enable
`-DGRPC_ASR_SYCL=ON`. This selects whisper's native GGML SYCL implementation
for both encoder and decoder graphs; it does not use the OpenVINO encoder.
The existing explicit CPU and OpenVINO configurations retain their behavior.

At startup select `GRPC_ASR_BACKEND=sycl`. `GRPC_ASR_SYCL_DEVICE` is a
zero-based GPU ordinal within the SYCL registry, translated to whisper's
ordinal across GPU registries. An unavailable registry/device fails startup.
The pinned whisper patch also rejects GPU initialization failure instead of
continuing with the CPU backend. Operator-level CPU execution remains a
separate qualification concern: GPU selection alone does not establish full
offload.

Set `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` when qualifying the Intel device.
`GRPC_ASR_TRACE_PLACEMENT=1` records actual scheduler assignments after each
successful graph computation for conv, encoder, cross, and decoder. It
excludes view operations; host preprocessing and sampling are outside those
graph counts. It does not use per-tensor readback callbacks. An assignment
count is execution-placement evidence, not a broad model accuracy claim.

The loopback RPC test accepts `GRPC_ASR_TEST_BACKEND=sycl` (default remains
CPU). Supply the existing `GRPC_ASR_TEST_MODEL`, `GRPC_ASR_TEST_SAMPLE`, and
`GRPC_ASR_SERVER_BINARY` inputs. It asserts fixture text, timestamps,
streaming, cancellation and recovery and prints five warm RPC timings. Run
the same binary/model/fixture with `GRPC_ASR_TEST_BACKEND=cpu` for a comparable
baseline. Missing prerequisites and any skipped tier must remain visible.

Qualification receipts are added only after actual device tests finish.

## Historical cached-source receipt, not current-pin qualification

`tools/sycl_whisper_probe.cpp` passed seven real tiny.en/JFK transcriptions,
text and timestamp assertions, an explicitly aborted inference, and recovery
on krick-1 Arc Pro B70 (Level Zero 1.15.39122, oneAPI 2025.3.3). The same
binary/model/fixture passed on explicit CPU. Each uses two CPU threads.
The model SHA256 is
`921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f`.
The initial run used an older dependency cache, not the configured
`d1be6fde11ac6e0407606b4e42fe72d34add8037` pin. Its VAD-patched source SHA256
was `a9ccd99e46d7fad713c442c94567eb4480bdaa6ac239643035a8ce76b4e7204a`.
These results are historical only and cannot qualify the current service.
Artifact hashes and raw timings are retained under
`docs/receipts/historical-intel-native-*`. Current-pin rerun is required.

Both logs passed `scripts/check-placement.sh`. GPU execution counts were
conv 8, encoder 97, cross 24, decoder 144; CPU/other counts zero. Explicit
CPU had the same stage counts on CPU and zero GPU/other. These are scheduler
execution assignments, not model-loading claims.

Warm runs 1..5 had median 442.902 ms SYCL and 1178.702 ms CPU (2.66x).
SYCL warm range 284.185..654.477 ms; CPU 833.873..2575.725 ms. Cold run 0
was 10.948 s SYCL versus 1.030 s CPU and is excluded from warm latency.
Concurrent CI/build activity caused substantial host contention, so these
are fixture-specific observations, not clean-host throughput guarantees.
Run 6 follows cancellation and also passed. Full RPC acceptance remains
pending and is not established by this native receipt.
