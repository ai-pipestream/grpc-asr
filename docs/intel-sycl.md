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
