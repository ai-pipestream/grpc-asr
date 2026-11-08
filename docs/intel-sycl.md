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

## Reproduce the native build and RPC gate

Use a private checkout and a compiler/runtime installation that exposes the
Intel GPU through Level Zero. The krick-1 native run uses Ubuntu26.10,
oneAPI2025.3.3, `libze1=1.32.0-1`, and
`libze-intel-gpu1=26.27.39122.14-1`. Preserve those identities in each receipt.
Install normal CMake/Ninja, OpenSSL, curl, and FFmpeg development dependencies
in the build environment; the test fixture generator also needs ffmpeg.

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGRPC_ASR_SYCL=ON
cmake --build build-sycl --target grpc-asr-server asr-service-test config-test -j2
build-sycl/config-test
# Run these serially on a reserved device; retain stdout and stderr separately.
export GRPC_ASR_TEST_MODEL=/models/ggml-tiny.en.bin
export GRPC_ASR_TEST_SAMPLE="$PWD/build-sycl/_deps/whisper-src/samples/jfk.wav"
export GRPC_ASR_SERVER_BINARY="$PWD/build-sycl/grpc-asr-server"
export GRPC_ASR_TRACE_PLACEMENT=1 ONEAPI_DEVICE_SELECTOR=level_zero:gpu
GRPC_ASR_TEST_BACKEND=sycl build-sycl/asr-service-test >sycl-rpc.stdout 2>sycl-rpc.stderr
scripts/check-placement.sh sycl sycl-rpc.stderr
GRPC_ASR_TEST_BACKEND=cpu build-sycl/asr-service-test >cpu-rpc.stdout 2>cpu-rpc.stderr
scripts/check-placement.sh cpu cpu-rpc.stderr
```

This test includes actual loopback gRPC, repeated content/timestamps,
cancellation during active speech-free processing, queue cancellation,
deadlines, recovery, and video/OGG cases when ffmpeg is available. Missing
video dependencies are a test limitation, not a pass for those cases.

## SYCL container recipe

`Dockerfile.sycl` is separate from the CUDA image. It pins the Ubuntu and
oneAPI image digests, carries the SYCL/oneMKL runtime and Level Zero driver,
and builds the existing LGPL FFmpeg recipe with bounded `BUILD_JOBS`.
It is a recipe, not a published or device-qualified image claim. Package
repositories must still supply the pinned driver versions. Native receipts
do not qualify a container built with a different toolchain or driver.

```bash
docker build --platform linux/amd64 -f Dockerfile.sycl \
  --build-arg BUILD_JOBS=2 -t grpc-asr:sycl-local .
docker run --rm --read-only --tmpfs /tmp --device=/dev/dri \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  -v /work/models/asr:/models:ro -p 127.0.0.1:50055:50055 grpc-asr:sycl-local
```

The host needs a working kernel driver and firmware for the B70; a container
cannot supply the host kernel driver. The numeric render-node group grants
the nonroot image user access to `/dev/dri`. Select the render node for the
intended card on hosts with more than one GPU. `GRPC_ASR_BACKEND=cpu` remains
an explicit operator choice; a failed SYCL selection must fail startup.
Validate live RPC output and actual placement after starting the image.
