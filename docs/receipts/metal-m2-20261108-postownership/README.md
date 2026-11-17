# Post-ownership-fix native Metal runtime proof

Tested source: `d393547ac92dd48ab5f96064cb66401d0d2ea920`, a clean tracked
checkout on the native Apple M2 Mac (24 GiB, arm64, macOS 27.2). Parent
integration: `f9b5eea052780f17866efc9e8ec44d36f9b201bf`. The ownership helper
and test are identical; the parent model-pool file additionally logs the
selected device. No backend-selection semantics differ.

The incremental build passed with two jobs at nice 10. The native
`state-ownership-test` passed (0.13 s). One actual Metal RPC suite then passed
with exit 0, including transcript/timestamps, five repeated requests,
streaming during upload, cancellation/deadline/queue recovery, video,
Opus/Vorbis, limits and startup failure assertions. No video cases skipped.
The suite took 14.64 s including fixture generation; this is functional
verification, not a new latency benchmark.

All 1724 native scheduler records had positive GPU operations and zero CPU
or other operations: convolution 149, encoder 149, cross-attention 149,
decoder 1277. Host preprocessing/sampling are outside these graph counts.
No OpenVINO device qualification is claimed.

`asr-postownership-manifest.json` was generated on the Mac after the run. All
85 source hashes were independently checked against the tested Git commit.
It includes exact model/fixture hashes, source tree, native binary hashes,
pristine Whisper hash, generated canonical patch-chain hash and CMake-cache
hash. The exact runtime remains Whisper
`d1be6fde11ac6e0407606b4e42fe72d34add8037` with the canonical patch chain.

- Server SHA256: `60158d7ffdb876a9aff662e3ddcb9c23ec33e4d5a2b007aec9493f5619a4493a`
- RPC-test SHA256: `9d1e3feafd5dd92a1c4e0d89040a4e98b83494c46fb98376e72714f3d97eb51a`
- Pristine Whisper: `891168a838301bd2baa5e8b87422a854dafeeb7da914d40a2e4c6bf33ff448ed`
- Generated Whisper: `5bb518beb0defd5098badee6c7c6e576f15e98a1ce7a8e233408f52ac51fccc6`

Run from the Mac checkout with `/opt/homebrew/bin` on PATH:

```sh
GRPC_ASR_TEST_FIXTURE_FFMPEG="$PWD/tests/macos_fixture_ffmpeg.py" \
GRPC_ASR_TEST_MODEL=/absolute/models/asr/ggml-tiny.en.bin \
GRPC_ASR_TEST_SAMPLE=/absolute/models/asr/jfk.wav \
GRPC_ASR_SERVER_BINARY="$PWD/build-metal/grpc-asr-server" \
GRPC_ASR_TEST_REQUIRE_MODELS=1 GRPC_ASR_TEST_BACKEND=metal \
GRPC_ASR_TRACE_PLACEMENT=1 VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2 \
  /usr/bin/time -l timeout 300 nice -n 10 ./build-metal/asr-service-test
```

The fixture wrapper was executed directly from its tracked 100755 file.
Complete unedited build/unit/RPC outputs are retained as deterministic gzip;
the exit-code file and `assertions.json` record the automated checks. Raw
`time -l` output is retained without interpreting wrapper footprint fields
as model allocation measurements.

No CPU benchmark or LightOn rerun occurred. The previous CPU/Metal latency
comparison in `../metal-m2-20261108` remains evidence for source
`17b85d25071c6d395ae9cceb8fb062d6c5d42270`, not a post-fix comparison.
After this suite, an independent process check found no owned ASR/LightOn
process; memory free was 78%, and the exclusive Mac slot was released.
