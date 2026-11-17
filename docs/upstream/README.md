# Changes to ggml's OpenVINO backend, for its maintainers

grpc-asr builds the pinned whisper.cpp with three edits to ggml's OpenVINO
backend (`ggml/src/ggml-openvino`). The build applies them from
`cmake/PatchGgmlOpenvino.cmake`. This directory holds the same edits as
patches that apply to `ggml-org/llama.cpp`, where the backend is developed,
each with a note on what was seen and how it was checked. None of them has
been sent upstream yet.

| Patch | Note | What it does |
|---|---|---|
| `ggml-openvino-q5_0.patch` | `ggml-openvino-q5_0.md` | Runs Q5_0 weights on the device |
| `ggml-openvino-flat-kv-reuse.patch` | `ggml-openvino-flat-kv-reuse.md` | Stops reusing a compiled graph after whisper.cpp replaces a KV buffer |
| `ggml-openvino-cache-mode.patch` | `ggml-openvino-cache-mode.md` | Stores weights in the plugin cache on CPU and GPU |

The patches are `git format-patch` output against llama.cpp `master` at
`d3c3e24` (2026-11-16) and apply with `git am` in that order; they are
independent of each other. The backend sources in whisper.cpp's vendored ggml
(the copy grpc-asr pins) take the same edits.

Test stack for every result below: OpenVINO 2026.4.1 (Intel's release
archive), Ubuntu 26.04, NEO compute runtime 26.05; the GPU is an Intel Arc
B70 on the xe driver, the CPU device an AMD Ryzen 9 9950X. The checks are
grpc-asr's `openvino-q5-0-test` and `openvino-full-test` (ctest, OpenVINO CPU
device in the image build, the same binaries on the GPU) and
`scripts/test/gpu-placement-test.sh` (the shipped server, two boots).
llama.cpp's `test-backend-ops` was not run against these patches.
