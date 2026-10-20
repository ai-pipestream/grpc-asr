# Receipts: feat/remote-asr-options

Box: krick. Build dir `/work/tmp/grpc-asr-remote-build`: Release, Ninja,
`-DBUILD_TESTING=ON`, CPU backend. The grpc v1.83.0 and whisper.cpp v1.9.3
sources are reused from the main checkout's `_deps` through
`FETCHCONTENT_SOURCE_DIR_*`; the pins are the same. `models/` is a symlink
to the main checkout's models (ggml-tiny.en.bin), so asr-service-test runs
for real. It is not skipped.

| Receipt | What it shows |
|---|---|
| `build-716098b.txt` | build exit 0 |
| `ctest-716098b.txt` | `ctest -L asr`: 11/11 passed. Includes remote-options-test and asr-service-test, whose error matrix now covers remote options (file:// url is INVALID_ARGUMENT, a valid set is UNIMPLEMENTED, no media info or segments). |
| `ctest-5432bc7.txt` | the same suite on the first commit, before the proto comment fix |
| `buf-lint-716098b.txt` | `buf lint` exit 0 |
| `form_fields_reference.py`, `form-fields-reference.txt` | docling-slim 2.140.0's own `_ApiAsrModel._form_fields` for the cases remote-options-test checks, including the `floats` case (Python `str()` in both notations). |
| `remote-options-test-standalone.txt` | the float format and validation commit: remote-options-test built standalone and passing, red with the old formatter, and 20000 random floats matching python3 `str()`. |
