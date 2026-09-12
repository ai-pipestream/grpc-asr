#!/usr/bin/env bash
# Boot-proofs a grpc-asr image: a green build is not "done" until the
# artifact actually starts. The service loads every model at startup and
# fails loud, so the hermetic checks below prove the whole static and early
# runtime path without GPU, Intel render devices, or model weights.
#
# Hermetic checks (safe for CI and publish, identical for the CUDA, CPU,
# and OpenVINO images):
#   1. closure: every shared library the server links resolves in the image
#   2. non-root: the image runs as the numeric user 65532
#   3. boot-to-model-load: with an empty models dir the server must fail
#      with its OWN "no ggml-*.bin model files" startup failure, proving
#      the loader, configuration parsing, backend verification, and model
#      discovery all ran, not a loader error.
#
# There is deliberately no --full mode: the image ships no client binary
# and a real transcription needs model weights, which CI contexts do not
# carry (the in-build ctest battery already covers the transcribe path on
# every leg, natively, inside the build).
#
# Nothing here needs a shell inside the image: the closure check asks the
# dynamic loader itself (LD_TRACE_LOADED_OBJECTS is what ldd does under the
# hood), so the checks pass on any base.
#
# The CUDA image links libcuda.so.1, which nvidia-container-toolkit injects
# on GPU hosts and no CI runner has. The image parks the CUDA stub library
# at /opt/cuda-stubs (off the default loader path; see Dockerfile), and
# every docker run below points LD_LIBRARY_PATH there so the loader
# resolves. The CPU and OpenVINO images have no such directory and ignore
# the setting. The boot check forces GRPC_ASR_BACKEND=cpu so model
# discovery is reached without a single CUDA call; the GPU path itself is
# out of scope here, as the header says.
set -euo pipefail

stub_path=/opt/cuda-stubs

usage() {
  echo "Usage: $0 IMAGE" >&2
  exit 64
}
[[ $# -eq 1 ]] || usage
image=$1

echo "== smoke: library closure of the shipped server"
trace=$(docker run --rm -e LD_TRACE_LOADED_OBJECTS=1 -e LD_LIBRARY_PATH="$stub_path" --entrypoint /usr/local/bin/grpc-asr-server "$image" 2>&1 || true)
if ! grep -q '=>' <<<"$trace"; then
  echo "the loader printed no dependency list for grpc-asr-server in $image:" >&2
  echo "$trace" >&2
  exit 1
fi
if grep -q "not found" <<<"$trace"; then
  echo "unresolved shared libraries for grpc-asr-server in $image:" >&2
  grep "not found" <<<"$trace" >&2
  exit 1
fi

echo "== smoke: image runs as the non-root user 65532"
image_user=$(docker inspect --format '{{.Config.User}}' "$image")
if [[ "$image_user" != "65532:65532" ]]; then
  echo "expected USER 65532:65532, image has '${image_user:-root}'" >&2
  exit 1
fi

echo "== smoke: server reaches model loading (expects its own no-models failure)"
# /models is a tmpfs with no weights: ModelPool must stop the boot with its
# own discovery failure on every image variant. The cpu backend keeps the
# check hermetic on the CUDA image (discovery precedes any device use).
boot_output=$(docker run --rm --tmpfs /models -e GRPC_ASR_BACKEND=cpu -e LD_LIBRARY_PATH="$stub_path" "$image" 2>&1 || true)
echo "$boot_output"
if grep -q "error while loading shared libraries" <<<"$boot_output"; then
  echo "the loader failed before main ran" >&2
  exit 1
fi
if ! grep -qF "Startup failed: no ggml-*.bin model files in /models" <<<"$boot_output"; then
  echo "expected the server's own startup failure for absent models" >&2
  exit 1
fi

echo "smoke-test: OK ($image)"
