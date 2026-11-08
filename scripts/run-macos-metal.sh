#!/bin/bash
set -euo pipefail
test "$(uname -s)" = Darwin && test "$(uname -m)" = arm64 || exit 1
test "$#" = 2 || { echo "usage: $0 READ_ONLY_MODEL_DIRECTORY MODEL_NAME" >&2; exit 64; }
task_models=$(cd "$1" && pwd)
[[ "$2" =~ ^[a-zA-Z0-9._-]+$ ]] || exit 64
test -r "$task_models/ggml-$2.bin" && test ! -w "$task_models/ggml-$2.bin" || {
  echo 'Selected model must exist and be read-only.' >&2; exit 1;
}
task_root=$(cd "$(dirname "$0")/.." && pwd)
export GRPC_ASR_MODELS_DIR="$task_models" GRPC_ASR_MODELS="$2"
export GRPC_ASR_BACKEND=metal GRPC_ASR_METAL_DEVICE=0
export GRPC_ASR_CONCURRENCY=1 GRPC_ASR_THREADS=2
export GRPC_ASR_LISTEN_ADDRESS=127.0.0.1:50055
exec nice -n 10 "$task_root/build-metal/grpc-asr-server"
