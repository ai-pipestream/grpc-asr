#!/bin/bash
set -euo pipefail
test "$(uname -s)" = Darwin && test "$(uname -m)" = arm64 || {
  echo 'Native Apple Silicon macOS is required; Linux Docker is not Metal.' >&2; exit 1;
}
test "$(sysctl -in sysctl.proc_translated 2>/dev/null || echo 0)" = 0 || exit 1
export PATH=/opt/homebrew/bin:$PATH
task_root=$(cd "$(dirname "$0")/.." && pwd)
cmake -S "$task_root" -B "$task_root/build-metal" -DCMAKE_BUILD_TYPE=Release \
  -DGRPC_ASR_METAL=ON -DGRPC_ASR_NATIVE_DEMUX=ON -DGRPC_ASR_SYSTEM_GRPC=ON "$@"
nice -n 10 cmake --build "$task_root/build-metal" --parallel 2
ctest --test-dir "$task_root/build-metal" --output-on-failure
