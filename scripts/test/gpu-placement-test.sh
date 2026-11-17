#!/usr/bin/env bash
# GPU placement receipt for the Intel image: proves Whisper runs whole on the
# Intel GPU (conv, encoder, cross-attention and decoder), that the GPU does
# the work, and that a missing or vague device stops startup instead of
# running on CPU.
#
#   scripts/test/gpu-placement-test.sh <openvino-image> <models-dir>
#
# <models-dir> holds the ggml-<name>.bin weights to serve, in a format the
# full mode runs (f16, or a q8_0/q5_1/q4_x/q4_k/q5_k/q6_k quantization; not
# q5_0). The server loads every ggml-*.bin it finds there; set
# GRPC_ASR_MODELS (e.g. "tiny.en,base") when the directory also holds files
# that are not Whisper weights, such as a VAD or Parakeet model.
# Needs a host with an Intel GPU render node (/dev/dri/renderD*);
# krick-1's Arc B70 is the fleet's. GRPC_ASR_RENDER_GID is the host gid of
# that node (default: read with stat). Without a render node the test FAILS;
# it never skips, so a host that lost its GPU cannot pass.
#
# Checks, in order:
#   1. no /dev/dri in the container: startup must fail with the server's own
#      "OpenVINO device 'GPU' ... is not available" (nothing is loaded, no
#      CPU run is attempted);
#   2. GRPC_ASR_OPENVINO_DEVICE=AUTO:GPU,CPU: startup must fail (device lists
#      let OpenVINO move work to CPU on its own);
#   3. GRPC_ASR_OPENVINO_MODE=full with GRPC_ASR_WARMUP=0: startup must fail
#      (the warmup pass carries the placement check);
#   4. with the GPU, two boots in a row: every loaded model must log
#      "full graph on OpenVINO GPU (<the card's name>)" with all compute
#      nodes of all four Whisper graphs on the device, no line may say
#      "decoder on CPU", the two boots must agree, the off-device counter
#      must read 0, and the server's own xe compute-engine cycles
#      (drm-cycles-ccs in /proc/1/fdinfo) must have advanced during warmup,
#      so a graph that quietly ran on CPU shows up as either a placement
#      failure or an idle engine.
set -euo pipefail
image=${1:?openvino image}
models=${2:?models dir}
models=$(cd "$models" && pwd)
work=$(mktemp -d)
server=grpc-asr-gpu-placement-$$
cleanup() {
  if docker inspect "$server" >/dev/null 2>&1; then
    echo "== server log (tail)"
    docker logs "$server" 2>&1 | tail -40 || true
    docker rm -f "$server" >/dev/null 2>&1 || true
  fi
  rm -rf "$work"
}
trap cleanup EXIT
fail() { echo "gpu-placement: FAIL: $*" >&2; exit 1; }

render=$(ls /dev/dri/renderD* 2>/dev/null | head -1 || true)
[ -n "$render" ] || fail "no /dev/dri/renderD* on this host; this test needs the Intel GPU"
ls "$models"/ggml-*.bin >/dev/null 2>&1 || fail "no ggml-*.bin weights in $models"
gid=${GRPC_ASR_RENDER_GID:-$(stat -c %g "$render")}
gpu_args=(--device /dev/dri:/dev/dri --group-add "$gid")
base_env=(-e GRPC_ASR_BACKEND=openvino -e GRPC_ASR_OPENVINO_MODE=full
          -e GRPC_ASR_OPENVINO_DEVICE=GPU -e GRPC_ASR_METRICS_ADDRESS=0.0.0.0:9464
          -e GRPC_ASR_OPENVINO_CACHE_DIR=/var/cache/openvino)
[ -z "${GRPC_ASR_MODELS:-}" ] || base_env+=(-e "GRPC_ASR_MODELS=$GRPC_ASR_MODELS")
cache=grpc-asr-gpu-placement-cache
docker volume create "$cache" >/dev/null
run_args=(--read-only --tmpfs /tmp -v "$models:/models:ro" -v "$cache:/var/cache/openvino")

# A startup that must fail: exits non-zero within the timeout, through the
# server's own "Startup failed" path, and its log carries the expected text.
# Later -e arguments override base_env.
expect_startup_failure() {
  local label=$1 text=$2; shift 2
  echo "== $label: startup must fail"
  set +e
  timeout 300 docker run --rm "${run_args[@]}" "${base_env[@]}" "$@" "$image" > "$work/fail.log" 2>&1
  local code=$?
  set -e
  grep -E "Startup failed|not available|GRPC_ASR_" "$work/fail.log" | head -3 || true
  [ "$code" != 0 ] && [ "$code" != 124 ] || fail "$label: startup exit $code, wanted a failure"
  grep -q "Startup failed" "$work/fail.log" || fail "$label: the server did not report a startup failure"
  grep -qF "$text" "$work/fail.log" || fail "$label: log lacks '$text'"
  ! grep -q "grpc-asr listening on" "$work/fail.log" || fail "$label: the server came up"
  echo "failed as required (exit $code)"
}

expect_startup_failure "no GPU device" "OpenVINO device 'GPU' (GRPC_ASR_OPENVINO_DEVICE) is not available"
expect_startup_failure "AUTO device list" "AUTO, HETERO, MULTI and BATCH device lists" \
  "${gpu_args[@]}" -e GRPC_ASR_OPENVINO_DEVICE=AUTO:GPU,CPU
expect_startup_failure "no warmup" "GRPC_ASR_WARMUP=0 cannot be combined with GRPC_ASR_OPENVINO_MODE=full" \
  "${gpu_args[@]}" -e GRPC_ASR_WARMUP=0

ccs_cycles() {
  docker exec "$server" sh -c \
    'total=0; for f in /proc/1/fdinfo/*; do
       v=$(grep -s "^drm-cycles-ccs:" "$f" | awk "{print \$2}");
       [ -n "$v" ] && total=$((total + v));
     done; echo $total'
}

boot_and_check() {
  local pass=$1
  echo "== boot $pass with the GPU"
  docker run -d --name "$server" -p 127.0.0.1::9464 "${run_args[@]}" "${base_env[@]}" \
    "${gpu_args[@]}" "$image" >/dev/null
  local listening=false
  for _ in $(seq 1 900); do
    if docker logs "$server" 2>&1 | grep -q "grpc-asr listening on"; then listening=true; break; fi
    [ "$(docker inspect -f '{{.State.Running}}' "$server")" = true ] || break
    sleep 1
  done
  [ "$listening" = true ] || fail "boot $pass: server did not come up"
  docker logs "$server" > "$work/boot-$pass.log" 2>&1
  grep -E "grpc-asr (openvino|placement):" "$work/boot-$pass.log" || true

  grep -q "grpc-asr openvino: mode full (GRPC_ASR_OPENVINO_MODE), device GPU (" "$work/boot-$pass.log" ||
    fail "boot $pass: the log does not state full mode on device GPU with the card's name"
  ! grep -q "decoder on CPU" "$work/boot-$pass.log" || fail "boot $pass: a decoder runs on CPU"
  grep "^grpc-asr placement: " "$work/boot-$pass.log" > "$work/placement-$pass.txt" ||
    fail "boot $pass: no placement line"
  local loaded
  loaded=$(grep -c "^grpc-asr model loaded: " "$work/boot-$pass.log")
  [ "$(wc -l < "$work/placement-$pass.txt")" = "$loaded" ] ||
    fail "boot $pass: $loaded model(s) loaded but not as many placement lines"
  # Every line: the full graph on the GPU, and for each of the four graphs
  # "<nodes on device>/<compute nodes>" equal and above zero.
  awk '
    !/ full graph on OpenVINO GPU \(.+\); compute nodes on the device: / { print "not the full graph on the GPU: " $0; bad = 1; next }
    {
      counts = $0; sub(/.*compute nodes on the device: /, "", counts)
      n = split(counts, stage, ", ")
      if (n != 4) { print "expected four graphs: " $0; bad = 1; next }
      split("conv encoder cross decoder", want, " ")
      for (i = 1; i <= 4; i++) {
        split(stage[i], part, " "); split(part[2], frac, "/")
        if (part[1] != want[i] || frac[1] != frac[2] || frac[2] + 0 <= 0) {
          print "graph " want[i] " is not fully on the device: " $0; bad = 1
        }
      }
    }
    END { exit bad }' "$work/placement-$pass.txt" || fail "boot $pass: placement is not complete"

  local port
  port=$(docker port "$server" 9464/tcp | head -1 | sed 's/.*://')
  curl -sf "http://127.0.0.1:$port/metrics" > "$work/metrics-$pass.prom"
  grep -qx 'grpc_asr_whisper_off_device_nodes_total 0' "$work/metrics-$pass.prom" ||
    fail "boot $pass: grpc_asr_whisper_off_device_nodes_total is not 0"

  local cycles
  cycles=$(ccs_cycles)
  echo "boot $pass: compute-engine cycles after warmup: $cycles"
  [ "$cycles" -gt 0 ] || fail "boot $pass: the GPU compute engine did no work for this server"
  docker rm -f "$server" >/dev/null
}

boot_and_check 1
boot_and_check 2
cmp -s "$work/placement-1.txt" "$work/placement-2.txt" || fail "placement changed between boots"
echo "gpu-placement: ok"
