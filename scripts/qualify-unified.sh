#!/usr/bin/env bash
# Exercise all three interfaces on one server, using pre-provisioned weights.
set -euo pipefail
build=${1:?build directory}
models=${2:?directory containing ggml-tiny.en.bin}
vad=${3:?Silero model path}
backend=${4:-whisper}
export GRPC_ASR_BACKEND=cpu
export GRPC_ASR_MODELS_DIR="$models" GRPC_ASR_MODELS=tiny.en
export GRPC_ASR_LISTEN_ADDRESS="127.0.0.1:${GRPC_ASR_QUALIFY_PORT:-50555}"
export GRPC_ASR_STREAM_BACKEND="$backend" GRPC_ASR_VAD_MODEL_PATH="$vad"
export GRPC_ASR_HTTP_ORIGINS="http://127.0.0.1:${GRPC_ASR_QUALIFY_HTTP_PORT:-18081}"
export GRPC_ASR_HTTP_ALLOW_PRIVATE=1
if [[ "$backend" == whisper ]]; then
  export GRPC_ASR_STREAM_MODEL=tiny.en
  test_model="$models/ggml-tiny.en.bin"
elif [[ "$backend" == parakeet ]]; then
  export GRPC_ASR_STREAM_MODEL=parakeet
  test_model=${GRPC_ASR_PARAKEET_MODEL_PATH:?Parakeet model path}
else
  echo 'backend must be whisper or parakeet' >&2
  exit 1
fi
mkdir -p "$build/qualification"
"$build/grpc-asr-server" > "$build/qualification/server-$backend.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
ready=0
for attempt in $(seq 1 120); do
  kill -0 "$server_pid"
  if grpcurl -plaintext "$GRPC_ASR_LISTEN_ADDRESS" list > "$build/qualification/rpcs-$backend.log" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 0.25
done
[[ "$ready" == 1 ]]
for service in ai.pipestream.asr.v1.AsrService ai.pipestream.media.v1.MediaService ai.pipestream.media.v1.SpeechWorkerService; do
  grep -Fx "$service" "$build/qualification/rpcs-$backend.log"
done
grpcurl -plaintext -d '{}' "$GRPC_ASR_LISTEN_ADDRESS" ai.pipestream.asr.v1.AsrService/GetServiceInfo > "$build/qualification/collector-$backend.json"
MEDIA_PIPELINE_TEST_TARGET="$GRPC_ASR_LISTEN_ADDRESS" \
MEDIA_PIPELINE_HTTP_TARGET="$GRPC_ASR_LISTEN_ADDRESS" \
MEDIA_PIPELINE_HTTP_PORT="${GRPC_ASR_QUALIFY_HTTP_PORT:-18081}" \
MEDIA_PIPELINE_TEST_MODEL_ID="$GRPC_ASR_STREAM_MODEL" \
  "$build/streaming/pipeline_tests" "$test_model" "$vad" "$build/_deps/whisper-src/samples/jfk.wav" "$backend"
