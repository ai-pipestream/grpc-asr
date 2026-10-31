#!/usr/bin/env bash
# Explicit qualification setup, never invoked by a serving request.
set -euo pipefail
destination=${1:-models}
mkdir -p "$destination"
fetch() {
  local name=$1 url=$2 digest=$3
  if [[ -f "$destination/$name" ]] && printf '%s  %s\n' "$digest" "$destination/$name" | sha256sum -c - >/dev/null 2>&1; then
    return
  fi
  curl -fsSL --retry 5 --retry-all-errors -o "$destination/$name" "$url"
  printf '%s  %s\n' "$digest" "$destination/$name" | sha256sum -c -
}
fetch ggml-tiny.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.en.bin 921e4cf8686fdd993dcd081a5da5b6c365bfde1162e72b08d75ac75289920b1f
fetch ggml-base.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-base.bin 60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe
fetch ggml-silero-v5.1.2.bin https://huggingface.co/ggml-org/whisper-vad/resolve/9ffd54a1e1ee413ddf265af9913beaf518d1639b/ggml-silero-v5.1.2.bin 29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf
fetch ggml-parakeet-tdt-0.6b-v3-q8_0.bin https://huggingface.co/ggml-org/parakeet-GGUF/resolve/35156454d1a39de06863303dd209fd2bed6ee079/ggml-parakeet-tdt-0.6b-v3-q8_0.bin 4d64e9e96c2792186d072fde0034df0ad670cf680a2f53069052ead827fd600e
