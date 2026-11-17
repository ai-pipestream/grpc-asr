# Native media timeline regression — 2026-11-08

Base: `add2a7a05f619cdc8515719d3a94633b74086f27`.
Branch: `fix/native-media-timeline-20261108`.

Audio and video now share the container origin. Without container metadata,
the earliest known A/V stream start is used, or zero if all starts are unknown.
Delayed audio emits zeros in bounded chunks; preroll is trimmed after
resampling. Cancellation applies during padding. The existing consumed-sample
limit includes padding (`src/engine/transcriber.cpp:174-184`, source checked;
no new model-backed duration-limit experiment).

## Results

9 Matroska/PCM cases combine origins +10s, -1s, 0 with aligned tracks, video
500ms late, and audio 500ms late. Encoded fixture metadata is independently
checked. Each case verifies exact keyframe times, PCM length and content.
Delayed audio includes cancellation during padding, reopening, and full reuse.
An attempted -10s fixture was excluded: FFmpeg rebases that output.

- Original source: exit 1, 7 failures, 2 passing controls (`red.log`).
- Fixed source: exit 0, 9 cases plus existing native demux tests (`green.log`).
- ASan/UBSan/leak detection: complete suite exit 0 (`sanitized.log`).

Successful logs include expected errors from intentional truncated MP4 tests.
Linux x86_64, GCC 15.3.0, FFmpeg CLI 8.1.2-2ubuntu2; libavformat 62.12.102,
libavcodec 62.28.102, libswresample 6.3.102, libswscale 9.5.102.

## Commands

Targeted native implementation build; no Whisper/gRPC dependency:

```bash
mkdir -p build-timeline
git show add2a7a:src/media/video_demux_native.cpp > build-timeline/video_demux_native_before.cpp
nice -n 10 g++ -std=c++23 -O1 -g -Wall -Wextra -Wnon-virtual-dtor -Werror \
  -Isrc -Itests tests/video_demux_native_test.cpp build-timeline/video_demux_native_before.cpp \
  -o build-timeline/video-demux-before \
  $(pkg-config --cflags --libs libavformat libavcodec libavutil libswresample libswscale)
timeout 60s nice -n 10 build-timeline/video-demux-before --timeline-only
# Expected exit 1.

nice -n 10 g++ -std=c++23 -O1 -g -Wall -Wextra -Wnon-virtual-dtor -Werror \
  -Isrc -Itests tests/video_demux_native_test.cpp src/media/video_demux_native.cpp \
  -o build-timeline/video-demux-test \
  $(pkg-config --cflags --libs libavformat libavcodec libavutil libswresample libswscale)
timeout 60s nice -n 10 build-timeline/video-demux-test

nice -n 10 g++ -std=c++23 -O1 -g -Wall -Wextra -Wnon-virtual-dtor -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer -Isrc -Itests \
  tests/video_demux_native_test.cpp src/media/video_demux_native.cpp \
  -o build-timeline/video-demux-sanitized \
  $(pkg-config --cflags --libs libavformat libavcodec libavutil libswresample libswscale)
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
  timeout 60s nice -n 10 build-timeline/video-demux-sanitized
```

No GPU, Mac, RPC, model, or performance qualification. Midstream timestamp
discontinuities are outside this initial-origin fix. Gauss review remains
pending. Parent owns the separate probe-root correction.
