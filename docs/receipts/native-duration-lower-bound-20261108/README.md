# Unknown-duration admission receipt

CPU Linux proof for P2 in d1d5809. Unknown packet timing previously erased
confirmed endpoints. PCM lasting 2s passed a 3s quota despite video ending at 3.5s.

ProbeInfo adds duration_lower_bound_ms. Unknown total remains zero; admission
checks both fields. PTS without packet duration also contributes an endpoint.
Absolute format metadata is not restored. Scan deadline/cancellation unchanged.
No proto or transcriber changes.

4 authored Matroska fixtures: origins +10s/-1s, relative/absolute Duration tags.
The fixture inspector verifies encoded origin, metadata, 1 untimed PNG packet,
and 3.5s confirmed endpoint. Main video 0.5–3.5s; audio 0–2s. Native assertions:
unknown total 0, minimum duration 3500ms, PCM 32000 samples, frames 500/1500/2500ms.

RED: d1d5809 demux with current header/admission/tests. TCP RPC returned OK;
RESOURCE_EXHAUSTED assertion failed, exit 1.

GREEN: current demux. 4 TCP requests return RESOURCE_EXHAUSTED before MediaInfo,
transcript or Complete. Same CPU pool then passes 5 existing timeline requests:
shifted JFK, delayed audio, video extending beyond PCM. Exit 0; elapsed 2.32s;
peak RSS 214604 KiB. Full native unit suite passes ASan/UBSan/leak detection and
WERROR: 9 existing origin/offset cases plus 4 new cases.

Build/dependencies: ../native-media-timeline-rpc-20261108/README.md and
cpu-build.cmake.txt. TIMELINE_DEMUX_SOURCE selects old demux for RED, current
src/media/video_demux_native.cpp for GREEN. Identical CPU run environment:
6GiB address-space limit, nice10, 240s timeout, 2 threads, MEDIA_TIMELINE_ONLY.
No ASR device runs. Busy-graph hard interruption unqualified. Independent review
pending. Raw logs, exits and hashes accompany this receipt.
