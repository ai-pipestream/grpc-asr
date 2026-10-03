#pragma once

#include <cstdint>

namespace asr::engine {

// Where one decoded window ends: segments [0, finalize) become final, and
// the next window starts resume_ms after this window's first sample.
struct WindowCut {
    int finalize = 0;
    uint64_t resume_ms = 0;
};

// Decides the cut for a window of window_ms that decoded `segments`
// segments, the last of which starts last_start_ms into the window (read
// only when there are two or more).
//
// The last window finalizes everything. Otherwise the last segment is held
// back, because the window edge may have cut it, and the next window
// re-decodes from its start. Holding back needs a progress point strictly
// inside the window: a window that decoded nothing (silence, music), a
// single segment spanning the window, and a last segment that starts at
// the window's first sample or past its end all finalize what there is and
// resume at the window end. Resuming in place would re-decode the same
// window forever and emit its finals again under new indexes.
inline WindowCut cut_window(int segments, bool last_window, uint64_t window_ms,
                            uint64_t last_start_ms) {
    if (segments <= 0) {
        return {.finalize = 0, .resume_ms = window_ms};
    }
    if (!last_window && segments >= 2 && last_start_ms > 0 && last_start_ms < window_ms) {
        return {.finalize = segments - 1, .resume_ms = last_start_ms};
    }
    return {.finalize = segments, .resume_ms = window_ms};
}

}  // namespace asr::engine
