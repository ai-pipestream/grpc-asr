// The window-cut rule over hand-picked decodes: which segments a window
// finalizes and where the next window resumes. Pure logic, so it runs on
// every build, model weights or not.

#include "engine/window_cut.h"

#include <print>

#include "fixture.h"

using asr::engine::cut_window;
using asr::engine::WindowCut;

namespace {

void verify_held_back_segment() {
    // The window edge may have cut the last segment, so it is held back
    // and the next window re-decodes from its start.
    const WindowCut cut = cut_window(3, /*last_window=*/false, 30000, 24000);
    require(cut.finalize == 2, "all but the last segment finalize");
    require(cut.resume_ms == 24000, "the next window starts at the held-back segment");
}

void verify_last_window() {
    const WindowCut cut = cut_window(3, /*last_window=*/true, 12000, 9000);
    require(cut.finalize == 3, "the last window finalizes every segment");
}

void verify_empty_window() {
    // Silence or music: whisper decodes no segment. This used to finalize
    // segment -1, stepping the index backwards and reading whisper's
    // segment array out of bounds for the resume point.
    WindowCut cut = cut_window(0, /*last_window=*/false, 30000, 0);
    require(cut.finalize == 0, "a window with no segment finalizes nothing");
    require(cut.resume_ms == 30000, "and the next window starts at its end");
    cut = cut_window(0, /*last_window=*/true, 30000, 0);
    require(cut.finalize == 0, "an empty last window finalizes nothing either");
}

void verify_single_segment() {
    // One segment spanning the window leaves no progress point to resume
    // from; holding it back would decode the same window again.
    const WindowCut cut = cut_window(1, /*last_window=*/false, 30000, 0);
    require(cut.finalize == 1 && cut.resume_ms == 30000,
            "a lone segment finalizes and the next window starts at the end");
}

void verify_no_progress_point() {
    // A last segment starting at the window's first sample would resume in
    // place: the same window decoded again and again, its finals emitted
    // again each time under new indexes.
    WindowCut cut = cut_window(4, /*last_window=*/false, 30000, 0);
    require(cut.finalize == 4 && cut.resume_ms == 30000,
            "a held-back segment at the window start finalizes instead");
    // Nor can a start at or past the window end be a resume point.
    cut = cut_window(2, /*last_window=*/false, 30000, 30000);
    require(cut.finalize == 2 && cut.resume_ms == 30000,
            "a held-back segment starting at the window end finalizes");
    cut = cut_window(2, /*last_window=*/false, 30000, 45000);
    require(cut.finalize == 2 && cut.resume_ms == 30000,
            "a held-back segment starting past the window end finalizes");
}

}  // namespace

int main() {
    try {
        verify_held_back_segment();
        verify_last_window();
        verify_empty_window();
        verify_single_segment();
        verify_no_progress_point();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("window-cut-test passed");
    return 0;
}
