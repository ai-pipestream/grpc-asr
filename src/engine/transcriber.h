#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

struct whisper_context;
struct whisper_state;

namespace asr::engine {

// Thrown when the media runs past the configured duration cap. The
// service maps this to RESOURCE_EXHAUSTED.
class DurationCapExceeded : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// A word with timings inside a segment, present when word timestamps were
// requested.
struct EngineWord {
    std::string text;
    uint64_t start_ms = 0;
    uint64_t end_ms = 0;
    float probability = 0.0f;
};

// One transcribed span in absolute media time.
struct EngineSegment {
    uint32_t index = 0;
    uint64_t start_ms = 0;
    uint64_t end_ms = 0;
    std::string text;
    // Mean token log-probability. Meaningful only when token_count > 0; a
    // segment that scored no token has no confidence to report, which is a
    // different statement from a confident zero.
    float avg_logprob = 0.0f;
    uint32_t token_count = 0;
    std::vector<EngineWord> words;
    // Stream-local speaker label ("S1", "S2", ...), empty unless
    // diarization is on.
    std::string speaker;
    // True when the decoder predicted a speaker change after this segment.
    bool speaker_turn_next = false;
};

// Aggregates for the TranscriptComplete trailer.
struct EngineResult {
    std::string language;
    uint64_t duration_ms = 0;
    uint32_t final_segments = 0;
    uint64_t tokens = 0;
    // True when the sink asked to stop (client went away) — the trailer
    // must not be sent.
    bool aborted = false;
};

// Per-run knobs, resolved from TranscribeOptions plus server config.
struct EngineOptions {
    // ISO 639-1 hint; empty means autodetect.
    std::string language;
    bool translate = false;
    bool word_timestamps = false;
    // Ask the decoder for speaker turns and label segments by speaker. A
    // model without speaker-turn training never predicts one, so every
    // segment stays on the first speaker.
    bool diarize = false;
    int threads = 4;
    // PCM window per whisper_full call; bounds resident PCM memory.
    size_t window_seconds = 480;
    // Hard cap on total media duration; exceeding it throws.
    size_t max_duration_seconds = 14400;
};

// Streams a transcription over a pull-based PCM source.
//
// The window discipline: fill up to window_seconds of PCM, run
// whisper_full over it, and emit each segment the decoder commits through
// the sink as it commits (is_final=false while the window is still open
// for revision). Segments wholly inside a window become final immediately
// after the window; the window's last segment stays partial and is
// re-decoded from its own start in the next window, so its final may
// extend it. Finals replace partials by index. PCM behind a final is
// dropped — memory never grows with media length.
class Transcriber {
  public:
    // Pulls up to max mono f32 samples at the model rate; 0 means EOF.
    using PcmRead = std::function<size_t(float* out, size_t max_samples)>;
    // Receives segments; returning false aborts the transcription (the
    // client is gone). is_final follows the contract above.
    using SegmentSink = std::function<bool(const EngineSegment& segment, bool is_final)>;

    // Runs to completion on the caller's thread. Throws DecodeError
    // propagated from the source, DurationCapExceeded past the cap, and
    // std::runtime_error on an internal whisper failure.
    static EngineResult run(whisper_context* ctx, whisper_state* state,
                            const EngineOptions& options, const PcmRead& read,
                            const SegmentSink& sink);
};

}  // namespace asr::engine
