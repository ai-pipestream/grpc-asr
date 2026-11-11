#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <vector>

#include "ai/pipestream/asr/v1/asr_service.pb.h"

struct whisper_context;
struct whisper_state;

namespace asr::engine {

// Thrown when the media runs past the configured duration cap. The
// service maps this to RESOURCE_EXHAUSTED.
class DurationCapExceeded : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// Local decoder sampling strategy.
enum class SamplingMode {
    kGreedy,
    kBeamSearch,
};

// Validated local decoder tuning from TranscribeOptions.decoding.
struct DecodingConfig {
    std::string initial_prompt;
    SamplingMode strategy = SamplingMode::kGreedy;
    uint32_t beam_size = 0;  // 0 -> whisper default (5), max 8
    uint32_t best_of = 0;    // 0 -> whisper default, max 8
    std::optional<float> no_speech_threshold;
    std::optional<float> logprob_threshold;
    std::optional<float> entropy_threshold;
    bool suppress_non_speech_tokens = false;
};

inline constexpr size_t kMaxInitialPromptBytes = 1024;
inline constexpr uint32_t kMaxBeamSize = 8;
inline constexpr uint32_t kMaxBestOf = 8;

// Validates `options` and populates `out` when non-null. Returns std::nullopt
// when valid, or a human-readable INVALID_ARGUMENT reason otherwise.
std::optional<std::string> parse_decoding_options(
    const ai::pipestream::asr::v1::DecodingOptions& options,
    DecodingConfig* out = nullptr);

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
    bool has_avg_logprob = false;
    uint32_t token_count = 0;
    float no_speech_prob = 0.0f;
    bool has_no_speech_prob = false;
    bool suppressed = false;
    std::optional<uint32_t> channel;
    std::string language;
    std::vector<EngineWord> words;
    // Stream-local speaker label ("S1", "S2", ... or "C1", "C1-S1", ...),
    // empty unless diarization or per-channel mode is on.
    std::string speaker;
    // True when the decoder predicted a speaker change after this segment.
    bool speaker_turn_next = false;
};

// Aggregates for the TranscriptComplete trailer.
struct EngineResult {
    std::string language;
    std::vector<std::string> languages;
    uint64_t duration_ms = 0;
    uint32_t final_segments = 0;
    uint64_t tokens = 0;
    uint64_t windows = 0;
    uint64_t window_redecodes = 0;
    uint64_t decode_wall_ns = 0;
    // True when the sink asked to stop or the stop token fired (the client
    // went away); the trailer must not be sent.
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
    // Upper bound on PCM window per whisper_full call; bounds resident PCM.
    size_t window_seconds = 480;
    // Minimum PCM buffered before a stalled upload triggers an early decode.
    size_t min_window_seconds = 30;
    // Milliseconds of source stall after `min_window_seconds` is buffered that
    // triggers an early decode pass (0 disables stall cuts).
    size_t window_stall_ms = 0;
    // Hard cap on total media duration; exceeding it throws.
    size_t max_duration_seconds = 14400;
    // Zero-based source channel when running under CHANNEL_MODE_PER_CHANNEL.
    std::optional<uint32_t> channel;
    DecodingConfig decoding;
};

// Streams a transcription over a pull-based PCM source.
class Transcriber {
  public:
    // Pulls up to max mono f32 samples at the model rate; 0 means EOF.
    using PcmRead = std::function<size_t(float* out, size_t max_samples)>;
    // Optional bounded readiness wait used once `min_window_seconds` of PCM is
    // buffered: returns true when more input (or EOF/stop) becomes available
    // within `timeout`, or false when the source stalled.
    using PcmWait = std::function<bool(std::chrono::milliseconds timeout)>;
    // Receives segments; returning false aborts the transcription (the
    // client is gone).
    using SegmentSink = std::function<bool(const EngineSegment& segment, bool is_final)>;

    static EngineResult run(whisper_context* ctx, whisper_state* state,
                            const EngineOptions& options, const PcmRead& read,
                            const SegmentSink& sink, std::stop_token stop = {},
                            const PcmWait& wait_for_more = {});
};

}  // namespace asr::engine
