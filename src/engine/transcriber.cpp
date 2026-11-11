#include "engine/transcriber.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#include "engine/window_cut.h"
#include "media/audio_decoder.h"
#include "whisper.h"

namespace asr::engine {

namespace {

using media::kModelSampleRate;

constexpr size_t kFillSliceSamples = 4096;  // 256 ms at 16 kHz
constexpr float kDefaultNoSpeechThreshold = 0.6f;
constexpr float kDefaultLogprobThreshold = -1.0f;
constexpr int kDefaultBeamSize = 5;

uint64_t centisec_to_ms(int64_t t) {
    return static_cast<uint64_t>(t) * 10ULL;
}

// State shared between the whisper callbacks and the window loop.
struct RunState {
    whisper_context* ctx = nullptr;
    const EngineOptions* options = nullptr;
    const Transcriber::SegmentSink* sink = nullptr;
    // Absolute media time of the current window's first sample.
    uint64_t window_base_ms = 0;
    // Global index of the current window's segment 0.
    uint32_t window_base_index = 0;
    // Zero-based speaker of the current window's segment 0; advances by
    // every speaker turn the finalized segments predicted.
    uint32_t window_base_speaker = 0;
    // Set when the sink returns false; makes whisper abort mid-window.
    std::atomic<bool> abort{false};
    // The stream's stop; whisper polls it through on_abort as well.
    std::stop_token stop;

    bool stopped() const { return abort.load() || stop.stop_requested(); }
};

// Builds an EngineSegment from whisper's committed segment i (window
// relative), shifting times to absolute media time.
EngineSegment read_segment(RunState& run, whisper_state* state, int i) {
    EngineSegment segment;
    segment.index = run.window_base_index + static_cast<uint32_t>(i);
    segment.start_ms =
        run.window_base_ms + centisec_to_ms(whisper_full_get_segment_t0_from_state(state, i));
    segment.end_ms =
        run.window_base_ms + centisec_to_ms(whisper_full_get_segment_t1_from_state(state, i));
    segment.text = whisper_full_get_segment_text_from_state(state, i);
    segment.channel = run.options->channel;

    const int lang_id = whisper_full_lang_id_from_state(state);
    if (lang_id >= 0) {
        const char* code = whisper_lang_str(lang_id);
        if (code != nullptr) {
            segment.language = code;
        }
    }
    if (segment.language.empty() && !run.options->language.empty()) {
        segment.language = run.options->language;
    }

    if (run.options->diarize) {
        // Speakers are numbered, not identified: the decoder reports where
        // one voice hands off to another, so the label is the running count
        // of handoffs. Segments earlier in this window are re-read rather
        // than remembered, which keeps a re-decoded window self-consistent.
        uint32_t speaker = run.window_base_speaker;
        for (int j = 0; j < i; j++) {
            if (whisper_full_get_segment_speaker_turn_next_from_state(state, j)) {
                speaker++;
            }
        }
        if (run.options->channel.has_value()) {
            segment.speaker = "C" + std::to_string(*run.options->channel + 1) + "-S" +
                              std::to_string(speaker + 1);
        } else {
            segment.speaker = "S" + std::to_string(speaker + 1);
        }
        segment.speaker_turn_next =
            whisper_full_get_segment_speaker_turn_next_from_state(state, i);
    } else if (run.options->channel.has_value()) {
        segment.speaker = "C" + std::to_string(*run.options->channel + 1);
    }

    const int n_tokens = whisper_full_n_tokens_from_state(state, i);
    const whisper_token eot = whisper_token_eot(run.ctx);
    double logprob_sum = 0.0;
    uint32_t counted = 0;
    EngineWord word;
    for (int t = 0; t < n_tokens; t++) {
        whisper_token_data data = whisper_full_get_token_data_from_state(state, i, t);
        if (data.id >= eot) {
            continue;  // timestamp/special tokens carry no text
        }
        logprob_sum += data.plog;
        counted++;
        if (!run.options->word_timestamps) {
            continue;
        }
        const char* text = whisper_full_get_token_text_from_state(run.ctx, state, i, t);
        // A token starting with a space starts a new word.
        if (text[0] == ' ' && !word.text.empty()) {
            segment.words.push_back(word);
            word = EngineWord{};
        }
        if (word.text.empty()) {
            word.start_ms = run.window_base_ms + centisec_to_ms(data.t0);
            word.probability = data.p;
        } else {
            word.probability = std::min(word.probability, data.p);
        }
        word.end_ms = run.window_base_ms + centisec_to_ms(data.t1);
        word.text += text;
    }
    if (!word.text.empty()) {
        segment.words.push_back(word);
    }
    segment.token_count = counted;
    segment.has_avg_logprob = counted > 0;
    segment.avg_logprob =
        counted == 0 ? 0.0f : static_cast<float>(logprob_sum / static_cast<double>(counted));
    segment.no_speech_prob = whisper_full_get_segment_no_speech_prob_from_state(state, i);
    segment.has_no_speech_prob = true;

    const float no_speech_gate =
        run.options->decoding.no_speech_threshold.value_or(kDefaultNoSpeechThreshold);
    const float logprob_gate =
        run.options->decoding.logprob_threshold.value_or(kDefaultLogprobThreshold);
    segment.suppressed = segment.no_speech_prob > no_speech_gate &&
                         (segment.token_count == 0 || segment.avg_logprob < logprob_gate);
    return segment;
}

// whisper commits segments as it decodes; each one is surfaced live as a
// partial. Finalization happens after the window, when we know which
// segments the window edge could not have cut.
void on_new_segment(whisper_context* /*ctx*/, whisper_state* state, int n_new, void* user_data) {
    RunState& run = *static_cast<RunState*>(user_data);
    const int total = whisper_full_n_segments_from_state(state);
    for (int i = total - n_new; i < total; i++) {
        EngineSegment segment = read_segment(run, state, i);
        if (!(*run.sink)(segment, /*is_final=*/false)) {
            run.abort.store(true);
        }
    }
}

bool on_abort(void* user_data) {
    return static_cast<RunState*>(user_data)->stopped();
}

}  // namespace

std::optional<std::string> parse_decoding_options(
    const ai::pipestream::asr::v1::DecodingOptions& options, DecodingConfig* out) {
    if (options.initial_prompt().size() > kMaxInitialPromptBytes) {
        return "decoding.initial_prompt exceeds " + std::to_string(kMaxInitialPromptBytes) +
               " bytes";
    }
    if (options.beam_size() > kMaxBeamSize) {
        return "decoding.beam_size exceeds maximum of " + std::to_string(kMaxBeamSize);
    }
    if (options.best_of() > kMaxBestOf) {
        return "decoding.best_of exceeds maximum of " + std::to_string(kMaxBestOf);
    }
    SamplingMode mode = SamplingMode::kGreedy;
    switch (options.strategy()) {
        case ai::pipestream::asr::v1::SAMPLING_STRATEGY_UNSPECIFIED:
        case ai::pipestream::asr::v1::SAMPLING_STRATEGY_GREEDY:
            mode = SamplingMode::kGreedy;
            break;
        case ai::pipestream::asr::v1::SAMPLING_STRATEGY_BEAM_SEARCH:
            mode = SamplingMode::kBeamSearch;
            break;
        default:
            return "unknown decoding.strategy value";
    }
    if (options.has_no_speech_threshold()) {
        const float v = options.no_speech_threshold();
        if (!std::isfinite(v) || v < 0.0f || v > 1.0f) {
            return "decoding.no_speech_threshold must be finite and in [0, 1]";
        }
    }
    if (options.has_logprob_threshold() && !std::isfinite(options.logprob_threshold())) {
        return "decoding.logprob_threshold must be finite";
    }
    if (options.has_entropy_threshold()) {
        const float v = options.entropy_threshold();
        if (!std::isfinite(v) || v <= 0.0f) {
            return "decoding.entropy_threshold must be finite and positive";
        }
    }
    if (out != nullptr) {
        out->initial_prompt = options.initial_prompt();
        out->strategy = mode;
        out->beam_size = options.beam_size();
        out->best_of = options.best_of();
        out->no_speech_threshold = options.has_no_speech_threshold()
                                       ? std::optional<float>(options.no_speech_threshold())
                                       : std::nullopt;
        out->logprob_threshold = options.has_logprob_threshold()
                                     ? std::optional<float>(options.logprob_threshold())
                                     : std::nullopt;
        out->entropy_threshold = options.has_entropy_threshold()
                                     ? std::optional<float>(options.entropy_threshold())
                                     : std::nullopt;
        out->suppress_non_speech_tokens = options.suppress_non_speech_tokens();
    }
    return std::nullopt;
}

EngineResult Transcriber::run(whisper_context* ctx, whisper_state* state,
                              const EngineOptions& options, const PcmRead& read,
                              const SegmentSink& sink, std::stop_token stop,
                              const PcmWait& wait_for_more) {
    RunState run{.ctx = ctx, .options = &options, .sink = &sink, .stop = std::move(stop)};

    const whisper_sampling_strategy sampling =
        options.decoding.strategy == SamplingMode::kBeamSearch ? WHISPER_SAMPLING_BEAM_SEARCH
                                                               : WHISPER_SAMPLING_GREEDY;
    whisper_full_params params = whisper_full_default_params(sampling);
    params.print_progress = false;
    params.print_realtime = false;
    params.print_special = false;
    params.print_timestamps = false;
    params.translate = options.translate;
    params.language = options.language.empty() ? "auto" : options.language.c_str();
    params.n_threads = options.threads;
    // Windows are independent decodes; carrying the previous window's text
    // as a prompt would let one hallucination poison the rest of the file.
    params.no_context = true;
    if (!options.decoding.initial_prompt.empty()) {
        params.initial_prompt = options.decoding.initial_prompt.c_str();
        params.carry_initial_prompt = true;
    }
    if (options.decoding.strategy == SamplingMode::kBeamSearch) {
        params.beam_search.beam_size = options.decoding.beam_size > 0
                                           ? static_cast<int>(options.decoding.beam_size)
                                           : kDefaultBeamSize;
    }
    if (options.decoding.best_of > 0) {
        params.greedy.best_of = static_cast<int>(options.decoding.best_of);
    }
    params.suppress_nst = options.decoding.suppress_non_speech_tokens;
    if (options.decoding.entropy_threshold.has_value()) {
        params.entropy_thold = *options.decoding.entropy_threshold;
    }
    // Keep whisper's internal drop gate at least as permissive as the defaults
    // (and loosen it if the caller asked for looser gates) so stricter caller
    // gates surface segments with Segment.suppressed = true instead of dropping
    // them inside whisper_full_with_state.
    const float eff_no_speech =
        options.decoding.no_speech_threshold.value_or(kDefaultNoSpeechThreshold);
    const float eff_logprob =
        options.decoding.logprob_threshold.value_or(kDefaultLogprobThreshold);
    params.no_speech_thold = std::max(kDefaultNoSpeechThreshold, eff_no_speech);
    params.logprob_thold = std::min(kDefaultLogprobThreshold, eff_logprob);

    params.token_timestamps = options.word_timestamps;
    params.tdrz_enable = options.diarize;
    params.new_segment_callback = on_new_segment;
    params.new_segment_callback_user_data = &run;
    params.abort_callback = on_abort;
    params.abort_callback_user_data = &run;

    const size_t eff_window_seconds = std::max<size_t>(1, options.window_seconds);
    const size_t eff_min_window_seconds =
        std::min(std::max<size_t>(1, options.min_window_seconds), eff_window_seconds);
    const size_t window_samples = eff_window_seconds * kModelSampleRate;
    const size_t min_window_samples = eff_min_window_seconds * kModelSampleRate;
    const auto stall_timeout =
        std::chrono::milliseconds(static_cast<int64_t>(options.window_stall_ms));
    const uint64_t max_samples =
        static_cast<uint64_t>(options.max_duration_seconds) * kModelSampleRate;

    std::vector<float> buffer;
    buffer.reserve(window_samples);
    uint64_t consumed_samples = 0;  // total samples pulled from the source
    bool source_done = false;
    std::string fallback_language;

    EngineResult result;

    while (true) {
        size_t fresh_samples = 0;
        // Fill the window. buffer may already hold the carried-over tail
        // of the previous window's unfinalized last segment.
        while (!source_done && buffer.size() < window_samples) {
            if (run.stopped()) {
                result.aborted = true;
                return result;
            }
            if (fresh_samples > 0 && buffer.size() >= min_window_samples && wait_for_more &&
                stall_timeout.count() > 0) {
                if (!wait_for_more(stall_timeout)) {
                    break;
                }
                if (run.stopped()) {
                    result.aborted = true;
                    return result;
                }
            }
            const size_t space = std::min(window_samples - buffer.size(), kFillSliceSamples);
            const size_t old_size = buffer.size();
            buffer.resize(old_size + space);
            const size_t got = read(buffer.data() + old_size, space);
            buffer.resize(old_size + got);
            if (got == 0) {
                source_done = true;
                break;
            }
            fresh_samples += got;
            consumed_samples += got;
            if (consumed_samples > max_samples) {
                throw DurationCapExceeded(
                    "media exceeds GRPC_ASR_MAX_DURATION_SECONDS=" +
                    std::to_string(options.max_duration_seconds));
            }
        }
        if (buffer.empty()) {
            break;
        }
        const bool last_window = source_done;

        // whisper refuses sub-second inputs; pad the final sliver with
        // silence rather than dropping committed speech.
        if (last_window && buffer.size() < kModelSampleRate) {
            buffer.resize(kModelSampleRate, 0.0f);
        }

        if (run.stopped()) {
            result.aborted = true;
            return result;
        }
        const auto decode_start = std::chrono::steady_clock::now();
        if (whisper_full_with_state(ctx, state, params, buffer.data(),
                                    static_cast<int>(buffer.size())) != 0) {
            if (run.stopped()) {
                result.aborted = true;
                return result;
            }
            throw std::runtime_error("whisper_full failed mid-transcription");
        }
        result.decode_wall_ns += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - decode_start)
                .count());
        result.windows++;
        if (run.stopped()) {
            result.aborted = true;
            return result;
        }

        if (fallback_language.empty()) {
            const int lang_id = whisper_full_lang_id_from_state(state);
            if (lang_id >= 0) {
                const char* code = whisper_lang_str(lang_id);
                if (code != nullptr) {
                    fallback_language = code;
                }
            }
        }

        // Finalize everything the window edge could not have cut: every
        // segment on the last window, all but the last otherwise, and
        // nothing from a window that decoded no segment at all.
        const int segments = whisper_full_n_segments_from_state(state);
        const uint64_t window_ms = buffer.size() * 1000ULL / kModelSampleRate;
        const uint64_t last_start_ms =
            segments >= 2
                ? centisec_to_ms(whisper_full_get_segment_t0_from_state(state, segments - 1))
                : 0;
        const WindowCut cut = cut_window(segments, last_window, window_ms, last_start_ms);
        if (segments > cut.finalize) {
            result.window_redecodes++;
        }

        uint32_t window_turns = 0;
        for (int i = 0; i < cut.finalize; i++) {
            EngineSegment segment = read_segment(run, state, i);
            result.tokens += segment.token_count;
            result.final_segments++;
            if (!segment.language.empty() &&
                std::find(result.languages.begin(), result.languages.end(), segment.language) ==
                    result.languages.end()) {
                result.languages.push_back(segment.language);
            }
            window_turns += segment.speaker_turn_next ? 1 : 0;
            if (!sink(segment, /*is_final=*/true)) {
                result.aborted = true;
                return result;
            }
        }
        run.window_base_index += static_cast<uint32_t>(cut.finalize);
        run.window_base_speaker += window_turns;

        if (last_window) {
            break;
        }

        // Carry the tail: resume the next window at the first unfinalized
        // sample (the held-back segment's own start, or the window end
        // when everything finalized).
        const uint64_t keep_from_sample =
            std::min<uint64_t>(cut.resume_ms * kModelSampleRate / 1000ULL, buffer.size());
        buffer.erase(buffer.begin(), buffer.begin() + static_cast<size_t>(keep_from_sample));
        run.window_base_ms += cut.resume_ms;
    }

    result.language =
        result.languages.empty() ? fallback_language : result.languages.front();
    result.duration_ms = consumed_samples * 1000ULL / kModelSampleRate;
    return result;
}

}  // namespace asr::engine
