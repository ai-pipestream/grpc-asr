#include "engine/transcriber.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <vector>

#include "media/audio_decoder.h"
#include "whisper.h"

namespace asr::engine {

namespace {

using media::kModelSampleRate;

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
        segment.speaker = "S" + std::to_string(speaker + 1);
        segment.speaker_turn_next =
            whisper_full_get_segment_speaker_turn_next_from_state(state, i);
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
    segment.avg_logprob =
        counted == 0 ? 0.0f : static_cast<float>(logprob_sum / static_cast<double>(counted));
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
    return static_cast<RunState*>(user_data)->abort.load();
}

}  // namespace

EngineResult Transcriber::run(whisper_context* ctx, whisper_state* state,
                              const EngineOptions& options, const PcmRead& read,
                              const SegmentSink& sink) {
    RunState run{.ctx = ctx, .options = &options, .sink = &sink};

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
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
    params.token_timestamps = options.word_timestamps;
    params.tdrz_enable = options.diarize;
    params.new_segment_callback = on_new_segment;
    params.new_segment_callback_user_data = &run;
    params.abort_callback = on_abort;
    params.abort_callback_user_data = &run;

    const size_t window_samples = options.window_seconds * kModelSampleRate;
    const uint64_t max_samples =
        static_cast<uint64_t>(options.max_duration_seconds) * kModelSampleRate;

    std::vector<float> buffer;
    buffer.reserve(window_samples);
    uint64_t consumed_samples = 0;  // total samples pulled from the source
    bool source_done = false;

    EngineResult result;

    while (true) {
        // Fill the window. buffer may already hold the carried-over tail
        // of the previous window's unfinalized last segment.
        while (!source_done && buffer.size() < window_samples) {
            size_t space = window_samples - buffer.size();
            size_t old_size = buffer.size();
            buffer.resize(old_size + space);
            size_t got = read(buffer.data() + old_size, space);
            buffer.resize(old_size + got);
            if (got == 0) {
                source_done = true;
                break;
            }
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

        if (whisper_full_with_state(ctx, state, params, buffer.data(),
                                    static_cast<int>(buffer.size())) != 0) {
            if (run.abort.load()) {
                result.aborted = true;
                return result;
            }
            throw std::runtime_error("whisper_full failed mid-transcription");
        }
        if (run.abort.load()) {
            result.aborted = true;
            return result;
        }

        if (result.language.empty()) {
            int lang_id = whisper_full_lang_id_from_state(state);
            if (lang_id >= 0) {
                result.language = whisper_lang_str(lang_id);
            }
        }

        const int segments = whisper_full_n_segments_from_state(state);
        // Finalize everything the window edge could not have cut: every
        // segment on the last window, all but the last otherwise.
        int finalize = last_window ? segments : segments - 1;

        // A single segment spanning the whole window means no progress
        // point to resume from; finalize it and move on rather than
        // re-decoding forever.
        if (!last_window && segments == 1) {
            finalize = 1;
        }

        uint32_t window_turns = 0;
        for (int i = 0; i < finalize; i++) {
            EngineSegment segment = read_segment(run, state, i);
            result.tokens += segment.token_count;
            result.final_segments++;
            window_turns += segment.speaker_turn_next ? 1 : 0;
            if (!sink(segment, /*is_final=*/true)) {
                result.aborted = true;
                return result;
            }
        }
        run.window_base_index += static_cast<uint32_t>(finalize);
        run.window_base_speaker += window_turns;

        if (last_window) {
            break;
        }

        // Carry the tail: resume the next window at the first unfinalized
        // sample (the cut segment's own start, or the window end when
        // everything finalized).
        uint64_t resume_ms;
        if (finalize < segments) {
            resume_ms = run.window_base_ms +
                        centisec_to_ms(whisper_full_get_segment_t0_from_state(state, finalize));
        } else {
            resume_ms = run.window_base_ms +
                        buffer.size() * 1000ULL / kModelSampleRate;
        }
        uint64_t keep_from_sample =
            (resume_ms - run.window_base_ms) * kModelSampleRate / 1000ULL;
        keep_from_sample = std::min<uint64_t>(keep_from_sample, buffer.size());
        buffer.erase(buffer.begin(), buffer.begin() + static_cast<size_t>(keep_from_sample));
        run.window_base_ms = resume_ms;
    }

    result.duration_ms = consumed_samples * 1000ULL / kModelSampleRate;
    return result;
}

}  // namespace asr::engine
