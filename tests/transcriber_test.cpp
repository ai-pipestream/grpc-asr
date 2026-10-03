// Engine-level transcription over the fixture wav, including the
// anti-batch assertion: with a window smaller than the media, a final
// segment must be delivered before the PCM source is fully consumed.
// Whoever turns the window loop into one big batch breaks this test.
//
// Needs GRPC_ASR_TEST_MODEL (ggml weights) and GRPC_ASR_TEST_SAMPLE
// (whisper.cpp's samples/jfk.wav); skips with 77 otherwise.

#include "engine/transcriber.h"

#include <algorithm>
#include <cctype>
#include <print>
#include <vector>

#include "fixture.h"
#include "media/audio_decoder.h"
#include "whisper.h"

using asr::engine::EngineOptions;
using asr::engine::EngineResult;
using asr::engine::EngineSegment;
using asr::engine::Transcriber;
using asr::media::AudioDecoder;

namespace {

whisper_context* g_ctx = nullptr;
whisper_state* g_state = nullptr;

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return text;
}

struct Collected {
    std::vector<EngineSegment> partials;
    std::vector<EngineSegment> finals;
    bool final_before_eof = false;
    EngineResult result;
};

Collected run_over(const std::string& media, const EngineOptions& options) {
    asr::media::ByteStream stream;
    stream.append(media.data(), media.size());
    stream.complete();
    AudioDecoder decoder(stream, /*header_declares_duration=*/true);
    Collected collected;
    bool source_exhausted = false;
    collected.result = Transcriber::run(
        g_ctx, g_state, options,
        [&](float* out, size_t max_samples) {
            size_t got = decoder.read(out, max_samples);
            if (got == 0) {
                source_exhausted = true;
            }
            return got;
        },
        [&](const EngineSegment& segment, bool is_final) {
            if (is_final) {
                if (!source_exhausted) {
                    collected.final_before_eof = true;
                }
                collected.finals.push_back(segment);
            } else {
                collected.partials.push_back(segment);
            }
            return true;
        });
    return collected;
}

void verify_jfk(const std::string& jfk) {
    EngineOptions options;
    options.language = "en";
    options.threads = 4;
    // jfk.wav is ~11s; a 10s window forces at least two windows, which is
    // what makes the anti-batch assertion meaningful.
    options.window_seconds = 10;
    Collected collected = run_over(jfk, options);

    require(!collected.finals.empty(), "final segments were emitted");
    require(!collected.partials.empty(), "partial segments were emitted");
    require(collected.final_before_eof,
            "a final segment must arrive before the PCM source is exhausted; the "
            "window loop has become a batch");

    std::string transcript;
    for (const EngineSegment& segment : collected.finals) {
        transcript += segment.text;
    }
    require(lower(transcript).find("country") != std::string::npos,
            "transcript contains 'country', got: " + transcript);

    // Index and timestamp discipline.
    uint64_t last_start = 0;
    for (size_t i = 0; i < collected.finals.size(); i++) {
        const EngineSegment& segment = collected.finals[i];
        require(segment.index == i, "final indexes are dense and ordered");
        require(segment.start_ms >= last_start, "final start times never go backward");
        require(segment.end_ms >= segment.start_ms, "segment end is not before its start");
        last_start = segment.start_ms;
    }
    // Every final index was previewed as a partial first (live paint).
    for (const EngineSegment& segment : collected.finals) {
        bool previewed = false;
        for (const EngineSegment& partial : collected.partials) {
            if (partial.index == segment.index) {
                previewed = true;
                break;
            }
        }
        require(previewed, "final index " + std::to_string(segment.index) +
                               " was previewed as a partial");
    }
    require(collected.result.language == "en", "language detected as en");
    require(collected.result.final_segments == collected.finals.size(),
            "trailer count matches finals");
    require(collected.result.duration_ms > 10000 && collected.result.duration_ms < 12000,
            "duration close to 11s");
}

void verify_word_timestamps(const std::string& jfk) {
    EngineOptions options;
    options.language = "en";
    options.word_timestamps = true;
    options.window_seconds = 30;
    Collected collected = run_over(jfk, options);
    require(!collected.finals.empty(), "segments with word timestamps");
    size_t words = 0;
    for (const EngineSegment& segment : collected.finals) {
        for (const auto& word : segment.words) {
            require(word.end_ms >= word.start_ms, "word end not before start");
            require(!word.text.empty(), "word has text");
            words++;
        }
    }
    require(words >= 10, "the quote yields at least 10 timed words, got " +
                             std::to_string(words));
}

void verify_speaker_labels(const std::string& jfk) {
    EngineOptions options;
    options.language = "en";
    options.window_seconds = 30;
    Collected quiet = run_over(jfk, options);
    for (const EngineSegment& segment : quiet.finals) {
        require(segment.speaker.empty(), "no speaker is claimed without diarization");
        require(!segment.speaker_turn_next, "no turn is reported without diarization");
    }

    // With diarization on, every segment is attributed. A model without
    // speaker-turn training never predicts a turn, so this fixture stays
    // on one voice; the labels must still be there and consistent.
    options.diarize = true;
    Collected diarized = run_over(jfk, options);
    require(!diarized.finals.empty(), "diarization does not break decoding");
    std::string transcript;
    for (const EngineSegment& segment : diarized.finals) {
        require(segment.speaker.starts_with("S"), "every segment carries a speaker label");
        transcript += segment.text;
    }
    require(lower(transcript).find("country") != std::string::npos,
            "the transcript survives diarization, got: " + transcript);
}

void verify_silence() {
    EngineOptions options;
    options.language = "en";
    options.window_seconds = 30;
    Collected collected = run_over(make_wav(12.0, 0.0), options);
    // Digital silence must not fail; whatever whisper emits for it must
    // carry no words (a lone [BLANK_AUDIO] marker is acceptable).
    std::string transcript;
    for (const EngineSegment& segment : collected.finals) {
        transcript += segment.text;
    }
    for (char c : transcript) {
        require(!std::isalnum(static_cast<unsigned char>(c)) || transcript.find('[') != std::string::npos,
                "silence produced words: " + transcript);
    }
}

// Decodes a whole media file to mono f32 at the model rate.
std::vector<float> decode_pcm(const std::string& media) {
    asr::media::ByteStream stream;
    stream.append(media.data(), media.size());
    stream.complete();
    AudioDecoder decoder(stream, /*header_declares_duration=*/true);
    std::vector<float> pcm;
    std::vector<float> chunk(16000);
    while (size_t got = decoder.read(chunk.data(), chunk.size())) {
        pcm.insert(pcm.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(got));
    }
    return pcm;
}

void verify_empty_windows_before_speech(const std::string& jfk) {
    // 25 s of low hiss, then the spoken fixture, through 10 s windows.
    // whisper's no-speech gate decodes a speech-free window like the
    // first two into no segment at all (see make_hiss_wav). Such a window
    // used to finalize segment -1, stepping the index back and reading
    // whisper's segment array out of bounds for the resume point, so
    // everything after it came out misnumbered and mistimed.
    const std::vector<float> hiss = decode_pcm(make_hiss_wav(25.0));
    const std::vector<float> speech = decode_pcm(jfk);
    const size_t lead_in = hiss.size();
    const size_t total = lead_in + speech.size();
    const uint64_t total_ms = total * 1000ULL / asr::media::kModelSampleRate;
    EngineOptions options;
    options.language = "en";
    options.window_seconds = 10;
    std::vector<EngineSegment> finals;
    size_t position = 0;
    const EngineResult result = Transcriber::run(
        g_ctx, g_state, options,
        [&](float* out, size_t max_samples) {
            const size_t count = std::min(max_samples, total - position);
            for (size_t i = 0; i < count; i++, position++) {
                out[i] = position < lead_in ? hiss[position] : speech[position - lead_in];
            }
            return count;
        },
        [&](const EngineSegment& segment, bool is_final) {
            if (is_final) {
                finals.push_back(segment);
            }
            return true;
        });

    std::string transcript;
    uint64_t last_start = 0;
    for (size_t i = 0; i < finals.size(); i++) {
        const EngineSegment& segment = finals[i];
        transcript += segment.text;
        require(segment.index == i, "final indexes stay dense from 0 across empty windows, got " +
                                        std::to_string(segment.index) + " at " +
                                        std::to_string(i));
        require(segment.start_ms >= last_start && segment.end_ms <= total_ms + 1000,
                "finals stay ordered and inside the media, got " +
                    std::to_string(segment.start_ms) + ".." + std::to_string(segment.end_ms));
        last_start = segment.start_ms;
        // Nothing was spoken in the first two windows; whisper may time a
        // window's first words from the window start, so the bound is the
        // first window holding speech. Bracketed tags ("[BLANK_AUDIO]")
        // are whisper's non-speech markers, not words.
        const size_t first = segment.text.find_first_not_of(' ');
        const bool spoken = first != std::string::npos && segment.text[first] != '[' &&
                            segment.text[first] != '(';
        require(!spoken || segment.start_ms >= 20000,
                "speech is timed after the empty windows, got " + std::to_string(segment.start_ms));
    }
    require(lower(transcript).find("country") != std::string::npos,
            "the speech after the empty windows is transcribed, got: " + transcript);
    require(result.final_segments == finals.size(), "trailer count matches finals");
    require(result.duration_ms + 100 >= total_ms, "duration covers the lead-in too");
}

void verify_duration_cap() {
    EngineOptions options;
    options.language = "en";
    options.window_seconds = 10;
    options.max_duration_seconds = 5;
    bool threw = false;
    try {
        run_over(make_wav(12.0, 0.0), options);
    } catch (const asr::engine::DurationCapExceeded&) {
        threw = true;
    }
    require(threw, "media past the duration cap throws");
}

}  // namespace

int main() {
    const char* model = env_or_null("GRPC_ASR_TEST_MODEL");
    const char* sample = env_or_null("GRPC_ASR_TEST_SAMPLE");
    if (model == nullptr || slurp(model).empty()) {
        return skip("no model weights at $GRPC_ASR_TEST_MODEL");
    }
    std::string jfk = sample == nullptr ? "" : slurp(sample);
    if (jfk.empty()) {
        return skip("no fixture wav at $GRPC_ASR_TEST_SAMPLE");
    }

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    g_ctx = whisper_init_from_file_with_params_no_state(model, cparams);
    require(g_ctx != nullptr, "model loads");
    g_state = whisper_init_state(g_ctx);
    require(g_state != nullptr, "state allocates");

    try {
        verify_jfk(jfk);
        verify_word_timestamps(jfk);
        verify_speaker_labels(jfk);
        verify_silence();
        verify_empty_windows_before_speech(jfk);
        verify_duration_cap();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    whisper_free_state(g_state);
    whisper_free(g_ctx);
    std::println("transcriber-test passed");
    return 0;
}
