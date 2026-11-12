// Engine-level transcription over the fixture wav, including the
// anti-batch assertion: with a window smaller than the media, a final
// segment must be delivered before the PCM source is fully consumed.
// Whoever turns the window loop into one big batch breaks this test.
//
// Needs GRPC_ASR_TEST_MODEL (ggml weights) and GRPC_ASR_TEST_SAMPLE
// (whisper.cpp's samples/jfk.wav); skips with 77 otherwise.

#include "engine/transcriber.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <print>
#include <thread>
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

void verify_stall_cut_default_window(const std::string& jfk) {
    // Issue #13: feed 40 s of PCM, stall the upload until a FinalSegment
    // arrives (bounded by 15 s so a regression fails promptly rather than
    // hanging), then deliver the rest using the default 480 s window
    // configuration (min_window_seconds = 30, window_stall_ms = 500). A
    // FinalSegment must arrive while the upload is still stalled, proving the
    // 500 ms stall threshold cut the 480 s window early without racing CPU
    // inference wall-clock speed.
    const std::vector<float> unit = decode_pcm(jfk);
    std::vector<float> full_pcm;
    while (full_pcm.size() < 52 * asr::media::kModelSampleRate) {
        full_pcm.insert(full_pcm.end(), unit.begin(), unit.end());
    }
    const size_t first_burst_samples = 40 * asr::media::kModelSampleRate;

    std::mutex mu;
    std::condition_variable cv;
    std::condition_variable stall_done_cv;
    size_t available = first_burst_samples;
    bool producer_done = false;
    bool in_stall = true;
    std::atomic<bool> final_during_stall{false};

    std::jthread producer([&] {
        {
            std::unique_lock<std::mutex> lock(mu);
            stall_done_cv.wait_for(lock, std::chrono::seconds(15), [&] {
                return final_during_stall.load(std::memory_order_acquire);
            });
            in_stall = false;
            available = full_pcm.size();
            producer_done = true;
        }
        cv.notify_all();
    });

    EngineOptions options;
    options.language = "en";
    options.threads = 4;
    options.window_seconds = 480;
    options.min_window_seconds = 30;
    options.window_stall_ms = 500;

    size_t cursor = 0;
    std::vector<EngineSegment> finals;
    const EngineResult result = Transcriber::run(
        g_ctx, g_state, options,
        [&](float* out, size_t max_samples) -> size_t {
            std::unique_lock<std::mutex> lock(mu);
            cv.wait(lock, [&] { return producer_done || available > cursor; });
            if (cursor >= available) {
                return 0;
            }
            const size_t count = std::min(max_samples, available - cursor);
            std::copy_n(full_pcm.data() + cursor, count, out);
            cursor += count;
            return count;
        },
        [&](const EngineSegment& segment, bool is_final) {
            if (is_final) {
                {
                    std::lock_guard<std::mutex> lock(mu);
                    if (in_stall) {
                        final_during_stall.store(true, std::memory_order_release);
                        stall_done_cv.notify_all();
                    }
                }
                finals.push_back(segment);
            }
            return true;
        },
        {},
        [&](std::chrono::milliseconds timeout) -> bool {
            std::unique_lock<std::mutex> lock(mu);
            return cv.wait_for(lock, timeout, [&] { return producer_done || available > cursor; });
        });

    require(final_during_stall.load(),
            "a FinalSegment must arrive during the upload stall with default 480 s window");
    require(!finals.empty(), "stall-cut run produced final segments");
    std::string transcript;
    for (size_t i = 0; i < finals.size(); ++i) {
        require(finals[i].index == i, "stall-cut final indexes remain dense");
        transcript += finals[i].text;
    }
    require(lower(transcript).find("country") != std::string::npos,
            "stall-cut transcript contains 'country', got: " + transcript);
    require(result.final_segments == finals.size(), "stall-cut trailer matches finals");
    require(result.windows >= 2, "stall-cut executed at least two windows");
}

void verify_decoding_options(const std::string& jfk) {
    // Issue #14:
    // 1. initial_prompt with a distinctive proper-noun spelling ("Amerikans")
    //    conditions the decoder to adopt that spelling across windows.
    {
        EngineOptions options;
        options.language = "en";
        options.threads = 4;
        options.window_seconds = 10;
        options.decoding.initial_prompt = "And so my fellow Amerikans, ask not.";
        Collected prompted = run_over(jfk, options);
        std::string transcript;
        for (const EngineSegment& segment : prompted.finals) {
            require(segment.has_no_speech_prob, "no_speech_prob populated on segment");
            require(segment.no_speech_prob >= 0.0f && segment.no_speech_prob <= 1.0f,
                    "no_speech_prob within [0, 1]");
            require(!segment.suppressed, "clear speech segment is not suppressed");
            transcript += segment.text;
        }
        require(lower(transcript).find("amerikans") != std::string::npos,
                "initial_prompt conditioned proper-noun spelling 'Amerikans', got: " +
                    transcript);
    }

    // 2. Beam search strategy produces the fixture text cleanly.
    {
        EngineOptions options;
        options.language = "en";
        options.threads = 4;
        options.window_seconds = 30;
        options.decoding.strategy = asr::engine::SamplingMode::kBeamSearch;
        options.decoding.beam_size = 3;
        Collected beamed = run_over(jfk, options);
        std::string transcript;
        for (const EngineSegment& segment : beamed.finals) {
            transcript += segment.text;
        }
        require(lower(transcript).find("country") != std::string::npos,
                "beam search transcript contains 'country', got: " + transcript);
    }

    // 3. Hiss fixture with default and explicit hallucination gates: when a
    //    segment appears on hiss/silence, no_speech_prob > 0.5 and suppressed.
    {
        EngineOptions options;
        options.language = "en";
        options.window_seconds = 30;
        Collected hiss_default = run_over(make_hiss_wav(6.0), options);
        for (const EngineSegment& segment : hiss_default.finals) {
            require(segment.has_no_speech_prob && segment.no_speech_prob > 0.5f,
                    "hiss segment reports high no_speech_prob");
            require(segment.suppressed, "hiss hallucinated segment is marked suppressed");
        }

        // Also deterministically trigger suppression on digital silence, where
        // whisper emits "[BLANK_AUDIO]" with high no_speech_prob and logprob < 0.
        options.decoding.no_speech_threshold = 0.4f;
        options.decoding.logprob_threshold = 0.0f;
        Collected gated_silence = run_over(make_wav(4.0, 0.0), options);
        require(!gated_silence.finals.empty(), "digital silence emits non-speech marker");
        for (const EngineSegment& segment : gated_silence.finals) {
            require(segment.has_no_speech_prob && segment.no_speech_prob > 0.5f,
                    "gated non-speech segment has no_speech_prob > 0.5");
            require(segment.suppressed, "gated non-speech segment has suppressed=true");
        }
    }
}

void verify_per_channel_labels(const std::string& jfk) {
    // Issue #15: channel=0 labels segments "C1" without diarize and "C1-S1"
    // with diarize.
    EngineOptions options;
    options.language = "en";
    options.window_seconds = 30;
    options.channel = 0;
    Collected ch0 = run_over(jfk, options);
    require(!ch0.finals.empty(), "per-channel run produced segments");
    for (const EngineSegment& segment : ch0.finals) {
        require(segment.channel == std::optional<uint32_t>(0), "segment carries channel 0");
        require(segment.speaker == "C1", "segment speaker is C1, got: " + segment.speaker);
    }

    options.diarize = true;
    options.channel = 1;
    Collected ch1_diarize = run_over(jfk, options);
    require(!ch1_diarize.finals.empty(), "per-channel + diarize run produced segments");
    for (const EngineSegment& segment : ch1_diarize.finals) {
        require(segment.channel == std::optional<uint32_t>(1), "segment carries channel 1");
        require(segment.speaker.starts_with("C2-S"),
                "per-channel + diarize speaker starts with C2-S, got: " + segment.speaker);
    }
}

void verify_language_switch(const std::string& jfk) {
    // Issue #16: concatenate the ~11 s English fixture (padded to 10 s boundary)
    // with the committed Spanish fixture on the multilingual model using a 10 s
    // window, and assert per-segment language switches to "es" and the trailer
    // lists both ["en", "es"] with language == "en".
    const char* multi_model = env_or_null("GRPC_ASR_TEST_MULTILINGUAL_MODEL");
    const char* es_sample = env_or_null("GRPC_ASR_TEST_ES_SAMPLE");
    if (multi_model == nullptr || slurp(multi_model).empty() || es_sample == nullptr ||
        slurp(es_sample).empty()) {
        std::println(stderr, "note: skipping multilingual switch sub-test (no multilingual model)");
        return;
    }

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    whisper_context* mctx = whisper_init_from_file_with_params_no_state(multi_model, cparams);
    require(mctx != nullptr, "multilingual model loads");
    whisper_state* mstate = whisper_init_state(mctx);
    require(mstate != nullptr, "multilingual state allocates");

    try {
        std::vector<float> en_pcm = decode_pcm(jfk);
        // Pad English part to 20 s (two clean 10 s windows, trailing silence so
        // cut_window does not hold English speech into the Spanish window).
        if (en_pcm.size() < 20 * asr::media::kModelSampleRate) {
            en_pcm.resize(20 * asr::media::kModelSampleRate, 0.0f);
        }
        const std::vector<float> es_pcm = decode_pcm(slurp(es_sample));
        require(!es_pcm.empty(), "Spanish fixture decoded to PCM");

        std::vector<float> combined = en_pcm;
        combined.insert(combined.end(), es_pcm.begin(), es_pcm.end());

        EngineOptions options;
        options.language.clear();  // autodetect per window
        options.threads = 4;
        options.window_seconds = 10;

        size_t pos = 0;
        std::vector<EngineSegment> finals;
        const EngineResult res = Transcriber::run(
            mctx, mstate, options,
            [&](float* out, size_t max_samples) -> size_t {
                const size_t n = std::min(max_samples, combined.size() - pos);
                std::copy_n(combined.data() + pos, n, out);
                pos += n;
                return n;
            },
            [&](const EngineSegment& segment, bool is_final) {
                if (is_final) {
                    finals.push_back(segment);
                }
                return true;
            });

        bool saw_en = false;
        bool saw_es_after_switch = false;
        for (const EngineSegment& seg : finals) {
            if (seg.start_ms < 15000 && seg.language == "en") {
                saw_en = true;
            }
            if (seg.start_ms >= 20000 && seg.language == "es") {
                saw_es_after_switch = true;
            }
        }
        require(saw_en, "English window segments tagged with language 'en'");
        require(saw_es_after_switch,
                "segments after the 20 s switch tagged with language 'es'");
        require(res.language == "en", "trailer primary language stays first ('en'), got: " +
                                          res.language);
        require(std::find(res.languages.begin(), res.languages.end(), "en") !=
                    res.languages.end(),
                "trailer languages lists 'en'");
        require(std::find(res.languages.begin(), res.languages.end(), "es") !=
                    res.languages.end(),
                "trailer languages lists 'es'");
    } catch (...) {
        whisper_free_state(mstate);
        whisper_free(mctx);
        throw;
    }
    whisper_free_state(mstate);
    whisper_free(mctx);
}

}  // namespace

int main() {
    const char* model = env_or_null("GRPC_ASR_TEST_MODEL");
    const char* sample = env_or_null("GRPC_ASR_TEST_SAMPLE");
    if (model == nullptr || slurp(model).empty()) {
        return missing_model_input("no model weights at $GRPC_ASR_TEST_MODEL");
    }
    std::string jfk = sample == nullptr ? "" : slurp(sample);
    if (jfk.empty()) {
        return missing_model_input("no fixture wav at $GRPC_ASR_TEST_SAMPLE");
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
        verify_stall_cut_default_window(jfk);
        verify_decoding_options(jfk);
        verify_per_channel_labels(jfk);
        verify_language_switch(jfk);
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    whisper_free_state(g_state);
    whisper_free(g_ctx);
    std::println("transcriber-test passed");
    return 0;
}
