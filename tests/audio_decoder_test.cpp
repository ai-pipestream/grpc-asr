// In-process audio decoding over in-memory fixtures, including the
// streaming case: decoding makes progress while the upload is still
// arriving.

#include "media/audio_decoder.h"

#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

#include "fixture.h"

using asr::media::AudioDecoder;
using asr::media::ByteStream;
using asr::media::DecodeError;
using asr::media::kModelSampleRate;

namespace {

// Completed-upload helper: the whole file is present before decoding.
void fill(ByteStream& stream, const std::string& bytes) {
    stream.append(bytes.data(), bytes.size());
    stream.complete();
}

void verify_wav_decode() {
    // 2 seconds of 440 Hz at 44.1 kHz: the decoder must resample to the
    // model rate and report the source rate.
    std::string wav = make_wav(2.0, 440.0, 44100);
    ByteStream stream;
    fill(stream, wav);
    AudioDecoder decoder(stream, /*header_declares_duration=*/true);
    require(decoder.info().sample_rate_hz == 44100, "source rate reported");
    require(decoder.info().channels == 1, "source channels reported");
    require(decoder.info().duration_ms > 1900 && decoder.info().duration_ms < 2100,
            "duration close to 2s, got " + std::to_string(decoder.info().duration_ms));

    std::vector<float> pcm(3 * kModelSampleRate);
    size_t total = 0;
    while (true) {
        size_t got = decoder.read(pcm.data() + total, 4096);
        if (got == 0) {
            break;
        }
        total += got;
    }
    require(total > 1900 * kModelSampleRate / 1000 && total < 2100 * kModelSampleRate / 1000,
            "sample count matches duration, got " + std::to_string(total));
    float peak = 0;
    for (size_t i = 0; i < total; i++) {
        peak = std::max(peak, std::abs(pcm[i]));
    }
    require(peak > 0.1f && peak <= 0.3f, "sine amplitude survived the resample");
}

void verify_streaming_decode() {
    // Feed the wav in small chunks from a producer thread; the decoder
    // must return PCM from the early chunks before the upload finishes.
    std::string wav = make_wav(4.0, 440.0);
    ByteStream stream;
    // Enough for the header and the first second of audio.
    size_t first_batch = 44 + 2 * kModelSampleRate;
    stream.append(wav.data(), first_batch);

    AudioDecoder decoder(stream, /*header_declares_duration=*/true);
    std::vector<float> pcm(kModelSampleRate / 2);
    size_t got = decoder.read(pcm.data(), pcm.size());
    require(got == pcm.size(), "a full read completes from the first chunks alone");
    require(!stream.is_complete(), "the upload is genuinely still open");

    // Trickle the rest from another thread while reads block on demand.
    std::thread producer([&] {
        size_t offset = first_batch;
        while (offset < wav.size()) {
            size_t n = std::min<size_t>(8192, wav.size() - offset);
            stream.append(wav.data() + offset, n);
            offset += n;
        }
        stream.complete();
    });
    size_t total = got;
    while (true) {
        size_t more = decoder.read(pcm.data(), pcm.size());
        if (more == 0) {
            break;
        }
        total += more;
    }
    producer.join();
    require(total > 3900 * kModelSampleRate / 1000, "the whole stream decoded, got " +
                                                        std::to_string(total));
}

void verify_aborted_upload() {
    std::string wav = make_wav(4.0, 440.0);
    ByteStream stream;
    stream.append(wav.data(), 44 + kModelSampleRate);
    AudioDecoder decoder(stream, /*header_declares_duration=*/true);
    stream.abort();
    // After an abort the decoder must wind down (end-of-data or a decode
    // error), never hang.
    std::vector<float> pcm(4 * kModelSampleRate);
    try {
        while (decoder.read(pcm.data(), pcm.size()) != 0) {
        }
    } catch (const DecodeError&) {
        // acceptable: truncation surfaced as a decode failure
    }
    require(stream.is_aborted(), "abort flag visible to the service layer");
}

void verify_silence() {
    ByteStream stream;
    fill(stream, make_wav(1.0, 0.0));
    AudioDecoder decoder(stream, /*header_declares_duration=*/true);
    std::vector<float> pcm(kModelSampleRate + 1024);
    size_t got = decoder.read(pcm.data(), pcm.size());
    require(got > 0, "silence decodes");
    for (size_t i = 0; i < got; i++) {
        require(pcm[i] == 0.0f, "silence is silent");
    }
}

void verify_bad_input() {
    ByteStream mp3_stream;
    fill(mp3_stream, make_truncated_mp3());
    bool threw = false;
    try {
        AudioDecoder decoder(mp3_stream, /*header_declares_duration=*/false);
    } catch (const DecodeError&) {
        threw = true;
    }
    require(threw, "frameless mp3 throws DecodeError");

    ByteStream torn_stream;
    fill(torn_stream, make_wav(1.0, 440.0).substr(0, 20));
    threw = false;
    try {
        AudioDecoder decoder(torn_stream, /*header_declares_duration=*/true);
    } catch (const DecodeError&) {
        threw = true;
    }
    require(threw, "torn wav header throws DecodeError");
}

}  // namespace

int main() {
    try {
        verify_wav_decode();
        verify_streaming_decode();
        verify_aborted_upload();
        verify_silence();
        verify_bad_input();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "audio-decoder-test passed\n";
    return 0;
}
