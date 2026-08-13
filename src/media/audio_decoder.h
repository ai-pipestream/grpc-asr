#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "media/byte_stream.h"

namespace asr::media {

// The whisper.cpp input contract: mono float32 at 16 kHz.
inline constexpr uint32_t kModelSampleRate = 16000;

// Thrown when the media bytes cannot be decoded (bad container, truncated
// stream, unsupported codec inside a known container). The service maps
// this to INVALID_ARGUMENT.
class DecodeError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// In-process decoder for the audio container families (wav/mp3/flac/ogg)
// producing mono f32 PCM at the model rate. Pulls encoded bytes from a
// ByteStream through blocking reads, so decoding (and therefore
// transcription) starts while the media is still uploading. Nothing
// touches the filesystem.
class AudioDecoder {
  public:
    // Container facts for the MediaInfo event.
    struct Info {
        // Total duration at the model rate. 0 when unknown — a container
        // that only reveals its length by scanning the whole stream
        // reports 0 while the upload is still in flight rather than
        // stalling the live path to find out.
        uint64_t duration_ms = 0;
        // Source sample rate before resampling.
        uint32_t sample_rate_hz = 0;
        // Source channel count before downmix.
        uint32_t channels = 0;
    };

    // Opens the stream, blocking until enough header bytes exist. wav_like
    // marks containers whose duration is declared in the header (wav), so
    // it is safe to query before the upload completes. Throws DecodeError
    // when the bytes cannot be opened as audio.
    AudioDecoder(ByteStream& stream, bool header_declares_duration);
    ~AudioDecoder();

    AudioDecoder(const AudioDecoder&) = delete;
    AudioDecoder& operator=(const AudioDecoder&) = delete;

    const Info& info() const { return info_; }

    // Decodes up to max_samples mono samples at the model rate into out,
    // blocking while the upload is behind. Returns the number written; 0
    // means end of stream (or an aborted upload — the caller checks the
    // ByteStream). Throws DecodeError on a mid-stream decode failure.
    size_t read(float* out, size_t max_samples);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Info info_;
};

}  // namespace asr::media
