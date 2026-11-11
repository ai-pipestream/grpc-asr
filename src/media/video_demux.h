#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>

#include "media/byte_stream.h"

namespace asr::media {

// Thrown when ffmpeg/ffprobe themselves misbehave: missing binary, crash,
// or an inactivity timeout. The service maps this to INTERNAL. Bad media
// throws DecodeError (audio_decoder.h) instead.
class ToolError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// Stream-level facts ffprobe reports about a media container.
struct ProbeInfo {
    // Declared container duration, 0 when unknown.
    uint64_t duration_ms = 0;
    // True when the container has at least one audio stream.
    bool has_audio = false;
    // Audio codec name ("aac", "opus", ...), empty without audio.
    std::string audio_codec;
    // Audio source sample rate, 0 without audio.
    uint32_t sample_rate_hz = 0;
    // Audio channel count, 0 without audio.
    uint32_t channels = 0;
    // True when the container has at least one video stream.
    bool has_video = false;
    // Video codec name ("h264", "vp9", ...), empty without video.
    std::string video_codec;
};

// Demuxes a media container held in memory through ffmpeg without ever touching
// a filesystem:
//   - Pipe mode: streams bytes directly from an in-flight ByteStream into
//     `ffmpeg -i pipe:0` on a background feeder thread, with zero memfd copy.
//   - Borrowed-memfd mode: duplicates a ByteStream's already-sealed memfd so
//     the upload buffer and ffmpeg share the same kernel pages (1x RAM).
//   - Owned-memfd mode: creates and seals a memfd from a contiguous buffer.
class VideoDemux {
  public:
    // Creates and seals a fresh memfd from [data, data + size).
    VideoDemux(const uint8_t* data, size_t size, std::string ffmpeg_path,
               std::string ffprobe_path, std::chrono::milliseconds inactivity_timeout);

    // Borrows an existing sealed memfd (duplicated with FD_CLOEXEC; zero byte
    // copy). Caller retains ownership of `sealed_memfd`.
    VideoDemux(int sealed_memfd, std::string ffmpeg_path, std::string ffprobe_path,
               std::chrono::milliseconds inactivity_timeout);

    // Pipe-only demuxer (no memfd allocated); use open_audio_from_pipe().
    VideoDemux(std::string ffmpeg_path, std::string ffprobe_path,
               std::chrono::milliseconds inactivity_timeout);

    ~VideoDemux();

    VideoDemux(VideoDemux&&) noexcept;
    VideoDemux& operator=(VideoDemux&&) noexcept;
    VideoDemux(const VideoDemux&) = delete;
    VideoDemux& operator=(const VideoDemux&) = delete;

    // Runs ffprobe over the memfd container. Throws DecodeError when ffprobe
    // rejects the media, ToolError when ffprobe itself fails.
    ProbeInfo probe(std::stop_token stop = {});

    // Opens an ffmpeg child decoding the first audio track from the memfd to
    // mono f32 PCM at 16 kHz. When `channel` is set, extracts only that 0-based
    // source channel (`-af pan=mono|c0=c<N>`); otherwise mixes down (`-ac 1`).
    void open_audio(std::stop_token stop = {}, std::optional<uint32_t> channel = std::nullopt);

    // Opens an ffmpeg child reading encoded bytes from `stream` via `pipe:0`
    // while the upload is still in flight.
    void open_audio_from_pipe(ByteStream& stream, std::stop_token stop = {},
                              std::optional<uint32_t> channel = std::nullopt);

    size_t read_audio(float* out, size_t max_samples);

    // Non-blocking check returning true when stdout already has PCM bytes (or
    // EOF) ready so read_audio() will not block waiting for the source.
    bool has_buffered_audio() const;

    // Blocks up to `timeout` until stdout has PCM bytes (or EOF/stop) ready.
    bool wait_for_audio(std::chrono::milliseconds timeout, std::stop_token stop = {}) const;

    void close_audio();
    void cancel_audio();

    // Extracts one PNG still roughly every interval_seconds from the memfd.
    void extract_keyframes(
        uint32_t interval_seconds,
        const std::function<bool(uint64_t timestamp_ms, uint32_t width, uint32_t height,
                                 std::string png)>& sink,
        std::stop_token stop = {});

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Parses the IHDR width/height from a complete PNG byte string. Exposed
// for tests; throws DecodeError on a malformed PNG.
void png_dimensions(const std::string& png, uint32_t* width, uint32_t* height);

}  // namespace asr::media
