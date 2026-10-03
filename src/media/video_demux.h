#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <string>

namespace asr::media {

// Thrown when ffmpeg/ffprobe themselves misbehave: missing binary, crash,
// or an inactivity timeout. The service maps this to INTERNAL. Bad media
// throws DecodeError (audio_decoder.h) instead.
class ToolError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// Stream-level facts ffprobe reports about a video container.
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

// Demuxes a video container held in memory through ffmpeg without ever
// touching a filesystem: the encoded bytes live in a memfd (anonymous RAM
// file, seekable, so mp4 trailing-moov layouts work), and every child
// reads it via /dev/fd. PCM and PNG output stream back through pipes with
// backpressure, so memory stays bounded no matter the media length.
//
// Every child watches the stop token its call was given: a stop request
// kills the child within a fraction of a second and throws Cancelled from
// that call (from read_audio for the audio child), whether the child was
// streaming or silent.
class VideoDemux {
  public:
    // Copies nothing to disk: creates a memfd and writes the media bytes
    // into it. inactivity_timeout bounds how long a child may go without
    // producing output before it is killed.
    VideoDemux(const uint8_t* data, size_t size, std::string ffmpeg_path,
               std::string ffprobe_path, std::chrono::milliseconds inactivity_timeout);
    ~VideoDemux();

    VideoDemux(const VideoDemux&) = delete;
    VideoDemux& operator=(const VideoDemux&) = delete;

    // Runs ffprobe over the container. Throws DecodeError when ffprobe
    // rejects the media, ToolError when ffprobe itself fails.
    ProbeInfo probe(std::stop_token stop = {});

    // Opens an ffmpeg child decoding the first audio track to mono f32
    // PCM at the model rate. Read pulls samples with pipe backpressure;
    // returns 0 at end of stream. A non-zero ffmpeg exit surfaces as
    // DecodeError from read() or close_audio().
    void open_audio(std::stop_token stop = {});
    size_t read_audio(float* out, size_t max_samples);
    void close_audio();
    // Kills and reaps the audio child without reading its exit status.
    // For a reader that stopped early: the child was cut off mid-stream
    // (blocked on the full pipe, then killed), so its status says nothing
    // about the media. No-op when no child is open.
    void cancel_audio();

    // Extracts one PNG still roughly every interval_seconds, invoking the
    // sink per frame as it is parsed from the child's output stream. The
    // timestamp is the frame's position on the sampling grid. A sink that
    // returns false stops the extraction: the child is killed and the call
    // returns normally.
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
