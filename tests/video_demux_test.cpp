// memfd + ffmpeg demux: probe, audio extraction, keyframes, and the
// video-without-audio case. The fixture mp4 is generated at test time by
// ffmpeg itself (testsrc2 video + sine audio) — nothing is committed.
// Skips with 77 when ffmpeg/ffprobe are not on PATH.

#include "media/video_demux.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <print>
#include <vector>

#include "fixture.h"
#include "media/audio_decoder.h"

using asr::media::AudioDecoder;
using asr::media::DecodeError;
using asr::media::VideoDemux;

namespace {

constexpr std::chrono::milliseconds kToolTimeout{60000};

bool have_ffmpeg() {
    return std::system("ffmpeg -version >/dev/null 2>&1") == 0 &&
           std::system("ffprobe -version >/dev/null 2>&1") == 0;
}

// Generates a fixture through ffmpeg into a temp file, slurps it, deletes
// it. The hot path under test stays diskless; only fixture *generation*
// touches a temp file, the same way a person would author a fixture.
std::string generate_media(const std::string& args, const std::string& suffix) {
    std::string path = "/tmp/grpc-asr-fixture-XXXXXX" + suffix;
    // mkstemps keeps the suffix so ffmpeg picks the muxer from it.
    int fd = mkstemps(path.data(), static_cast<int>(suffix.size()));
    require(fd >= 0, "temp fixture path");
    ::close(fd);
    std::string command = "ffmpeg -v error -y " + args + " " + path + " >/dev/null 2>&1";
    require(std::system(command.c_str()) == 0, "fixture generation: " + command);
    std::string bytes = slurp(path);
    std::remove(path.c_str());
    require(!bytes.empty(), "fixture has bytes");
    return bytes;
}

std::string make_av_mp4() {
    return generate_media(
        "-f lavfi -i testsrc2=duration=8:size=320x240:rate=10 "
        "-f lavfi -i sine=frequency=440:duration=8 -shortest -pix_fmt yuv420p", ".mp4");
}

std::string make_video_only_mp4() {
    return generate_media(
        "-f lavfi -i testsrc2=duration=4:size=320x240:rate=10 -an -pix_fmt yuv420p", ".mp4");
}

VideoDemux open(const std::string& media) {
    return VideoDemux(reinterpret_cast<const uint8_t*>(media.data()), media.size(), "ffmpeg",
                      "ffprobe", kToolTimeout);
}

void verify_probe(const std::string& media) {
    VideoDemux demux = open(media);
    asr::media::ProbeInfo info = demux.probe();
    require(info.has_audio, "fixture has audio");
    require(info.has_video, "fixture has video");
    require(info.audio_codec == "aac", "audio codec reported, got " + info.audio_codec);
    require(!info.video_codec.empty(), "video codec reported");
    require(info.duration_ms > 7000 && info.duration_ms < 9000,
            "duration close to 8s, got " + std::to_string(info.duration_ms));
    require(info.sample_rate_hz > 0 && info.channels > 0, "audio stream facts reported");
}

void verify_audio_extraction(const std::string& media) {
    VideoDemux demux = open(media);
    demux.open_audio();
    std::vector<float> pcm(16000);
    size_t total = 0;
    float peak = 0;
    while (true) {
        size_t got = demux.read_audio(pcm.data(), pcm.size());
        if (got == 0) {
            break;
        }
        for (size_t i = 0; i < got; i++) {
            peak = std::max(peak, std::abs(pcm[i]));
        }
        total += got;
    }
    demux.close_audio();
    require(total > 7 * asr::media::kModelSampleRate, "extracted close to 8s of PCM");
    require(peak > 0.1f, "the sine survived the demux");
}

void verify_keyframes(const std::string& media) {
    VideoDemux demux = open(media);
    size_t count = 0;
    uint64_t last_ts = 0;
    demux.extract_keyframes(2, [&](uint64_t timestamp_ms, uint32_t width, uint32_t height,
                                   std::string png) {
        require(width == 320 && height == 240, "keyframe dimensions match the fixture");
        require(png.size() > 100, "keyframe has PNG bytes");
        require(png.starts_with("\x89PNG"), "keyframe is a PNG");
        require(count == 0 || timestamp_ms > last_ts, "keyframe timestamps advance");
        last_ts = timestamp_ms;
        count++;
    });
    // 8 seconds at one frame per 2 seconds: allow the fencepost.
    require(count >= 3 && count <= 5,
            "keyframe count matches the interval, got " + std::to_string(count));
}

void verify_video_without_audio() {
    std::string media = make_video_only_mp4();
    VideoDemux demux = open(media);
    asr::media::ProbeInfo info = demux.probe();
    require(info.has_video, "video-only fixture has video");
    require(!info.has_audio, "video-only fixture reports no audio");
}

void verify_garbage_rejected() {
    std::string garbage(std::string("\x00\x00\x00\x20", 4) + "ftypisom");
    garbage.append(4096, '\x42');
    VideoDemux demux = open(garbage);
    bool threw = false;
    try {
        demux.probe();
    } catch (const DecodeError&) {
        threw = true;
    }
    require(threw, "a torn mp4 fails probe with DecodeError");
}

void verify_png_dimensions() {
    bool threw = false;
    try {
        uint32_t w = 0;
        uint32_t h = 0;
        asr::media::png_dimensions("not a png", &w, &h);
    } catch (const DecodeError&) {
        threw = true;
    }
    require(threw, "malformed PNG throws");
}

}  // namespace

int main() {
    if (!have_ffmpeg()) {
        return skip("ffmpeg/ffprobe not on PATH");
    }
    try {
        std::string media = make_av_mp4();
        verify_probe(media);
        verify_audio_extraction(media);
        verify_keyframes(media);
        verify_video_without_audio();
        verify_garbage_rejected();
        verify_png_dimensions();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("video-demux-test passed");
    return 0;
}
