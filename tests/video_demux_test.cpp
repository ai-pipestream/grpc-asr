// memfd + ffmpeg demux: probe, audio extraction, keyframes, and the
// video-without-audio case. The fixture mp4 is generated at test time by
// ffmpeg itself (testsrc2 video + sine audio) — nothing is committed.
// Skips with 77 when ffmpeg/ffprobe are not on PATH.

#include "media/video_demux.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <print>
#include <stop_token>
#include <string_view>
#include <thread>
#include <vector>

#include "cancellation.h"
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
        return true;
    });
    // 8 seconds at one frame per 2 seconds: allow the fencepost.
    require(count >= 3 && count <= 5,
            "keyframe count matches the interval, got " + std::to_string(count));
}

// No ffmpeg child may outlive the call that started it: once every child
// was reaped, waitpid has nobody left to report.
bool no_children_left() {
    int status = 0;
    return ::waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD;
}

void verify_keyframes_stop(const std::string& media) {
    VideoDemux demux = open(media);
    size_t count = 0;
    demux.extract_keyframes(1, [&](uint64_t, uint32_t, uint32_t, std::string) {
        count++;
        return false;  // the reader went away after the first still
    });
    require(count == 1, "a sink that returns false is sent no further stills");
    require(no_children_left(), "the stopped keyframe child was killed and reaped");
}

void verify_audio_cancel(const std::string& media) {
    VideoDemux demux = open(media);
    demux.open_audio();
    std::vector<float> pcm(1024);
    require(demux.read_audio(pcm.data(), pcm.size()) > 0, "the audio child streams PCM");
    // A reader that stops early leaves the child blocked on its pipe;
    // cancelling must neither wait it out nor judge its exit status.
    demux.cancel_audio();
    require(no_children_left(), "the cancelled audio child was killed and reaped");
    demux.close_audio();  // nothing left to close: a no-op, not an error
}

// A descriptor of this process whose /proc link starts with a prefix.
struct Descriptor {
    int fd;
    std::string target;
    bool cloexec;
};

// Descriptors already in `before` (the test's own stdout and stderr are
// pipes too) are left out.
std::vector<Descriptor> open_descriptors(std::string_view prefix,
                                         const std::vector<Descriptor>& before = {}) {
    std::vector<Descriptor> found;
    for (const auto& entry : std::filesystem::directory_iterator("/proc/self/fd")) {
        std::error_code error;
        const std::string target = std::filesystem::read_symlink(entry.path(), error).string();
        if (error || !target.starts_with(prefix)) {
            continue;
        }
        const int fd = std::stoi(entry.path().filename().string());
        const int flags = ::fcntl(fd, F_GETFD);
        const bool known = std::ranges::any_of(
            before, [&](const Descriptor& old) { return old.fd == fd && old.target == target; });
        if (flags >= 0 && !known) {
            found.push_back({fd, target, (flags & FD_CLOEXEC) != 0});
        }
    }
    return found;
}

// What a child forked right now inherits, as /proc lists its descriptors.
std::string inherited_by_a_child() {
    std::string listing;
    FILE* child = ::popen("ls -l /proc/self/fd/ 2>&1", "r");
    require(child != nullptr, "listing child started");
    char buf[4096];
    while (size_t n = std::fread(buf, 1, sizeof buf, child)) {
        listing.append(buf, n);
    }
    ::pclose(child);
    return listing;
}

void verify_descriptors_stay_private(const std::string& media) {
    // Another stream's fork must inherit none of this stream's pipe ends
    // or its memfd: a leaked pipe write end kept the reader here from ever
    // seeing end of file until the unrelated child exited (the read then
    // waited out the inactivity timeout and failed INTERNAL), and a leaked
    // memfd pinned this stream's media in memory.
    const std::vector<Descriptor> own_pipes = open_descriptors("pipe:");
    VideoDemux demux = open(media);
    demux.open_audio();
    const std::vector<Descriptor> memfds = open_descriptors("/memfd:grpc-asr-media");
    const std::vector<Descriptor> pipes = open_descriptors("pipe:", own_pipes);
    require(memfds.size() == 1, "the demux holds one media memfd");
    require(pipes.size() >= 2, "the audio child's stdout and stderr pipes are open");
    const std::string listing = inherited_by_a_child();
    for (const Descriptor& descriptor : memfds) {
        require(descriptor.cloexec, "the media memfd is close-on-exec");
        require(listing.find("grpc-asr-media") == std::string::npos,
                "an unrelated child inherits no media memfd");
    }
    for (const Descriptor& descriptor : pipes) {
        require(descriptor.cloexec, descriptor.target + " is close-on-exec");
        require(listing.find(descriptor.target) == std::string::npos,
                "an unrelated child inherits no " + descriptor.target);
    }

    // Sealed, as the docs say: the bytes the children read cannot change.
    const int seals = ::fcntl(memfds.front().fd, F_GET_SEALS);
    const int wanted = F_SEAL_WRITE | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL;
    require(seals >= 0 && (seals & wanted) == wanted, "the media memfd is sealed");
    require(::pwrite(memfds.front().fd, "x", 1, 0) < 0, "a write to the sealed media fails");
    demux.cancel_audio();
}

void verify_media_on_fd_3(const std::string& media) {
    // Children read the media on fd 3. When the memfd was itself given fd 3
    // (the lowest free descriptor), dup2(3, 3) is a no-op that leaves
    // close-on-exec set, and exec would close the descriptor the tool has
    // to read; the child moves its descriptors clear of 1-3 first.
    int parked = -1;
    if (::fcntl(3, F_GETFD) >= 0) {
        parked = ::fcntl(3, F_DUPFD_CLOEXEC, 10);
        ::close(3);
    }
    {
        VideoDemux demux = open(media);
        const std::vector<Descriptor> memfds = open_descriptors("/memfd:grpc-asr-media");
        require(memfds.size() == 1 && memfds.front().fd == 3, "the media memfd landed on fd 3");
        const asr::media::ProbeInfo info = demux.probe();
        require(info.has_audio && info.has_video, "the tool still reads the media on fd 3");
    }
    if (parked >= 0) {
        ::dup2(parked, 3);
        ::close(parked);
    }
}

// A stand-in tool that never writes a byte, like ffmpeg seeking or
// decoding toward its next still on a long video.
std::string write_silent_tool() {
    std::string path = "/tmp/grpc-asr-silent-tool-XXXXXX";
    int fd = mkstemp(path.data());
    require(fd >= 0, "temp tool path");
    const std::string script = "#!/bin/sh\nexec sleep 60\n";
    require(::write(fd, script.data(), script.size()) == static_cast<ssize_t>(script.size()),
            "tool script written");
    ::close(fd);
    require(::chmod(path.c_str(), 0700) == 0, "tool script executable");
    return path;
}

void verify_stop_reaches_a_silent_child(const std::string& media) {
    // The stop lands while the child produces nothing: the read must not
    // sit out the (60 s) inactivity timeout before noticing.
    const std::string tool = write_silent_tool();
    VideoDemux demux(reinterpret_cast<const uint8_t*>(media.data()), media.size(), "ffmpeg", tool,
                     kToolTimeout);
    std::stop_source stop;
    std::thread stopper([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        stop.request_stop();
    });
    const auto started = std::chrono::steady_clock::now();
    bool cancelled = false;
    try {
        demux.probe(stop.get_token());
    } catch (const asr::Cancelled&) {
        cancelled = true;
    }
    const auto waited = std::chrono::steady_clock::now() - started;
    stopper.join();
    std::remove(tool.c_str());
    require(cancelled, "a stopped probe throws Cancelled");
    require(waited < std::chrono::seconds(2),
            "the stop reached the silent child within a poll interval, not the inactivity timeout");
    require(no_children_left(), "the stopped child was killed and reaped");
}

void verify_stop_before_start(const std::string& media) {
    VideoDemux demux = open(media);
    std::stop_source stop;
    stop.request_stop();
    bool cancelled = false;
    try {
        demux.probe(stop.get_token());
    } catch (const asr::Cancelled&) {
        cancelled = true;
    }
    require(cancelled, "a probe for a stream that is already gone throws Cancelled");
    require(no_children_left(), "and never starts ffprobe");
}

void verify_stop_mid_audio(const std::string& media) {
    VideoDemux demux = open(media);
    std::stop_source stop;
    demux.open_audio(stop.get_token());
    std::vector<float> pcm(1024);
    require(demux.read_audio(pcm.data(), pcm.size()) > 0, "the audio child streams PCM");
    stop.request_stop();
    bool cancelled = false;
    try {
        while (demux.read_audio(pcm.data(), pcm.size()) != 0) {
        }
    } catch (const asr::Cancelled&) {
        cancelled = true;
    }
    require(cancelled, "the next audio read after a stop throws Cancelled");
    demux.cancel_audio();
    require(no_children_left(), "the stopped audio child was killed and reaped");
}

void verify_ogg_audio() {
    // Ogg goes through ffmpeg: miniaudio decodes neither codec here.
    for (const char* codec : {"libopus", "libvorbis"}) {
        const std::string ogg = generate_media(
            std::string("-f lavfi -i sine=frequency=440:duration=4 -c:a ") + codec, ".ogg");
        VideoDemux demux = open(ogg);
        const asr::media::ProbeInfo info = demux.probe();
        const std::string expected = std::string(codec).substr(3);  // opus, vorbis
        require(info.has_audio && info.audio_codec == expected,
                "ogg audio probed as " + expected + ", got " + info.audio_codec);
        require(!info.has_video, "audio-only ogg has no video");
        demux.open_audio();
        std::vector<float> pcm(16000);
        size_t total = 0;
        while (size_t got = demux.read_audio(pcm.data(), pcm.size())) {
            total += got;
        }
        demux.close_audio();
        require(total > 3 * asr::media::kModelSampleRate,
                expected + " in ogg decodes to PCM, got " + std::to_string(total) + " samples");
    }
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
        verify_keyframes_stop(media);
        verify_audio_cancel(media);
        verify_stop_reaches_a_silent_child(media);
        verify_stop_before_start(media);
        verify_stop_mid_audio(media);
        verify_descriptors_stay_private(media);
        verify_media_on_fd_3(media);
        verify_ogg_audio();
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
