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
#include <cmath>
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
#include "media/sniff.h"
#include "metrics.h"

using asr::media::AudioDecoder;
using asr::media::DecodeError;
using asr::media::VideoDemux;

namespace {

constexpr std::chrono::milliseconds kToolTimeout{60000};

bool have_ffmpeg() {
    const std::string generator = fixture_ffmpeg() + " -version >/dev/null 2>&1";
    return std::system("ffmpeg -version >/dev/null 2>&1") == 0 &&
           std::system("ffprobe -version >/dev/null 2>&1") == 0 &&
           std::system(generator.c_str()) == 0;
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
    std::string command =
        fixture_ffmpeg() + " -v error -y " + args + " " + path + " >/dev/null 2>&1";
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

// Reads the audio to the end and returns what ffmpeg said on the way.
asr::media::AudioDecodeReport decode_report(const std::string& media, size_t* samples = nullptr) {
    VideoDemux demux = open(media);
    demux.open_audio();
    std::vector<float> pcm(16000);
    size_t total = 0;
    while (true) {
        const size_t got = demux.read_audio(pcm.data(), pcm.size());
        if (got == 0) {
            break;
        }
        total += got;
    }
    if (samples != nullptr) {
        *samples = total;
    }
    return demux.close_audio();
}

// #38: ffmpeg skips what it cannot read, says so on stderr and exits 0.
// What it said used to be read and dropped, so a file cut short or corrupt
// in the middle came back as a clean, shorter decode.
void verify_decode_report() {
    const std::string webm = generate_media(
        "-f lavfi -i sine=frequency=440:duration=8 -c:a libopus", ".webm");
    {
        const asr::media::AudioDecodeReport report = decode_report(webm);
        require(report.error_lines == 0 && report.first_error.empty(),
                "a clean decode reports nothing, got " + std::to_string(report.error_lines) +
                    " line(s): " + report.first_error);
    }
    {
        size_t samples = 0;
        const asr::media::AudioDecodeReport report =
            decode_report(webm.substr(0, webm.size() * 6 / 10), &samples);
        require(samples > asr::media::kModelSampleRate,
                "the readable part of a cut webm is still decoded");
        require(report.error_lines >= 1, "a cut webm reports ffmpeg's error lines");
        require(!report.first_error.empty(), "the first error line is kept");
        require(report.first_error.find(" @ 0x") == std::string::npos,
                "the context address is dropped: " + report.first_error);
        require(report.first_error.find('\n') == std::string::npos &&
                    report.first_error.size() <= 512,
                "the first error is one bounded line");
    }
    {
        // Garbage over the middle of the file: many packets are lost, and
        // every line is counted though only the first is kept.
        std::string corrupt = webm;
        for (size_t i = corrupt.size() / 2; i < corrupt.size() / 2 + corrupt.size() / 5; ++i) {
            corrupt[i] = static_cast<char>(0xA5 ^ (i * 31));
        }
        size_t samples = 0;
        const asr::media::AudioDecodeReport report = decode_report(corrupt, &samples);
        require(samples > asr::media::kModelSampleRate,
                "the readable part of a corrupt webm is still decoded");
        require(report.error_lines >= 1,
                "a webm with a corrupt middle reports ffmpeg's error lines");
        require(!report.first_error.empty(), "the first error line of a corrupt webm is kept");
    }
    {
        // A fast-start mp4 keeps its index when its tail is cut: the audio
        // stream still says 8 s, which is what the decode is held against.
        const std::string mp4 = generate_media(
            "-f lavfi -i sine=frequency=440:duration=8 -c:a aac -movflags +faststart", ".mp4");
        const std::string cut = mp4.substr(0, mp4.size() * 6 / 10);
        VideoDemux demux = open(cut);
        const asr::media::ProbeInfo info = demux.probe();
        require(info.audio_duration_ms > 7900 && info.audio_duration_ms < 8200,
                "a cut mp4 still states its audio stream's 8 s, got " +
                    std::to_string(info.audio_duration_ms));
        size_t samples = 0;
        const asr::media::AudioDecodeReport report = decode_report(cut, &samples);
        require(samples < 7 * asr::media::kModelSampleRate, "the cut mp4 decodes less than 8 s");
        require(report.error_lines >= 1, "a cut mp4 reports ffmpeg's error lines");
    }
}

// A stand-in ffmpeg that writes one second of PCM, says `stderr_text` and
// exits 0: what ffmpeg prints is then exactly what the test chose.
asr::media::AudioDecodeReport report_of_stderr(const std::string& stderr_text) {
    std::string said = "/tmp/grpc-asr-said-XXXXXX";
    const int said_fd = mkstemp(said.data());
    require(said_fd >= 0, "temp stderr text path");
    require(::write(said_fd, stderr_text.data(), stderr_text.size()) ==
                static_cast<ssize_t>(stderr_text.size()),
            "stderr text written");
    ::close(said_fd);
    std::string tool = "/tmp/grpc-asr-talking-tool-XXXXXX";
    const int tool_fd = mkstemp(tool.data());
    require(tool_fd >= 0, "temp tool path");
    const std::string script =
        "#!/bin/sh\nhead -c 64000 /dev/zero\ncat " + said + " >&2\nexit 0\n";
    require(::write(tool_fd, script.data(), script.size()) ==
                static_cast<ssize_t>(script.size()),
            "tool script written");
    ::close(tool_fd);
    require(::chmod(tool.c_str(), 0700) == 0, "tool script executable");

    const std::string media = "not read by the stand-in";
    VideoDemux demux(reinterpret_cast<const uint8_t*>(media.data()), media.size(), tool,
                     "ffprobe", std::chrono::seconds(10));
    demux.open_audio();
    std::vector<float> pcm(16000);
    size_t total = 0;
    while (true) {
        const size_t got = demux.read_audio(pcm.data(), pcm.size());
        if (got == 0) {
            break;
        }
        total += got;
    }
    const asr::media::AudioDecodeReport report = demux.close_audio();
    ::unlink(tool.c_str());
    ::unlink(said.c_str());
    require(total == 16000, "the stand-in's second of PCM was read whole");
    return report;
}

// The raw PCM muxer writes no timestamps, so when libavformat's timestamp
// check complains on its behalf the packet is written all the same: every
// sample is there. An AMR file read by an ffmpeg built without the AMR
// parser gets one such line per packet, and so does any recording whose
// clock restarts. Those lines say nothing about the audio, and counting
// them reported whole media as damaged.
void verify_output_timestamp_notes_are_not_decode_errors() {
    const std::string note_a =
        "[f32le @ 0x5618e67f5e00] Application provided invalid, non monotonically "
        "increasing dts to muxer in stream 0: 9888 >= 288\n";
    const std::string note_b =
        "[f32le @ 0x5618e67f5e00] Application provided invalid, non monotonically "
        "increasing dts to muxer in stream 0: 9888 >= 608\n";
    const std::string note_c = "[f32le @ 0x5618e67f5e00] pts (288) < dts (608) in stream 0\n";
    const std::string real =
        "[opus @ 0x55d0c1e2f940] Error parsing the packet header.\n";
    {
        const asr::media::AudioDecodeReport report = report_of_stderr(note_a + note_b + note_c);
        require(report.error_lines == 0 && report.first_error.empty(),
                "timestamp notes from the raw PCM muxer are not decode errors, got " +
                    std::to_string(report.error_lines) + " line(s): " + report.first_error);
    }
    {
        // A real complaint among them still counts, once, and it is the
        // one kept: not the note that happened to be printed first.
        const asr::media::AudioDecodeReport report =
            report_of_stderr(note_a + real + note_b + real);
        require(report.error_lines == 2,
                "decoder errors between the notes are counted, got " +
                    std::to_string(report.error_lines));
        require(report.first_error == "[opus] Error parsing the packet header.",
                "the first error kept is the decoder's, got: " + report.first_error);
    }
    {
        // The same words from anything but the raw PCM muxer are not this
        // case: a muxer that does write timestamps drops the packet.
        const asr::media::AudioDecodeReport report = report_of_stderr(
            "[matroska @ 0x1] Application provided invalid, non monotonically increasing "
            "dts to muxer in stream 0: 9888 >= 288\n");
        require(report.error_lines == 1,
                "only the raw PCM muxer's timestamp notes are set aside");
    }
    {
        // A last line with no newline is still a line, and a long first
        // line is still cut at the bound.
        const asr::media::AudioDecodeReport report =
            report_of_stderr("[aac @ 0x2] " + std::string(2000, 'x'));
        require(report.error_lines == 1 && report.first_error.size() == 512 &&
                    report.first_error.starts_with("[aac] xxx"),
                "an unterminated long line counts once and is bounded, got " +
                    std::to_string(report.error_lines) + " line(s), " +
                    std::to_string(report.first_error.size()) + " bytes");
    }
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

void verify_stray_descriptors_stay_out(const std::string& media) {
    // A descriptor some library opened without close-on-exec must not
    // reach the tool: the child closes everything above fd 3 before exec.
    // The stand-in ffprobe records the descriptors it was started with.
    std::string listing_path = "/tmp/grpc-asr-tool-fds-XXXXXX";
    int listing_fd = mkstemp(listing_path.data());
    require(listing_fd >= 0, "temp listing path");
    ::close(listing_fd);
    std::string tool = "/tmp/grpc-asr-fd-tool-XXXXXX";
    int fd = mkstemp(tool.data());
    require(fd >= 0, "temp tool path");
    const std::string script = "#!/bin/sh\nls /proc/$$/fd > " + listing_path + "\n";
    require(::write(fd, script.data(), script.size()) == static_cast<ssize_t>(script.size()),
            "tool script written");
    ::close(fd);
    require(::chmod(tool.c_str(), 0700) == 0, "tool script executable");

    // High enough to stay clear of the descriptors sh opens for itself.
    const int null_fd = ::open("/dev/null", O_RDONLY);
    require(null_fd >= 0, "/dev/null opened");
    const int stray = ::fcntl(null_fd, F_DUPFD, 100);
    ::close(null_fd);
    require(stray >= 100, "stray descriptor without close-on-exec");

    VideoDemux demux(reinterpret_cast<const uint8_t*>(media.data()), media.size(), "ffmpeg", tool,
                     kToolTimeout);
    demux.probe();
    ::close(stray);
    const std::string listing = slurp(listing_path);
    std::remove(listing_path.c_str());
    std::remove(tool.c_str());

    std::vector<int> inherited;
    size_t start = 0;
    while (start < listing.size()) {
        size_t end = listing.find('\n', start);
        if (end == std::string::npos) {
            end = listing.size();
        }
        if (end > start) {
            inherited.push_back(std::stoi(listing.substr(start, end - start)));
        }
        start = end + 1;
    }
    require(std::ranges::find(inherited, 3) != inherited.end(), "the tool still gets the media");
    require(std::ranges::find(inherited, stray) == inherited.end(),
            "a descriptor without close-on-exec stays out of the tool, got: " + listing);
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

void verify_non_finite_samples_scrubbed() {
    // A float PCM track in a mov carries its bit patterns verbatim through
    // ffmpeg: NaN in, NaN out on the f32le pipe. The demux path hands the
    // model the same contract the in-process decoder does, finite samples
    // in [-1, 1], so the scrub has to happen on this side of the pipe too.
    std::string wav_path = "/tmp/grpc-asr-fixture-XXXXXX.wav";
    int fd = mkstemps(wav_path.data(), 4);
    require(fd >= 0, "temp wav path");
    {
        const std::string wav = make_float_wav(hostile_float_samples());
        require(::write(fd, wav.data(), wav.size()) == static_cast<ssize_t>(wav.size()),
                "wrote the float wav");
    }
    ::close(fd);
    const std::string media = generate_media("-i " + wav_path + " -c:a pcm_f32le", ".mov");
    std::remove(wav_path.c_str());

    VideoDemux demux = open(media);
    const asr::media::ProbeInfo info = demux.probe();
    require(info.has_audio && info.audio_codec == "pcm_f32le",
            "float pcm track probed, got " + info.audio_codec);
    demux.open_audio();
    std::vector<float> pcm(16000);
    size_t total = 0;
    size_t bad = 0;
    while (size_t got = demux.read_audio(pcm.data(), pcm.size())) {
        for (size_t i = 0; i < got; i++) {
            if (!std::isfinite(pcm[i]) || pcm[i] < -1.0f || pcm[i] > 1.0f) {
                bad++;
            }
        }
        total += got;
    }
    demux.close_audio();
    require(total > 15000, "the whole second demuxed, got " + std::to_string(total));
    require(bad == 0, std::to_string(bad) + " samples reached the caller non-finite or out of range");
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

uint64_t read_rss_bytes() {
    std::ifstream statm("/proc/self/statm");
    uint64_t total_pages = 0;
    uint64_t resident_pages = 0;
    if (statm >> total_pages >> resident_pages) {
        const long page_size = ::sysconf(_SC_PAGESIZE);
        return resident_pages * static_cast<uint64_t>(page_size > 0 ? page_size : 4096);
    }
    return 0;
}

void verify_pipe_streaming_and_single_copy() {
    // ~12.8 MiB Matroska fixture (100 s stereo 32-bit PCM at 16 kHz = 12.8 MB)
    // so any duplicate full-file memfd copy would jump RSS and memfd count.
    const std::string mkv = generate_media(
        "-f lavfi -i sine=frequency=440:duration=100 -ac 2 -ar 16000 -c:a pcm_f32le", ".mkv");
    require(mkv.size() > 10 * 1024 * 1024, "large mkv fixture > 10 MiB");

    const auto base_metrics = asr::metrics_summary();
    const uint64_t rss_before = read_rss_bytes();

    {
        asr::media::ByteStream stream;
        // Append the first half only: open_audio_from_pipe must stream PCM while
        // the second half has not been uploaded yet.
        const size_t half = mkv.size() / 2;
        stream.append(mkv.data(), half);

        VideoDemux demux("ffmpeg", "ffprobe", kToolTimeout);
        demux.open_audio_from_pipe(stream);

        std::vector<float> pcm(16000);
        const size_t early_samples = demux.read_audio(pcm.data(), pcm.size());
        require(early_samples > 0, "pipe demux produces PCM while upload is still in flight");

        // Zero memfds exist on the pipe path.
        require(open_descriptors("/memfd:grpc-asr-media").empty(),
                "pipe demux allocates zero media memfds");
        const auto mid_metrics = asr::metrics_summary();
        require(mid_metrics.memfd_bytes_resident == base_metrics.memfd_bytes_resident,
                "pipe demux adds 0 bytes to memfd gauge");

        // Feed the second half and finish.
        stream.append(mkv.data() + half, mkv.size() - half);
        stream.complete();

        const uint64_t rss_peak = read_rss_bytes();
        if (rss_peak > rss_before) {
            const uint64_t rss_delta = rss_peak - rss_before;
            require(rss_delta < mkv.size() * 2 + (8ULL << 20),
                    "pipe demux RSS growth stays within single-copy budget: delta=" +
                        std::to_string(rss_delta) + " media=" + std::to_string(mkv.size()));
        }

        size_t total = early_samples;
        while (size_t got = demux.read_audio(pcm.data(), pcm.size())) {
            total += got;
        }
        demux.close_audio();
        require(total > 95 * asr::media::kModelSampleRate, "full 100 s decoded over pipe");
        require(no_children_left(), "pipe ffmpeg child reaped cleanly");
    }

    // Borrowed-memfd path: ByteStream switches to memfd, freeing heap copy so
    // only one copy lives in the memfd shared with VideoDemux.
    {
        asr::media::ByteStream memfd_stream;
        memfd_stream.append(mkv.data(), 4096);
        memfd_stream.switch_to_memfd();
        memfd_stream.append(mkv.data() + 4096, mkv.size() - 4096);
        memfd_stream.complete();

        VideoDemux borrowed(memfd_stream.memfd(), "ffmpeg", "ffprobe", kToolTimeout);
        const auto m = asr::metrics_summary();
        require(m.upload_bytes_resident == base_metrics.upload_bytes_resident,
                "borrowed-memfd mode keeps 0 heap upload bytes");
        require(m.memfd_bytes_resident == base_metrics.memfd_bytes_resident + mkv.size(),
                "borrowed-memfd mode keeps strictly 1x media bytes in memfd");
        const asr::media::ProbeInfo info = borrowed.probe();
        require(info.has_audio && info.channels == 2, "borrowed memfd probes cleanly");
    }
}

void verify_new_routed_families() {
    using asr::media::MediaFamily;
    // AVI (mp3), ASF (wmav2), MPEG-PS (mp2), MPEG-TS (aac), raw ADTS AAC, and AMR-NB.
    struct Case {
        const char* label;
        std::string bytes;
        MediaFamily expected_family;
        bool pipeable;
    };
    std::vector<Case> cases;
    cases.push_back(
        {"avi",
         generate_media("-f lavfi -i sine=frequency=440:duration=2 -c:a libmp3lame", ".avi"),
         MediaFamily::kAvi, false});
    cases.push_back(
        {"asf", generate_media("-f lavfi -i sine=frequency=440:duration=2 -c:a wmav2", ".asf"),
         MediaFamily::kAsf, false});
    cases.push_back(
        {"mpegps", generate_media("-f lavfi -i sine=frequency=440:duration=2 -c:a mp2", ".mpg"),
         MediaFamily::kMpegPs, true});
    cases.push_back(
        {"mpegts", generate_media("-f lavfi -i sine=frequency=440:duration=2 -c:a aac", ".ts"),
         MediaFamily::kMpegTs, true});
    cases.push_back({"adts",
                     generate_media(
                         "-f lavfi -i sine=frequency=440:duration=2 -c:a aac -f adts", ".aac"),
                     MediaFamily::kAac, true});

    // Author 2 s of 440 Hz 8 kHz AMR-NB in memory via OpenCORE AMR-NB.
    std::vector<int16_t> tone_8k(16000);
    for (size_t i = 0; i < tone_8k.size(); ++i) {
        tone_8k[i] = static_cast<int16_t>(
            8000.0 * std::sin(2.0 * 3.14159265358979 * 440.0 * static_cast<double>(i) / 8000.0));
    }
    cases.push_back({"amr", encode_amr_nb_8k(tone_8k), MediaFamily::kAmr, true});

    for (const Case& c : cases) {
        const MediaFamily got_family = asr::media::sniff(
            reinterpret_cast<const uint8_t*>(c.bytes.data()),
            std::min(c.bytes.size(), asr::media::kSniffPrefixBytes));
        require(got_family == c.expected_family,
                std::string(c.label) + " sniffs as expected family");

        std::vector<float> all_pcm;
        std::vector<float> buf(8192);
        if (c.pipeable) {
            asr::media::ByteStream stream;
            stream.append(c.bytes.data(), c.bytes.size());
            stream.complete();
            VideoDemux demux("ffmpeg", "ffprobe", kToolTimeout);
            demux.open_audio_from_pipe(stream);
            while (size_t n = demux.read_audio(buf.data(), buf.size())) {
                all_pcm.insert(all_pcm.end(), buf.begin(), buf.begin() + n);
            }
            demux.close_audio();
        } else {
            VideoDemux demux = open(c.bytes);
            const asr::media::ProbeInfo info = demux.probe();
            require(info.has_audio, std::string(c.label) + " probe reports audio");
            demux.open_audio();
            while (size_t n = demux.read_audio(buf.data(), buf.size())) {
                all_pcm.insert(all_pcm.end(), buf.begin(), buf.begin() + n);
            }
            demux.close_audio();
        }
        require(all_pcm.size() > asr::media::kModelSampleRate,
                std::string(c.label) + " decodes > 1 s of 16 kHz PCM, got " +
                    std::to_string(all_pcm.size()));
    }
}

void verify_stereo_per_channel() {
    // Stereo Matroska with lossless 16-bit PCM: 440 Hz on left (ch 0), 880 Hz on right (ch 1).
    std::string wav_path = "/tmp/grpc-asr-stereo-XXXXXX.wav";
    int fd = mkstemps(wav_path.data(), 4);
    require(fd >= 0, "temp stereo wav path");
    {
        const std::string stereo = make_multichannel_wav(2.0, {440.0, 880.0});
        require(::write(fd, stereo.data(), stereo.size()) == static_cast<ssize_t>(stereo.size()),
                "stereo wav written");
    }
    ::close(fd);
    const std::string mkv = generate_media("-i " + wav_path + " -c:a pcm_s16le", ".mkv");
    std::remove(wav_path.c_str());

    VideoDemux demux = open(mkv);
    const asr::media::ProbeInfo info = demux.probe();
    require(info.channels == 2, "stereo mkv probed with 2 channels");

    auto read_channel = [&](uint32_t ch) {
        demux.open_audio({}, ch);
        std::vector<float> out;
        std::vector<float> buf(8192);
        while (size_t n = demux.read_audio(buf.data(), buf.size())) {
            out.insert(out.end(), buf.begin(), buf.begin() + n);
        }
        demux.close_audio();
        return out;
    };

    const std::vector<float> left = read_channel(0);
    const std::vector<float> right = read_channel(1);
    require(left.size() > 30000 && right.size() > 30000, "both channels produced 2 s of PCM");
    require(tone_power(left, 440.0) > 0.005 && tone_power(left, 880.0) < 1e-6,
            "channel 0 carries 440 Hz and rejects 880 Hz");
    require(tone_power(right, 880.0) > 0.005 && tone_power(right, 440.0) < 1e-6,
            "channel 1 carries 880 Hz and rejects 440 Hz");
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
        verify_decode_report();
        verify_output_timestamp_notes_are_not_decode_errors();
        verify_stop_reaches_a_silent_child(media);
        verify_stop_before_start(media);
        verify_stop_mid_audio(media);
        verify_descriptors_stay_private(media);
        verify_media_on_fd_3(media);
        verify_stray_descriptors_stay_out(media);
        verify_ogg_audio();
        verify_non_finite_samples_scrubbed();
        verify_video_without_audio();
        verify_garbage_rejected();
        verify_png_dimensions();
        verify_pipe_streaming_and_single_copy();
        verify_new_routed_families();
        verify_stereo_per_channel();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("video-demux-test passed");
    return 0;
}
