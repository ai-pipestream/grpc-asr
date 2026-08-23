#include "media/video_demux.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string_view>
#include <vector>

#include "media/audio_decoder.h"

namespace asr::media {

namespace {

// The child inherits the memfd on this fixed descriptor and opens it as a
// fresh, independently-seekable file through /dev/fd.
constexpr int kMediaFd = 3;
constexpr char kMediaPath[] = "/dev/fd/3";
// Sentinel exit code the child uses when execvp itself fails, so a missing
// binary is distinguishable from ffmpeg rejecting the media.
constexpr int kExecFailed = 127;
constexpr std::chrono::milliseconds kReapGrace{2000};

// One ffmpeg/ffprobe child with its stdout and stderr pipes. Reads apply
// the inactivity timeout; stderr is drained alongside stdout (so the child
// can never block on a full stderr pipe) and its tail kept for errors.
class ToolProcess {
  public:
    ToolProcess(const std::vector<std::string>& argv, int media_fd,
                std::chrono::milliseconds inactivity_timeout)
        : inactivity_timeout_(inactivity_timeout) {
        int out_pipe[2];
        int err_pipe[2];
        if (::pipe(out_pipe) != 0 || ::pipe(err_pipe) != 0) {
            throw ToolError("pipe creation failed");
        }
        pid_ = ::fork();
        if (pid_ < 0) {
            throw ToolError("fork failed");
        }
        if (pid_ == 0) {
            // dup2 clears CLOEXEC on the duplicate, which is exactly what
            // lets the exec'd tool see the memfd.
            ::dup2(media_fd, kMediaFd);
            ::dup2(out_pipe[1], STDOUT_FILENO);
            ::dup2(err_pipe[1], STDERR_FILENO);
            ::close(out_pipe[0]);
            ::close(out_pipe[1]);
            ::close(err_pipe[0]);
            ::close(err_pipe[1]);
            std::vector<char*> args;
            args.reserve(argv.size() + 1);
            for (const std::string& arg : argv) {
                args.push_back(const_cast<char*>(arg.c_str()));
            }
            args.push_back(nullptr);
            ::execvp(args[0], args.data());
            ::_exit(kExecFailed);
        }
        ::close(out_pipe[1]);
        ::close(err_pipe[1]);
        out_fd_ = out_pipe[0];
        err_fd_ = err_pipe[0];
        tool_ = argv.empty() ? "tool" : argv[0];
    }

    ~ToolProcess() {
        if (out_fd_ >= 0) {
            ::close(out_fd_);
        }
        if (err_fd_ >= 0) {
            ::close(err_fd_);
        }
        if (pid_ > 0 && !reaped_) {
            ::kill(pid_, SIGKILL);
            ::waitpid(pid_, nullptr, 0);
        }
    }

    // Reads up to max_bytes of the child's stdout. Returns 0 on clean end
    // of stream. Kills the child and throws ToolError when it produces no
    // output within the inactivity timeout.
    size_t read(uint8_t* out, size_t max_bytes) {
        while (true) {
            struct pollfd fds[2];
            fds[0] = {out_fd_, POLLIN, 0};
            fds[1] = {err_fd_, POLLIN, 0};
            nfds_t nfds = err_fd_ >= 0 ? 2 : 1;
            int ready = ::poll(fds, nfds, static_cast<int>(inactivity_timeout_.count()));
            if (ready == 0) {
                ::kill(pid_, SIGKILL);
                throw ToolError(tool_ + " produced no output for " +
                                std::to_string(inactivity_timeout_.count()) + "ms; killed");
            }
            if (ready < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw ToolError(tool_ + " poll failed: " + std::strerror(errno));
            }
            if (err_fd_ >= 0 && (fds[1].revents & (POLLIN | POLLHUP)) != 0) {
                drain_stderr();
            }
            if ((fds[0].revents & (POLLIN | POLLHUP)) != 0) {
                ssize_t n = ::read(out_fd_, out, max_bytes);
                if (n < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    throw ToolError(tool_ + " read failed: " + std::strerror(errno));
                }
                return static_cast<size_t>(n);
            }
        }
    }

    // Reaps the child after EOF (killing it if it lingers past the reap
    // grace) and returns its exit code; negative when signal-killed.
    int wait_exit() {
        // Drain any stderr the child wrote after our last poll.
        drain_stderr();
        // A pidfd makes "wait with a deadline" one poll: the descriptor
        // turns readable the moment the child exits, no sleep loop. Raw
        // syscall because glibc's sys/pidfd.h lacks extern "C" guards.
        if (int pidfd = static_cast<int>(::syscall(SYS_pidfd_open, pid_, 0)); pidfd >= 0) {
            auto deadline = std::chrono::steady_clock::now() + kReapGrace;
            while (true) {
                auto left = std::chrono::ceil<std::chrono::milliseconds>(
                    deadline - std::chrono::steady_clock::now());
                if (left.count() <= 0) {
                    break;
                }
                struct pollfd fd = {pidfd, POLLIN, 0};
                if (::poll(&fd, 1, static_cast<int>(left.count())) >= 0) {
                    break;
                }
                if (errno != EINTR) {
                    break;
                }
            }
            ::close(pidfd);
        }
        int status = 0;
        if (::waitpid(pid_, &status, WNOHANG) != pid_) {
            ::kill(pid_, SIGKILL);
            ::waitpid(pid_, &status, 0);
        }
        reaped_ = true;
        if (WIFSIGNALED(status)) {
            return -WTERMSIG(status);
        }
        return WEXITSTATUS(status);
    }

    // Last stderr bytes, for attaching the tool's own words to an error.
    const std::string& stderr_tail() const { return stderr_tail_; }

  private:
    void drain_stderr() {
        if (err_fd_ < 0) {
            return;
        }
        // Non-blocking sweep: keep only the final 4 KiB.
        int flags = ::fcntl(err_fd_, F_GETFL);
        ::fcntl(err_fd_, F_SETFL, flags | O_NONBLOCK);
        char buf[4096];
        while (true) {
            ssize_t n = ::read(err_fd_, buf, sizeof buf);
            if (n <= 0) {
                break;
            }
            stderr_tail_.append(buf, static_cast<size_t>(n));
            if (stderr_tail_.size() > 4096) {
                stderr_tail_.erase(0, stderr_tail_.size() - 4096);
            }
        }
        ::fcntl(err_fd_, F_SETFL, flags);
    }

    pid_t pid_ = -1;
    int out_fd_ = -1;
    int err_fd_ = -1;
    bool reaped_ = false;
    std::string tool_;
    std::string stderr_tail_;
    std::chrono::milliseconds inactivity_timeout_;
};

// Reads the whole (small) stdout of a tool run, e.g. an ffprobe query.
std::string collect_output(ToolProcess& tool, size_t cap = 1 << 20) {
    std::string out;
    uint8_t buf[8192];
    while (true) {
        size_t n = tool.read(buf, sizeof buf);
        if (n == 0) {
            break;
        }
        out.append(reinterpret_cast<char*>(buf), n);
        if (out.size() > cap) {
            throw ToolError("tool output exceeded cap");
        }
    }
    return out;
}

std::string first_line(const std::string& text) {
    size_t end = text.find_first_of("\r\n");
    return end == std::string::npos ? text : text.substr(0, end);
}

uint32_t read_be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

}  // namespace

void png_dimensions(const std::string& png, uint32_t* width, uint32_t* height) {
    // Signature (8) + IHDR length/type (8) + width (4) + height (4).
    if (png.size() < 24 || png.compare(12, 4, "IHDR") != 0) {
        throw DecodeError("malformed PNG from keyframe extraction");
    }
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(png.data());
    *width = read_be32(bytes + 16);
    *height = read_be32(bytes + 20);
}

struct VideoDemux::Impl {
    int media_fd = -1;
    std::string ffmpeg;
    std::string ffprobe;
    std::chrono::milliseconds inactivity_timeout;
    std::unique_ptr<ToolProcess> audio;
    // Carry-over between read_audio calls: bytes that did not fill a
    // whole float.
    std::string partial_sample;

    ~Impl() {
        if (media_fd >= 0) {
            ::close(media_fd);
        }
    }
};

VideoDemux::VideoDemux(const uint8_t* data, size_t size, std::string ffmpeg_path,
                       std::string ffprobe_path, std::chrono::milliseconds inactivity_timeout)
    : impl_(std::make_unique<Impl>()) {
    impl_->ffmpeg = std::move(ffmpeg_path);
    impl_->ffprobe = std::move(ffprobe_path);
    impl_->inactivity_timeout = inactivity_timeout;
    impl_->media_fd = static_cast<int>(::memfd_create("grpc-asr-media", 0));
    if (impl_->media_fd < 0) {
        throw ToolError("memfd_create failed");
    }
    size_t written = 0;
    while (written < size) {
        ssize_t n = ::write(impl_->media_fd, data + written, size - written);
        if (n <= 0) {
            throw ToolError("writing media to memfd failed");
        }
        written += static_cast<size_t>(n);
    }
}

VideoDemux::~VideoDemux() = default;

ProbeInfo VideoDemux::probe() {
    ProbeInfo info;

    auto run_query = [&](const std::vector<std::string>& argv) -> std::string {
        ToolProcess tool(argv, impl_->media_fd, impl_->inactivity_timeout);
        std::string out = collect_output(tool);
        int code = tool.wait_exit();
        if (code == kExecFailed) {
            throw ToolError(impl_->ffprobe + " could not be executed");
        }
        if (code != 0) {
            throw DecodeError("ffprobe rejected the media: " + first_line(tool.stderr_tail()));
        }
        return out;
    };

    // Three tiny line-oriented queries beat parsing JSON without a JSON
    // library: format duration, audio stream, video stream.
    std::string duration = first_line(run_query(
        {impl_->ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         kMediaPath}));
    if (!duration.empty() && duration != "N/A") {
        info.duration_ms = static_cast<uint64_t>(std::stod(duration) * 1000.0);
    }

    std::string audio = first_line(run_query(
        {impl_->ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels", "-of", "csv=p=0", kMediaPath}));
    if (!audio.empty()) {
        info.has_audio = true;
        size_t first_comma = audio.find(',');
        size_t second_comma = audio.find(',', first_comma + 1);
        if (first_comma == std::string::npos || second_comma == std::string::npos) {
            throw ToolError("unexpected ffprobe audio output: " + audio);
        }
        info.audio_codec = audio.substr(0, first_comma);
        info.sample_rate_hz = static_cast<uint32_t>(
            std::stoul(audio.substr(first_comma + 1, second_comma - first_comma - 1)));
        info.channels = static_cast<uint32_t>(std::stoul(audio.substr(second_comma + 1)));
    }

    std::string video = first_line(run_query(
        {impl_->ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name", "-of", "csv=p=0", kMediaPath}));
    if (!video.empty()) {
        info.has_video = true;
        info.video_codec = video;
    }

    return info;
}

void VideoDemux::open_audio() {
    impl_->audio = std::make_unique<ToolProcess>(
        std::vector<std::string>{impl_->ffmpeg, "-v", "error", "-i", kMediaPath, "-map", "a:0",
                                 "-f", "f32le", "-ac", "1", "-ar",
                                 std::to_string(kModelSampleRate), "pipe:1"},
        impl_->media_fd, impl_->inactivity_timeout);
    impl_->partial_sample.clear();
}

size_t VideoDemux::read_audio(float* out, size_t max_samples) {
    uint8_t* bytes = reinterpret_cast<uint8_t*>(out);
    size_t want_bytes = max_samples * sizeof(float);
    size_t have = impl_->partial_sample.size();
    std::memcpy(bytes, impl_->partial_sample.data(), have);
    impl_->partial_sample.clear();

    while (have < sizeof(float)) {
        size_t n = impl_->audio->read(bytes + have, want_bytes - have);
        if (n == 0) {
            if (have != 0) {
                throw DecodeError("audio stream ended mid-sample");
            }
            return 0;
        }
        have += n;
    }
    size_t whole = have / sizeof(float);
    size_t leftover = have - whole * sizeof(float);
    if (leftover != 0) {
        impl_->partial_sample.assign(reinterpret_cast<char*>(bytes) + whole * sizeof(float),
                                     leftover);
    }
    return whole;
}

void VideoDemux::close_audio() {
    if (!impl_->audio) {
        return;
    }
    int code = impl_->audio->wait_exit();
    std::string tail = first_line(impl_->audio->stderr_tail());
    impl_->audio.reset();
    if (code == kExecFailed) {
        throw ToolError(impl_->ffmpeg + " could not be executed");
    }
    if (code != 0) {
        throw DecodeError("ffmpeg audio demux failed: " + tail);
    }
}

void VideoDemux::extract_keyframes(
    uint32_t interval_seconds,
    const std::function<void(uint64_t, uint32_t, uint32_t, std::string)>& sink) {
    // fps=1/N picks the frame nearest each N-second grid point starting at
    // zero, so frame n sits at n*N seconds of media time.
    ToolProcess tool(
        {impl_->ffmpeg, "-v", "error", "-i", kMediaPath, "-map", "v:0", "-vf",
         "fps=1/" + std::to_string(interval_seconds), "-f", "image2pipe", "-c:v", "png",
         "pipe:1"},
        impl_->media_fd, impl_->inactivity_timeout);

    constexpr std::string_view kSignature{"\x89PNG\r\n\x1a\n", 8};
    std::string buffer;
    uint64_t frame_index = 0;
    uint8_t chunk[64 * 1024];

    // Walk PNG chunks to find each image's end; everything up to and
    // including IEND+CRC is one still.
    auto emit_complete = [&]() {
        while (true) {
            if (buffer.size() < 8) {
                return;
            }
            if (!buffer.starts_with(kSignature)) {
                throw DecodeError("keyframe stream lost PNG framing");
            }
            size_t offset = 8;
            while (true) {
                if (buffer.size() < offset + 8) {
                    return;  // need more bytes for the next chunk header
                }
                uint32_t length =
                    read_be32(reinterpret_cast<const uint8_t*>(buffer.data()) + offset);
                bool is_end = buffer.compare(offset + 4, 4, "IEND") == 0;
                size_t chunk_total = 8ULL + length + 4ULL;  // header + data + crc
                if (buffer.size() < offset + chunk_total) {
                    return;
                }
                offset += chunk_total;
                if (is_end) {
                    std::string png = buffer.substr(0, offset);
                    buffer.erase(0, offset);
                    uint32_t width = 0;
                    uint32_t height = 0;
                    png_dimensions(png, &width, &height);
                    sink(frame_index * interval_seconds * 1000ULL, width, height, std::move(png));
                    frame_index++;
                    break;  // scan the buffer again from the top
                }
            }
        }
    };

    while (true) {
        size_t n = tool.read(chunk, sizeof chunk);
        if (n == 0) {
            break;
        }
        buffer.append(reinterpret_cast<char*>(chunk), n);
        emit_complete();
    }
    int code = tool.wait_exit();
    if (code == kExecFailed) {
        throw ToolError(impl_->ffmpeg + " could not be executed");
    }
    if (code != 0) {
        throw DecodeError("ffmpeg keyframe extraction failed: " +
                          first_line(tool.stderr_tail()));
    }
    if (!buffer.empty()) {
        throw DecodeError("keyframe stream ended mid-frame");
    }
}

}  // namespace asr::media
