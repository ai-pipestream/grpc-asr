#include "media/video_demux.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <string_view>
#include <vector>

#include "cancellation.h"
#include "media/audio_decoder.h"

namespace asr::media {

namespace {

// The child inherits the memfd on this fixed descriptor and opens it as a
// fresh, independently-seekable file through /dev/fd.
constexpr int kMediaFd = 3;
constexpr char kMediaPath[] = "/dev/fd/3";
// Sentinel exit code the child uses when execvp (or the descriptor setup
// before it) fails, so a missing binary is distinguishable from ffmpeg
// rejecting the media.
constexpr int kExecFailed = 127;
constexpr std::chrono::milliseconds kReapGrace{2000};
// Highest descriptor count the pre-close_range fallback walks.
constexpr long kCloseLimitCap = 65536;
// How often a waiting read re-checks its stop token: the bound on how long
// a child outlives the call that wanted its output.
constexpr std::chrono::milliseconds kStopPollInterval{100};

// One ffmpeg/ffprobe child with its stdout and stderr pipes. Reads apply
// the inactivity timeout and the stop token; stderr is drained alongside
// stdout (so the child can never block on a full stderr pipe) and its tail
// kept for errors.
class ToolProcess {
  public:
    ToolProcess(const std::vector<std::string>& argv, int media_fd,
                std::chrono::milliseconds inactivity_timeout, std::stop_token stop)
        : inactivity_timeout_(inactivity_timeout), stop_(std::move(stop)) {
        tool_ = argv.empty() ? "tool" : argv[0];
        if (stop_.stop_requested()) {
            throw Cancelled(tool_ + " not started: the stream is gone");
        }
        // argv is built before fork: the child of a multithreaded process
        // may only make async-signal-safe calls, so it cannot allocate.
        std::vector<char*> args;
        args.reserve(argv.size() + 1);
        for (const std::string& arg : argv) {
            args.push_back(const_cast<char*>(arg.c_str()));
        }
        args.push_back(nullptr);
        // Close-on-exec, like the memfd: a child another stream forks
        // concurrently must not inherit these ends, since an inherited
        // write end keeps this reader from ever seeing end of file.
        int out_pipe[2];
        int err_pipe[2];
        if (::pipe2(out_pipe, O_CLOEXEC) != 0) {
            throw ToolError("pipe creation failed");
        }
        if (::pipe2(err_pipe, O_CLOEXEC) != 0) {
            ::close(out_pipe[0]);
            ::close(out_pipe[1]);
            throw ToolError("pipe creation failed");
        }
        // Bound for the fallback close loop in the child, read here because
        // sysconf is not async-signal-safe.
        const long open_max = ::sysconf(_SC_OPEN_MAX);
        const int close_limit = open_max > kMediaFd && open_max < kCloseLimitCap
                                    ? static_cast<int>(open_max)
                                    : kCloseLimitCap;
        pid_ = ::fork();
        if (pid_ < 0) {
            for (int fd : {out_pipe[0], out_pipe[1], err_pipe[0], err_pipe[1]}) {
                ::close(fd);
            }
            throw ToolError("fork failed");
        }
        if (pid_ == 0) {
            // First lift every source above the target slots, so no dup2
            // below overwrites one still waiting to be copied: the memfd
            // or a pipe end sits on 1, 2 or 3 whenever those were free.
            // Then dup2 clears close-on-exec on the copies, and every
            // descriptor above fd 3 is closed, so the tool keeps exactly
            // its stdin, stdout, stderr, and the media on fd 3. Close-on-exec
            // alone would leave in place whatever a library (a GPU runtime,
            // say) opened without it.
            int media = media_fd;
            int out = out_pipe[1];
            int err = err_pipe[1];
            for (int* fd : {&media, &out, &err}) {
                if (*fd <= kMediaFd) {
                    *fd = ::fcntl(*fd, F_DUPFD_CLOEXEC, kMediaFd + 1);
                    if (*fd < 0) {
                        ::_exit(kExecFailed);
                    }
                }
            }
            if (::dup2(media, kMediaFd) < 0 || ::dup2(out, STDOUT_FILENO) < 0 ||
                ::dup2(err, STDERR_FILENO) < 0) {
                ::_exit(kExecFailed);
            }
            if (::close_range(kMediaFd + 1, ~0U, 0) != 0) {
                // Kernels before 5.9 have no close_range.
                for (int fd = kMediaFd + 1; fd < close_limit; ++fd) {
                    ::close(fd);
                }
            }
            ::execvp(args[0], args.data());
            ::_exit(kExecFailed);
        }
        ::close(out_pipe[1]);
        ::close(err_pipe[1]);
        out_fd_ = out_pipe[0];
        err_fd_ = err_pipe[0];
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
    // output (stdout or stderr) within the inactivity timeout, and
    // Cancelled once the stop token fires.
    size_t read(uint8_t* out, size_t max_bytes) {
        auto idle_deadline = std::chrono::steady_clock::now() + inactivity_timeout_;
        while (true) {
            if (stop_.stop_requested()) {
                ::kill(pid_, SIGKILL);
                throw Cancelled(tool_ + " stopped: the stream is gone");
            }
            const auto idle_left = std::chrono::ceil<std::chrono::milliseconds>(
                idle_deadline - std::chrono::steady_clock::now());
            if (idle_left.count() <= 0) {
                ::kill(pid_, SIGKILL);
                throw ToolError(tool_ + " produced no output for " +
                                std::to_string(inactivity_timeout_.count()) + "ms; killed");
            }
            struct pollfd fds[2];
            fds[0] = {out_fd_, POLLIN, 0};
            fds[1] = {err_fd_, POLLIN, 0};
            nfds_t nfds = err_fd_ >= 0 ? 2 : 1;
            // Short slices, so a stop request lands even while the child
            // is silent (seeking, decoding toward the next still).
            int ready = ::poll(fds, nfds,
                               static_cast<int>(std::min(idle_left, kStopPollInterval).count()));
            if (ready == 0) {
                continue;
            }
            if (ready < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw ToolError(tool_ + " poll failed: " + std::strerror(errno));
            }
            if (err_fd_ >= 0 && (fds[1].revents & (POLLIN | POLLHUP)) != 0) {
                if (drain_stderr()) {
                    idle_deadline = std::chrono::steady_clock::now() + inactivity_timeout_;
                }
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
    // Returns true when the sweep read anything. At end of file the pipe
    // is closed, so poll stops reporting its hangup over and over.
    bool drain_stderr() {
        if (err_fd_ < 0) {
            return false;
        }
        // Non-blocking sweep: keep only the final 4 KiB.
        int flags = ::fcntl(err_fd_, F_GETFL);
        ::fcntl(err_fd_, F_SETFL, flags | O_NONBLOCK);
        char buf[4096];
        bool got = false;
        while (true) {
            ssize_t n = ::read(err_fd_, buf, sizeof buf);
            if (n == 0) {
                ::close(err_fd_);
                err_fd_ = -1;
                return got;
            }
            if (n < 0) {
                break;
            }
            got = true;
            stderr_tail_.append(buf, static_cast<size_t>(n));
            if (stderr_tail_.size() > 4096) {
                stderr_tail_.erase(0, stderr_tail_.size() - 4096);
            }
        }
        ::fcntl(err_fd_, F_SETFL, flags);
        return got;
    }

    pid_t pid_ = -1;
    int out_fd_ = -1;
    int err_fd_ = -1;
    bool reaped_ = false;
    std::string tool_;
    std::string stderr_tail_;
    std::chrono::milliseconds inactivity_timeout_;
    std::stop_token stop_;
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
    // Close-on-exec, so the children of other streams never inherit (and
    // pin) this media; each tool gets its own copy on fd 3.
    impl_->media_fd = static_cast<int>(
        ::memfd_create("grpc-asr-media", MFD_CLOEXEC | MFD_ALLOW_SEALING));
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
    // Seal it: the bytes every child reads can no longer change or move.
    if (::fcntl(impl_->media_fd, F_ADD_SEALS,
                F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL) != 0) {
        throw ToolError(std::string("sealing the media memfd failed: ") + std::strerror(errno));
    }
}

VideoDemux::~VideoDemux() = default;

ProbeInfo VideoDemux::probe(std::stop_token stop) {
    ProbeInfo info;

    auto run_query = [&](const std::vector<std::string>& argv) -> std::string {
        ToolProcess tool(argv, impl_->media_fd, impl_->inactivity_timeout, stop);
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

void VideoDemux::open_audio(std::stop_token stop) {
    impl_->audio = std::make_unique<ToolProcess>(
        std::vector<std::string>{impl_->ffmpeg, "-v", "error", "-i", kMediaPath, "-map", "a:0",
                                 "-f", "f32le", "-ac", "1", "-ar",
                                 std::to_string(kModelSampleRate), "pipe:1"},
        impl_->media_fd, impl_->inactivity_timeout, std::move(stop));
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

void VideoDemux::cancel_audio() {
    // ToolProcess's destructor closes the pipes, then kills and reaps.
    impl_->audio.reset();
}

void VideoDemux::extract_keyframes(
    uint32_t interval_seconds,
    const std::function<bool(uint64_t, uint32_t, uint32_t, std::string)>& sink,
    std::stop_token stop) {
    // fps=1/N picks the frame nearest each N-second grid point starting at
    // zero, so frame n sits at n*N seconds of media time.
    ToolProcess tool(
        {impl_->ffmpeg, "-v", "error", "-i", kMediaPath, "-map", "v:0", "-vf",
         "fps=1/" + std::to_string(interval_seconds), "-f", "image2pipe", "-c:v", "png",
         "pipe:1"},
        impl_->media_fd, impl_->inactivity_timeout, std::move(stop));

    constexpr std::string_view kSignature{"\x89PNG\r\n\x1a\n", 8};
    std::string buffer;
    uint64_t frame_index = 0;
    uint8_t chunk[64 * 1024];

    // Walk PNG chunks to find each image's end; everything up to and
    // including IEND+CRC is one still. Returns false once the sink asked
    // to stop.
    auto emit_complete = [&]() {
        while (true) {
            if (buffer.size() < 8) {
                return true;
            }
            if (!buffer.starts_with(kSignature)) {
                throw DecodeError("keyframe stream lost PNG framing");
            }
            size_t offset = 8;
            while (true) {
                if (buffer.size() < offset + 8) {
                    return true;  // need more bytes for the next chunk header
                }
                uint32_t length =
                    read_be32(reinterpret_cast<const uint8_t*>(buffer.data()) + offset);
                bool is_end = buffer.compare(offset + 4, 4, "IEND") == 0;
                size_t chunk_total = 8ULL + length + 4ULL;  // header + data + crc
                if (buffer.size() < offset + chunk_total) {
                    return true;
                }
                offset += chunk_total;
                if (is_end) {
                    std::string png = buffer.substr(0, offset);
                    buffer.erase(0, offset);
                    uint32_t width = 0;
                    uint32_t height = 0;
                    png_dimensions(png, &width, &height);
                    if (!sink(frame_index * interval_seconds * 1000ULL, width, height,
                              std::move(png))) {
                        return false;
                    }
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
        if (!emit_complete()) {
            return;  // the sink stopped us; ~ToolProcess kills and reaps the child
        }
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
