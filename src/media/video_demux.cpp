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
#include <mutex>
#include <span>
#include <string_view>
#include <thread>
#include <vector>

#include "cancellation.h"
#include "media/audio_decoder.h"
#include "media/pcm_scrub.h"
#include "metrics.h"

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

void ensure_sigpipe_ignored() {
    static std::once_flag once;
    std::call_once(once, [] { ::signal(SIGPIPE, SIG_IGN); });
}

class ToolProcess {
  public:
    // Exactly one of `media_fd` (>= 0 -> passed as fd 3) or `stdin_fd`
    // (>= 0 -> passed as fd 0) is inherited by the child.
    ToolProcess(ToolKind kind, const std::vector<std::string>& argv, int media_fd, int stdin_fd,
                std::chrono::milliseconds inactivity_timeout, std::stop_token stop)
        : kind_(kind), inactivity_timeout_(inactivity_timeout), stop_(std::move(stop)) {
        ensure_sigpipe_ignored();
        tool_ = argv.empty() ? "tool" : argv[0];
        if (stop_.stop_requested()) {
            record_outcome(ToolOutcome::kCancelled);
            throw Cancelled(tool_ + " not started: the stream is gone");
        }
        std::vector<char*> args;
        args.reserve(argv.size() + 1);
        for (const std::string& arg : argv) {
            args.push_back(const_cast<char*>(arg.c_str()));
        }
        args.push_back(nullptr);

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
        const int highest_kept = media_fd >= 0 ? kMediaFd : STDERR_FILENO;
        const long open_max = ::sysconf(_SC_OPEN_MAX);
        const int close_limit = open_max > highest_kept && open_max < kCloseLimitCap
                                    ? static_cast<int>(open_max)
                                    : kCloseLimitCap;
        pid_ = ::fork();
        if (pid_ < 0) {
            for (int fd : {out_pipe[0], out_pipe[1], err_pipe[0], err_pipe[1]}) {
                ::close(fd);
            }
            record_outcome(ToolOutcome::kExecFailed);
            throw ToolError("fork failed");
        }
        if (pid_ == 0) {
            int media = media_fd;
            int in = stdin_fd;
            int out = out_pipe[1];
            int err = err_pipe[1];
            for (int* fd : {&media, &in, &out, &err}) {
                if (*fd >= 0 && *fd <= highest_kept) {
                    *fd = ::fcntl(*fd, F_DUPFD_CLOEXEC, highest_kept + 1);
                    if (*fd < 0) {
                        ::_exit(kExecFailed);
                    }
                }
            }
            if (in >= 0 && ::dup2(in, STDIN_FILENO) < 0) {
                ::_exit(kExecFailed);
            }
            if (::dup2(out, STDOUT_FILENO) < 0 || ::dup2(err, STDERR_FILENO) < 0) {
                ::_exit(kExecFailed);
            }
            if (media >= 0 && ::dup2(media, kMediaFd) < 0) {
                ::_exit(kExecFailed);
            }
            if (::close_range(static_cast<unsigned int>(highest_kept + 1), ~0U, 0) != 0) {
                for (int fd = highest_kept + 1; fd < close_limit; ++fd) {
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
            record_outcome(ToolOutcome::kCancelled);
            ::kill(pid_, SIGKILL);
            ::waitpid(pid_, nullptr, 0);
        }
    }

    bool has_buffered_stdout() const {
        if (out_fd_ < 0) {
            return false;
        }
        struct pollfd pfd = {out_fd_, POLLIN, 0};
        const int ready = ::poll(&pfd, 1, 0);
        return ready > 0 && (pfd.revents & (POLLIN | POLLHUP)) != 0;
    }

    bool wait_for_stdout(std::chrono::milliseconds timeout, std::stop_token stop) const {
        if (out_fd_ < 0) {
            return true;
        }
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (true) {
            if (stop.stop_requested() || stop_.stop_requested()) {
                return true;
            }
            const auto left = std::chrono::ceil<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now());
            if (left.count() <= 0) {
                return has_buffered_stdout();
            }
            struct pollfd pfd = {out_fd_, POLLIN, 0};
            const int slice_ms =
                static_cast<int>(std::min(left, kStopPollInterval).count());
            const int ready = ::poll(&pfd, 1, slice_ms);
            if (ready > 0 && (pfd.revents & (POLLIN | POLLHUP)) != 0) {
                return true;
            }
            if (ready < 0 && errno != EINTR) {
                return true;
            }
        }
    }

    size_t read(uint8_t* out, size_t max_bytes) {
        auto idle_deadline = std::chrono::steady_clock::now() + inactivity_timeout_;
        while (true) {
            if (stop_.stop_requested()) {
                record_outcome(ToolOutcome::kCancelled);
                ::kill(pid_, SIGKILL);
                throw Cancelled(tool_ + " stopped: the stream is gone");
            }
            const auto idle_left = std::chrono::ceil<std::chrono::milliseconds>(
                idle_deadline - std::chrono::steady_clock::now());
            if (idle_left.count() <= 0) {
                record_outcome(ToolOutcome::kInactivityKilled);
                ::kill(pid_, SIGKILL);
                throw ToolError(tool_ + " produced no output for " +
                                std::to_string(inactivity_timeout_.count()) + "ms; killed");
            }
            struct pollfd fds[2];
            fds[0] = {out_fd_, POLLIN, 0};
            fds[1] = {err_fd_, POLLIN, 0};
            const nfds_t nfds = err_fd_ >= 0 ? 2 : 1;
            const int ready = ::poll(
                fds, nfds, static_cast<int>(std::min(idle_left, kStopPollInterval).count()));
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
                const ssize_t n = ::read(out_fd_, out, max_bytes);
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

    int wait_exit() {
        drain_stderr();
        if (int pidfd = static_cast<int>(::syscall(SYS_pidfd_open, pid_, 0)); pidfd >= 0) {
            const auto deadline = std::chrono::steady_clock::now() + kReapGrace;
            while (true) {
                const auto left = std::chrono::ceil<std::chrono::milliseconds>(
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
        const int code = WIFSIGNALED(status) ? -WTERMSIG(status) : WEXITSTATUS(status);
        if (code == 0) {
            record_outcome(ToolOutcome::kOk);
        } else if (code == kExecFailed) {
            record_outcome(ToolOutcome::kExecFailed);
        } else {
            record_outcome(ToolOutcome::kRejected);
        }
        return code;
    }

    const std::string& stderr_tail() const { return stderr_tail_; }

  private:
    void record_outcome(ToolOutcome outcome) {
        if (!recorded_outcome_) {
            recorded_outcome_ = true;
            metrics_record_tool_run(kind_, outcome);
        }
    }

    bool drain_stderr() {
        if (err_fd_ < 0) {
            return false;
        }
        const int flags = ::fcntl(err_fd_, F_GETFL);
        ::fcntl(err_fd_, F_SETFL, flags | O_NONBLOCK);
        char buf[4096];
        bool got = false;
        while (true) {
            const ssize_t n = ::read(err_fd_, buf, sizeof buf);
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

    ToolKind kind_;
    pid_t pid_ = -1;
    int out_fd_ = -1;
    int err_fd_ = -1;
    bool reaped_ = false;
    bool recorded_outcome_ = false;
    std::string tool_;
    std::string stderr_tail_;
    std::chrono::milliseconds inactivity_timeout_;
    std::stop_token stop_;
};

std::string collect_output(ToolProcess& tool, size_t cap = 1 << 20) {
    std::string out;
    uint8_t buf[8192];
    while (true) {
        const size_t n = tool.read(buf, sizeof buf);
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
    const size_t end = text.find_first_of("\r\n");
    return end == std::string::npos ? text : text.substr(0, end);
}

uint32_t read_be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

std::vector<std::string> build_audio_argv(const std::string& ffmpeg, const char* input_spec,
                                          std::optional<uint32_t> channel) {
    std::vector<std::string> argv = {ffmpeg, "-v", "error", "-i", input_spec, "-map", "a:0"};
    if (channel.has_value()) {
        argv.push_back("-af");
        argv.push_back("pan=mono|c0=c" + std::to_string(*channel));
    } else {
        argv.push_back("-ac");
        argv.push_back("1");
    }
    argv.push_back("-f");
    argv.push_back("f32le");
    argv.push_back("-ar");
    argv.push_back(std::to_string(kModelSampleRate));
    argv.push_back("pipe:1");
    return argv;
}

}  // namespace

void png_dimensions(const std::string& png, uint32_t* width, uint32_t* height) {
    if (png.size() < 24 || png.compare(12, 4, "IHDR") != 0) {
        throw DecodeError("malformed PNG from keyframe extraction");
    }
    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(png.data());
    *width = read_be32(bytes + 16);
    *height = read_be32(bytes + 20);
}

struct VideoDemux::Impl {
    int media_fd = -1;
    size_t owned_memfd_bytes = 0;
    std::string ffmpeg;
    std::string ffprobe;
    std::chrono::milliseconds inactivity_timeout{};
    std::stop_source feeder_stop;
    std::jthread feeder;
    std::unique_ptr<ToolProcess> audio;
    std::string partial_sample;

    void stop_feeder_and_audio() {
        feeder_stop.request_stop();
        audio.reset();
        if (feeder.joinable()) {
            feeder.join();
        }
    }

    ~Impl() {
        stop_feeder_and_audio();
        if (owned_memfd_bytes > 0) {
            metrics_add_memfd_bytes(-static_cast<int64_t>(owned_memfd_bytes));
        }
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
    impl_->media_fd =
        static_cast<int>(::memfd_create("grpc-asr-media", MFD_CLOEXEC | MFD_ALLOW_SEALING));
    if (impl_->media_fd < 0) {
        throw ToolError("memfd_create failed");
    }
    size_t written = 0;
    while (written < size) {
        const ssize_t n = ::write(impl_->media_fd, data + written, size - written);
        if (n <= 0) {
            throw ToolError("writing media to memfd failed");
        }
        written += static_cast<size_t>(n);
    }
    if (::fcntl(impl_->media_fd, F_ADD_SEALS,
                F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL) != 0) {
        throw ToolError(std::string("sealing the media memfd failed: ") + std::strerror(errno));
    }
    impl_->owned_memfd_bytes = size;
    metrics_add_memfd_bytes(static_cast<int64_t>(size));
}

VideoDemux::VideoDemux(int sealed_memfd, std::string ffmpeg_path, std::string ffprobe_path,
                       std::chrono::milliseconds inactivity_timeout)
    : impl_(std::make_unique<Impl>()) {
    impl_->ffmpeg = std::move(ffmpeg_path);
    impl_->ffprobe = std::move(ffprobe_path);
    impl_->inactivity_timeout = inactivity_timeout;
    if (sealed_memfd < 0) {
        throw ToolError("invalid borrowed media memfd");
    }
    impl_->media_fd = ::fcntl(sealed_memfd, F_DUPFD_CLOEXEC, 0);
    if (impl_->media_fd < 0) {
        throw ToolError(std::string("duplicating media memfd failed: ") + std::strerror(errno));
    }
}

VideoDemux::VideoDemux(std::string ffmpeg_path, std::string ffprobe_path,
                       std::chrono::milliseconds inactivity_timeout)
    : impl_(std::make_unique<Impl>()) {
    impl_->ffmpeg = std::move(ffmpeg_path);
    impl_->ffprobe = std::move(ffprobe_path);
    impl_->inactivity_timeout = inactivity_timeout;
}

VideoDemux::~VideoDemux() = default;
VideoDemux::VideoDemux(VideoDemux&&) noexcept = default;
VideoDemux& VideoDemux::operator=(VideoDemux&&) noexcept = default;

ProbeInfo VideoDemux::probe(std::stop_token stop) {
    if (impl_->media_fd < 0) {
        throw ToolError("probe requires a memfd-backed VideoDemux");
    }
    ProbeInfo info;

    auto run_query = [&](const std::vector<std::string>& argv) -> std::string {
        ToolProcess tool(ToolKind::kFfprobe, argv, impl_->media_fd, -1, impl_->inactivity_timeout,
                         stop);
        std::string out = collect_output(tool);
        const int code = tool.wait_exit();
        if (code == kExecFailed) {
            throw ToolError(impl_->ffprobe + " could not be executed");
        }
        if (code != 0) {
            throw DecodeError("ffprobe rejected the media: " + first_line(tool.stderr_tail()));
        }
        return out;
    };

    const std::string duration = first_line(run_query(
        {impl_->ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         kMediaPath}));
    if (!duration.empty() && duration != "N/A") {
        info.duration_ms = static_cast<uint64_t>(std::stod(duration) * 1000.0);
    }

    const std::string audio = first_line(run_query(
        {impl_->ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_name,sample_rate,channels", "-of", "csv=p=0", kMediaPath}));
    if (!audio.empty()) {
        info.has_audio = true;
        const size_t first_comma = audio.find(',');
        const size_t second_comma = audio.find(',', first_comma + 1);
        if (first_comma == std::string::npos || second_comma == std::string::npos) {
            throw ToolError("unexpected ffprobe audio output: " + audio);
        }
        info.audio_codec = audio.substr(0, first_comma);
        const std::string rate_str =
            audio.substr(first_comma + 1, second_comma - first_comma - 1);
        if (!rate_str.empty() && rate_str != "N/A") {
            info.sample_rate_hz = static_cast<uint32_t>(std::stoul(rate_str));
        }
        const std::string ch_str = audio.substr(second_comma + 1);
        if (!ch_str.empty() && ch_str != "N/A") {
            info.channels = static_cast<uint32_t>(std::stoul(ch_str));
        }
    }

    const std::string video = first_line(run_query(
        {impl_->ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name", "-of", "csv=p=0", kMediaPath}));
    if (!video.empty()) {
        info.has_video = true;
        info.video_codec = video;
    }

    return info;
}

void VideoDemux::open_audio(std::stop_token stop, std::optional<uint32_t> channel) {
    if (impl_->media_fd < 0) {
        throw ToolError("open_audio requires a memfd-backed VideoDemux");
    }
    impl_->stop_feeder_and_audio();
    impl_->audio = std::make_unique<ToolProcess>(
        ToolKind::kFfmpeg, build_audio_argv(impl_->ffmpeg, kMediaPath, channel), impl_->media_fd,
        -1, impl_->inactivity_timeout, std::move(stop));
    impl_->partial_sample.clear();
}

void VideoDemux::open_audio_from_pipe(ByteStream& stream, std::stop_token stop,
                                      std::optional<uint32_t> channel) {
    ensure_sigpipe_ignored();
    impl_->stop_feeder_and_audio();
    impl_->feeder_stop = std::stop_source{};

    int in_pipe[2];
    if (::pipe2(in_pipe, O_CLOEXEC) != 0) {
        throw ToolError("pipe creation failed");
    }
    try {
        impl_->audio = std::make_unique<ToolProcess>(
            ToolKind::kFfmpeg, build_audio_argv(impl_->ffmpeg, "pipe:0", channel), -1, in_pipe[0],
            impl_->inactivity_timeout, stop);
    } catch (...) {
        ::close(in_pipe[0]);
        ::close(in_pipe[1]);
        throw;
    }
    ::close(in_pipe[0]);
    impl_->partial_sample.clear();

    const int write_fd = in_pipe[1];
    std::stop_token local_stop = impl_->feeder_stop.get_token();
    impl_->feeder = std::jthread([&stream, write_fd, stop, local_stop]() {
        // Forward caller cancellation into a combined stop_source so
        // ByteStream::read_at wakes on either caller cancel or close/cancel_audio.
        std::stop_source combined;
        std::stop_callback on_caller(stop, [&combined] { combined.request_stop(); });
        std::stop_callback on_local(local_stop, [&combined] { combined.request_stop(); });
        std::stop_token token = combined.get_token();

        uint64_t offset = 0;
        char buf[64 * 1024];
        while (!token.stop_requested()) {
            const size_t got = stream.read_at(offset, buf, sizeof(buf), token);
            if (got == 0) {
                break;
            }
            size_t written = 0;
            bool pipe_closed = false;
            while (written < got && !token.stop_requested()) {
                const ssize_t n = ::write(write_fd, buf + written, got - written);
                if (n < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    pipe_closed = true;
                    break;
                }
                written += static_cast<size_t>(n);
            }
            if (pipe_closed) {
                break;
            }
            offset += written;
        }
        ::close(write_fd);
    });
}

size_t VideoDemux::read_audio(float* out, size_t max_samples) {
    uint8_t* bytes = reinterpret_cast<uint8_t*>(out);
    const size_t want_bytes = max_samples * sizeof(float);
    size_t have = impl_->partial_sample.size();
    std::memcpy(bytes, impl_->partial_sample.data(), have);
    impl_->partial_sample.clear();

    while (have < sizeof(float)) {
        const size_t n = impl_->audio->read(bytes + have, want_bytes - have);
        if (n == 0) {
            if (have != 0) {
                throw DecodeError("audio stream ended mid-sample");
            }
            return 0;
        }
        have += n;
    }
    const size_t whole = have / sizeof(float);
    const size_t leftover = have - whole * sizeof(float);
    if (leftover != 0) {
        impl_->partial_sample.assign(reinterpret_cast<char*>(bytes) + whole * sizeof(float),
                                     leftover);
    }
    scrub_pcm(std::span<float>(out, whole));
    return whole;
}

bool VideoDemux::has_buffered_audio() const {
    if (!impl_->partial_sample.empty()) {
        return true;
    }
    return impl_->audio && impl_->audio->has_buffered_stdout();
}

bool VideoDemux::wait_for_audio(std::chrono::milliseconds timeout,
                                std::stop_token stop) const {
    if (!impl_->partial_sample.empty()) {
        return true;
    }
    if (!impl_->audio) {
        return true;
    }
    return impl_->audio->wait_for_stdout(timeout, std::move(stop));
}

void VideoDemux::close_audio() {
    if (!impl_->audio) {
        if (impl_->feeder.joinable()) {
            impl_->feeder_stop.request_stop();
            impl_->feeder.join();
        }
        return;
    }
    impl_->feeder_stop.request_stop();
    const int code = impl_->audio->wait_exit();
    const std::string tail = first_line(impl_->audio->stderr_tail());
    impl_->audio.reset();
    if (impl_->feeder.joinable()) {
        impl_->feeder.join();
    }
    if (code == kExecFailed) {
        throw ToolError(impl_->ffmpeg + " could not be executed");
    }
    if (code != 0) {
        throw DecodeError("ffmpeg audio demux failed: " + tail);
    }
}

void VideoDemux::cancel_audio() {
    impl_->stop_feeder_and_audio();
}

void VideoDemux::extract_keyframes(
    uint32_t interval_seconds,
    const std::function<bool(uint64_t, uint32_t, uint32_t, std::string)>& sink,
    std::stop_token stop) {
    if (impl_->media_fd < 0) {
        throw ToolError("extract_keyframes requires a memfd-backed VideoDemux");
    }
    ToolProcess tool(
        ToolKind::kFfmpeg,
        {impl_->ffmpeg, "-v", "error", "-i", kMediaPath, "-map", "v:0", "-vf",
         "fps=1/" + std::to_string(interval_seconds), "-f", "image2pipe", "-c:v", "png",
         "pipe:1"},
        impl_->media_fd, -1, impl_->inactivity_timeout, std::move(stop));

    constexpr std::string_view kSignature{"\x89PNG\r\n\x1a\n", 8};
    std::string buffer;
    uint64_t frame_index = 0;
    uint8_t chunk[64 * 1024];

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
                    return true;
                }
                const uint32_t length =
                    read_be32(reinterpret_cast<const uint8_t*>(buffer.data()) + offset);
                const bool is_end = buffer.compare(offset + 4, 4, "IEND") == 0;
                const size_t chunk_total = 8ULL + length + 4ULL;
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
                    break;
                }
            }
        }
    };

    while (true) {
        const size_t n = tool.read(chunk, sizeof chunk);
        if (n == 0) {
            break;
        }
        buffer.append(reinterpret_cast<char*>(chunk), n);
        if (!emit_complete()) {
            return;
        }
    }
    const int code = tool.wait_exit();
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
