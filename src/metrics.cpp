#include "metrics.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <print>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "config.h"

namespace asr {
namespace {

constexpr const char* kContentType = "text/plain; version=0.0.4; charset=utf-8";

constexpr std::array<uint64_t, 12> kDurationBoundsNs = {
    10'000'000ULL,     25'000'000ULL,     50'000'000ULL,     100'000'000ULL,
    250'000'000ULL,    500'000'000ULL,    1'000'000'000ULL,  2'500'000'000ULL,
    5'000'000'000ULL,  10'000'000'000ULL, 30'000'000'000ULL, 60'000'000'000ULL,
};

constexpr std::array<std::string_view, 12> kDurationBoundLabels = {
    "0.01", "0.025", "0.05", "0.1", "0.25", "0.5", "1", "2.5", "5", "10", "30", "60",
};

// Fixed-point micro-units (1e-6 RTF) for lock-free integer sum accumulation.
constexpr uint64_t kRtfScale = 1'000'000ULL;

constexpr std::array<uint64_t, 10> kRtfBoundsScaled = {
    250'000ULL,     500'000ULL,    1'000'000ULL,  2'000'000ULL,   5'000'000ULL,
    10'000'000ULL,  20'000'000ULL, 50'000'000ULL, 100'000'000ULL, 200'000'000ULL,
};

constexpr std::array<std::string_view, 10> kRtfBoundLabels = {
    "0.25", "0.5", "1", "2", "5", "10", "20", "50", "100", "200",
};

template <size_t N>
size_t find_bucket(const std::array<uint64_t, N>& bounds, uint64_t value) {
    for (size_t i = 0; i < N; ++i) {
        if (value <= bounds[i]) {
            return i;
        }
    }
    return N;
}

template <size_t N>
struct AtomicHistogram {
    std::array<std::atomic<uint64_t>, N + 1> buckets{};
    std::atomic<uint64_t> sum_scaled{0};

    void observe(const std::array<uint64_t, N>& bounds, uint64_t scaled_value) {
        buckets[find_bucket(bounds, scaled_value)].fetch_add(1, std::memory_order_relaxed);
        sum_scaled.fetch_add(scaled_value, std::memory_order_relaxed);
    }
};

struct ModelTelemetry {
    std::atomic<int64_t> states_total{0};
    std::atomic<int64_t> states_busy{0};
    std::atomic<int64_t> queue_depth{0};
    std::atomic<int64_t> queue_depth_max{0};
    AtomicHistogram<kDurationBoundsNs.size()> queue_wait{};
    std::array<AtomicHistogram<kDurationBoundsNs.size()>,
               static_cast<size_t>(FirstSegmentPath::kCount)>
        first_segment{};
    std::atomic<uint64_t> audio_ms_total{0};
};

struct GlobalTelemetry {
    std::mutex mutex;
    std::map<std::string, std::shared_ptr<ModelTelemetry>, std::less<>> models;
    std::map<std::pair<std::string, std::string>,
             std::shared_ptr<AtomicHistogram<kRtfBoundsScaled.size()>>>
        rtf_by_model_backend;

    std::atomic<int64_t> streams_active{0};
    std::atomic<int64_t> upload_bytes_resident{0};
    std::atomic<int64_t> memfd_bytes_resident{0};
    std::atomic<uint64_t> windows_total{0};
    std::atomic<uint64_t> window_redecodes_total{0};
    std::atomic<uint64_t> keyframes_total{0};
    std::atomic<uint64_t> warmup_ms{0};
    std::atomic<uint32_t> last_window_seconds{0};

    std::array<std::atomic<uint64_t>, static_cast<size_t>(StreamEndResult::kCount)> stream_ends{};
    std::array<std::array<std::atomic<uint64_t>, static_cast<size_t>(ToolOutcome::kCount)>,
               static_cast<size_t>(ToolKind::kCount)>
        tool_runs{};
};

GlobalTelemetry& telemetry() {
    static GlobalTelemetry instance;
    return instance;
}

std::shared_ptr<ModelTelemetry> model_slot(std::string_view model) {
    GlobalTelemetry& t = telemetry();
    std::lock_guard<std::mutex> lock(t.mutex);
    auto found = t.models.find(model);
    if (found != t.models.end()) {
        return found->second;
    }
    auto created = std::make_shared<ModelTelemetry>();
    t.models.emplace(std::string(model), created);
    return created;
}

std::shared_ptr<AtomicHistogram<kRtfBoundsScaled.size()>> rtf_slot(std::string_view model,
                                                                   std::string_view backend) {
    GlobalTelemetry& t = telemetry();
    std::lock_guard<std::mutex> lock(t.mutex);
    auto key = std::make_pair(std::string(model), std::string(backend));
    auto found = t.rtf_by_model_backend.find(key);
    if (found != t.rtf_by_model_backend.end()) {
        return found->second;
    }
    auto created = std::make_shared<AtomicHistogram<kRtfBoundsScaled.size()>>();
    t.rtf_by_model_backend.emplace(std::move(key), created);
    return created;
}

std::string escape_label(std::string_view raw) {
    std::string out;
    out.reserve(raw.size());
    for (const char c : raw) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            default:
                out.push_back(c);
                break;
        }
    }
    return out;
}

uint64_t clamp_non_negative(int64_t value) {
    return value > 0 ? static_cast<uint64_t>(value) : 0ULL;
}

template <size_t N>
void render_histogram_series(std::ostringstream& out, std::string_view metric_name,
                             std::string_view labels_prefix,
                             const AtomicHistogram<N>& hist,
                             const std::array<std::string_view, N>& bound_labels,
                             double unit_divisor) {
    uint64_t cumulative = 0;
    for (size_t i = 0; i < N + 1; ++i) {
        cumulative += hist.buckets[i].load(std::memory_order_relaxed);
        out << metric_name << "_bucket{";
        if (!labels_prefix.empty()) {
            out << labels_prefix << ',';
        }
        out << "le=\"" << (i < N ? bound_labels[i] : std::string_view("+Inf")) << "\"} "
            << cumulative << '\n';
    }
    const double sum =
        static_cast<double>(hist.sum_scaled.load(std::memory_order_relaxed)) / unit_divisor;
    out << metric_name << "_sum";
    if (!labels_prefix.empty()) {
        out << '{' << labels_prefix << '}';
    }
    out << ' ' << sum << '\n';
    out << metric_name << "_count";
    if (!labels_prefix.empty()) {
        out << '{' << labels_prefix << '}';
    }
    out << ' ' << cumulative << '\n';
}

struct ProcessUsage {
    uint64_t resident_bytes = 0;
    double cpu_seconds = 0.0;
};

ProcessUsage read_process_usage() {
    ProcessUsage usage;
    if (std::ifstream statm("/proc/self/statm"); statm) {
        uint64_t total_pages = 0;
        uint64_t resident_pages = 0;
        if (statm >> total_pages >> resident_pages) {
            const long page_size = ::sysconf(_SC_PAGESIZE);
            usage.resident_bytes =
                resident_pages * static_cast<uint64_t>(page_size > 0 ? page_size : 4096);
        }
    }
    struct rusage self {};
    if (::getrusage(RUSAGE_SELF, &self) == 0) {
        usage.cpu_seconds = static_cast<double>(self.ru_utime.tv_sec) +
                            static_cast<double>(self.ru_utime.tv_usec) / 1e6 +
                            static_cast<double>(self.ru_stime.tv_sec) +
                            static_cast<double>(self.ru_stime.tv_usec) / 1e6;
    }
    return usage;
}

std::string http_response(const char* status_line, const std::string& content_type,
                          const std::string& body) {
    std::ostringstream response;
    response << "HTTP/1.1 " << status_line << "\r\nContent-Type: " << content_type
             << "\r\nContent-Length: " << body.size() << "\r\nConnection: close\r\n\r\n"
             << body;
    return response.str();
}

void write_all(int fd, const std::string& data) {
    size_t sent = 0;
    while (sent < data.size()) {
        const ssize_t wrote = ::send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
        if (wrote <= 0) {
            return;
        }
        sent += static_cast<size_t>(wrote);
    }
}

uint16_t bound_port_of(int fd) {
    sockaddr_storage storage{};
    socklen_t len = sizeof(storage);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&storage), &len) != 0) {
        return 0;
    }
    if (storage.ss_family == AF_INET) {
        return ntohs(reinterpret_cast<const sockaddr_in*>(&storage)->sin_port);
    }
    if (storage.ss_family == AF_INET6) {
        return ntohs(reinterpret_cast<const sockaddr_in6*>(&storage)->sin6_port);
    }
    return 0;
}

}  // namespace

void metrics_add_model_states(std::string_view model, uint64_t count) {
    model_slot(model)->states_total.fetch_add(static_cast<int64_t>(count),
                                              std::memory_order_relaxed);
}

void metrics_remove_model_states(std::string_view model, uint64_t count) {
    model_slot(model)->states_total.fetch_sub(static_cast<int64_t>(count),
                                              std::memory_order_relaxed);
}

void metrics_inc_model_busy(std::string_view model) {
    model_slot(model)->states_busy.fetch_add(1, std::memory_order_relaxed);
}

void metrics_dec_model_busy(std::string_view model) {
    model_slot(model)->states_busy.fetch_sub(1, std::memory_order_relaxed);
}

void metrics_inc_queue_depth(std::string_view model) {
    auto slot = model_slot(model);
    const int64_t current = slot->queue_depth.fetch_add(1, std::memory_order_relaxed) + 1;
    int64_t prev_max = slot->queue_depth_max.load(std::memory_order_relaxed);
    while (current > prev_max &&
           !slot->queue_depth_max.compare_exchange_weak(prev_max, current,
                                                        std::memory_order_relaxed)) {
    }
}

void metrics_dec_queue_depth(std::string_view model) {
    model_slot(model)->queue_depth.fetch_sub(1, std::memory_order_relaxed);
}

void metrics_observe_queue_wait(std::string_view model, uint64_t duration_ns) {
    model_slot(model)->queue_wait.observe(kDurationBoundsNs, duration_ns);
}

void metrics_record_warmup_ms(uint64_t warmup_ms) {
    telemetry().warmup_ms.store(warmup_ms, std::memory_order_relaxed);
}

void metrics_inc_streams_active() {
    telemetry().streams_active.fetch_add(1, std::memory_order_relaxed);
}

void metrics_dec_streams_active() {
    telemetry().streams_active.fetch_sub(1, std::memory_order_relaxed);
}

void metrics_add_upload_bytes(int64_t delta) {
    telemetry().upload_bytes_resident.fetch_add(delta, std::memory_order_relaxed);
}

void metrics_add_memfd_bytes(int64_t delta) {
    telemetry().memfd_bytes_resident.fetch_add(delta, std::memory_order_relaxed);
}

void metrics_observe_first_segment(std::string_view model, FirstSegmentPath path,
                                   uint64_t duration_ns) {
    model_slot(model)->first_segment[static_cast<size_t>(path)].observe(kDurationBoundsNs,
                                                                        duration_ns);
}

void metrics_observe_realtime_factor(std::string_view model, std::string_view backend,
                                     double audio_seconds_per_wall_second) {
    if (!std::isfinite(audio_seconds_per_wall_second) || audio_seconds_per_wall_second < 0.0) {
        return;
    }
    const double clamped = std::min(audio_seconds_per_wall_second, 1e9);
    const auto scaled = static_cast<uint64_t>(clamped * static_cast<double>(kRtfScale));
    rtf_slot(model, backend)->observe(kRtfBoundsScaled, scaled);
}

void metrics_record_stream_end(const StreamMetrics& stream) {
    GlobalTelemetry& t = telemetry();
    const auto idx = static_cast<size_t>(stream.result);
    if (idx < t.stream_ends.size()) {
        t.stream_ends[idx].fetch_add(1, std::memory_order_relaxed);
    }
    if (stream.windows > 0) {
        t.windows_total.fetch_add(stream.windows, std::memory_order_relaxed);
    }
    if (stream.window_redecodes > 0) {
        t.window_redecodes_total.fetch_add(stream.window_redecodes, std::memory_order_relaxed);
    }
    if (stream.keyframes > 0) {
        t.keyframes_total.fetch_add(stream.keyframes, std::memory_order_relaxed);
    }
    if (stream.chosen_window_seconds > 0) {
        t.last_window_seconds.store(stream.chosen_window_seconds, std::memory_order_relaxed);
    }
    if (stream.audio_ms > 0 && !stream.model.empty()) {
        model_slot(stream.model)->audio_ms_total.fetch_add(stream.audio_ms,
                                                           std::memory_order_relaxed);
    }
}

void metrics_record_tool_run(ToolKind tool, ToolOutcome outcome) {
    GlobalTelemetry& t = telemetry();
    const auto tool_idx = static_cast<size_t>(tool);
    const auto out_idx = static_cast<size_t>(outcome);
    if (tool_idx < t.tool_runs.size() && out_idx < t.tool_runs[tool_idx].size()) {
        t.tool_runs[tool_idx][out_idx].fetch_add(1, std::memory_order_relaxed);
    }
}

MetricsSummary metrics_summary() {
    GlobalTelemetry& t = telemetry();
    MetricsSummary out;
    out.streams_active = clamp_non_negative(t.streams_active.load(std::memory_order_relaxed));
    out.upload_bytes_resident =
        clamp_non_negative(t.upload_bytes_resident.load(std::memory_order_relaxed));
    out.memfd_bytes_resident =
        clamp_non_negative(t.memfd_bytes_resident.load(std::memory_order_relaxed));
    out.windows_total = t.windows_total.load(std::memory_order_relaxed);
    out.window_redecodes_total = t.window_redecodes_total.load(std::memory_order_relaxed);
    out.keyframes_total = t.keyframes_total.load(std::memory_order_relaxed);
    out.warmup_ms = t.warmup_ms.load(std::memory_order_relaxed);
    out.last_window_seconds = t.last_window_seconds.load(std::memory_order_relaxed);
    for (size_t i = 0; i < out.stream_ends.size(); ++i) {
        out.stream_ends[i] = t.stream_ends[i].load(std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> lock(t.mutex);
    for (const auto& [name, slot] : t.models) {
        (void)name;
        out.audio_ms_total += slot->audio_ms_total.load(std::memory_order_relaxed);
    }
    return out;
}

std::string render_prometheus_metrics() {
    GlobalTelemetry& t = telemetry();
    std::vector<std::pair<std::string, std::shared_ptr<ModelTelemetry>>> models_copy;
    std::vector<std::pair<std::pair<std::string, std::string>,
                          std::shared_ptr<AtomicHistogram<kRtfBoundsScaled.size()>>>>
        rtf_copy;
    {
        std::lock_guard<std::mutex> lock(t.mutex);
        models_copy.assign(t.models.begin(), t.models.end());
        rtf_copy.assign(t.rtf_by_model_backend.begin(), t.rtf_by_model_backend.end());
    }

    std::ostringstream out;
    out.precision(15);

    out << "# HELP grpc_asr_states_total Configured whisper states per loaded model.\n"
           "# TYPE grpc_asr_states_total gauge\n";
    for (const auto& [name, slot] : models_copy) {
        out << "grpc_asr_states_total{model=\"" << escape_label(name) << "\"} "
            << clamp_non_negative(slot->states_total.load(std::memory_order_relaxed)) << '\n';
    }

    out << "# HELP grpc_asr_states_busy Whisper states currently leased to active streams.\n"
           "# TYPE grpc_asr_states_busy gauge\n";
    for (const auto& [name, slot] : models_copy) {
        out << "grpc_asr_states_busy{model=\"" << escape_label(name) << "\"} "
            << clamp_non_negative(slot->states_busy.load(std::memory_order_relaxed)) << '\n';
    }

    out << "# HELP grpc_asr_queue_depth Streams currently waiting for a free model state.\n"
           "# TYPE grpc_asr_queue_depth gauge\n";
    for (const auto& [name, slot] : models_copy) {
        out << "grpc_asr_queue_depth{model=\"" << escape_label(name) << "\"} "
            << clamp_non_negative(slot->queue_depth.load(std::memory_order_relaxed)) << '\n';
    }

    out << "# HELP grpc_asr_queue_depth_max Peak streams waiting concurrently for a model state.\n"
           "# TYPE grpc_asr_queue_depth_max gauge\n";
    for (const auto& [name, slot] : models_copy) {
        out << "grpc_asr_queue_depth_max{model=\"" << escape_label(name) << "\"} "
            << clamp_non_negative(slot->queue_depth_max.load(std::memory_order_relaxed)) << '\n';
    }

    out << "# HELP grpc_asr_queue_wait_seconds Time spent waiting to lease a model state.\n"
           "# TYPE grpc_asr_queue_wait_seconds histogram\n";
    for (const auto& [name, slot] : models_copy) {
        const std::string label = "model=\"" + escape_label(name) + "\"";
        render_histogram_series(out, "grpc_asr_queue_wait_seconds", label, slot->queue_wait,
                                kDurationBoundLabels, 1e9);
    }

    out << "# HELP grpc_asr_streams_active Active Transcribe RPC handlers.\n"
           "# TYPE grpc_asr_streams_active gauge\n"
        << "grpc_asr_streams_active "
        << clamp_non_negative(t.streams_active.load(std::memory_order_relaxed)) << '\n';

    out << "# HELP grpc_asr_upload_bytes_resident Bytes currently resident in in-memory upload buffers.\n"
           "# TYPE grpc_asr_upload_bytes_resident gauge\n"
        << "grpc_asr_upload_bytes_resident "
        << clamp_non_negative(t.upload_bytes_resident.load(std::memory_order_relaxed)) << '\n';

    out << "# HELP grpc_asr_memfd_bytes_resident Bytes currently resident in memfd-backed media buffers.\n"
           "# TYPE grpc_asr_memfd_bytes_resident gauge\n"
        << "grpc_asr_memfd_bytes_resident "
        << clamp_non_negative(t.memfd_bytes_resident.load(std::memory_order_relaxed)) << '\n';

    out << "# HELP grpc_asr_first_segment_seconds Latency from stream start to first emitted segment.\n"
           "# TYPE grpc_asr_first_segment_seconds histogram\n";
    for (const auto& [name, slot] : models_copy) {
        for (size_t p = 0; p < kFirstSegmentPathLabels.size(); ++p) {
            const std::string label = "model=\"" + escape_label(name) + "\",path=\"" +
                                      std::string(kFirstSegmentPathLabels[p]) + "\"";
            render_histogram_series(out, "grpc_asr_first_segment_seconds", label,
                                    slot->first_segment[p], kDurationBoundLabels, 1e9);
        }
    }

    out << "# HELP grpc_asr_realtime_factor Decoded audio seconds per wall-clock decode second.\n"
           "# TYPE grpc_asr_realtime_factor histogram\n";
    for (const auto& [key, hist] : rtf_copy) {
        const std::string label = "model=\"" + escape_label(key.first) + "\",backend=\"" +
                                  escape_label(key.second) + "\"";
        render_histogram_series(out, "grpc_asr_realtime_factor", label, *hist, kRtfBoundLabels,
                                static_cast<double>(kRtfScale));
    }

    out << "# HELP grpc_asr_windows_total Total PCM windows decoded.\n"
           "# TYPE grpc_asr_windows_total counter\n"
        << "grpc_asr_windows_total " << t.windows_total.load(std::memory_order_relaxed) << '\n';

    out << "# HELP grpc_asr_window_redecodes_total Window boundary hold-back segments re-decoded in the next window.\n"
           "# TYPE grpc_asr_window_redecodes_total counter\n"
        << "grpc_asr_window_redecodes_total "
        << t.window_redecodes_total.load(std::memory_order_relaxed) << '\n';

    out << "# HELP grpc_asr_stream_end_total Completed Transcribe streams by terminal outcome.\n"
           "# TYPE grpc_asr_stream_end_total counter\n";
    for (size_t i = 0; i < kStreamEndResultLabels.size(); ++i) {
        out << "grpc_asr_stream_end_total{result=\"" << kStreamEndResultLabels[i] << "\"} "
            << t.stream_ends[i].load(std::memory_order_relaxed) << '\n';
    }

    out << "# HELP grpc_asr_tool_runs_total Child ffmpeg and ffprobe invocations by outcome.\n"
           "# TYPE grpc_asr_tool_runs_total counter\n";
    for (size_t tool = 0; tool < kToolKindLabels.size(); ++tool) {
        for (size_t outcome = 0; outcome < kToolOutcomeLabels.size(); ++outcome) {
            out << "grpc_asr_tool_runs_total{tool=\"" << kToolKindLabels[tool] << "\",outcome=\""
                << kToolOutcomeLabels[outcome] << "\"} "
                << t.tool_runs[tool][outcome].load(std::memory_order_relaxed) << '\n';
        }
    }

    out << "# HELP grpc_asr_audio_seconds_total Transcribed audio duration in seconds by model.\n"
           "# TYPE grpc_asr_audio_seconds_total counter\n";
    for (const auto& [name, slot] : models_copy) {
        const double secs =
            static_cast<double>(slot->audio_ms_total.load(std::memory_order_relaxed)) / 1000.0;
        out << "grpc_asr_audio_seconds_total{model=\"" << escape_label(name) << "\"} " << secs
            << '\n';
    }

    out << "# HELP grpc_asr_keyframes_total Video keyframes extracted and emitted.\n"
           "# TYPE grpc_asr_keyframes_total counter\n"
        << "grpc_asr_keyframes_total " << t.keyframes_total.load(std::memory_order_relaxed) << '\n';

    out << "# HELP grpc_asr_warmup_seconds Startup model-state warmup duration in seconds.\n"
           "# TYPE grpc_asr_warmup_seconds gauge\n"
        << "grpc_asr_warmup_seconds "
        << (static_cast<double>(t.warmup_ms.load(std::memory_order_relaxed)) / 1000.0) << '\n';

    out << "# HELP grpc_asr_window_seconds_last Effective PCM window seconds used by the last stream.\n"
           "# TYPE grpc_asr_window_seconds_last gauge\n"
        << "grpc_asr_window_seconds_last " << t.last_window_seconds.load(std::memory_order_relaxed)
        << '\n';

    const ProcessUsage usage = read_process_usage();
    out << "# HELP process_resident_memory_bytes Resident memory size in bytes.\n"
           "# TYPE process_resident_memory_bytes gauge\n"
        << "process_resident_memory_bytes " << usage.resident_bytes << '\n'
        << "# HELP process_cpu_seconds_total Total user and system CPU time spent in seconds.\n"
           "# TYPE process_cpu_seconds_total counter\n"
        << "process_cpu_seconds_total " << usage.cpu_seconds << '\n';

    return out.str();
}

MetricsHttpServer::MetricsHttpServer(const std::string& listen_address, Renderer render)
    : render_(std::move(render)) {
    uint16_t requested_port = 0;
    parse_host_port_endpoint(listen_address, "GRPC_ASR_METRICS_ADDRESS", &host_, &requested_port);

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_ADDRCONFIG;
    addrinfo* results = nullptr;
    const std::string port_str = std::to_string(requested_port);
    const int gai_rc = ::getaddrinfo(host_.c_str(), port_str.c_str(), &hints, &results);
    if (gai_rc != 0 || results == nullptr) {
        throw std::runtime_error("Metrics exporter: cannot resolve '" + listen_address +
                                 "': " + ::gai_strerror(gai_rc));
    }

    std::string last_error = "no usable address";
    for (addrinfo* cur = results; cur != nullptr; cur = cur->ai_next) {
        const int fd =
            ::socket(cur->ai_family, cur->ai_socktype | SOCK_CLOEXEC, cur->ai_protocol);
        if (fd < 0) {
            last_error = std::strerror(errno);
            continue;
        }
        const int enable = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
        if (cur->ai_family == AF_INET6 && host_ == "::") {
            const int disable = 0;
            ::setsockopt(fd, IPPROTO_IPV6, IPV6_V6ONLY, &disable, sizeof(disable));
        }
        if (::bind(fd, cur->ai_addr, cur->ai_addrlen) == 0 && ::listen(fd, 8) == 0) {
            listen_fd_ = fd;
            break;
        }
        last_error = std::strerror(errno);
        ::close(fd);
    }
    ::freeaddrinfo(results);

    if (listen_fd_ < 0) {
        throw std::runtime_error("Metrics exporter: could not listen on " + listen_address + ": " +
                                 last_error);
    }
    port_ = bound_port_of(listen_fd_);
    thread_ = std::thread([this] { serve(); });
}

MetricsHttpServer::~MetricsHttpServer() {
    stopping_.store(true);
    ::shutdown(listen_fd_, SHUT_RDWR);
    if (thread_.joinable()) {
        thread_.join();
    }
    if (listen_fd_ >= 0) {
        ::close(listen_fd_);
    }
}

void MetricsHttpServer::serve() {
    constexpr std::chrono::milliseconds kMaximumBackoff{1000};
    std::chrono::milliseconds backoff{0};
    while (!stopping_.load()) {
        const int client = ::accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
        if (client < 0) {
            if (stopping_.load()) {
                return;
            }
            if (errno == EINTR || errno == ECONNABORTED) {
                continue;
            }
            if (backoff.count() == 0) {
                std::println(stderr, "Metrics exporter: accept failed ({}); retrying",
                             std::strerror(errno));
            }
            backoff =
                std::min(kMaximumBackoff, std::max(std::chrono::milliseconds(10), backoff * 2));
            std::this_thread::sleep_for(backoff);
            continue;
        }
        backoff = std::chrono::milliseconds(0);
        timeval timeout{};
        timeout.tv_sec = 5;
        ::setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        ::setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

        std::string request;
        char buffer[1024];
        while (request.size() < 8192 && !request.contains("\r\n\r\n")) {
            const ssize_t received = ::recv(client, buffer, sizeof(buffer), 0);
            if (received <= 0) {
                break;
            }
            request.append(buffer, static_cast<size_t>(received));
        }

        const size_t line_end = request.find("\r\n");
        const std::string request_line = request.substr(0, line_end);
        if (!request_line.starts_with("GET ")) {
            write_all(client, http_response("405 Method Not Allowed", "text/plain",
                                            "only GET is supported\n"));
        } else if (request_line.starts_with("GET /metrics ") ||
                   request_line.starts_with("GET /metrics?")) {
            try {
                write_all(client, http_response("200 OK", kContentType, render_()));
            } catch (const std::exception& error) {
                write_all(client, http_response("500 Internal Server Error", "text/plain",
                                                std::string("metrics rendering failed: ") +
                                                    error.what() + "\n"));
            }
        } else {
            write_all(client, http_response("404 Not Found", "text/plain", "see /metrics\n"));
        }
        ::close(client);
    }
}

}  // namespace asr
