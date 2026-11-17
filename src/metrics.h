#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace asr {

// Terminal outcome classes for grpc_asr_stream_end_total{result=...}.
enum class StreamEndResult : size_t {
    kOk = 0,
    kCancelled,
    kDeadline,
    kQueueTimeout,
    kByteCap,
    kDurationCap,
    kInvalidMedia,
    kUnknownContainer,
    kToolError,
    kInternal,
    kCount,
};

constexpr std::array<std::string_view, static_cast<size_t>(StreamEndResult::kCount)>
    kStreamEndResultLabels = {
        "ok",          "cancelled", "deadline",          "queue_timeout", "byte_cap",
        "duration_cap", "invalid_media", "unknown_container", "tool_error", "internal",
};

// Child process kinds and outcomes for grpc_asr_tool_runs_total{tool,outcome}.
enum class ToolKind : size_t {
    kFfmpeg = 0,
    kFfprobe,
    kCount,
};

constexpr std::array<std::string_view, static_cast<size_t>(ToolKind::kCount)> kToolKindLabels = {
    "ffmpeg",
    "ffprobe",
};

enum class ToolOutcome : size_t {
    kOk = 0,
    kRejected,
    kExecFailed,
    kInactivityKilled,
    kCancelled,
    kCount,
};

constexpr std::array<std::string_view, static_cast<size_t>(ToolOutcome::kCount)>
    kToolOutcomeLabels = {
        "ok", "rejected", "exec_failed", "inactivity_killed", "cancelled",
};

// Ingest path label for grpc_asr_first_segment_seconds{model,path}.
enum class FirstSegmentPath : size_t {
    kLive = 0,
    kFfmpeg,
    kCount,
};

constexpr std::array<std::string_view, static_cast<size_t>(FirstSegmentPath::kCount)>
    kFirstSegmentPathLabels = {
        "live",
        "ffmpeg",
};

// Per-stream telemetry returned alongside grpc::Status by process_stream so
// AsrServiceImpl records every counter from one structured summary.
struct StreamMetrics {
    StreamEndResult result = StreamEndResult::kInternal;
    std::string model;
    uint64_t audio_ms = 0;
    uint64_t keyframes = 0;
    uint64_t windows = 0;
    uint64_t window_redecodes = 0;
    uint32_t chosen_window_seconds = 0;
};

// Snapshot of process-wide counters for the periodic stdout metrics line.
struct MetricsSummary {
    uint64_t streams_active = 0;
    uint64_t upload_bytes_resident = 0;
    uint64_t memfd_bytes_resident = 0;
    uint64_t windows_total = 0;
    uint64_t window_redecodes_total = 0;
    uint64_t keyframes_total = 0;
    uint64_t warmup_ms = 0;
    uint32_t last_window_seconds = 0;
    std::array<uint64_t, static_cast<size_t>(StreamEndResult::kCount)> stream_ends{};
    uint64_t audio_ms_total = 0;
};

// Model-pool state/queue telemetry hooks.
void metrics_add_model_states(std::string_view model, uint64_t count);
void metrics_remove_model_states(std::string_view model, uint64_t count);
void metrics_inc_model_busy(std::string_view model);
void metrics_dec_model_busy(std::string_view model);
void metrics_inc_queue_depth(std::string_view model);
void metrics_dec_queue_depth(std::string_view model);
void metrics_observe_queue_wait(std::string_view model, uint64_t duration_ns);
void metrics_record_warmup_ms(uint64_t warmup_ms);

// Stream, resident-memory, latency, RTF, and child-process hooks.
void metrics_inc_streams_active();
void metrics_dec_streams_active();
void metrics_add_upload_bytes(int64_t delta);
void metrics_add_memfd_bytes(int64_t delta);
void metrics_observe_first_segment(std::string_view model, FirstSegmentPath path,
                                   uint64_t duration_ns);
void metrics_observe_realtime_factor(std::string_view model, std::string_view backend,
                                     double audio_seconds_per_wall_second);
void metrics_record_stream_end(const StreamMetrics& stream);
void metrics_record_tool_run(ToolKind tool, ToolOutcome outcome);

// Snapshot and Prometheus text exposition renderer.
MetricsSummary metrics_summary();
std::string render_prometheus_metrics();

// Minimal single-threaded HTTP/1.1 listener serving Prometheus text exposition
// at GET /metrics, matching gRParse's MetricsHttpServer shape.
class MetricsHttpServer {
  public:
    using Renderer = std::function<std::string()>;

    explicit MetricsHttpServer(const std::string& listen_address,
                               Renderer render = render_prometheus_metrics);
    ~MetricsHttpServer();

    MetricsHttpServer(const MetricsHttpServer&) = delete;
    MetricsHttpServer& operator=(const MetricsHttpServer&) = delete;

    uint16_t port() const { return port_; }
    const std::string& host() const { return host_; }

  private:
    void serve();

    Renderer render_;
    std::string host_;
    int listen_fd_ = -1;
    uint16_t port_ = 0;
    std::atomic<bool> stopping_{false};
    std::thread thread_;
};

}  // namespace asr
