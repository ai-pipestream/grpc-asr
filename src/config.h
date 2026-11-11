#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace asr {

// Process configuration, entirely from GRPC_ASR_* environment variables.
// Malformed values throw at startup; nothing is silently defaulted away
// from what the operator wrote.
struct Config {
    std::string listen_address = "0.0.0.0:50055";
    // cuda | openvino | cpu. CUDA is the default and fails loud when this
    // build or host cannot provide it; cpu is an explicit choice, never a
    // fallback.
    std::string backend = "cuda";
    int cuda_device = 0;
    // OpenVINO device the whisper encoder compiles for under the openvino
    // backend ("GPU", "GPU.1", ...; other devices need their OpenVINO
    // plugin, and the OpenVINO image ships only the GPU one). An
    // unavailable device fails the boot, like every other backend choice.
    std::string openvino_device = "GPU";
    // Directory where OpenVINO caches compiled encoder blobs; mount a
    // persistent volume here to skip recompiling on container restart.
    std::string openvino_cache_dir = "/tmp/grpc-asr-openvino-cache";
    std::string models_dir = "/models";
    // Model names to serve ("tiny.en", "large-v3", ...). Empty means
    // discover every ggml-*.bin in models_dir. Each name must resolve to
    // an existing weight file or startup fails.
    std::vector<std::string> models;
    // Concurrent transcriptions per model: whisper states created at
    // startup per loaded context.
    size_t concurrency = 2;
    // Run one second of digital silence through every loaded whisper state
    // at startup so backend kernels and buffers initialize before SERVING.
    bool warmup = true;
    // How long a stream may wait for a free whisper state before it fails
    // RESOURCE_EXHAUSTED; 0 waits as long as the client does. A stream
    // also leaves the queue the moment its client cancels or its deadline
    // passes. The default matches gRParse's ASR deadline.
    size_t queue_timeout_seconds = 1800;
    size_t max_media_bytes = 256ULL * 1024 * 1024;
    size_t max_duration_seconds = 14400;
    // Upper bound on the PCM window fed to whisper_full per iteration;
    // bounds resident PCM.
    size_t window_seconds = 480;
    // Minimum buffered PCM window (seconds) before a stalled upload starts
    // an early decode pass, and lower bound when clamping client hints.
    size_t min_window_seconds = 30;
    // Milliseconds a PCM read may block once min_window_seconds is buffered
    // before the transcriber decodes what it has; 0 disables stall cuts.
    size_t window_stall_ms = 500;
    // Maximum source channels accepted under CHANNEL_MODE_PER_CHANNEL.
    size_t max_channels = 2;
    // whisper decode threads per transcription.
    size_t threads = 4;
    // Server default when TranscribeOptions.keyframe_interval_seconds is 0.
    size_t keyframe_interval_seconds = 10;
    // Whether the folded Document locates every word, not just every
    // segment: one ProvenanceItem per word on the segment's text item.
    // On by default; the cost is one provenance entry per word (roughly
    // 150 per transcribed minute), which a many-hour transcript may not
    // want to carry in one message. Words only exist when the client asked
    // for TranscribeOptions.word_timestamps, so this narrows that further,
    // it never adds work of its own.
    bool document_word_provenance = true;
    // 0 disables the stdout metrics line.
    size_t metrics_interval_seconds = 60;
    // Optional HTTP/1.1 host:port serving Prometheus text metrics at
    // /metrics (e.g. "0.0.0.0:9090"). Empty disables the listener.
    std::string metrics_address;
    // Allowlist of host[:port] patterns that TranscribeOptions.remote may
    // call. Empty disables remote transcription (FAILED_PRECONDITION).
    std::vector<std::string> remote_allowed_hosts;
    // ffmpeg/ffprobe children with no output for this long are killed.
    size_t tool_inactivity_seconds = 120;
    std::string ffmpeg = "ffmpeg";
    std::string ffprobe = "ffprobe";
};

// Parses a "<host>:<port>" or "[<ipv6>]:<port>" endpoint string, validating
// that host is non-empty and port is in 0..65535. Throws std::invalid_argument
// naming `var_name` when malformed.
void parse_host_port_endpoint(const std::string& endpoint, const char* var_name,
                              std::string* host_out, uint16_t* port_out);

// Reads and validates the environment. Throws std::invalid_argument with
// the variable name and accepted range on any malformed value.
Config load_config_from_env();

}  // namespace asr
