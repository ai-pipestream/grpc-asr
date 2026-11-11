// Environment parsing: defaults, the models list, and the fail-loud
// contract on malformed values. Every case pins the full GRPC_ASR_*
// environment first so ambient variables cannot leak into a check.

#include "config.h"

#include <cstdlib>
#include <print>
#include <stdexcept>
#include <string>

#include "fixture.h"

using asr::Config;
using asr::load_config_from_env;

namespace {

void clear_env() {
    for (const char* name :
         {"GRPC_ASR_LISTEN_ADDRESS", "GRPC_ASR_BACKEND", "GRPC_ASR_CUDA_DEVICE",
          "GRPC_ASR_OPENVINO_DEVICE", "GRPC_ASR_OPENVINO_CACHE_DIR",
          "GRPC_ASR_MODELS_DIR", "GRPC_ASR_MODELS", "GRPC_ASR_CONCURRENCY",
          "GRPC_ASR_WARMUP", "GRPC_ASR_QUEUE_TIMEOUT_SECONDS",
          "GRPC_ASR_MAX_MEDIA_BYTES", "GRPC_ASR_MAX_DURATION_SECONDS",
          "GRPC_ASR_WINDOW_SECONDS", "GRPC_ASR_MIN_WINDOW_SECONDS",
          "GRPC_ASR_WINDOW_STALL_MS", "GRPC_ASR_MAX_CHANNELS",
          "GRPC_ASR_THREADS", "GRPC_ASR_KEYFRAME_INTERVAL_SECONDS",
          "GRPC_ASR_METRICS_INTERVAL_SECONDS", "GRPC_ASR_METRICS_ADDRESS",
          "GRPC_ASR_REMOTE_ALLOWED_HOSTS", "GRPC_ASR_TOOL_INACTIVITY_SECONDS",
          "GRPC_ASR_DOCUMENT_WORD_PROVENANCE", "GRPC_ASR_FFMPEG", "GRPC_ASR_FFPROBE"}) {
        ::unsetenv(name);
    }
}

// True when load_config_from_env throws std::invalid_argument naming the
// offending variable.
bool rejects(const char* name, const char* value) {
    clear_env();
    ::setenv(name, value, 1);
    try {
        (void)load_config_from_env();
    } catch (const std::invalid_argument& error) {
        return std::string(error.what()).contains(name);
    }
    return false;
}

void verify_defaults() {
    clear_env();
    const Config config = load_config_from_env();
    require(config.listen_address == "0.0.0.0:50055", "default listen address");
    require(config.backend == "cuda", "cuda is the default backend");
    require(config.openvino_device == "GPU", "the OpenVINO encoder targets the GPU by default");
    require(config.openvino_cache_dir == "/tmp/grpc-asr-openvino-cache",
            "default OpenVINO compile cache dir");
    require(config.models_dir == "/models", "default models dir");
    require(config.models.empty(), "no models named means discovery");
    require(config.concurrency == 2, "default concurrency");
    require(config.warmup, "startup warmup is enabled by default");
    require(config.queue_timeout_seconds == 1800, "default queue timeout");
    require(config.max_media_bytes == 256ULL * 1024 * 1024, "default media cap");
    require(config.window_seconds == 480, "default window");
    require(config.min_window_seconds == 30, "default min window");
    require(config.window_stall_ms == 500, "default window stall threshold");
    require(config.max_channels == 2, "default per-channel cap");
    require(config.metrics_address.empty(), "Prometheus listener is off by default");
    require(config.remote_allowed_hosts.empty(), "remote transcription is off by default");
    require(config.ffmpeg == "ffmpeg" && config.ffprobe == "ffprobe", "default tool names");
    require(config.document_word_provenance, "word provenance is on unless turned off");
}

void verify_word_provenance_switch() {
    for (const char* off : {"0", "false", "FALSE"}) {
        clear_env();
        ::setenv("GRPC_ASR_DOCUMENT_WORD_PROVENANCE", off, 1);
        require(!load_config_from_env().document_word_provenance,
                std::string("word provenance off via ") + off);
    }
    for (const char* on : {"1", "true", "True"}) {
        clear_env();
        ::setenv("GRPC_ASR_DOCUMENT_WORD_PROVENANCE", on, 1);
        require(load_config_from_env().document_word_provenance,
                std::string("word provenance on via ") + on);
    }
    require(rejects("GRPC_ASR_DOCUMENT_WORD_PROVENANCE", "maybe"),
            "a non-boolean word provenance switch fails loud");
}

void verify_overrides() {
    clear_env();
    ::setenv("GRPC_ASR_BACKEND", "cpu", 1);
    ::setenv("GRPC_ASR_LISTEN_ADDRESS", "127.0.0.1:9", 1);
    ::setenv("GRPC_ASR_CONCURRENCY", "5", 1);
    ::setenv("GRPC_ASR_THREADS", "8", 1);
    ::setenv("GRPC_ASR_OPENVINO_DEVICE", "NPU", 1);
    const Config config = load_config_from_env();
    require(config.backend == "cpu", "backend override");
    require(config.listen_address == "127.0.0.1:9", "listen address override");
    require(config.concurrency == 5, "concurrency override");
    require(config.threads == 8, "threads override");
    require(config.openvino_device == "NPU", "OpenVINO device override");
}

void verify_models_list() {
    clear_env();
    ::setenv("GRPC_ASR_MODELS", " tiny.en , large-v3 ,", 1);
    const Config config = load_config_from_env();
    require(config.models.size() == 2, "two models parsed");
    require(config.models[0] == "tiny.en" && config.models[1] == "large-v3",
            "names are trimmed and empty items dropped");
}

void verify_rejects() {
    require(rejects("GRPC_ASR_BACKEND", "metal"), "unknown backend fails loud");
    require(rejects("GRPC_ASR_MODELS", " , "), "a models list naming no models fails loud");
    require(rejects("GRPC_ASR_CONCURRENCY", "0"), "below-minimum concurrency fails loud");
    require(rejects("GRPC_ASR_CONCURRENCY", "65"), "above-maximum concurrency fails loud");
    require(rejects("GRPC_ASR_CONCURRENCY", "two"), "non-numeric size fails loud");
    require(rejects("GRPC_ASR_CONCURRENCY", "2x"), "trailing junk fails loud");
    require(rejects("GRPC_ASR_WINDOW_SECONDS", "5"), "sub-minimum window fails loud");
    require(rejects("GRPC_ASR_MAX_MEDIA_BYTES", "512"), "sub-minimum media cap fails loud");
}

void verify_queue_timeout() {
    clear_env();
    ::setenv("GRPC_ASR_QUEUE_TIMEOUT_SECONDS", "30", 1);
    require(load_config_from_env().queue_timeout_seconds == 30, "queue timeout override");
    clear_env();
    ::setenv("GRPC_ASR_QUEUE_TIMEOUT_SECONDS", "0", 1);
    require(load_config_from_env().queue_timeout_seconds == 0,
            "queue timeout 0 waits as long as the client does");
    require(rejects("GRPC_ASR_QUEUE_TIMEOUT_SECONDS", "-1"), "a negative queue timeout fails loud");
    require(rejects("GRPC_ASR_QUEUE_TIMEOUT_SECONDS", "soon"),
            "a non-numeric queue timeout fails loud");
}

void verify_metrics_can_be_disabled() {
    clear_env();
    ::setenv("GRPC_ASR_METRICS_INTERVAL_SECONDS", "0", 1);
    require(load_config_from_env().metrics_interval_seconds == 0,
            "metrics interval 0 is a valid off switch");
}

void verify_warmup_and_openvino_cache() {
    clear_env();
    ::setenv("GRPC_ASR_WARMUP", "0", 1);
    ::setenv("GRPC_ASR_OPENVINO_CACHE_DIR", "/var/cache/openvino", 1);
    const Config config = load_config_from_env();
    require(!config.warmup, "warmup disabled via 0");
    require(config.openvino_cache_dir == "/var/cache/openvino",
            "OpenVINO cache dir overridden");
    require(rejects("GRPC_ASR_WARMUP", "later"), "non-boolean warmup fails loud");
}

void verify_window_and_channel_knobs() {
    clear_env();
    ::setenv("GRPC_ASR_WINDOW_SECONDS", "15", 1);
    const Config auto_clamped = load_config_from_env();
    require(auto_clamped.window_seconds == 15 && auto_clamped.min_window_seconds == 15,
            "default min_window_seconds clamps down to window_seconds when unset");

    clear_env();
    ::setenv("GRPC_ASR_WINDOW_SECONDS", "120", 1);
    ::setenv("GRPC_ASR_MIN_WINDOW_SECONDS", "20", 1);
    ::setenv("GRPC_ASR_WINDOW_STALL_MS", "250", 1);
    ::setenv("GRPC_ASR_MAX_CHANNELS", "4", 1);
    const Config tuned = load_config_from_env();
    require(tuned.min_window_seconds == 20, "explicit min_window_seconds");
    require(tuned.window_stall_ms == 250, "explicit window_stall_ms");
    require(tuned.max_channels == 4, "explicit max_channels");

    clear_env();
    ::setenv("GRPC_ASR_WINDOW_SECONDS", "20", 1);
    ::setenv("GRPC_ASR_MIN_WINDOW_SECONDS", "30", 1);
    bool inverted_rejected = false;
    try {
        (void)load_config_from_env();
    } catch (const std::invalid_argument& error) {
        inverted_rejected = std::string(error.what()).contains("GRPC_ASR_MIN_WINDOW_SECONDS");
    }
    require(inverted_rejected, "min_window_seconds above window_seconds fails loud");
    require(rejects("GRPC_ASR_MAX_CHANNELS", "0"), "zero max_channels fails loud");
    require(rejects("GRPC_ASR_MAX_CHANNELS", "17"), "above-cap max_channels fails loud");
}

void verify_metrics_and_remote_allowlist() {
    clear_env();
    ::setenv("GRPC_ASR_METRICS_ADDRESS", "127.0.0.1:0", 1);
    ::setenv("GRPC_ASR_REMOTE_ALLOWED_HOSTS", " whisper:8000 , api.example.com , [::1]:9000 ", 1);
    const Config config = load_config_from_env();
    require(config.metrics_address == "127.0.0.1:0", "metrics address parsed");
    require(config.remote_allowed_hosts.size() == 3 &&
                config.remote_allowed_hosts[0] == "whisper:8000" &&
                config.remote_allowed_hosts[1] == "api.example.com" &&
                config.remote_allowed_hosts[2] == "[::1]:9000",
            "remote allowlist trimmed and split");

    clear_env();
    ::setenv("GRPC_ASR_REMOTE_ALLOWED_HOSTS", "", 1);
    require(load_config_from_env().remote_allowed_hosts.empty(),
            "empty GRPC_ASR_REMOTE_ALLOWED_HOSTS disables remote cleanly");

    require(rejects("GRPC_ASR_METRICS_ADDRESS", "9090"), "port-only metrics address fails loud");
    require(rejects("GRPC_ASR_METRICS_ADDRESS", "0.0.0.0:99999"),
            "out-of-range metrics port fails loud");
    require(rejects("GRPC_ASR_REMOTE_ALLOWED_HOSTS", " , "),
            "blank non-empty remote allowlist fails loud");
    require(rejects("GRPC_ASR_REMOTE_ALLOWED_HOSTS", "https://api.example.com"),
            "scheme in remote allowlist fails loud");
    require(rejects("GRPC_ASR_REMOTE_ALLOWED_HOSTS", "api.example.com:70000"),
            "bad port in remote allowlist fails loud");
}

}  // namespace

int main() {
    try {
        verify_defaults();
        verify_overrides();
        verify_models_list();
        verify_rejects();
        verify_word_provenance_switch();
        verify_queue_timeout();
        verify_metrics_can_be_disabled();
        verify_warmup_and_openvino_cache();
        verify_window_and_channel_knobs();
        verify_metrics_and_remote_allowlist();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("config-test passed");
    return 0;
}
