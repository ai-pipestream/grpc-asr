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
          "GRPC_ASR_MODELS_DIR", "GRPC_ASR_MODELS", "GRPC_ASR_CONCURRENCY",
          "GRPC_ASR_MAX_MEDIA_BYTES", "GRPC_ASR_MAX_DURATION_SECONDS",
          "GRPC_ASR_WINDOW_SECONDS", "GRPC_ASR_THREADS", "GRPC_ASR_KEYFRAME_INTERVAL_SECONDS",
          "GRPC_ASR_METRICS_INTERVAL_SECONDS", "GRPC_ASR_TOOL_INACTIVITY_SECONDS",
          "GRPC_ASR_FFMPEG", "GRPC_ASR_FFPROBE"}) {
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
    require(config.models_dir == "/models", "default models dir");
    require(config.models.empty(), "no models named means discovery");
    require(config.concurrency == 2, "default concurrency");
    require(config.max_media_bytes == 256ULL * 1024 * 1024, "default media cap");
    require(config.window_seconds == 480, "default window");
    require(config.ffmpeg == "ffmpeg" && config.ffprobe == "ffprobe", "default tool names");
}

void verify_overrides() {
    clear_env();
    ::setenv("GRPC_ASR_BACKEND", "cpu", 1);
    ::setenv("GRPC_ASR_LISTEN_ADDRESS", "127.0.0.1:9", 1);
    ::setenv("GRPC_ASR_CONCURRENCY", "5", 1);
    ::setenv("GRPC_ASR_THREADS", "8", 1);
    const Config config = load_config_from_env();
    require(config.backend == "cpu", "backend override");
    require(config.listen_address == "127.0.0.1:9", "listen address override");
    require(config.concurrency == 5, "concurrency override");
    require(config.threads == 8, "threads override");
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

void verify_metrics_can_be_disabled() {
    clear_env();
    ::setenv("GRPC_ASR_METRICS_INTERVAL_SECONDS", "0", 1);
    require(load_config_from_env().metrics_interval_seconds == 0,
            "metrics interval 0 is a valid off switch");
}

}  // namespace

int main() {
    try {
        verify_defaults();
        verify_overrides();
        verify_models_list();
        verify_rejects();
        verify_metrics_can_be_disabled();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("config-test passed");
    return 0;
}
