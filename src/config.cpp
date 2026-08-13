#include "config.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

namespace asr {

namespace {

size_t configured_size(const char* name, size_t fallback, size_t minimum, size_t maximum) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    char* end = nullptr;
    unsigned long long value = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || value < minimum || value > maximum) {
        throw std::invalid_argument(std::string(name) + " must be an integer between " +
                                    std::to_string(minimum) + " and " + std::to_string(maximum));
    }
    return static_cast<size_t>(value);
}

std::string configured_string(const char* name, const std::string& fallback) {
    const char* raw = std::getenv(name);
    return raw == nullptr || *raw == '\0' ? fallback : raw;
}

std::vector<std::string> configured_list(const char* name) {
    std::vector<std::string> values;
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return values;
    }
    std::string buffer(raw);
    size_t start = 0;
    while (start <= buffer.size()) {
        size_t comma = buffer.find(',', start);
        if (comma == std::string::npos) {
            comma = buffer.size();
        }
        std::string item = buffer.substr(start, comma - start);
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        if (!item.empty()) {
            values.push_back(item);
        }
        start = comma + 1;
    }
    if (values.empty()) {
        throw std::invalid_argument(std::string(name) + " is set but names no models");
    }
    return values;
}

}  // namespace

Config load_config_from_env() {
    Config config;
    config.listen_address = configured_string("GRPC_ASR_LISTEN_ADDRESS", config.listen_address);
    config.backend = configured_string("GRPC_ASR_BACKEND", config.backend);
    if (config.backend != "cuda" && config.backend != "openvino" && config.backend != "cpu") {
        throw std::invalid_argument("GRPC_ASR_BACKEND must be cuda, openvino, or cpu");
    }
    config.cuda_device =
        static_cast<int>(configured_size("GRPC_ASR_CUDA_DEVICE", 0, 0, 63));
    config.models_dir = configured_string("GRPC_ASR_MODELS_DIR", config.models_dir);
    config.models = configured_list("GRPC_ASR_MODELS");
    config.concurrency = configured_size("GRPC_ASR_CONCURRENCY", config.concurrency, 1, 64);
    config.max_media_bytes = configured_size("GRPC_ASR_MAX_MEDIA_BYTES", config.max_media_bytes,
                                             1024, 4ULL * 1024 * 1024 * 1024);
    config.max_duration_seconds = configured_size("GRPC_ASR_MAX_DURATION_SECONDS",
                                                  config.max_duration_seconds, 1, 86400 * 7);
    config.window_seconds =
        configured_size("GRPC_ASR_WINDOW_SECONDS", config.window_seconds, 10, 3600);
    size_t hw = std::max<size_t>(1, std::thread::hardware_concurrency());
    config.threads =
        configured_size("GRPC_ASR_THREADS", std::min<size_t>(4, hw), 1, 256);
    config.keyframe_interval_seconds = configured_size(
        "GRPC_ASR_KEYFRAME_INTERVAL_SECONDS", config.keyframe_interval_seconds, 1, 3600);
    config.metrics_interval_seconds = configured_size(
        "GRPC_ASR_METRICS_INTERVAL_SECONDS", config.metrics_interval_seconds, 0, 86400);
    config.tool_inactivity_seconds = configured_size(
        "GRPC_ASR_TOOL_INACTIVITY_SECONDS", config.tool_inactivity_seconds, 1, 3600);
    config.ffmpeg = configured_string("GRPC_ASR_FFMPEG", config.ffmpeg);
    config.ffprobe = configured_string("GRPC_ASR_FFPROBE", config.ffprobe);
    return config;
}

}  // namespace asr
