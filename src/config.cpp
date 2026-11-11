#include "config.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
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
    if (*raw == '-' || *raw == '+') {
        throw std::invalid_argument(std::string(name) + " must be an integer between " +
                                    std::to_string(minimum) + " and " + std::to_string(maximum));
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

bool configured_bool(const char* name, bool fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return fallback;
    }
    std::string value(raw);
    std::ranges::transform(value, value.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "1" || value == "true") {
        return true;
    }
    if (value == "0" || value == "false") {
        return false;
    }
    throw std::invalid_argument(std::string(name) + " must be 0, 1, true, or false");
}

std::vector<std::string> configured_list(const char* name, bool allow_empty_env,
                                         const char* item_noun) {
    std::vector<std::string> values;
    const char* raw = std::getenv(name);
    if (raw == nullptr || (allow_empty_env && *raw == '\0')) {
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
        const size_t first = item.find_first_not_of(" \t");
        if (first != std::string::npos) {
            const size_t last = item.find_last_not_of(" \t");
            values.push_back(item.substr(first, last - first + 1));
        }
        start = comma + 1;
    }
    if (values.empty()) {
        throw std::invalid_argument(std::string(name) + " is set but names no " + item_noun);
    }
    return values;
}

void validate_host_pattern(const std::string& pattern, const char* name) {
    if (pattern.empty() || pattern.contains("://") || pattern.contains('/') ||
        pattern.contains('@') ||
        std::ranges::any_of(pattern, [](unsigned char c) { return std::isspace(c) != 0; })) {
        throw std::invalid_argument(std::string(name) + " contains invalid host[:port] pattern '" +
                                    pattern + "'");
    }
    std::string host;
    std::string port_str;
    if (pattern.front() == '[') {
        const size_t close = pattern.find(']');
        if (close == std::string::npos || close == 1) {
            throw std::invalid_argument(std::string(name) + " contains malformed IPv6 pattern '" +
                                        pattern + "'");
        }
        host = pattern.substr(1, close - 1);
        if (close + 1 < pattern.size()) {
            if (pattern[close + 1] != ':') {
                throw std::invalid_argument(std::string(name) +
                                            " contains malformed IPv6 pattern '" + pattern + "'");
            }
            port_str = pattern.substr(close + 2);
            if (port_str.empty()) {
                throw std::invalid_argument(std::string(name) + " pattern '" + pattern +
                                            "' has an empty port");
            }
        }
    } else {
        const size_t colon = pattern.rfind(':');
        if (colon != std::string::npos) {
            if (pattern.find(':') != colon) {
                throw std::invalid_argument(std::string(name) +
                                            " IPv6 literals must be bracketed: '" + pattern + "'");
            }
            host = pattern.substr(0, colon);
            port_str = pattern.substr(colon + 1);
            if (port_str.empty()) {
                throw std::invalid_argument(std::string(name) + " pattern '" + pattern +
                                            "' has an empty port");
            }
        } else {
            host = pattern;
        }
    }
    if (host.empty()) {
        throw std::invalid_argument(std::string(name) + " pattern '" + pattern +
                                    "' has an empty host");
    }
    if (!port_str.empty() && port_str != "*") {
        char* end = nullptr;
        unsigned long port = std::strtoul(port_str.c_str(), &end, 10);
        if (end == port_str.c_str() || *end != '\0' || port < 1 || port > 65535) {
            throw std::invalid_argument(std::string(name) + " pattern '" + pattern +
                                        "' port must be 1..65535 or *");
        }
    }
}

}  // namespace

void parse_host_port_endpoint(const std::string& endpoint, const char* var_name,
                              std::string* host_out, uint16_t* port_out) {
    std::string host;
    std::string port_str;
    if (!endpoint.empty() && endpoint.front() == '[') {
        const size_t close = endpoint.find(']');
        if (close == std::string::npos || close == 1 || close + 2 > endpoint.size() ||
            endpoint[close + 1] != ':') {
            throw std::invalid_argument(std::string(var_name) +
                                        " must be <host>:<port> or [<ipv6>]:<port>");
        }
        host = endpoint.substr(1, close - 1);
        port_str = endpoint.substr(close + 2);
    } else {
        const size_t colon = endpoint.rfind(':');
        if (colon == std::string::npos || colon == 0 || endpoint.find(':') != colon) {
            throw std::invalid_argument(std::string(var_name) +
                                        " must be <host>:<port> or [<ipv6>]:<port>");
        }
        host = endpoint.substr(0, colon);
        port_str = endpoint.substr(colon + 1);
    }
    if (port_str.empty() || port_str.front() == '-' || port_str.front() == '+') {
        throw std::invalid_argument(std::string(var_name) + " port must be 0..65535");
    }
    char* end = nullptr;
    unsigned long port = std::strtoul(port_str.c_str(), &end, 10);
    if (end == port_str.c_str() || *end != '\0' || port > 65535) {
        throw std::invalid_argument(std::string(var_name) + " port must be 0..65535");
    }
    if (host_out != nullptr) {
        *host_out = std::move(host);
    }
    if (port_out != nullptr) {
        *port_out = static_cast<uint16_t>(port);
    }
}

Config load_config_from_env() {
    Config config;
    config.listen_address = configured_string("GRPC_ASR_LISTEN_ADDRESS", config.listen_address);
    config.backend = configured_string("GRPC_ASR_BACKEND", config.backend);
    if (config.backend != "cuda" && config.backend != "openvino" && config.backend != "cpu") {
        throw std::invalid_argument("GRPC_ASR_BACKEND must be cuda, openvino, or cpu");
    }
    config.cuda_device =
        static_cast<int>(configured_size("GRPC_ASR_CUDA_DEVICE", 0, 0, 63));
    config.openvino_device = configured_string("GRPC_ASR_OPENVINO_DEVICE", config.openvino_device);
    config.openvino_cache_dir =
        configured_string("GRPC_ASR_OPENVINO_CACHE_DIR", config.openvino_cache_dir);
    config.models_dir = configured_string("GRPC_ASR_MODELS_DIR", config.models_dir);
    config.models = configured_list("GRPC_ASR_MODELS", /*allow_empty_env=*/false, "models");
    config.concurrency = configured_size("GRPC_ASR_CONCURRENCY", config.concurrency, 1, 64);
    config.warmup = configured_bool("GRPC_ASR_WARMUP", config.warmup);
    config.queue_timeout_seconds = configured_size("GRPC_ASR_QUEUE_TIMEOUT_SECONDS",
                                                   config.queue_timeout_seconds, 0, 86400 * 7);
    config.max_media_bytes = configured_size("GRPC_ASR_MAX_MEDIA_BYTES", config.max_media_bytes,
                                             1024, 4ULL * 1024 * 1024 * 1024);
    config.max_duration_seconds = configured_size("GRPC_ASR_MAX_DURATION_SECONDS",
                                                  config.max_duration_seconds, 1, 86400 * 7);
    config.window_seconds =
        configured_size("GRPC_ASR_WINDOW_SECONDS", config.window_seconds, 10, 3600);
    config.min_window_seconds = configured_size(
        "GRPC_ASR_MIN_WINDOW_SECONDS",
        std::min<size_t>(config.min_window_seconds, config.window_seconds), 1, 3600);
    if (config.min_window_seconds > config.window_seconds) {
        throw std::invalid_argument(
            "GRPC_ASR_MIN_WINDOW_SECONDS must not exceed GRPC_ASR_WINDOW_SECONDS");
    }
    config.window_stall_ms =
        configured_size("GRPC_ASR_WINDOW_STALL_MS", config.window_stall_ms, 0, 60000);
    config.max_channels = configured_size("GRPC_ASR_MAX_CHANNELS", config.max_channels, 1, 16);
    size_t hw = std::max<size_t>(1, std::thread::hardware_concurrency());
    config.threads =
        configured_size("GRPC_ASR_THREADS", std::min<size_t>(4, hw), 1, 256);
    config.keyframe_interval_seconds = configured_size(
        "GRPC_ASR_KEYFRAME_INTERVAL_SECONDS", config.keyframe_interval_seconds, 1, 3600);
    config.document_word_provenance = configured_bool("GRPC_ASR_DOCUMENT_WORD_PROVENANCE",
                                                      config.document_word_provenance);
    config.metrics_interval_seconds = configured_size(
        "GRPC_ASR_METRICS_INTERVAL_SECONDS", config.metrics_interval_seconds, 0, 86400);
    config.metrics_address = configured_string("GRPC_ASR_METRICS_ADDRESS", config.metrics_address);
    if (!config.metrics_address.empty()) {
        parse_host_port_endpoint(config.metrics_address, "GRPC_ASR_METRICS_ADDRESS", nullptr,
                                 nullptr);
    }
    config.remote_allowed_hosts =
        configured_list("GRPC_ASR_REMOTE_ALLOWED_HOSTS", /*allow_empty_env=*/true, "hosts");
    for (const std::string& pattern : config.remote_allowed_hosts) {
        validate_host_pattern(pattern, "GRPC_ASR_REMOTE_ALLOWED_HOSTS");
    }
    config.tool_inactivity_seconds = configured_size(
        "GRPC_ASR_TOOL_INACTIVITY_SECONDS", config.tool_inactivity_seconds, 1, 3600);
    config.ffmpeg = configured_string("GRPC_ASR_FFMPEG", config.ffmpeg);
    config.ffprobe = configured_string("GRPC_ASR_FFPROBE", config.ffprobe);
    return config;
}

}  // namespace asr
