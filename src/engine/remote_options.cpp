#include "engine/remote_options.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <string_view>

namespace asr::engine {
namespace asrv1 = ai::pipestream::asr::v1;

namespace {

grpc::Status invalid(std::string message) {
    return {grpc::StatusCode::INVALID_ARGUMENT, std::move(message)};
}

bool starts_with_ci(std::string_view text, std::string_view prefix) {
    if (text.size() < prefix.size()) return false;
    for (size_t i = 0; i < prefix.size(); ++i) {
        const char c = text[i];
        const char lower = (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
        if (lower != prefix[i]) return false;
    }
    return true;
}

// RFC 9110 token characters.
bool is_token(std::string_view name) {
    if (name.empty()) return false;
    return std::ranges::all_of(name, [](char c) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) return true;
        return std::string_view("!#$%&'*+-.^_`|~").find(c) != std::string_view::npos;
    });
}

// Python's repr of a float, which str() uses: shortest round trip, with
// ".0" for integral values and inf/nan spelled as Python spells them.
std::string python_float(double value) {
    if (std::isnan(value)) return "nan";
    if (std::isinf(value)) return value > 0 ? "inf" : "-inf";
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    std::string out(buffer, result.ptr);
    if (out.find_first_of(".en") == std::string::npos) out += ".0";
    return out;
}

}  // namespace

grpc::Status validate_remote_options(const asrv1::RemoteTranscriptionOptions& options) {
    const std::string& url = options.url();
    const bool http = starts_with_ci(url, "http://");
    const bool https = starts_with_ci(url, "https://");
    if (!http && !https) {
        return invalid("RemoteTranscriptionOptions.url must be an http or https URL");
    }
    const std::string_view rest = std::string_view(url).substr(http ? 7 : 8);
    if (rest.empty() || rest.front() == '/' || rest.front() == '?' || rest.front() == '#') {
        return invalid("RemoteTranscriptionOptions.url has no host");
    }
    if (url.find_first_of(" \r\n\t") != std::string::npos) {
        return invalid("RemoteTranscriptionOptions.url must not contain whitespace");
    }
    for (const auto& [name, value] : options.headers()) {
        if (!is_token(name)) {
            return invalid("RemoteTranscriptionOptions.headers key '" + name + "' is not an HTTP token");
        }
        if (value.find_first_of("\r\n") != std::string::npos) {
            return invalid("RemoteTranscriptionOptions.headers['" + name + "'] contains CR or LF");
        }
    }
    for (const auto& [key, value] : options.params()) {
        if (key.empty()) return invalid("RemoteTranscriptionOptions.params has an empty key");
        if (value.kind_case() == asrv1::ScalarValue::KIND_NOT_SET) {
            return invalid("RemoteTranscriptionOptions.params['" + key + "'] has no value");
        }
    }
    if (options.has_timeout_seconds()) {
        const double timeout = options.timeout_seconds();
        if (!std::isfinite(timeout) || timeout <= 0.0) {
            return invalid("RemoteTranscriptionOptions.timeout_seconds must be finite and > 0");
        }
    }
    return grpc::Status::OK;
}

std::string scalar_form_value(const asrv1::ScalarValue& value) {
    switch (value.kind_case()) {
        case asrv1::ScalarValue::kStringValue:
            return value.string_value();
        case asrv1::ScalarValue::kIntValue:
            return std::to_string(value.int_value());
        case asrv1::ScalarValue::kUintValue:
            return std::to_string(value.uint_value());
        case asrv1::ScalarValue::kDoubleValue:
            return python_float(value.double_value());
        case asrv1::ScalarValue::kBoolValue:
            return value.bool_value() ? "true" : "false";
        case asrv1::ScalarValue::KIND_NOT_SET:
            break;
    }
    return std::string();
}

std::vector<std::pair<std::string, std::string>> remote_form_fields(
    const asrv1::RemoteTranscriptionOptions& options) {
    std::vector<std::pair<std::string, std::string>> fields{{"response_format", "verbose_json"}};
    if (!options.model().empty()) fields.emplace_back("model", options.model());
    if (!options.language().empty()) fields.emplace_back("language", options.language());
    if (!options.initial_prompt().empty()) fields.emplace_back("prompt", options.initial_prompt());
    if (options.word_timestamps()) {
        fields.emplace_back("timestamp_granularities[]", "segment");
        fields.emplace_back("timestamp_granularities[]", "word");
    }
    std::vector<std::string> keys;
    keys.reserve(options.params_size());
    for (const auto& [key, value] : options.params()) keys.push_back(key);
    std::ranges::sort(keys);
    for (const std::string& key : keys) {
        if (key == "response_format") continue;
        fields.emplace_back(key, scalar_form_value(options.params().at(key)));
    }
    return fields;
}

double remote_timeout_seconds(const asrv1::RemoteTranscriptionOptions& options) {
    return options.has_timeout_seconds() ? options.timeout_seconds() : kDefaultRemoteTimeoutSeconds;
}

}  // namespace asr::engine
