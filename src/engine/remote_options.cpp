#include "engine/remote_options.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <set>
#include <string>
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

// Python's repr of a float, which str() uses: the shortest digits that
// round-trip, fixed notation when the decimal exponent is in [-4, 16)
// (with ".0" for integral values), otherwise scientific with a signed,
// at least two digit exponent (1e+16, 1.5e-07). inf and nan are spelled
// as Python spells them.
std::string python_float(double value) {
    if (std::isnan(value)) return "nan";
    if (std::isinf(value)) return value > 0 ? "inf" : "-inf";
    char buffer[64];
    const auto result =
        std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::scientific);
    const std::string_view text(buffer, result.ptr);
    const size_t e = text.find('e');
    std::string_view mantissa = text.substr(0, e);
    const int exponent = std::stoi(std::string(text.substr(e + 1)));
    std::string sign;
    if (!mantissa.empty() && mantissa.front() == '-') {
        sign = "-";
        mantissa.remove_prefix(1);
    }
    std::string digits;
    for (const char c : mantissa) {
        if (c != '.') digits.push_back(c);
    }
    if (exponent >= -4 && exponent < 16) {
        const int point = exponent + 1;
        const int count = static_cast<int>(digits.size());
        if (point <= 0) return sign + "0." + std::string(static_cast<size_t>(-point), '0') + digits;
        if (point >= count) return sign + digits + std::string(static_cast<size_t>(point - count), '0') + ".0";
        return sign + digits.substr(0, static_cast<size_t>(point)) + "." +
               digits.substr(static_cast<size_t>(point));
    }
    std::string out = sign + digits.substr(0, 1);
    if (digits.size() > 1) out += "." + digits.substr(1);
    const int magnitude = exponent < 0 ? -exponent : exponent;
    out += exponent < 0 ? "e-" : "e+";
    if (magnitude < 10) out += "0";
    out += std::to_string(magnitude);
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
    if (url.find_first_of(" \r\n\t") != std::string::npos) {
        return invalid("RemoteTranscriptionOptions.url must not contain whitespace");
    }
    const std::string_view rest = std::string_view(url).substr(http ? 7 : 8);
    const std::string_view authority = rest.substr(0, rest.find_first_of("/?#"));
    if (authority.find('@') != std::string_view::npos) {
        return invalid(
            "RemoteTranscriptionOptions.url must not carry userinfo; send credentials in headers");
    }
    std::string_view host = authority;
    std::string_view port;
    if (!authority.empty() && authority.front() == '[') {
        const size_t close = authority.find(']');
        if (close == std::string_view::npos) {
            return invalid("RemoteTranscriptionOptions.url has an unterminated IPv6 host");
        }
        host = authority.substr(1, close - 1);
        const std::string_view after = authority.substr(close + 1);
        if (!after.empty()) {
            if (after.front() != ':') return invalid("RemoteTranscriptionOptions.url has a malformed host");
            port = after.substr(1);
            if (port.empty()) return invalid("RemoteTranscriptionOptions.url has an empty port");
        }
    } else if (const size_t colon = authority.find(':'); colon != std::string_view::npos) {
        host = authority.substr(0, colon);
        port = authority.substr(colon + 1);
        if (port.empty()) return invalid("RemoteTranscriptionOptions.url has an empty port");
    }
    if (host.empty()) {
        return invalid("RemoteTranscriptionOptions.url has no host");
    }
    if (!port.empty() &&
        (port.size() > 5 || !std::ranges::all_of(port, [](char c) { return c >= '0' && c <= '9'; }) ||
         std::stoi(std::string(port)) == 0 || std::stoi(std::string(port)) > 65535)) {
        return invalid("RemoteTranscriptionOptions.url port must be a number in 1-65535");
    }
    std::set<std::string> seen_headers;
    for (const auto& [name, value] : options.headers()) {
        if (!is_token(name)) {
            return invalid("RemoteTranscriptionOptions.headers key '" + name + "' is not an HTTP token");
        }
        std::string folded = name;
        std::ranges::transform(folded, folded.begin(), [](char c) {
            return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
        });
        if (!seen_headers.insert(folded).second) {
            return invalid("RemoteTranscriptionOptions.headers key '" + name +
                           "' repeats another header name (names are case-insensitive)");
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
        if (!std::isfinite(timeout) || timeout <= 0.0 || timeout > kMaxRemoteTimeoutSeconds) {
            return invalid("RemoteTranscriptionOptions.timeout_seconds must be > 0 and at most 3600");
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
