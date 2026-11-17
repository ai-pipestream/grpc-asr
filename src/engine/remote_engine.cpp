#include "engine/remote_engine.h"

#include <arpa/inet.h>
#include <curl/curl.h>
#include <netdb.h>
#include <sys/socket.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "engine/remote_options.h"
#include "whisper.h"

namespace asr::engine {

namespace asrv1 = ai::pipestream::asr::v1;

namespace {

void ensure_curl_initialized() {
    static std::once_flag once;
    std::call_once(once, [] {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
            throw std::runtime_error("libcurl global initialization failed");
        }
    });
}

std::string to_lower_ascii(std::string_view input) {
    std::string out(input);
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

struct ParsedAuthority {
    bool https = false;
    std::string host;  // brackets stripped for IPv6
    uint16_t port = 0;
    bool has_port = false;
};

std::optional<ParsedAuthority> parse_url_authority(std::string_view url) {
    ParsedAuthority out;
    const std::string lower_prefix = to_lower_ascii(url.substr(0, 8));
    size_t rest_pos = 0;
    if (lower_prefix.starts_with("http://")) {
        out.https = false;
        out.port = 80;
        rest_pos = 7;
    } else if (lower_prefix.starts_with("https://")) {
        out.https = true;
        out.port = 443;
        rest_pos = 8;
    } else {
        return std::nullopt;
    }
    const std::string_view rest = url.substr(rest_pos);
    const std::string_view authority = rest.substr(0, rest.find_first_of("/?#"));
    if (authority.empty() || authority.find('@') != std::string_view::npos) {
        return std::nullopt;
    }
    std::string_view host_part = authority;
    std::string_view port_part;
    if (authority.front() == '[') {
        const size_t close = authority.find(']');
        if (close == std::string_view::npos) {
            return std::nullopt;
        }
        host_part = authority.substr(1, close - 1);
        const std::string_view after = authority.substr(close + 1);
        if (!after.empty()) {
            if (after.front() != ':') {
                return std::nullopt;
            }
            port_part = after.substr(1);
            out.has_port = true;
        }
    } else if (const size_t colon = authority.find(':'); colon != std::string_view::npos) {
        host_part = authority.substr(0, colon);
        port_part = authority.substr(colon + 1);
        out.has_port = true;
    }
    if (host_part.empty()) {
        return std::nullopt;
    }
    out.host = to_lower_ascii(host_part);
    if (out.has_port) {
        if (port_part.empty() || port_part.size() > 5) {
            return std::nullopt;
        }
        unsigned int p = 0;
        const auto [ptr, ec] =
            std::from_chars(port_part.data(), port_part.data() + port_part.size(), p);
        if (ec != std::errc{} || ptr != port_part.data() + port_part.size() || p == 0 ||
            p > 65535) {
            return std::nullopt;
        }
        out.port = static_cast<uint16_t>(p);
    }
    return out;
}

struct HostPattern {
    std::string host;  // lowercased, brackets stripped
    std::optional<uint16_t> port;
};

std::optional<HostPattern> parse_host_pattern(std::string_view raw) {
    while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.front()))) {
        raw.remove_prefix(1);
    }
    while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.back()))) {
        raw.remove_suffix(1);
    }
    if (raw.empty()) {
        return std::nullopt;
    }
    HostPattern out;
    std::string_view host_part = raw;
    std::string_view port_part;
    if (raw.front() == '[') {
        const size_t close = raw.find(']');
        if (close == std::string_view::npos) {
            return std::nullopt;
        }
        host_part = raw.substr(1, close - 1);
        const std::string_view after = raw.substr(close + 1);
        if (!after.empty()) {
            if (after.front() != ':') {
                return std::nullopt;
            }
            port_part = after.substr(1);
        }
    } else if (const size_t colon = raw.find(':'); colon != std::string_view::npos) {
        // Unbracketed IPv6 literals contain more than one colon; require brackets
        // when combining with a port, or treat multi-colon strings as bare IPv6.
        if (raw.find(':', colon + 1) == std::string_view::npos) {
            host_part = raw.substr(0, colon);
            port_part = raw.substr(colon + 1);
        }
    }
    if (host_part.empty()) {
        return std::nullopt;
    }
    out.host = to_lower_ascii(host_part);
    if (!port_part.empty()) {
        unsigned int p = 0;
        const auto [ptr, ec] =
            std::from_chars(port_part.data(), port_part.data() + port_part.size(), p);
        if (ec != std::errc{} || ptr != port_part.data() + port_part.size() || p == 0 ||
            p > 65535) {
            return std::nullopt;
        }
        out.port = static_cast<uint16_t>(p);
    }
    return out;
}

bool pattern_matches(const HostPattern& pattern, const std::string& host, uint16_t port) {
    if (pattern.port.has_value() && *pattern.port != port) {
        return false;
    }
    if (pattern.host == "*") {
        return true;
    }
    if (pattern.host.starts_with("*.")) {
        const std::string_view suffix = std::string_view(pattern.host).substr(1);  // ".example.com"
        return host.size() > suffix.size() && host.ends_with(suffix);
    }
    return host == pattern.host;
}

std::optional<std::string> canonical_ip_literal(const std::string& text) {
    in_addr v4{};
    if (::inet_pton(AF_INET, text.c_str(), &v4) == 1) {
        char buf[INET_ADDRSTRLEN] = {};
        if (::inet_ntop(AF_INET, &v4, buf, sizeof(buf)) != nullptr) {
            return std::string(buf);
        }
    }
    in6_addr v6{};
    if (::inet_pton(AF_INET6, text.c_str(), &v6) == 1) {
        // Normalize IPv4-mapped IPv6 literals (::ffff:a.b.c.d) to IPv4 text so
        // dual-stack socket checks compare apples to apples.
        const uint8_t* b = v6.s6_addr;
        bool v4_mapped = true;
        for (int i = 0; i < 10; ++i) {
            if (b[i] != 0) {
                v4_mapped = false;
                break;
            }
        }
        if (v4_mapped && b[10] == 0xff && b[11] == 0xff) {
            in_addr mapped{};
            std::memcpy(&mapped, b + 12, 4);
            char buf[INET_ADDRSTRLEN] = {};
            if (::inet_ntop(AF_INET, &mapped, buf, sizeof(buf)) != nullptr) {
                return std::string(buf);
            }
        }
        char buf[INET6_ADDRSTRLEN] = {};
        if (::inet_ntop(AF_INET6, &v6, buf, sizeof(buf)) != nullptr) {
            return std::string(buf);
        }
    }
    return std::nullopt;
}

struct IpClassification {
    std::string canonical;
    bool is_public = false;
    bool is_loopback = false;
};

IpClassification classify_ipv4_host_order(uint32_t ip) {
    in_addr addr{};
    addr.s_addr = htonl(ip);
    char buf[INET_ADDRSTRLEN] = {};
    ::inet_ntop(AF_INET, &addr, buf, sizeof(buf));

    IpClassification out;
    out.canonical = buf;
    const uint32_t first = ip >> 24;
    out.is_loopback = (first == 127);
    out.is_public = first != 0 && first != 10 && first != 127 && first < 224 &&
                    (ip & 0xffff0000U) != 0xa9fe0000U &&   // 169.254.0.0/16 link-local
                    (ip & 0xfff00000U) != 0xac100000U &&   // 172.16.0.0/12 RFC 1918
                    (ip & 0xffff0000U) != 0xc0a80000U &&   // 192.168.0.0/16 RFC 1918
                    (ip & 0xffc00000U) != 0x64400000U &&   // 100.64.0.0/10 CGNAT
                    (ip & 0xfffe0000U) != 0xc6120000U &&   // 198.18.0.0/15 benchmark
                    (ip & 0xffffff00U) != 0xc0000000U &&   // 192.0.0.0/24 IETF
                    (ip & 0xffffff00U) != 0xc0000200U &&   // 192.0.2.0/24 TEST-NET-1
                    (ip & 0xffffff00U) != 0xc6336400U &&   // 198.51.100.0/24 TEST-NET-2
                    (ip & 0xffffff00U) != 0xcb007100U;     // 203.0.113.0/24 TEST-NET-3
    return out;
}

std::optional<IpClassification> classify_sockaddr(const sockaddr* sa) {
    if (sa == nullptr) {
        return std::nullopt;
    }
    if (sa->sa_family == AF_INET) {
        const auto* sin = reinterpret_cast<const sockaddr_in*>(sa);
        return classify_ipv4_host_order(ntohl(sin->sin_addr.s_addr));
    }
    if (sa->sa_family == AF_INET6) {
        const auto* sin6 = reinterpret_cast<const sockaddr_in6*>(sa);
        const uint8_t* b = sin6->sin6_addr.s6_addr;
        bool prefix_zero = true;
        for (int i = 0; i < 10; ++i) {
            if (b[i] != 0) {
                prefix_zero = false;
                break;
            }
        }
        if (prefix_zero && b[10] == 0xff && b[11] == 0xff) {
            uint32_t v4_net = 0;
            std::memcpy(&v4_net, b + 12, 4);
            return classify_ipv4_host_order(ntohl(v4_net));
        }

        char buf[INET6_ADDRSTRLEN] = {};
        ::inet_ntop(AF_INET6, &sin6->sin6_addr, buf, sizeof(buf));
        IpClassification out;
        out.canonical = buf;
        bool first_15_zero = prefix_zero;
        for (int i = 10; i < 15 && first_15_zero; ++i) {
            if (b[i] != 0) {
                first_15_zero = false;
            }
        }
        out.is_loopback = first_15_zero && b[15] == 1;
        out.is_public = (b[0] & 0xe0) == 0x20 &&
                        !(b[0] == 0x20 && b[1] == 0x02) &&
                        !(b[0] == 0x20 && b[1] == 0x01 && b[2] == 0x00 && b[3] == 0x00) &&
                        !(b[0] == 0x20 && b[1] == 0x01 && b[2] == 0x0d && b[3] == 0xb8);
        return out;
    }
    return std::nullopt;
}

bool address_allowed(const IpClassification& ip, const RemoteEndpointPolicy& policy) {
    if (ip.is_public) {
        return true;
    }
    if (ip.is_loopback && policy.allow_loopback) {
        return true;
    }
    return std::find(policy.explicit_ip_literals.begin(), policy.explicit_ip_literals.end(),
                     ip.canonical) != policy.explicit_ip_literals.end();
}

// ---------------------------------------------------------------------------
// Minimal RFC 8259 recursive-descent JSON parser for verbose_json responses.
// ---------------------------------------------------------------------------
struct JsonValue {
    using Object = std::map<std::string, JsonValue>;
    using Array = std::vector<JsonValue>;
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> data{nullptr};

    bool is_object() const { return std::holds_alternative<Object>(data); }
    bool is_array() const { return std::holds_alternative<Array>(data); }
    bool is_string() const { return std::holds_alternative<std::string>(data); }
    bool is_number() const { return std::holds_alternative<double>(data); }

    const Object& as_object() const { return std::get<Object>(data); }
    const Array& as_array() const { return std::get<Array>(data); }
    const std::string& as_string() const { return std::get<std::string>(data); }
    double as_number() const { return std::get<double>(data); }

    const JsonValue* find(std::string_view key) const {
        if (!is_object()) {
            return nullptr;
        }
        const auto& obj = as_object();
        const auto it = obj.find(std::string(key));
        return it == obj.end() ? nullptr : &it->second;
    }
};

class JsonParser {
  public:
    explicit JsonParser(std::string_view input) : in_(input) {}

    std::optional<JsonValue> parse() {
        skip_ws();
        auto val = parse_value(0);
        if (!val.has_value()) {
            return std::nullopt;
        }
        skip_ws();
        if (pos_ != in_.size()) {
            return std::nullopt;
        }
        return val;
    }

  private:
    static constexpr int kMaxDepth = 32;

    void skip_ws() {
        while (pos_ < in_.size()) {
            const char c = in_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp <= 0x7FU) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FFU) {
            out.push_back(static_cast<char>(0xC0U | (cp >> 6)));
            out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        } else if (cp <= 0xFFFFU) {
            out.push_back(static_cast<char>(0xE0U | (cp >> 12)));
            out.push_back(static_cast<char>(0x80U | ((cp >> 6) & 0x3FU)));
            out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        } else if (cp <= 0x10FFFFU) {
            out.push_back(static_cast<char>(0xF0U | (cp >> 18)));
            out.push_back(static_cast<char>(0x80U | ((cp >> 12) & 0x3FU)));
            out.push_back(static_cast<char>(0x80U | ((cp >> 6) & 0x3FU)));
            out.push_back(static_cast<char>(0x80U | (cp & 0x3FU)));
        }
    }

    std::optional<uint16_t> parse_hex4() {
        if (pos_ + 4 > in_.size()) {
            return std::nullopt;
        }
        uint16_t v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = in_[pos_++];
            v = static_cast<uint16_t>(v << 4);
            if (c >= '0' && c <= '9') {
                v = static_cast<uint16_t>(v | (c - '0'));
            } else if (c >= 'a' && c <= 'f') {
                v = static_cast<uint16_t>(v | (c - 'a' + 10));
            } else if (c >= 'A' && c <= 'F') {
                v = static_cast<uint16_t>(v | (c - 'A' + 10));
            } else {
                return std::nullopt;
            }
        }
        return v;
    }

    std::optional<std::string> parse_string() {
        if (pos_ >= in_.size() || in_[pos_] != '"') {
            return std::nullopt;
        }
        ++pos_;
        std::string out;
        while (pos_ < in_.size()) {
            const unsigned char c = static_cast<unsigned char>(in_[pos_++]);
            if (c == '"') {
                return out;
            }
            if (c < 0x20U) {
                return std::nullopt;
            }
            if (c != '\\') {
                out.push_back(static_cast<char>(c));
                continue;
            }
            if (pos_ >= in_.size()) {
                return std::nullopt;
            }
            const char esc = in_[pos_++];
            switch (esc) {
                case '"':
                case '\\':
                case '/':
                    out.push_back(esc);
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u': {
                    const auto hi = parse_hex4();
                    if (!hi.has_value()) {
                        return std::nullopt;
                    }
                    uint32_t cp = *hi;
                    if (cp >= 0xD800U && cp <= 0xDBFFU) {
                        if (pos_ + 2 > in_.size() || in_[pos_] != '\\' || in_[pos_ + 1] != 'u') {
                            return std::nullopt;
                        }
                        pos_ += 2;
                        const auto lo = parse_hex4();
                        if (!lo.has_value() || *lo < 0xDC00U || *lo > 0xDFFFU) {
                            return std::nullopt;
                        }
                        cp = 0x10000U + ((cp - 0xD800U) << 10) + (*lo - 0xDC00U);
                    } else if (cp >= 0xDC00U && cp <= 0xDFFFU) {
                        return std::nullopt;
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    return std::nullopt;
            }
        }
        return std::nullopt;
    }

    std::optional<double> parse_number() {
        const size_t start = pos_;
        if (pos_ < in_.size() && in_[pos_] == '-') {
            ++pos_;
        }
        if (pos_ >= in_.size()) {
            return std::nullopt;
        }
        if (in_[pos_] == '0') {
            ++pos_;
        } else if (in_[pos_] >= '1' && in_[pos_] <= '9') {
            while (pos_ < in_.size() && in_[pos_] >= '0' && in_[pos_] <= '9') {
                ++pos_;
            }
        } else {
            return std::nullopt;
        }
        if (pos_ < in_.size() && in_[pos_] == '.') {
            ++pos_;
            if (pos_ >= in_.size() || in_[pos_] < '0' || in_[pos_] > '9') {
                return std::nullopt;
            }
            while (pos_ < in_.size() && in_[pos_] >= '0' && in_[pos_] <= '9') {
                ++pos_;
            }
        }
        if (pos_ < in_.size() && (in_[pos_] == 'e' || in_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < in_.size() && (in_[pos_] == '+' || in_[pos_] == '-')) {
                ++pos_;
            }
            if (pos_ >= in_.size() || in_[pos_] < '0' || in_[pos_] > '9') {
                return std::nullopt;
            }
            while (pos_ < in_.size() && in_[pos_] >= '0' && in_[pos_] <= '9') {
                ++pos_;
            }
        }
        double value = 0.0;
        const auto [ptr, ec] = std::from_chars(in_.data() + start, in_.data() + pos_, value);
        if (ec != std::errc{} || ptr != in_.data() + pos_ || !std::isfinite(value)) {
            return std::nullopt;
        }
        return value;
    }

    std::optional<JsonValue> parse_value(int depth) {
        if (depth > kMaxDepth) {
            return std::nullopt;
        }
        skip_ws();
        if (pos_ >= in_.size()) {
            return std::nullopt;
        }
        const char c = in_[pos_];
        if (c == '"') {
            auto s = parse_string();
            if (!s.has_value()) {
                return std::nullopt;
            }
            return JsonValue{*std::move(s)};
        }
        if (c == '{') {
            ++pos_;
            JsonValue::Object obj;
            skip_ws();
            if (pos_ < in_.size() && in_[pos_] == '}') {
                ++pos_;
                return JsonValue{std::move(obj)};
            }
            while (true) {
                skip_ws();
                auto key = parse_string();
                if (!key.has_value()) {
                    return std::nullopt;
                }
                skip_ws();
                if (pos_ >= in_.size() || in_[pos_] != ':') {
                    return std::nullopt;
                }
                ++pos_;
                auto val = parse_value(depth + 1);
                if (!val.has_value()) {
                    return std::nullopt;
                }
                obj[*std::move(key)] = *std::move(val);
                skip_ws();
                if (pos_ < in_.size() && in_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                if (pos_ < in_.size() && in_[pos_] == '}') {
                    ++pos_;
                    return JsonValue{std::move(obj)};
                }
                return std::nullopt;
            }
        }
        if (c == '[') {
            ++pos_;
            JsonValue::Array arr;
            skip_ws();
            if (pos_ < in_.size() && in_[pos_] == ']') {
                ++pos_;
                return JsonValue{std::move(arr)};
            }
            while (true) {
                auto elem = parse_value(depth + 1);
                if (!elem.has_value()) {
                    return std::nullopt;
                }
                arr.push_back(*std::move(elem));
                skip_ws();
                if (pos_ < in_.size() && in_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                if (pos_ < in_.size() && in_[pos_] == ']') {
                    ++pos_;
                    return JsonValue{std::move(arr)};
                }
                return std::nullopt;
            }
        }
        if (c == 't' && in_.substr(pos_, 4) == "true") {
            pos_ += 4;
            return JsonValue{true};
        }
        if (c == 'f' && in_.substr(pos_, 5) == "false") {
            pos_ += 5;
            return JsonValue{false};
        }
        if (c == 'n' && in_.substr(pos_, 4) == "null") {
            pos_ += 4;
            return JsonValue{nullptr};
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            auto num = parse_number();
            if (!num.has_value()) {
                return std::nullopt;
            }
            return JsonValue{*num};
        }
        return std::nullopt;
    }

    std::string_view in_;
    size_t pos_ = 0;
};

uint64_t seconds_to_ms(double sec) {
    if (!std::isfinite(sec) || sec <= 0.0) {
        return 0;
    }
    return static_cast<uint64_t>(std::llround(sec * 1000.0));
}

std::string normalize_language_code(const std::string& raw) {
    if (raw.empty()) {
        return {};
    }
    const std::string lowered = to_lower_ascii(raw);
    const int id = whisper_lang_id(lowered.c_str());
    if (id >= 0) {
        const char* code = whisper_lang_str(id);
        if (code != nullptr) {
            return code;
        }
    }
    return lowered;
}

EngineWord parse_word_object(const JsonValue& w) {
    EngineWord word;
    if (const auto* t = w.find("word"); t != nullptr && t->is_string()) {
        word.text = t->as_string();
    } else if (const auto* t2 = w.find("text"); t2 != nullptr && t2->is_string()) {
        word.text = t2->as_string();
    }
    if (const auto* s = w.find("start"); s != nullptr && s->is_number()) {
        word.start_ms = seconds_to_ms(s->as_number());
    }
    if (const auto* e = w.find("end"); e != nullptr && e->is_number()) {
        word.end_ms = std::max(word.start_ms, seconds_to_ms(e->as_number()));
    } else {
        word.end_ms = word.start_ms;
    }
    if (const auto* p = w.find("probability"); p != nullptr && p->is_number()) {
        word.probability = static_cast<float>(std::clamp(p->as_number(), 0.0, 1.0));
    } else if (const auto* sc = w.find("score"); sc != nullptr && sc->is_number()) {
        word.probability = static_cast<float>(std::clamp(sc->as_number(), 0.0, 1.0));
    }
    return word;
}

bool error_body_names_field(const std::string& body,
                            const asrv1::RemoteTranscriptionOptions& options) {
    const std::string lower = to_lower_ascii(body);
    static constexpr std::array<std::string_view, 7> kBuiltins = {
        "model",          "language",                "prompt",      "file",
        "response_format", "timestamp_granularities", "temperature",
    };
    for (const std::string_view field : kBuiltins) {
        if (lower.find(field) != std::string::npos) {
            return true;
        }
    }
    for (const auto& [key, _] : options.params()) {
        if (!key.empty() && lower.find(to_lower_ascii(key)) != std::string::npos) {
            return true;
        }
    }
    return false;
}

struct UploadReader {
    media::ByteStream* stream = nullptr;
    uint64_t offset = 0;

    static size_t read_cb(char* buffer, size_t size, size_t nitems, void* arg) noexcept {
        auto& self = *static_cast<UploadReader*>(arg);
        const size_t max_bytes = size * nitems;
        if (max_bytes == 0) {
            return 0;
        }
        const size_t got = self.stream->read_at(self.offset, buffer, max_bytes);
        self.offset += got;
        return got;
    }

    static int seek_cb(void* arg, curl_off_t offset, int origin) noexcept {
        auto& self = *static_cast<UploadReader*>(arg);
        const uint64_t total = self.stream->size();
        int64_t next = 0;
        if (origin == SEEK_SET) {
            next = static_cast<int64_t>(offset);
        } else if (origin == SEEK_CUR) {
            next = static_cast<int64_t>(self.offset) + static_cast<int64_t>(offset);
        } else if (origin == SEEK_END) {
            next = static_cast<int64_t>(total) + static_cast<int64_t>(offset);
        } else {
            return CURL_SEEKFUNC_FAIL;
        }
        if (next < 0 || static_cast<uint64_t>(next) > total) {
            return CURL_SEEKFUNC_FAIL;
        }
        self.offset = static_cast<uint64_t>(next);
        return CURL_SEEKFUNC_OK;
    }
};

struct TransferContext {
    const RemoteEndpointPolicy* policy = nullptr;
    std::stop_token stop;
    std::string response_body;
    bool body_too_large = false;
    bool address_rejected = false;
    bool cancelled = false;

    static size_t write_cb(char* ptr, size_t size, size_t nmemb, void* userdata) noexcept {
        auto& self = *static_cast<TransferContext*>(userdata);
        const size_t n = size * nmemb;
        if (n == 0) {
            return 0;
        }
        if (n > kMaxRemoteResponseBytes ||
            self.response_body.size() > kMaxRemoteResponseBytes - n) {
            self.body_too_large = true;
            return 0;
        }
        self.response_body.append(ptr, n);
        return n;
    }

    static int progress_cb(void* clientp, curl_off_t /*dltotal*/, curl_off_t /*dlnow*/,
                           curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) noexcept {
        auto& self = *static_cast<TransferContext*>(clientp);
        if (self.stop.stop_requested()) {
            self.cancelled = true;
            return 1;
        }
        return 0;
    }

    static curl_socket_t opensocket_cb(void* clientp, curlsocktype purpose,
                                       curl_sockaddr* address) noexcept {
        auto& self = *static_cast<TransferContext*>(clientp);
        if (purpose != CURLSOCKTYPE_IPCXN) {
            self.address_rejected = true;
            return CURL_SOCKET_BAD;
        }
        const auto classified = classify_sockaddr(&address->addr);
        if (!classified.has_value() || !address_allowed(*classified, *self.policy)) {
            self.address_rejected = true;
            return CURL_SOCKET_BAD;
        }
        return ::socket(address->family, address->socktype, address->protocol);
    }
};

}  // namespace

grpc::Status authorize_remote_endpoint(const std::string& url,
                                       const std::vector<std::string>& allowed_hosts,
                                       RemoteEndpointPolicy* out) {
    if (allowed_hosts.empty()) {
        return {grpc::StatusCode::FAILED_PRECONDITION,
                "remote transcription is disabled: GRPC_ASR_REMOTE_ALLOWED_HOSTS is not set"};
    }
    const auto parsed = parse_url_authority(url);
    if (!parsed.has_value()) {
        return {grpc::StatusCode::INVALID_ARGUMENT, "invalid RemoteTranscriptionOptions.url"};
    }

    bool matched = false;
    RemoteEndpointPolicy policy;
    policy.host = parsed->host;
    policy.port = parsed->port;

    for (const std::string& raw : allowed_hosts) {
        const auto pat = parse_host_pattern(raw);
        if (!pat.has_value()) {
            continue;
        }
        if (!pattern_matches(*pat, parsed->host, parsed->port)) {
            continue;
        }
        matched = true;
        if (pat->host == "localhost") {
            policy.allow_loopback = true;
        }
        if (const auto ip = canonical_ip_literal(pat->host); ip.has_value()) {
            policy.explicit_ip_literals.push_back(*ip);
        }
    }
    if (!matched) {
        return {grpc::StatusCode::PERMISSION_DENIED,
                "remote transcription host '" + parsed->host + ":" +
                    std::to_string(parsed->port) +
                    "' is not allowed by GRPC_ASR_REMOTE_ALLOWED_HOSTS"};
    }

    addrinfo hints{};
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_family = AF_UNSPEC;
    addrinfo* res = nullptr;
    const std::string port_str = std::to_string(parsed->port);
    const int rc = ::getaddrinfo(parsed->host.c_str(), port_str.c_str(), &hints, &res);
    if (rc != 0 || res == nullptr) {
        return {grpc::StatusCode::UNAVAILABLE,
                "could not resolve remote transcription host '" + parsed->host + "'"};
    }
    std::unique_ptr<addrinfo, decltype(&::freeaddrinfo)> guard(res, &::freeaddrinfo);

    std::string first_ip;
    for (const addrinfo* ai = res; ai != nullptr; ai = ai->ai_next) {
        const auto classified = classify_sockaddr(ai->ai_addr);
        if (!classified.has_value()) {
            continue;
        }
        if (!address_allowed(*classified, policy)) {
            return {grpc::StatusCode::PERMISSION_DENIED,
                    "remote transcription target '" + parsed->host +
                        "' resolves to a non-public address (" + classified->canonical +
                        ") not explicitly named in GRPC_ASR_REMOTE_ALLOWED_HOSTS"};
        }
        if (first_ip.empty()) {
            first_ip = classified->canonical;
        }
    }
    if (first_ip.empty()) {
        return {grpc::StatusCode::UNAVAILABLE,
                "no usable network address resolved for '" + parsed->host + "'"};
    }
    policy.pinned_ip = first_ip;
    if (out != nullptr) {
        *out = std::move(policy);
    }
    return grpc::Status::OK;
}

grpc::Status run_remote_transcription(const asrv1::RemoteTranscriptionOptions& options,
                                      const RemoteEndpointPolicy& policy,
                                      media::ByteStream& upload, const std::string& filename,
                                      const std::string& content_type,
                                      const Transcriber::SegmentSink& sink, EngineResult* result,
                                      std::stop_token stop) {
    ensure_curl_initialized();
    if (stop.stop_requested()) {
        return {grpc::StatusCode::CANCELLED, "remote transcription cancelled before start"};
    }

    CURL* easy = curl_easy_init();
    CURLM* multi = curl_multi_init();
    if (easy == nullptr || multi == nullptr) {
        if (easy != nullptr) curl_easy_cleanup(easy);
        if (multi != nullptr) curl_multi_cleanup(multi);
        return {grpc::StatusCode::INTERNAL, "curl handle allocation failed"};
    }

    curl_slist* headers = nullptr;
    curl_slist* resolve_list = nullptr;
    curl_mime* mime = nullptr;
    bool added_to_multi = false;
    auto cleanup = [&]() {
        if (added_to_multi) {
            curl_multi_remove_handle(multi, easy);
        }
        if (mime != nullptr) {
            curl_mime_free(mime);
        }
        if (headers != nullptr) {
            curl_slist_free_all(headers);
        }
        if (resolve_list != nullptr) {
            curl_slist_free_all(resolve_list);
        }
        curl_easy_cleanup(easy);
        curl_multi_cleanup(multi);
    };

    // Disable Expect: 100-continue so single-pass HTTP servers answer without
    // stalling for a 100 Continue round-trip, and apply caller headers verbatim
    // without ever logging them.
    headers = curl_slist_append(headers, "Expect:");
    for (const auto& [name, value] : options.headers()) {
        const std::string line = name + ": " + value;
        headers = curl_slist_append(headers, line.c_str());
    }

    // Pin the pre-validated IP when the URL host is a DNS name rather than a
    // literal IP, eliminating TOCTOU DNS rebinding.
    if (!canonical_ip_literal(policy.host).has_value() && !policy.pinned_ip.empty()) {
        const std::string formatted_ip =
            policy.pinned_ip.find(':') != std::string::npos ? "[" + policy.pinned_ip + "]"
                                                            : policy.pinned_ip;
        const std::string entry =
            policy.host + ":" + std::to_string(policy.port) + ":" + formatted_ip;
        resolve_list = curl_slist_append(resolve_list, entry.c_str());
    }

    UploadReader reader{.stream = &upload, .offset = 0};
    mime = curl_mime_init(easy);
    if (mime == nullptr) {
        cleanup();
        return {grpc::StatusCode::INTERNAL, "curl_mime_init failed"};
    }
    for (const auto& [field_name, field_val] : remote_form_fields(options)) {
        curl_mimepart* part = curl_mime_addpart(mime);
        curl_mime_name(part, field_name.c_str());
        curl_mime_data(part, field_val.data(), field_val.size());
    }
    {
        curl_mimepart* file_part = curl_mime_addpart(mime);
        curl_mime_name(file_part, "file");
        curl_mime_filename(file_part, filename.empty() ? "media.bin" : filename.c_str());
        if (!content_type.empty()) {
            curl_mime_type(file_part, content_type.c_str());
        }
        curl_mime_data_cb(file_part, static_cast<curl_off_t>(upload.size()),
                          &UploadReader::read_cb, &UploadReader::seek_cb, nullptr, &reader);
    }

    TransferContext transfer{};
    transfer.policy = &policy;
    transfer.stop = std::move(stop);
    const double timeout_sec = remote_timeout_seconds(options);
    const long timeout_ms =
        static_cast<long>(std::clamp(timeout_sec * 1000.0, 1.0, 3600.0 * 1000.0));

    curl_easy_setopt(easy, CURLOPT_URL, options.url().c_str());
    curl_easy_setopt(easy, CURLOPT_PROTOCOLS_STR, "http,https");
    curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 0L);
    curl_easy_setopt(easy, CURLOPT_PROXY, "");
    curl_easy_setopt(easy, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(easy, CURLOPT_VERBOSE, 0L);
    curl_easy_setopt(easy, CURLOPT_HTTPHEADER, headers);
    if (resolve_list != nullptr) {
        curl_easy_setopt(easy, CURLOPT_RESOLVE, resolve_list);
    }
    curl_easy_setopt(easy, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(easy, CURLOPT_TIMEOUT_MS, timeout_ms);
    curl_easy_setopt(easy, CURLOPT_CONNECTTIMEOUT_MS, std::min<long>(timeout_ms, 10000L));
    curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, &TransferContext::write_cb);
    curl_easy_setopt(easy, CURLOPT_WRITEDATA, &transfer);
    curl_easy_setopt(easy, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(easy, CURLOPT_XFERINFOFUNCTION, &TransferContext::progress_cb);
    curl_easy_setopt(easy, CURLOPT_XFERINFODATA, &transfer);
    curl_easy_setopt(easy, CURLOPT_OPENSOCKETFUNCTION, &TransferContext::opensocket_cb);
    curl_easy_setopt(easy, CURLOPT_OPENSOCKETDATA, &transfer);

    if (curl_multi_add_handle(multi, easy) != CURLM_OK) {
        cleanup();
        return {grpc::StatusCode::INTERNAL, "curl_multi_add_handle failed"};
    }
    added_to_multi = true;

    const auto start_wall = std::chrono::steady_clock::now();
    CURLcode curl_rc = CURLE_OK;
    bool finished = false;
    while (!finished) {
        if (transfer.stop.stop_requested()) {
            transfer.cancelled = true;
            break;
        }
        int running = 0;
        if (curl_multi_perform(multi, &running) != CURLM_OK) {
            cleanup();
            return {grpc::StatusCode::UNAVAILABLE, "remote transcription HTTP poll failed"};
        }
        int msgs = 0;
        while (CURLMsg* msg = curl_multi_info_read(multi, &msgs)) {
            if (msg->msg == CURLMSG_DONE) {
                curl_rc = msg->data.result;
                finished = true;
            }
        }
        if (finished || running == 0) {
            break;
        }
        if (curl_multi_poll(multi, nullptr, 0, 50, nullptr) != CURLM_OK) {
            cleanup();
            return {grpc::StatusCode::UNAVAILABLE, "remote transcription HTTP wait failed"};
        }
    }
    const uint64_t wall_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start_wall)
            .count());

    long http_status = 0;
    curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &http_status);
    cleanup();

    if (transfer.cancelled || transfer.stop.stop_requested()) {
        result->aborted = true;
        return {grpc::StatusCode::CANCELLED, "remote transcription cancelled"};
    }
    if (transfer.body_too_large) {
        return {grpc::StatusCode::RESOURCE_EXHAUSTED,
                "remote transcription response exceeded 16 MiB"};
    }
    if (transfer.address_rejected) {
        return {grpc::StatusCode::PERMISSION_DENIED,
                "remote transcription destination address is not allowed"};
    }
    if (curl_rc == CURLE_OPERATION_TIMEDOUT) {
        return {grpc::StatusCode::DEADLINE_EXCEEDED,
                "remote transcription timed out"};
    }
    if (curl_rc != CURLE_OK) {
        return {grpc::StatusCode::UNAVAILABLE,
                std::string("remote transcription transport failed: ") +
                    curl_easy_strerror(curl_rc)};
    }
    if (http_status < 200 || http_status >= 300) {
        if (http_status >= 400 && http_status < 500 &&
            error_body_names_field(transfer.response_body, options)) {
            return {grpc::StatusCode::INVALID_ARGUMENT,
                    "remote transcription rejected request field (HTTP " +
                        std::to_string(http_status) + ")"};
        }
        return {grpc::StatusCode::UNAVAILABLE,
                "remote transcription endpoint returned HTTP " + std::to_string(http_status)};
    }

    JsonParser parser(transfer.response_body);
    const auto root = parser.parse();
    if (!root.has_value() || !root->is_object()) {
        return {grpc::StatusCode::UNAVAILABLE,
                "remote transcription endpoint returned malformed JSON"};
    }

    result->decode_wall_ns = wall_ns;
    std::string top_language;
    if (const auto* lang = root->find("language"); lang != nullptr && lang->is_string()) {
        top_language = normalize_language_code(lang->as_string());
    } else if (!options.language().empty()) {
        top_language = normalize_language_code(options.language());
    }

    if (const auto* dur = root->find("duration"); dur != nullptr && dur->is_number()) {
        result->duration_ms = seconds_to_ms(dur->as_number());
    }

    std::vector<EngineWord> top_words;
    if (const auto* words_val = root->find("words");
        words_val != nullptr && words_val->is_array()) {
        for (const JsonValue& w : words_val->as_array()) {
            if (w.is_object()) {
                top_words.push_back(parse_word_object(w));
            }
        }
    }

    std::vector<EngineSegment> segments;
    if (const auto* segs_val = root->find("segments");
        segs_val != nullptr && segs_val->is_array()) {
        uint32_t idx = 0;
        for (const JsonValue& s : segs_val->as_array()) {
            if (!s.is_object()) {
                continue;
            }
            EngineSegment seg;
            seg.index = idx++;
            if (const auto* st = s.find("start"); st != nullptr && st->is_number()) {
                seg.start_ms = seconds_to_ms(st->as_number());
            }
            if (const auto* en = s.find("end"); en != nullptr && en->is_number()) {
                seg.end_ms = std::max(seg.start_ms, seconds_to_ms(en->as_number()));
            } else {
                seg.end_ms = seg.start_ms;
            }
            if (const auto* tx = s.find("text"); tx != nullptr && tx->is_string()) {
                seg.text = tx->as_string();
            }
            if (const auto* lp = s.find("avg_logprob"); lp != nullptr && lp->is_number()) {
                seg.avg_logprob = static_cast<float>(lp->as_number());
                seg.has_avg_logprob = true;
            }
            if (const auto* nsp = s.find("no_speech_prob"); nsp != nullptr && nsp->is_number()) {
                seg.no_speech_prob = static_cast<float>(std::clamp(nsp->as_number(), 0.0, 1.0));
                seg.has_no_speech_prob = true;
            }
            if (const auto* toks = s.find("tokens"); toks != nullptr && toks->is_array()) {
                seg.token_count = static_cast<uint32_t>(toks->as_array().size());
            }
            if (const auto* sl = s.find("language"); sl != nullptr && sl->is_string()) {
                seg.language = normalize_language_code(sl->as_string());
            } else {
                seg.language = top_language;
            }
            if (const auto* sw = s.find("words"); sw != nullptr && sw->is_array()) {
                for (const JsonValue& w : sw->as_array()) {
                    if (w.is_object()) {
                        seg.words.push_back(parse_word_object(w));
                    }
                }
            }
            segments.push_back(std::move(seg));
        }
    } else if (const auto* tx = root->find("text");
               tx != nullptr && tx->is_string() && !tx->as_string().empty()) {
        EngineSegment seg;
        seg.index = 0;
        seg.start_ms = 0;
        seg.end_ms = result->duration_ms;
        seg.text = tx->as_string();
        seg.language = top_language;
        segments.push_back(std::move(seg));
    }

    // Attach top-level words to segments when segments did not embed their own.
    if (!top_words.empty() && !segments.empty()) {
        bool any_segment_words = false;
        for (const auto& s : segments) {
            if (!s.words.empty()) {
                any_segment_words = true;
                break;
            }
        }
        if (!any_segment_words) {
            for (const EngineWord& w : top_words) {
                const uint64_t mid = w.start_ms + (w.end_ms - w.start_ms) / 2;
                size_t best = segments.size() - 1;
                for (size_t i = 0; i < segments.size(); ++i) {
                    if (mid <= segments[i].end_ms) {
                        best = i;
                        break;
                    }
                }
                segments[best].words.push_back(w);
            }
        }
    }

    uint64_t max_end_ms = 0;
    for (const EngineSegment& seg : segments) {
        max_end_ms = std::max(max_end_ms, seg.end_ms);
        result->final_segments++;
        result->tokens += seg.token_count;
        if (!seg.language.empty() &&
            std::find(result->languages.begin(), result->languages.end(), seg.language) ==
                result->languages.end()) {
            result->languages.push_back(seg.language);
        }
        if (!sink(seg, /*is_final=*/true)) {
            result->aborted = true;
            return {grpc::StatusCode::CANCELLED, "stream closed by client"};
        }
    }
    if (result->duration_ms == 0) {
        result->duration_ms = max_end_ms;
    }
    result->language =
        result->languages.empty() ? top_language : result->languages.front();
    return grpc::Status::OK;
}

}  // namespace asr::engine
