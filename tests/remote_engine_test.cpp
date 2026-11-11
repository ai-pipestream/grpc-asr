// Unit and loopback HTTP integration tests for src/engine/remote_engine.{h,cpp}:
// allowlist/SSRF policy, multipart field order and streaming body, header
// confidentiality, verbose_json segment/word parsing, 16 MiB cap, HTTP status
// mapping, and cancellation latency.

#include "engine/remote_engine.h"

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <print>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "engine/remote_options.h"
#include "fixture.h"

namespace asrv1 = ai::pipestream::asr::v1;

namespace {

struct MultipartPart {
    std::string name;
    std::string filename;
    std::string body;
};

struct RecordedHttpRequest {
    std::string raw_headers;
    std::string body;
};

// Minimal single-request loopback HTTP/1.1 server for deterministic tests.
class LoopbackHttpServer {
  public:
    using Handler = std::function<void(int client_fd, const RecordedHttpRequest& req,
                                       std::atomic<bool>& stop_flag)>;

    explicit LoopbackHttpServer(Handler handler) {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
        require(listen_fd_ >= 0, "loopback socket creation succeeded");
        int one = 1;
        ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        require(::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0,
                "loopback bind succeeded");
        socklen_t len = sizeof(addr);
        require(::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len) == 0,
                "getsockname succeeded");
        port_ = ntohs(addr.sin_port);
        require(::listen(listen_fd_, 4) == 0, "listen succeeded");

        thread_ = std::jthread([this, h = std::move(handler)] {
            pollfd pfd{listen_fd_, POLLIN, 0};
            while (!stop_.load()) {
                const int r = ::poll(&pfd, 1, 50);
                if (r <= 0) {
                    continue;
                }
                const int fd = ::accept4(listen_fd_, nullptr, nullptr, SOCK_CLOEXEC);
                if (fd < 0) {
                    return;
                }
                RecordedHttpRequest rec = read_http_request(fd);
                h(fd, rec, stop_);
                ::shutdown(fd, SHUT_RDWR);
                ::close(fd);
                return;
            }
        });
    }

    ~LoopbackHttpServer() {
        stop_.store(true);
        if (listen_fd_ >= 0) {
            ::shutdown(listen_fd_, SHUT_RDWR);
        }
        if (thread_.joinable()) {
            thread_.join();
        }
        if (listen_fd_ >= 0) {
            ::close(listen_fd_);
        }
    }

    uint16_t port() const { return port_; }
    std::string host_port() const { return "127.0.0.1:" + std::to_string(port_); }
    std::string url() const {
        return "http://" + host_port() + "/v1/audio/transcriptions";
    }

  private:
    static RecordedHttpRequest read_http_request(int fd) {
        std::string raw;
        char buf[8192];
        size_t header_end = std::string::npos;
        size_t content_length = 0;
        while (true) {
            pollfd pfd{fd, POLLIN, 0};
            if (::poll(&pfd, 1, 5000) <= 0) {
                break;
            }
            const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
            if (n <= 0) {
                break;
            }
            raw.append(buf, static_cast<size_t>(n));
            if (header_end == std::string::npos) {
                header_end = raw.find("\r\n\r\n");
                if (header_end != std::string::npos) {
                    std::string lower_hdr = raw.substr(0, header_end);
                    std::transform(lower_hdr.begin(), lower_hdr.end(), lower_hdr.begin(),
                                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                    const size_t cl_pos = lower_hdr.find("content-length:");
                    if (cl_pos != std::string::npos) {
                        size_t val_pos = cl_pos + 15;
                        while (val_pos < lower_hdr.size() && lower_hdr[val_pos] == ' ') {
                            ++val_pos;
                        }
                        const size_t eol = lower_hdr.find("\r\n", val_pos);
                        content_length = std::stoull(lower_hdr.substr(val_pos, eol - val_pos));
                    }
                }
            }
            if (header_end != std::string::npos &&
                raw.size() >= header_end + 4 + content_length) {
                break;
            }
        }
        RecordedHttpRequest out;
        if (header_end != std::string::npos) {
            out.raw_headers = raw.substr(0, header_end);
            out.body = raw.substr(header_end + 4, content_length);
        }
        return out;
    }

    int listen_fd_ = -1;
    uint16_t port_ = 0;
    std::atomic<bool> stop_{false};
    std::jthread thread_;
};

void send_all(int fd, std::string_view data) {
    size_t sent = 0;
    while (sent < data.size()) {
        const ssize_t n =
            ::send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            return;
        }
        sent += static_cast<size_t>(n);
    }
}

std::vector<MultipartPart> parse_multipart_parts(const RecordedHttpRequest& req) {
    // Extract boundary from Content-Type: multipart/form-data; boundary=...
    const size_t bpos = req.raw_headers.find("boundary=");
    require(bpos != std::string::npos, "multipart Content-Type carries boundary");
    size_t bend = req.raw_headers.find_first_of("\r\n; ", bpos + 9);
    if (bend == std::string::npos) {
        bend = req.raw_headers.size();
    }
    std::string boundary = req.raw_headers.substr(bpos + 9, bend - (bpos + 9));
    if (boundary.size() >= 2 && boundary.front() == '"' && boundary.back() == '"') {
        boundary = boundary.substr(1, boundary.size() - 2);
    }
    const std::string delim = "--" + boundary;
    std::vector<MultipartPart> parts;
    size_t pos = req.body.find(delim);
    while (pos != std::string::npos) {
        pos += delim.size();
        if (req.body.compare(pos, 2, "--") == 0) {
            break;
        }
        if (req.body.compare(pos, 2, "\r\n") == 0) {
            pos += 2;
        }
        const size_t next = req.body.find("\r\n" + delim, pos);
        if (next == std::string::npos) {
            break;
        }
        const std::string_view chunk(req.body.data() + pos, next - pos);
        const size_t hdr_end = chunk.find("\r\n\r\n");
        require(hdr_end != std::string_view::npos, "multipart part has header separator");
        const std::string_view hdr = chunk.substr(0, hdr_end);
        MultipartPart part;
        part.body = std::string(chunk.substr(hdr_end + 4));
        const size_t npos = hdr.find("name=\"");
        if (npos != std::string_view::npos) {
            const size_t nend = hdr.find('"', npos + 6);
            part.name = std::string(hdr.substr(npos + 6, nend - (npos + 6)));
        }
        const size_t fpos = hdr.find("filename=\"");
        if (fpos != std::string_view::npos) {
            const size_t fend = hdr.find('"', fpos + 10);
            part.filename = std::string(hdr.substr(fpos + 10, fend - (fpos + 10)));
        }
        parts.push_back(std::move(part));
        pos = next + 2;
    }
    return parts;
}

void verify_allowlist_and_ssrf_policy() {
    asr::engine::RemoteEndpointPolicy policy;

    // 1. Empty allowlist -> FAILED_PRECONDITION.
    {
        const grpc::Status s = asr::engine::authorize_remote_endpoint(
            "http://127.0.0.1:8000/v1/audio/transcriptions", {}, &policy);
        require(s.error_code() == grpc::StatusCode::FAILED_PRECONDITION,
                "empty allowlist returns FAILED_PRECONDITION");
    }

    // 2. Host outside allowlist -> PERMISSION_DENIED.
    {
        const grpc::Status s = asr::engine::authorize_remote_endpoint(
            "http://127.0.0.1:8000/v1/audio/transcriptions", {"whisper.internal:8000"}, &policy);
        require(s.error_code() == grpc::StatusCode::PERMISSION_DENIED,
                "unlisted host returns PERMISSION_DENIED");
    }

    // 3. Loopback URL under wildcard "*" without explicit loopback pattern -> PERMISSION_DENIED.
    {
        const grpc::Status s1 = asr::engine::authorize_remote_endpoint(
            "http://127.0.0.1:8000/v1/audio/transcriptions", {"*"}, &policy);
        require(s1.error_code() == grpc::StatusCode::PERMISSION_DENIED,
                "loopback IP without explicit pattern returns PERMISSION_DENIED");

        const grpc::Status s2 = asr::engine::authorize_remote_endpoint(
            "http://localhost:8000/v1/audio/transcriptions", {"*"}, &policy);
        require(s2.error_code() == grpc::StatusCode::PERMISSION_DENIED,
                "localhost without explicit pattern returns PERMISSION_DENIED");

        const grpc::Status s3 = asr::engine::authorize_remote_endpoint(
            "http://169.254.169.254/latest/meta-data", {"*"}, &policy);
        require(s3.error_code() == grpc::StatusCode::PERMISSION_DENIED,
                "link-local metadata IP without explicit pattern returns PERMISSION_DENIED");

        const grpc::Status s4 = asr::engine::authorize_remote_endpoint(
            "http://10.1.2.3:8000/v1/audio/transcriptions", {"*"}, &policy);
        require(s4.error_code() == grpc::StatusCode::PERMISSION_DENIED,
                "RFC 1918 IP without explicit pattern returns PERMISSION_DENIED");
    }

    // 4. Explicit loopback IP + matching port succeeds; wrong port fails.
    {
        const grpc::Status ok = asr::engine::authorize_remote_endpoint(
            "http://127.0.0.1:8000/v1/audio/transcriptions", {"127.0.0.1:8000"}, &policy);
        require(ok.ok(), "explicit 127.0.0.1:8000 pattern succeeds");

        const grpc::Status bad_port = asr::engine::authorize_remote_endpoint(
            "http://127.0.0.1:8001/v1/audio/transcriptions", {"127.0.0.1:8000"}, &policy);
        require(bad_port.error_code() == grpc::StatusCode::PERMISSION_DENIED,
                "mismatched port returns PERMISSION_DENIED");
    }
}

void verify_multipart_and_verbose_json() {
    const std::string kSecret = "Bearer super-secret-token-do-not-leak-98765";
    RecordedHttpRequest captured;

    const std::string canned_json = R"json({
      "task": "transcribe",
      "language": "english",
      "duration": 4.25,
      "text": "Hello world. Second turn.",
      "segments": [
        {
          "id": 0,
          "start": 0.0,
          "end": 1.8,
          "text": " Hello world.",
          "tokens": [101, 102, 103],
          "avg_logprob": -0.22,
          "no_speech_prob": 0.03,
          "words": [
            {"word": " Hello", "start": 0.0, "end": 0.8, "probability": 0.95},
            {"word": " world.", "start": 0.8, "end": 1.8, "probability": 0.91}
          ]
        },
        {
          "id": 1,
          "start": 2.0,
          "end": 4.25,
          "text": " Second turn.",
          "tokens": [201, 202],
          "avg_logprob": -0.35,
          "no_speech_prob": 0.08,
          "words": [
            {"word": " Second", "start": 2.0, "end": 3.1, "probability": 0.88},
            {"word": " turn.", "start": 3.1, "end": 4.25, "probability": 0.84}
          ]
        }
      ]
    })json";

    LoopbackHttpServer server([&](int fd, const RecordedHttpRequest& req, std::atomic<bool>&) {
        captured = req;
        const std::string resp =
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
            std::to_string(canned_json.size()) + "\r\nConnection: close\r\n\r\n" + canned_json;
        send_all(fd, resp);
    });

    asrv1::RemoteTranscriptionOptions opts;
    opts.set_url(server.url());
    (*opts.mutable_headers())["Authorization"] = kSecret;
    opts.set_model("whisper-1");
    opts.set_language("en");
    opts.set_initial_prompt("ProperNouns");
    opts.set_word_timestamps(true);
    (*opts.mutable_params())["temperature"].set_double_value(0.25);
    (*opts.mutable_params())["vad_filter"].set_bool_value(true);

    asr::engine::RemoteEndpointPolicy policy;
    require(asr::engine::authorize_remote_endpoint(opts.url(), {server.host_port()}, &policy).ok(),
            "loopback server authorized");

    const std::string wav = make_wav(0.5, 440.0);
    asr::media::ByteStream upload;
    upload.append(wav.data(), wav.size());
    upload.complete();

    std::vector<asr::engine::EngineSegment> segments;
    asr::engine::EngineResult result;
    const grpc::Status status = asr::engine::run_remote_transcription(
        opts, policy, upload, "sample.wav", "audio/wav",
        [&](const asr::engine::EngineSegment& seg, bool is_final) {
            require(is_final, "remote engine emits final segments");
            segments.push_back(seg);
            return true;
        },
        &result);

    require(status.ok(), "remote transcription succeeded: " + status.error_message());
    require(status.error_message().find(kSecret) == std::string::npos,
            "Authorization secret absent from status");

    // Verify Authorization header arrived on the wire.
    require(captured.raw_headers.find("Authorization: " + kSecret) != std::string::npos,
            "server received verbatim Authorization header");

    // Verify multipart form field order and values match remote_form_fields(opts),
    // followed by the file part.
    const auto expected_fields = asr::engine::remote_form_fields(opts);
    const auto parts = parse_multipart_parts(captured);
    require(parts.size() == expected_fields.size() + 1,
            "multipart has expected form fields + 1 file part");
    for (size_t i = 0; i < expected_fields.size(); ++i) {
        require(parts[i].name == expected_fields[i].first,
                "field " + std::to_string(i) + " name matches remote_form_fields order");
        require(parts[i].body == expected_fields[i].second,
                "field " + parts[i].name + " value matches remote_form_fields");
    }
    const MultipartPart& file_part = parts.back();
    require(file_part.name == "file", "last part is file");
    require(file_part.filename == "sample.wav", "file part carries filename");
    require(file_part.body == wav, "file part streamed exact media bytes");

    // Verify segments, words, and trailer fields.
    require(segments.size() == 2, "two segments emitted");
    require(segments[0].index == 0 && segments[1].index == 1, "dense segment indexes");
    require(segments[0].start_ms == 0 && segments[0].end_ms == 1800, "segment 0 timing");
    require(segments[0].text == " Hello world.", "segment 0 text");
    require(segments[0].has_avg_logprob && segments[0].has_no_speech_prob, "segment 0 scores set");
    require(segments[0].language == "en", "language 'english' normalized to 'en'");
    require(segments[0].words.size() == 2 && segments[1].words.size() == 2, "words parsed");
    require(segments[0].words[0].text == " Hello" && segments[0].words[0].end_ms == 800,
            "word 0 text and end_ms");
    require(result.language == "en", "trailer language is en");
    require(result.duration_ms == 4250, "trailer duration_ms is 4250");
    require(result.final_segments == 2 && result.tokens == 5, "trailer counts");
}

void verify_error_mapping_and_secret_redaction() {
    const std::string kSecret = "Bearer ultra-secret-key-never-echo";

    // 1. 400 naming a field -> INVALID_ARGUMENT, secret never echoed.
    {
        LoopbackHttpServer server([&](int fd, const RecordedHttpRequest&, std::atomic<bool>&) {
            const std::string body = R"({"error":{"message":"invalid model 'bad'","param":"model"}})";
            const std::string resp =
                "HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
            send_all(fd, resp);
        });
        asrv1::RemoteTranscriptionOptions opts;
        opts.set_url(server.url());
        (*opts.mutable_headers())["Authorization"] = kSecret;
        opts.set_model("bad");
        asr::engine::RemoteEndpointPolicy policy;
        require(asr::engine::authorize_remote_endpoint(opts.url(), {server.host_port()}, &policy).ok(),
                "authorized");
        asr::media::ByteStream upload;
        const std::string wav = make_wav(0.2, 440.0);
        upload.append(wav.data(), wav.size());
        upload.complete();
        asr::engine::EngineResult res;
        const grpc::Status s = asr::engine::run_remote_transcription(
            opts, policy, upload, "a.wav", "audio/wav",
            [](const auto&, bool) { return true; }, &res);
        require(s.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
                "400 naming field maps to INVALID_ARGUMENT");
        require(s.error_message().find(kSecret) == std::string::npos,
                "400 error message never echoes Authorization header");
    }

    // 2. 503 -> UNAVAILABLE, secret never echoed.
    {
        LoopbackHttpServer server([&](int fd, const RecordedHttpRequest&, std::atomic<bool>&) {
            const std::string body = "upstream overloaded";
            const std::string resp =
                "HTTP/1.1 503 Service Unavailable\r\nContent-Length: " +
                std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
            send_all(fd, resp);
        });
        asrv1::RemoteTranscriptionOptions opts;
        opts.set_url(server.url());
        (*opts.mutable_headers())["Authorization"] = kSecret;
        asr::engine::RemoteEndpointPolicy policy;
        require(asr::engine::authorize_remote_endpoint(opts.url(), {server.host_port()}, &policy).ok(),
                "authorized");
        asr::media::ByteStream upload;
        const std::string wav = make_wav(0.2, 440.0);
        upload.append(wav.data(), wav.size());
        upload.complete();
        asr::engine::EngineResult res;
        const grpc::Status s = asr::engine::run_remote_transcription(
            opts, policy, upload, "a.wav", "audio/wav",
            [](const auto&, bool) { return true; }, &res);
        require(s.error_code() == grpc::StatusCode::UNAVAILABLE, "503 maps to UNAVAILABLE");
        require(s.error_message().find(kSecret) == std::string::npos,
                "503 error message never echoes Authorization header");
    }

    // 3. Response > 16 MiB -> RESOURCE_EXHAUSTED.
    {
        LoopbackHttpServer server([&](int fd, const RecordedHttpRequest&, std::atomic<bool>& stop) {
            const size_t total = asr::engine::kMaxRemoteResponseBytes + 65536;
            const std::string hdr =
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
                std::to_string(total) + "\r\nConnection: close\r\n\r\n";
            send_all(fd, hdr);
            const std::string chunk(65536, ' ');
            size_t sent = 0;
            while (sent < total && !stop.load()) {
                const ssize_t n =
                    ::send(fd, chunk.data(), std::min(chunk.size(), total - sent), MSG_NOSIGNAL);
                if (n <= 0) {
                    break;
                }
                sent += static_cast<size_t>(n);
            }
        });
        asrv1::RemoteTranscriptionOptions opts;
        opts.set_url(server.url());
        asr::engine::RemoteEndpointPolicy policy;
        require(asr::engine::authorize_remote_endpoint(opts.url(), {server.host_port()}, &policy).ok(),
                "authorized");
        asr::media::ByteStream upload;
        const std::string wav = make_wav(0.2, 440.0);
        upload.append(wav.data(), wav.size());
        upload.complete();
        asr::engine::EngineResult res;
        const grpc::Status s = asr::engine::run_remote_transcription(
            opts, policy, upload, "a.wav", "audio/wav",
            [](const auto&, bool) { return true; }, &res);
        require(s.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED,
                ">16 MiB response maps to RESOURCE_EXHAUSTED");
    }
}

void verify_cancel_aborts_promptly() {
    std::atomic<bool> request_arrived{false};
    LoopbackHttpServer server([&](int /*fd*/, const RecordedHttpRequest&, std::atomic<bool>& stop) {
        request_arrived.store(true);
        // Hold the connection open without answering until cancelled.
        for (int i = 0; i < 100 && !stop.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    asrv1::RemoteTranscriptionOptions opts;
    opts.set_url(server.url());
    opts.set_timeout_seconds(30.0);
    asr::engine::RemoteEndpointPolicy policy;
    require(asr::engine::authorize_remote_endpoint(opts.url(), {server.host_port()}, &policy).ok(),
            "authorized");

    asr::media::ByteStream upload;
    const std::string wav = make_wav(0.2, 440.0);
    upload.append(wav.data(), wav.size());
    upload.complete();

    std::stop_source cancel_source;
    std::chrono::steady_clock::time_point cancel_time;
    std::jthread canceller([&] {
        while (!request_arrived.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cancel_time = std::chrono::steady_clock::now();
        cancel_source.request_stop();
    });

    asr::engine::EngineResult res;
    const grpc::Status s = asr::engine::run_remote_transcription(
        opts, policy, upload, "a.wav", "audio/wav",
        [](const auto&, bool) { return true; }, &res, cancel_source.get_token());
    const auto abort_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - cancel_time);

    require(s.error_code() == grpc::StatusCode::CANCELLED && res.aborted,
            "cancelled transfer returns CANCELLED");
    require(abort_elapsed.count() < 250,
            "transfer aborted within poll interval, took " +
                std::to_string(abort_elapsed.count()) + "ms");
}

}  // namespace

int main() {
    try {
        verify_allowlist_and_ssrf_policy();
        verify_multipart_and_verbose_json();
        verify_error_mapping_and_secret_redaction();
        verify_cancel_aborts_promptly();
    } catch (const std::exception& error) {
        std::println(stderr, "{}", error.what());
        return 1;
    }
    std::println("remote-engine-test passed");
    return 0;
}
