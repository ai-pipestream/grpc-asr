#pragma once

#include <stop_token>
#include <string>
#include <vector>

#include <grpcpp/support/status.h>

#include "ai/pipestream/asr/v1/asr_service.pb.h"
#include "config.h"
#include "engine/transcriber.h"
#include "media/byte_stream.h"

namespace asr::engine {

inline constexpr size_t kMaxRemoteResponseBytes = 16ULL * 1024ULL * 1024ULL;

// Validated, DNS-pinned connection target produced by authorize_remote_endpoint.
struct RemoteEndpointPolicy {
    std::string host;
    uint16_t port = 0;
    std::string pinned_ip;
    bool allow_loopback = false;
    std::vector<std::string> explicit_ip_literals;
};

// Validates `url` against `allowed_hosts`, resolves DNS, and enforces the
// non-public address policy (loopback, link-local, RFC 1918, ULA, CGNAT, and
// reserved ranges are refused unless explicitly named in the matched pattern).
// Returns FAILED_PRECONDITION when `allowed_hosts` is empty, PERMISSION_DENIED
// when the host or resolved IP is disallowed, UNAVAILABLE on DNS failure.
grpc::Status authorize_remote_endpoint(const std::string& url,
                                       const std::vector<std::string>& allowed_hosts,
                                       RemoteEndpointPolicy* out = nullptr);

// Posts the completed `upload` as multipart/form-data to `options.url()`,
// parses the OpenAI-compatible `verbose_json` response, emits FinalSegment
// events through `sink`, and populates `result`.
// Never logs or echoes HTTP header values.
grpc::Status run_remote_transcription(
    const ai::pipestream::asr::v1::RemoteTranscriptionOptions& options,
    const RemoteEndpointPolicy& policy, media::ByteStream& upload,
    const std::string& filename, const std::string& content_type,
    const Transcriber::SegmentSink& sink, EngineResult* result,
    std::stop_token stop = {});

}  // namespace asr::engine
