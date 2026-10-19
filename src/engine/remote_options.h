// Typed remote transcription options (docling ApiAsrOptions): validation
// and the multipart form fields docling's API transcriber sends.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include <grpcpp/support/status.h>

#include "ai/pipestream/asr/v1/asr_service.pb.h"

namespace asr::engine {

// docling's ApiAsrOptions.timeout default, in seconds.
inline constexpr double kDefaultRemoteTimeoutSeconds = 600.0;

// OK, or INVALID_ARGUMENT naming the first bad field.
grpc::Status validate_remote_options(const ai::pipestream::asr::v1::RemoteTranscriptionOptions& options);

// The form fields in docling's order: response_format=verbose_json, then
// model, language, prompt (each only when non-empty), the two
// timestamp_granularities[] entries when word_timestamps is set, then
// params sorted by key (protobuf maps are unordered; docling keeps dict
// order), skipping response_format.
std::vector<std::pair<std::string, std::string>> remote_form_fields(
    const ai::pipestream::asr::v1::RemoteTranscriptionOptions& options);

// timeout_seconds, or the docling default when unset.
double remote_timeout_seconds(const ai::pipestream::asr::v1::RemoteTranscriptionOptions& options);

// A ScalarValue as docling's str(value) would print it (bool lowercased).
std::string scalar_form_value(const ai::pipestream::asr::v1::ScalarValue& value);

}  // namespace asr::engine
