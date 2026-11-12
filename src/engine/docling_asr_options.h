#pragma once

#include <optional>
#include <string>

#include <grpcpp/grpcpp.h>

#include "ai/pipestream/asr/v1/asr_service.pb.h"

namespace asr::engine {

// Resolves TranscribeOptions.model when empty from a whisper.cpp-servable
// AsrModelType preset. Returns nullopt when the preset needs another engine.
std::optional<std::string> whisper_cpp_model_for(ai::pipestream::asr::v1::AsrModelType type);

// Validates docling-slim typed ASR fields on TranscribeOptions. Returns
// UNIMPLEMENTED for engines and presets this build cannot run; OK when the
// request stays on the local whisper.cpp pool (or only uses remote, which
// the caller already handles).
grpc::Status validate_docling_asr_options(const ai::pipestream::asr::v1::TranscribeOptions& options);

// Fills options.model from model_type when model is empty and the preset
// maps to a whisper.cpp id, and resolves options.task from options.task_raw
// when task is unspecified. Call after validate_docling_asr_options succeeds.
void apply_model_type_default(ai::pipestream::asr::v1::TranscribeOptions* options);

}  // namespace asr::engine
