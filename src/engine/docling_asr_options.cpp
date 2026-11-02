#include "engine/docling_asr_options.h"

namespace asrv1 = ai::pipestream::asr::v1;

namespace asr::engine {
namespace {

bool has_unsupported_inline(const asrv1::TranscribeOptions& options) {
  return options.has_transformers() || options.has_mlx_whisper() ||
         options.has_native_whisper();
}

}  // namespace

std::optional<std::string> whisper_cpp_model_for(asrv1::AsrModelType type) {
  switch (type) {
    case asrv1::ASR_MODEL_TYPE_WHISPER_TINY:
      return "tiny";
    case asrv1::ASR_MODEL_TYPE_WHISPER_BASE:
      return "base";
    case asrv1::ASR_MODEL_TYPE_WHISPER_SMALL:
      return "small";
    case asrv1::ASR_MODEL_TYPE_WHISPER_MEDIUM:
      return "medium";
    case asrv1::ASR_MODEL_TYPE_WHISPER_LARGE:
      return "large-v3";
    case asrv1::ASR_MODEL_TYPE_WHISPER_TURBO:
      return "large-v3-turbo";
    default:
      return std::nullopt;
  }
}

grpc::Status validate_docling_asr_options(const asrv1::TranscribeOptions& options) {
  if (options.has_engine()) {
    switch (options.engine()) {
      case asrv1::ASR_ENGINE_KIND_UNSPECIFIED:
      case asrv1::ASR_ENGINE_KIND_WHISPER_CPP:
        break;
      case asrv1::ASR_ENGINE_KIND_REMOTE:
        if (!options.has_remote()) {
          return {grpc::StatusCode::INVALID_ARGUMENT,
                  "TranscribeOptions.engine=REMOTE requires TranscribeOptions.remote"};
        }
        break;
      case asrv1::ASR_ENGINE_KIND_TRANSFORMERS:
        return {grpc::StatusCode::UNIMPLEMENTED,
                "TranscribeOptions.engine=TRANSFORMERS is not served by this build "
                "(whisper.cpp only); omit engine or set WHISPER_CPP"};
      case asrv1::ASR_ENGINE_KIND_MLX_WHISPER:
        return {grpc::StatusCode::UNIMPLEMENTED,
                "TranscribeOptions.engine=MLX_WHISPER is not served by this build "
                "(whisper.cpp only); omit engine or set WHISPER_CPP"};
      case asrv1::ASR_ENGINE_KIND_NATIVE_WHISPER:
        return {grpc::StatusCode::UNIMPLEMENTED,
                "TranscribeOptions.engine=NATIVE_WHISPER is not served by this build "
                "(whisper.cpp only); omit engine or set WHISPER_CPP"};
      default:
        return {grpc::StatusCode::INVALID_ARGUMENT,
                "TranscribeOptions.engine has an unknown value"};
    }
  }

  if (has_unsupported_inline(options)) {
    if (options.has_transformers()) {
      return {grpc::StatusCode::UNIMPLEMENTED,
              "TranscribeOptions.transformers is not served by this build"};
    }
    if (options.has_mlx_whisper()) {
      return {grpc::StatusCode::UNIMPLEMENTED,
              "TranscribeOptions.mlx_whisper is not served by this build"};
    }
    return {grpc::StatusCode::UNIMPLEMENTED,
            "TranscribeOptions.native_whisper is not served by this build"};
  }

  if (options.has_model_type() &&
      options.model_type() != asrv1::ASR_MODEL_TYPE_UNSPECIFIED && options.model().empty()) {
    if (!whisper_cpp_model_for(options.model_type()).has_value()) {
      return {grpc::StatusCode::UNIMPLEMENTED,
              "TranscribeOptions.model_type preset is not mapped to a whisper.cpp "
              "model in this build; set TranscribeOptions.model explicitly for a "
              "loaded whisper.cpp id, or omit model_type"};
    }
  }

  if (options.has_model_type_raw() && !options.model_type_raw().empty() &&
      (!options.has_model_type() || options.model_type() == asrv1::ASR_MODEL_TYPE_UNSPECIFIED) &&
      options.model().empty()) {
    return {grpc::StatusCode::UNIMPLEMENTED,
            "TranscribeOptions.model_type_raw has no closed-enum match in this build; "
            "set TranscribeOptions.model for a loaded whisper.cpp id"};
  }

  return grpc::Status::OK;
}

void apply_model_type_default(asrv1::TranscribeOptions* options) {
  if (options == nullptr || !options->model().empty()) return;
  if (!options->has_model_type()) return;
  if (auto mapped = whisper_cpp_model_for(options->model_type())) {
    options->set_model(*mapped);
  }
}

}  // namespace asr::engine
