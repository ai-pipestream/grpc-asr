// Docling-slim typed ASR fields on TranscribeOptions: engine / model_type /
// inline options. Unsupported engines and presets return UNIMPLEMENTED by
// name; whisper.cpp-servable presets resolve to a model id.

#include "engine/docling_asr_options.h"

#include <print>
#include <string>

#include "fixture.h"

namespace asrv1 = ai::pipestream::asr::v1;
using asr::engine::apply_model_type_default;
using asr::engine::validate_docling_asr_options;
using asr::engine::whisper_cpp_model_for;

namespace {

void expect_ok(const asrv1::TranscribeOptions& options, const std::string& name) {
  const grpc::Status status = validate_docling_asr_options(options);
  require(status.ok(), name + ": expected OK, got " + status.error_message());
}

void expect_unimplemented(const asrv1::TranscribeOptions& options, const std::string& needle,
                          const std::string& name) {
  const grpc::Status status = validate_docling_asr_options(options);
  require(status.error_code() == grpc::StatusCode::UNIMPLEMENTED,
          name + ": expected UNIMPLEMENTED, got " + std::to_string(status.error_code()) + " " +
              status.error_message());
  require(status.error_message().find(needle) != std::string::npos,
          name + ": message missing '" + needle + "': " + status.error_message());
}

void expect_invalid(const asrv1::TranscribeOptions& options, const std::string& needle,
                    const std::string& name) {
  const grpc::Status status = validate_docling_asr_options(options);
  require(status.error_code() == grpc::StatusCode::INVALID_ARGUMENT,
          name + ": expected INVALID_ARGUMENT, got " + status.error_message());
  require(status.error_message().find(needle) != std::string::npos,
          name + ": message missing '" + needle + "': " + status.error_message());
}

void verify_whisper_cpp_presets_map() {
  require(whisper_cpp_model_for(asrv1::ASR_MODEL_TYPE_WHISPER_TINY) == "tiny", "tiny");
  require(whisper_cpp_model_for(asrv1::ASR_MODEL_TYPE_WHISPER_BASE) == "base", "base");
  require(whisper_cpp_model_for(asrv1::ASR_MODEL_TYPE_WHISPER_SMALL) == "small", "small");
  require(whisper_cpp_model_for(asrv1::ASR_MODEL_TYPE_WHISPER_MEDIUM) == "medium", "medium");
  require(whisper_cpp_model_for(asrv1::ASR_MODEL_TYPE_WHISPER_LARGE) == "large-v3", "large");
  require(whisper_cpp_model_for(asrv1::ASR_MODEL_TYPE_WHISPER_TURBO) == "large-v3-turbo", "turbo");
  require(!whisper_cpp_model_for(asrv1::ASR_MODEL_TYPE_WHISPER_TINY_MLX).has_value(), "mlx");
  require(!whisper_cpp_model_for(asrv1::ASR_MODEL_TYPE_GRANITE_SPEECH_5_0_2B).has_value(),
          "granite");
  require(!whisper_cpp_model_for(asrv1::ASR_MODEL_TYPE_WHISPER_TINY_TRANSFORMERS).has_value(),
          "transformers preset");
}

void verify_engine_arms() {
  asrv1::TranscribeOptions options;
  options.set_model("tiny.en");
  expect_ok(options, "plain whisper.cpp");

  options.set_engine(asrv1::ASR_ENGINE_KIND_WHISPER_CPP);
  expect_ok(options, "explicit whisper_cpp");

  options.set_engine(asrv1::ASR_ENGINE_KIND_TRANSFORMERS);
  expect_unimplemented(options, "TRANSFORMERS", "transformers engine");

  options.set_engine(asrv1::ASR_ENGINE_KIND_MLX_WHISPER);
  expect_unimplemented(options, "MLX_WHISPER", "mlx engine");

  options.set_engine(asrv1::ASR_ENGINE_KIND_NATIVE_WHISPER);
  expect_unimplemented(options, "NATIVE_WHISPER", "native engine");

  options.set_engine(asrv1::ASR_ENGINE_KIND_REMOTE);
  expect_invalid(options, "REMOTE requires", "remote without remote options");
}

void verify_inline_options_unimplemented() {
  asrv1::TranscribeOptions options;
  options.set_model("tiny.en");
  options.mutable_transformers()->set_language("en");
  expect_unimplemented(options, "transformers", "transformers inline");

  options.Clear();
  options.set_model("tiny.en");
  options.mutable_mlx_whisper()->set_condition_on_previous_text(false);
  options.mutable_mlx_whisper()->set_hallucination_silence_threshold(1.5);
  options.mutable_mlx_whisper()->set_initial_prompt("vocab");
  expect_unimplemented(options, "mlx_whisper", "mlx inline with slim 2.145.0 fields");

  options.Clear();
  options.set_model("tiny.en");
  options.mutable_native_whisper()->set_condition_on_previous_text(true);
  expect_unimplemented(options, "native_whisper", "native inline");
}

void verify_model_type_resolution() {
  asrv1::TranscribeOptions options;
  options.set_model_type(asrv1::ASR_MODEL_TYPE_WHISPER_TINY);
  expect_ok(options, "tiny preset");
  apply_model_type_default(&options);
  require(options.model() == "tiny", "tiny preset fills model");

  options.Clear();
  options.set_model_type(asrv1::ASR_MODEL_TYPE_WHISPER_TINY_MLX);
  expect_unimplemented(options, "model_type", "mlx preset without model");

  options.Clear();
  options.set_model("tiny.en");
  options.set_model_type(asrv1::ASR_MODEL_TYPE_WHISPER_TINY_MLX);
  expect_unimplemented(options, "model_type", "explicit model cannot select an unsupported preset");

  options.Clear();
  options.set_model_type_raw("some-future-preset");
  expect_unimplemented(options, "model_type_raw", "raw without enum");
}

void verify_conflicts_and_unknown_values() {
  asrv1::TranscribeOptions options;
  options.set_model("tiny.en");
  options.set_model_type(static_cast<asrv1::AsrModelType>(999));
  expect_invalid(options, "model_type", "unknown enum with explicit model");
  options.clear_model_type();
  options.set_model_type_raw("future-backend");
  expect_unimplemented(options, "model_type_raw", "raw preset with explicit model");
  options.clear_model_type_raw();
  options.set_engine(asrv1::ASR_ENGINE_KIND_WHISPER_CPP);
  options.mutable_remote();
  expect_invalid(options, "engine", "local engine with remote options");
  options.clear_remote();
  options.mutable_mlx_whisper();
  expect_invalid(options, "engine", "local engine with mlx options");
  options.clear_engine();
  options.mutable_transformers();
  expect_invalid(options, "options", "multiple inline option messages");
  options.clear_transformers();
  options.mutable_remote();
  expect_invalid(options, "options", "remote and inline options");
}

void verify_task_and_task_raw() {
  asrv1::TranscribeOptions options;
  options.set_model("tiny.en");
  options.set_task(static_cast<asrv1::Task>(12345));
  expect_invalid(options, "TranscribeOptions.task", "unknown numeric task");

  options.Clear();
  options.set_model("tiny.en");
  options.set_task_raw("translate");
  expect_ok(options, "task_raw translate");
  apply_model_type_default(&options);
  require(options.task() == asrv1::TASK_TRANSLATE, "task_raw translate populates TASK_TRANSLATE");

  options.Clear();
  options.set_model("tiny.en");
  options.set_task_raw("  TASK_TRANSCRIBE ");
  expect_ok(options, "task_raw TASK_TRANSCRIBE");
  apply_model_type_default(&options);
  require(options.task() == asrv1::TASK_TRANSCRIBE,
          "task_raw TASK_TRANSCRIBE populates TASK_TRANSCRIBE");

  options.Clear();
  options.set_model("tiny.en");
  options.set_task(asrv1::TASK_TRANSCRIBE);
  options.set_task_raw("translate");
  expect_invalid(options, "conflicts with TranscribeOptions.task_raw",
                 "conflicting task and task_raw");

  options.Clear();
  options.set_model("tiny.en");
  options.set_task_raw("diarize-only");
  expect_invalid(options, "TranscribeOptions.task_raw", "unknown task_raw");
}

}  // namespace

int main() {
  try {
    verify_conflicts_and_unknown_values();
    verify_whisper_cpp_presets_map();
    verify_engine_arms();
    verify_inline_options_unimplemented();
    verify_model_type_resolution();
    verify_task_and_task_raw();
  } catch (const std::exception& ex) {
    std::println(stderr, "FAIL: {}", ex.what());
    return 1;
  }
  std::println("ok");
  return 0;
}
