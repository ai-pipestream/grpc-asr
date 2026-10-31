#include "worker.h"
#include <bit>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <thread>

namespace media {
namespace {
void require(bool ok, const char* reason) {
  if (!ok) throw Failure(grpc::StatusCode::INVALID_ARGUMENT, reason);
}
// The watchdog exists only during a blocking read, never during inference or
// response construction. At most one watchdog per admitted worker stream.
class ReadDeadline {
 public:
  ReadDeadline(grpc::ServerContext& context, std::chrono::milliseconds timeout)
      : thread_([this, &context, timeout] {
          std::unique_lock lock(mutex_);
          if (!changed_.wait_for(lock, timeout, [&] { return done_; })) context.TryCancel();
        }) {}
  ~ReadDeadline() {
    { std::lock_guard lock(mutex_); done_ = true; }
    changed_.notify_one();
    thread_.join();
  }
 private:
  std::mutex mutex_;
  std::condition_variable changed_;
  bool done_ = false;
  std::thread thread_;
};
}
SpeechWorker::SpeechWorker(SpeechBackend& model, std::string machine_id, std::chrono::milliseconds input_idle_timeout)
    : model_(model), input_idle_timeout_(input_idle_timeout) {
  require(input_idle_timeout.count() > 0 && input_idle_timeout <= std::chrono::hours(1), "invalid worker input idle timeout");
  if (machine_id.empty()) machine_id = uuid();
  require(valid_uuid(machine_id), "invalid worker machine UUID");
  producer_.set_machine_id(machine_id);
  producer_.set_service("grpc-asr-speech-worker");
  producer_.set_app_version(MEDIA_APP_VERSION);
  producer_.set_build_id(MEDIA_BUILD_ID);
  producer_.set_library_version(model.engine_version());
  producer_.set_execution_backend(model.execution_backend());
  producer_.set_model_id(model.model_id());
  producer_.set_model_sha256(model.model_hash());
}
grpc::Status SpeechWorker::GetModelInfo(grpc::ServerContext*, const mediapb::GetModelInfoRequest*, mediapb::GetModelInfoResponse* info) {
  *info->mutable_producer() = producer_;
  const auto multilingual = model_.multilingual_capability();
  info->set_multilingual_known(multilingual.has_value());
  info->set_multilingual(multilingual.value_or(false));
  info->set_backend(model_.backend_kind());
  info->set_accepts_language_hint(model_.supports_language_hint());
  info->set_reports_language(model_.reports_language());
  if (const auto minimum = model_.minimum_encoder_context_ms())
    info->set_minimum_encoder_context_ms(*minimum);
  info->add_tasks(mediapb::SPEECH_TASK_TRANSCRIBE);
  if (model_.supports_translation()) info->add_tasks(mediapb::SPEECH_TASK_TRANSLATE_TO_ENGLISH);
  info->set_max_window_samples(max_window_samples);
  info->set_sample_rate_hz(16000);
  info->set_max_streams(max_streams);
  info->set_max_concurrent_inferences(1);
  info->set_inference_threads(model_.threads());
  info->set_input_idle_timeout_ms(input_idle_timeout_.count());
  return grpc::Status::OK;
}
grpc::Status SpeechWorker::Transcribe(grpc::ServerContext* context,
    grpc::ServerReaderWriter<mediapb::TranscribeResponse, mediapb::TranscribeRequest>* wire) {
  if (streams_.fetch_add(1) >= max_streams) {
    streams_.fetch_sub(1);
    return {grpc::StatusCode::RESOURCE_EXHAUSTED, "speech worker stream limit reached"};
  }
  struct Release { std::atomic<unsigned>& streams; ~Release() { streams.fetch_sub(1); } } release{streams_};
  try {
    auto read = [&](mediapb::TranscribeRequest* request) {
      ReadDeadline deadline(*context, input_idle_timeout_);
      return wire->Read(request);
    };
    mediapb::TranscribeRequest request;
    require(read(&request) && request.has_open(), "first worker message must open a track");
    auto accepted = request.open();
    require(valid_uuid(accepted.stream_id()), "invalid source stream UUID");
    require(!accepted.options().has_windowing(), "worker accepts pre-cut windows, not a windowing policy");
    model_.validate_options(accepted.options());
    if (accepted.options().task() == mediapb::SPEECH_TASK_UNSPECIFIED)
      accepted.mutable_options()->set_task(mediapb::SPEECH_TASK_TRANSCRIBE);
    mediapb::TranscribeResponse response;
    auto* opened = response.mutable_opened();
    *opened->mutable_accepted() = accepted;
    *opened->mutable_producer() = producer_;
    opened->mutable_producer()->set_configuration_sha256(
        sha256(accepted.options().SerializeAsString() + ":threads=" + std::to_string(model_.threads()) +
               ":input_idle_ms=" + std::to_string(input_idle_timeout_.count())));
    opened->set_max_window_samples(max_window_samples);
    const auto producer = opened->producer();
    if (!wire->Write(response)) return {grpc::StatusCode::CANCELLED, "worker handshake disconnected"};
    uint64_t count = 0, end = 0;
    bool after_discontinuity = false, after_end = false;
    while (read(&request)) {
      if (request.has_finish()) {
        require(request.finish().window_count() == count, "worker finish count mismatch");
        response.Clear(); response.mutable_closed()->set_window_count(count);
        if (!wire->Write(response)) return {grpc::StatusCode::CANCELLED, "worker close disconnected"};
        return grpc::Status::OK;
      }
      require(request.has_window(), "expected worker window or finish");
      const auto& input = request.window();
      require(!after_end, "audio after end boundary");
      require(count != std::numeric_limits<uint64_t>::max() && input.window_number() == count + 1,
              "noncontiguous worker window number");
      require(!input.pcm_f32le().empty() && input.pcm_f32le().size() % 4 == 0 && input.pcm_f32le().size() <= max_window_samples * 4,
              "invalid inference PCM size");
      const uint64_t samples = input.pcm_f32le().size() / 4;
      require(input.first_sample() <= UINT64_C(100000000000000), "worker sample position exceeds limit");
      require(input.first_new_sample() >= input.first_sample() && input.first_new_sample() < input.first_sample() + samples,
              "window contains no new audio or has invalid overlap");
      require(count ? input.first_new_sample() == end : input.first_new_sample() == input.first_sample(),
              "worker new-audio boundary is not contiguous");
      require(!after_discontinuity || input.first_sample() == end, "overlap crossed a discontinuity");
      require(input.boundary() >= mediapb::INFERENCE_BOUNDARY_FIXED && input.boundary() <= mediapb::INFERENCE_BOUNDARY_DISCONTINUITY,
              "invalid inference boundary");
      SpeechWindow window{input.window_number(), input.first_sample(), input.first_new_sample(), WindowBoundary::Fixed, {}};
      window.samples.reserve(samples);
      const auto& bytes = input.pcm_f32le();
      for (size_t i = 0; i < bytes.size(); i += 4) {
        uint32_t bits = 0;
        for (int b = 0; b < 4; ++b) bits |= static_cast<uint32_t>(static_cast<unsigned char>(bytes[i+b])) << (8*b);
        window.samples.push_back(std::bit_cast<float>(bits));
      }
      if (busy_.test_and_set()) return {grpc::StatusCode::RESOURCE_EXHAUSTED, "speech inference slot is busy"};
      auto segments = [&] {
        struct InferenceRelease { std::atomic_flag& flag; ~InferenceRelease() { flag.clear(); } } slot{busy_};
        return model_.infer(window, accepted.track_id(), accepted.options(), [&] { return context->IsCancelled(); });
      }();
      response.Clear();
      auto* hypothesis = response.mutable_hypothesis();
      hypothesis->set_window_number(input.window_number()); hypothesis->set_track_id(accepted.track_id());
      hypothesis->set_first_sample(input.first_sample()); hypothesis->set_first_new_sample(input.first_new_sample());
      hypothesis->set_sample_count(samples); hypothesis->set_boundary(input.boundary());
      if (const auto preparation = model_.encoder_context(samples))
        *hypothesis->mutable_encoder_context() = *preparation;
      *hypothesis->mutable_producer() = producer;
      for (auto& segment : segments) *hypothesis->add_segments() = std::move(segment);
      if (response.ByteSizeLong() > 4 * 1024 * 1024)
        return {grpc::StatusCode::RESOURCE_EXHAUSTED, "speech hypothesis exceeds output limit"};
      if (!wire->Write(response)) return {grpc::StatusCode::CANCELLED, "worker hypothesis disconnected"};
      ++count; end = input.first_sample() + samples;
      after_discontinuity = input.boundary() == mediapb::INFERENCE_BOUNDARY_DISCONTINUITY;
      after_end = input.boundary() == mediapb::INFERENCE_BOUNDARY_END;
    }
    return {grpc::StatusCode::CANCELLED, "worker input ended without explicit finish"};
  } catch (const Failure& error) { return {error.code, error.what()}; }
    catch (const std::exception&) { return {grpc::StatusCode::INTERNAL, "speech worker failed"}; }
}
}  // namespace media
