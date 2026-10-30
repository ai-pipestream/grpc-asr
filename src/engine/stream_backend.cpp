#include "engine/stream_backend.h"
#include "whisper.h"
namespace asr::engine {
StreamBackend::StreamBackend(ModelPool& pool, const Config& config, std::string model)
    : pool_(pool), config_(config), id_(std::move(model)), hash_(pool.model_hash(id_)) {
  auto lease = pool_.acquire(id_);
  multilingual_ = whisper_is_multilingual(lease.context());
}
void StreamBackend::validate_options(const ::media::mediapb::SpeechOptions& o) const {
  namespace mediapb = ::media::mediapb;
  if (o.model_id() != id_ || o.language().find('\0') != std::string::npos ||
      (!o.language().empty() && whisper_lang_id(o.language().c_str()) < 0) ||
      (o.task() != mediapb::SPEECH_TASK_UNSPECIFIED && o.task() != mediapb::SPEECH_TASK_TRANSCRIBE &&
       o.task() != mediapb::SPEECH_TASK_TRANSLATE_TO_ENGLISH) ||
      (!multilingual_ && ((!o.language().empty() && o.language() != "en") ||
       o.task() == mediapb::SPEECH_TASK_TRANSLATE_TO_ENGLISH)))
    throw ::media::Failure(grpc::StatusCode::INVALID_ARGUMENT, "invalid pooled speech options");
}
std::vector<::media::mediapb::TranscriptSegment> StreamBackend::infer(const ::media::SpeechWindow& window,
    uint32_t track, const ::media::mediapb::SpeechOptions& options, const std::function<bool()>& cancelled) {
  validate_options(options);
  const auto deadline = config_.queue_timeout_seconds ? std::chrono::steady_clock::now() +
      std::chrono::seconds(config_.queue_timeout_seconds) : std::chrono::steady_clock::time_point::max();
  for (;;) {
    if (cancelled()) throw ::media::Failure(grpc::StatusCode::CANCELLED, "speech cancelled while waiting for pool");
    if (std::chrono::steady_clock::now() >= deadline)
      throw ::media::Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "speech pool queue timeout");
    try {
      auto lease = pool_.acquire(id_, {}, std::min(deadline,
          std::chrono::steady_clock::now() + std::chrono::milliseconds(20)));
      ::media::SpeechModel adapter(lease.context(), lease.state(), id_, hash_, config_.threads);
      return adapter.infer(window, track, options, cancelled);
    } catch (const QueueTimeout&) { /* Recheck cancellation before waiting again. */ }
  }
}
}
