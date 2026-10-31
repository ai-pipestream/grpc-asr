#pragma once
#include "engine/model_pool.h"
#include "speech.h"

namespace asr::engine {
// The streaming worker borrows the same states as the collector API.
class StreamBackend final : public ::media::SpeechBackend {
 public:
  StreamBackend(ModelPool&, const Config&, std::string model);
  void validate_options(const ::media::mediapb::SpeechOptions&) const override;
  std::vector<::media::mediapb::TranscriptSegment> infer(const ::media::SpeechWindow&, uint32_t,
      const ::media::mediapb::SpeechOptions&, const std::function<bool()>&) override;
  const std::string& model_hash() const override { return hash_; }
  const std::string& model_id() const override { return id_; }
  int threads() const override { return config_.threads; }
  std::string execution_backend() const override { return pool_.backend(); }
  std::optional<bool> multilingual_capability() const override { return multilingual_; }
  bool supports_translation() const override { return multilingual_; }
  bool supports_language_hint() const override { return true; }
  bool reports_language() const override { return true; }
  ::media::mediapb::SpeechBackendKind backend_kind() const override { return ::media::mediapb::SPEECH_BACKEND_KIND_WHISPER; }
  std::string engine_version() const override { return ::media::SpeechModel::library_version(); }
 private:
  ModelPool& pool_;
  Config config_;
  std::string id_, hash_;
  bool multilingual_;
};
}
