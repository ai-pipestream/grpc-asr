#pragma once
#include "registry.h"
#include "windowing.h"
#include <functional>
#include <memory>
#include <span>
#include <optional>

struct whisper_context;
struct whisper_state;

namespace media {
class SpeechBackend {
 public:
  virtual ~SpeechBackend() = default;
  virtual void validate_options(const mediapb::SpeechOptions&) const = 0;
  virtual std::vector<mediapb::TranscriptSegment> infer(const SpeechWindow&, uint32_t,
      const mediapb::SpeechOptions&, const std::function<bool()>&) = 0;
  virtual const std::string& model_hash() const = 0;
  virtual const std::string& model_id() const = 0;
  virtual int threads() const = 0;
  virtual std::string execution_backend() const { return "cpu"; }
  virtual std::optional<bool> multilingual_capability() const = 0;
  virtual bool supports_translation() const = 0;
  virtual bool supports_language_hint() const = 0;
  virtual bool reports_language() const = 0;
  virtual mediapb::SpeechBackendKind backend_kind() const = 0;
  virtual std::string engine_version() const = 0;
  virtual std::optional<uint32_t> minimum_encoder_context_ms() const { return std::nullopt; }
  virtual std::optional<mediapb::EncoderContext> encoder_context(size_t) const { return std::nullopt; }
};
// A native CPU model instance. Model bytes are supplied by operator-controlled
// startup configuration, never by an inference request. Calls are serialized.
class SpeechModel : public SpeechBackend {
 public:
  SpeechModel(std::span<const char> model, std::string model_id, int threads);
  // Borrows a checked-out pool state; the caller owns its lifetime.
  SpeechModel(whisper_context*, whisper_state*, std::string model_id,
              std::string model_hash, int threads);
  ~SpeechModel();
  void validate_options(const mediapb::SpeechOptions& options) const;
  SpeechModel(const SpeechModel&) = delete;
  SpeechModel& operator=(const SpeechModel&) = delete;
  std::vector<mediapb::TranscriptSegment> infer(const SpeechWindow& window, uint32_t track,
                                         const mediapb::SpeechOptions& options,
                                         const std::function<bool()>& cancelled);
  const std::string& model_hash() const;
  const std::string& model_id() const;
  int threads() const;
  bool multilingual() const;
  static std::string library_version();
  std::optional<bool> multilingual_capability() const override { return multilingual(); }
  bool supports_translation() const override { return multilingual(); }
  bool supports_language_hint() const override { return true; }
  bool reports_language() const override { return true; }
  mediapb::SpeechBackendKind backend_kind() const override { return mediapb::SPEECH_BACKEND_KIND_WHISPER; }
  std::string engine_version() const override { return library_version(); }
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace media
