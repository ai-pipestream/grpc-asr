#include "vad.h"
#include "registry.h"
#include "whisper.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace media {
struct VoiceActivity::Impl {
  whisper_vad_context* context = nullptr;
  Emit emit;
  std::string hash;
  std::array<float, 512> pending{};
  size_t used = 0;
  bool closed = false;
  ~Impl() { if (context) whisper_vad_free(context); }
  void flush() {
    if (!used) return;
    const auto real = used;
    std::fill(pending.begin() + used, pending.end(), 0.0f);
    if (!whisper_vad_detect_speech_no_reset(context, pending.data(), 512) || whisper_vad_n_probs(context) != 1)
      throw Failure(grpc::StatusCode::INTERNAL, "native VAD inference failed");
    const float probability = whisper_vad_probs(context)[0];
    if (!std::isfinite(probability) || probability < 0 || probability > 1)
      throw Failure(grpc::StatusCode::INTERNAL, "invalid VAD probability");
    emit(std::span(pending.data(), real), probability);
    used = 0;
  }
};
VoiceActivity::VoiceActivity(std::span<const char> model, Emit emit) : impl_(std::make_unique<Impl>()) {
  if (!emit) throw std::invalid_argument("VAD output callback required");
  impl_->hash = sha256(std::string(model.data(), model.size()));
  std::string hex;
  for (unsigned char byte : impl_->hash) { hex += "0123456789abcdef"[byte >> 4]; hex += "0123456789abcdef"[byte & 15]; }
  // This qualified model has a 512-sample input and stateful Silero architecture.
  // Reject other weights until their framing and behavior are qualified.
  if (hex != "29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf")
    throw std::invalid_argument("unsupported VAD weights; expected qualified Silero v5.1.2");
  struct Reader { std::span<const char> bytes; size_t offset = 0; } reader{model};
  whisper_model_loader loader{};
  loader.context = &reader;
  loader.read = [](void* opaque, void* output, size_t requested) -> size_t {
    auto& r = *static_cast<Reader*>(opaque);
    const auto count = std::min(requested, r.bytes.size() - r.offset);
    std::memcpy(output, r.bytes.data() + r.offset, count); r.offset += count; return count;
  };
  loader.eof = [](void* opaque) { auto& r = *static_cast<Reader*>(opaque); return r.offset == r.bytes.size(); };
  loader.close = [](void*) {};
  auto params = whisper_vad_default_context_params();
  params.use_gpu = false; params.n_threads = 1;
  impl_->context = whisper_vad_init_with_params(&loader, params);
  if (!impl_->context) throw std::runtime_error("native VAD model load failed");
  whisper_vad_reset_state(impl_->context);
  impl_->emit = std::move(emit);
}
VoiceActivity::~VoiceActivity() = default;
const std::string& VoiceActivity::model_hash() const { return impl_->hash; }
void VoiceActivity::append(std::span<const float> samples) {
  if (impl_->closed) throw std::logic_error("VAD input after close or failure");
  try {
    for (float sample : samples) {
      if (!std::isfinite(sample)) throw std::invalid_argument("nonfinite VAD sample");
      impl_->pending[impl_->used++] = sample;
      if (impl_->used == impl_->pending.size()) impl_->flush();
    }
  } catch (...) { impl_->closed = true; throw; }
}
void VoiceActivity::finish() {
  if (impl_->closed) throw std::logic_error("VAD stream already closed");
  impl_->closed = true;
  impl_->flush();
}
void VoiceActivity::discontinuity() {
  if (impl_->closed) throw std::logic_error("VAD discontinuity after close");
  try { impl_->flush(); whisper_vad_reset_state(impl_->context); }
  catch (...) { impl_->closed = true; throw; }
}
}  // namespace media
