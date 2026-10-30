#include "parakeet_model.h"
#include "parakeet.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <mutex>
#include <stdexcept>

namespace media {
struct ParakeetModel::Impl {
  parakeet_context* context = nullptr;
  parakeet_state* state = nullptr;
  std::string hash, id;
  int threads;
  std::mutex mutex;
  ~Impl() { if (state) parakeet_free_state(state); if (context) parakeet_free(context); }
};
ParakeetModel::ParakeetModel(std::span<const char> model, std::string model_id, int threads)
    : impl_(std::make_unique<Impl>()) {
  if (model.empty() || model_id.empty() || threads < 1 || threads > 64)
    throw std::invalid_argument("invalid speech model configuration");
  impl_->id = std::move(model_id);
  impl_->threads = threads;
  impl_->hash = sha256(std::string(model.data(), model.size()));
  auto parameters = parakeet_context_default_params();
  parameters.use_gpu = false;
  impl_->context = parakeet_init_from_buffer_with_params_no_state(
      const_cast<char*>(model.data()), model.size(), parameters);
  if (!impl_->context) throw std::runtime_error("native Parakeet model load failed");
  impl_->state = parakeet_init_state(impl_->context);
  if (!impl_->state) throw std::runtime_error("native Parakeet state allocation failed");
}
ParakeetModel::~ParakeetModel() = default;
const std::string& ParakeetModel::model_hash() const { return impl_->hash; }
const std::string& ParakeetModel::model_id() const { return impl_->id; }
int ParakeetModel::threads() const { return impl_->threads; }
std::string ParakeetModel::engine_version() const { return std::string("whisper.cpp/parakeet/") + parakeet_version(); }
std::optional<mediapb::EncoderContext> ParakeetModel::encoder_context(size_t samples) const {
  if (!samples || samples > 30 * 16000)
    throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "invalid encoder input length");
  mediapb::EncoderContext context;
  // The native normalizer requires at least two valid feature frames.
  context.set_analysis_samples(std::max<size_t>(samples, 320));
  context.set_frame_hop_samples(160);
  context.set_feature_frames(context.analysis_samples() / context.frame_hop_samples() + 1);
  context.set_encoder_frames(std::max(400u, context.feature_frames()));
  return context;
}
void ParakeetModel::validate_options(const mediapb::SpeechOptions& options) const {
  if (options.model_id() != impl_->id || !options.language().empty() ||
      (options.task() != mediapb::SPEECH_TASK_UNSPECIFIED && options.task() != mediapb::SPEECH_TASK_TRANSCRIBE))
    throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "Parakeet accepts transcription without language hints only");
}
std::vector<mediapb::TranscriptSegment> ParakeetModel::infer(const SpeechWindow& window, uint32_t track,
    const mediapb::SpeechOptions& options, const std::function<bool()>& cancelled) {
  validate_options(options);
  if (window.samples.empty() || window.samples.size() > 30 * 16000 ||
      window.first_sample > UINT64_C(100000000000000))
    throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "invalid speech inference request");
  for (float sample : window.samples)
    if (!std::isfinite(sample)) throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "nonfinite speech PCM");
  std::lock_guard lock(impl_->mutex);
  if (cancelled()) throw Failure(grpc::StatusCode::CANCELLED, "speech cancelled");
  struct Abort {
    const std::function<bool()>& cancelled;
    std::exception_ptr error;
  } abort{cancelled, {}};
  auto params = parakeet_full_default_params(PARAKEET_SAMPLING_GREEDY);
  params.n_threads = impl_->threads;
  params.no_context = true;
  const auto preparation = *encoder_context(window.samples.size());
  params.audio_ctx = static_cast<int>(preparation.encoder_frames());
  std::vector<float> padded;
  std::span<const float> analysis = window.samples;
  if (preparation.analysis_samples() > window.samples.size()) {
    padded = window.samples;
    padded.resize(preparation.analysis_samples(), 0.0f);
    analysis = padded;
  }
  params.abort_callback = [](void* opaque) noexcept {
    auto& a = *static_cast<Abort*>(opaque);
    try { return a.cancelled(); } catch (...) { a.error = std::current_exception(); return true; }
  };
  params.abort_callback_user_data = &abort;
  const int result = parakeet_full_with_state(impl_->context, impl_->state, params,
                                              analysis.data(), static_cast<int>(analysis.size()));
  if (abort.error) std::rethrow_exception(abort.error);
  if (cancelled()) throw Failure(grpc::StatusCode::CANCELLED, "speech cancelled");
  if (result) throw Failure(grpc::StatusCode::INTERNAL, "native Parakeet inference failed");
  std::vector<mediapb::TranscriptSegment> output;
  const int64_t base = static_cast<int64_t>((window.first_sample / 16000) * 1000000 +
                                          (window.first_sample % 16000) * 1000000 / 16000);
  for (int i = 0; i < parakeet_full_n_segments_from_state(impl_->state); ++i) {
    auto& segment = output.emplace_back();
    segment.set_segment_id(uuid());
    segment.set_revision(1);
    // Reconciliation owns finalization across windows, not this model adapter.
    segment.set_final(false);
    segment.set_track_id(track);
    segment.set_start_us(base + parakeet_full_get_segment_t0_from_state(impl_->state, i) * 10000);
    segment.set_end_us(base + parakeet_full_get_segment_t1_from_state(impl_->state, i) * 10000);
    segment.set_text(parakeet_full_get_segment_text_from_state(impl_->state, i));
    // The native API exposes neither detected language nor a language-hint input.
    segment.set_task(mediapb::SPEECH_TASK_TRANSCRIBE);
    double sum = 0;
    const int count = parakeet_full_n_tokens_from_state(impl_->state, i);
    for (int t = 0; t < count; ++t) {
      const auto native = parakeet_full_get_token_data_from_state(impl_->state, i, t);
      auto* token = segment.add_tokens();
      token->set_token_id(native.id);
      token->set_text(parakeet_full_get_token_text_from_state(impl_->context, impl_->state, i, t));
      token->set_probability(native.p);
      const bool control = native.id == parakeet_token_blank(impl_->context) || native.id == parakeet_token_bos(impl_->context);
      token->set_kind(control ? mediapb::SPEECH_TOKEN_KIND_CONTROL : mediapb::SPEECH_TOKEN_KIND_TEXT);
      const int length = control ? 0 : parakeet_token_to_text(token->text().c_str(), t == 0, nullptr, 0);
      if (length < 0 || length > 65535)
        throw Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "native token rendering exceeds limit");
      std::string rendered(length + 1, '\0');
      if (!control && parakeet_token_to_text(token->text().c_str(), t == 0, rendered.data(), rendered.size()) != length)
        throw Failure(grpc::StatusCode::DATA_LOSS, "native token rendering changed size");
      rendered.resize(length); token->set_rendered_text(rendered);
      auto* details = token->mutable_parakeet();
      details->set_duration_index(native.duration_idx); details->set_duration_value(native.duration_value);
      details->set_frame_index(native.frame_index); details->set_log_probability(native.plog);
      details->set_start_centiseconds(native.t0); details->set_end_centiseconds(native.t1);
      details->set_is_word_start(native.is_word_start);
      if (native.t0 >= 0 && native.t1 >= native.t0) {
        token->set_start_us(base + native.t0 * 10000);
        token->set_end_us(base + native.t1 * 10000);
      }
      sum += native.plog;
    }
    if (count) segment.set_average_log_probability(sum / count);
  }
  return output;
}
}  // namespace media
