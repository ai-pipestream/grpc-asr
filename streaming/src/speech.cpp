#include "speech.h"
#include "whisper.h"
#include <cmath>
#include <exception>
#include <mutex>
#include <stdexcept>

namespace media {
struct SpeechModel::Impl {
  whisper_context* context = nullptr;
  whisper_state* state = nullptr;
  std::string hash, id;
  int threads;
  std::mutex mutex;
  bool owns_state = true;
  ~Impl() { if (!owns_state) return; if (state) whisper_free_state(state); if (context) whisper_free(context); }
};
SpeechModel::SpeechModel(std::span<const char> model, std::string model_id, int threads)
    : impl_(std::make_unique<Impl>()) {
  if (model.empty() || model_id.empty() || threads < 1 || threads > 64)
    throw std::invalid_argument("invalid speech model configuration");
  impl_->id = std::move(model_id);
  impl_->threads = threads;
  impl_->hash = sha256(std::string(model.data(), model.size()));
  auto parameters = whisper_context_default_params();
  parameters.use_gpu = false;
  impl_->context = whisper_init_from_buffer_with_params_no_state(
      const_cast<char*>(model.data()), model.size(), parameters);
  if (!impl_->context) throw std::runtime_error("native Whisper model load failed");
  impl_->state = whisper_init_state(impl_->context);
  if (!impl_->state) throw std::runtime_error("native Whisper state allocation failed");
}
SpeechModel::SpeechModel(whisper_context* context, whisper_state* state,
    std::string id, std::string hash, int threads) : impl_(std::make_unique<Impl>()) {
  if (!context || !state || id.empty() || hash.size() != 32 || threads < 1 || threads > 256)
    throw std::invalid_argument("invalid borrowed speech state");
  impl_->owns_state = false;
  impl_->context = context; impl_->state = state;
  impl_->id = std::move(id); impl_->hash = std::move(hash); impl_->threads = threads;
}
SpeechModel::~SpeechModel() = default;
const std::string& SpeechModel::model_hash() const { return impl_->hash; }
const std::string& SpeechModel::model_id() const { return impl_->id; }
int SpeechModel::threads() const { return impl_->threads; }
bool SpeechModel::multilingual() const { return whisper_is_multilingual(impl_->context); }
std::string SpeechModel::library_version() { return std::string("whisper.cpp/") + whisper_version(); }
void SpeechModel::validate_options(const mediapb::SpeechOptions& options) const {
  if (options.task() == mediapb::SPEECH_TASK_TRANSLATE_TO_ENGLISH && !whisper_is_multilingual(impl_->context))
    throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "translation requires multilingual Whisper weights");
  if (options.model_id() != impl_->id || options.language().find('\0') != std::string::npos ||
      (!options.language().empty() && whisper_lang_id(options.language().c_str()) < 0) ||
      (options.task() != mediapb::SPEECH_TASK_UNSPECIFIED && options.task() != mediapb::SPEECH_TASK_TRANSCRIBE &&
       options.task() != mediapb::SPEECH_TASK_TRANSLATE_TO_ENGLISH) ||
      (!whisper_is_multilingual(impl_->context) && !options.language().empty() && options.language() != "en"))
    throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "invalid speech model options");
}
std::vector<mediapb::TranscriptSegment> SpeechModel::infer(const SpeechWindow& window, uint32_t track,
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
  auto params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
  params.n_threads = impl_->threads;
  params.print_progress = params.print_realtime = params.print_timestamps = params.print_special = false;
  params.no_context = true;
  // English-only vocabularies do not contain language-selection tokens.
  // Running multilingual detection on them can return unrelated language IDs.
  params.language = !whisper_is_multilingual(impl_->context) ? "en" :
      (options.language().empty() ? "auto" : options.language().c_str());
  params.translate = options.task() == mediapb::SPEECH_TASK_TRANSLATE_TO_ENGLISH;
  params.token_timestamps = options.word_timestamps();
  params.abort_callback = [](void* opaque) noexcept {
    auto& a = *static_cast<Abort*>(opaque);
    try { return a.cancelled(); } catch (...) { a.error = std::current_exception(); return true; }
  };
  params.abort_callback_user_data = &abort;
  const int result = whisper_full_with_state(impl_->context, impl_->state, params,
                                              window.samples.data(), static_cast<int>(window.samples.size()));
  if (abort.error) std::rethrow_exception(abort.error);
  if (cancelled()) throw Failure(grpc::StatusCode::CANCELLED, "speech cancelled");
  if (result) throw Failure(grpc::StatusCode::INTERNAL, "native Whisper inference failed");
  std::vector<mediapb::TranscriptSegment> output;
  const int64_t base = static_cast<int64_t>((window.first_sample / 16000) * 1000000 +
                                          (window.first_sample % 16000) * 1000000 / 16000);
  const char* language = whisper_lang_str(whisper_full_lang_id_from_state(impl_->state));
  for (int i = 0; i < whisper_full_n_segments_from_state(impl_->state); ++i) {
    auto& segment = output.emplace_back();
    segment.set_segment_id(uuid());
    segment.set_revision(1);
    // Reconciliation owns finalization across windows, not this model adapter.
    segment.set_final(false);
    segment.set_track_id(track);
    segment.set_start_us(base + whisper_full_get_segment_t0_from_state(impl_->state, i) * 10000);
    segment.set_end_us(base + whisper_full_get_segment_t1_from_state(impl_->state, i) * 10000);
    segment.set_text(whisper_full_get_segment_text_from_state(impl_->state, i));
    segment.set_source_language(language ? language : "");
    segment.set_output_language(params.translate ? "en" : segment.source_language());
    segment.set_task(params.translate ? mediapb::SPEECH_TASK_TRANSLATE_TO_ENGLISH : mediapb::SPEECH_TASK_TRANSCRIBE);
    double sum = 0;
    const int count = whisper_full_n_tokens_from_state(impl_->state, i);
    for (int t = 0; t < count; ++t) {
      const auto native = whisper_full_get_token_data_from_state(impl_->state, i, t);
      auto* token = segment.add_tokens();
      token->set_token_id(native.id);
      token->set_text(whisper_full_get_token_text_from_state(impl_->context, impl_->state, i, t));
      token->set_probability(native.p);
      token->set_kind(native.id < whisper_token_eot(impl_->context) ? mediapb::SPEECH_TOKEN_KIND_TEXT : mediapb::SPEECH_TOKEN_KIND_CONTROL);
      auto* details = token->mutable_whisper();
      details->set_timestamp_token_id(native.tid);
      details->set_log_probability(native.plog);
      details->set_timestamp_probability(native.pt);
      details->set_timestamp_probability_sum(native.ptsum);
      if (options.word_timestamps()) {
        details->set_start_centiseconds(native.t0); details->set_end_centiseconds(native.t1);
        details->set_voice_length(native.vlen);
      }
      if (options.word_timestamps() && native.t0 >= 0 && native.t1 >= native.t0) {
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
