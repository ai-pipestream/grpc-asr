#include "pipeline.h"
#include "reconcile.h"
#include "vad.h"
#include "windowing.h"
#include "speech.h"
#include <grpcpp/grpcpp.h>
#include <bit>
#include <map>
#include <optional>
#include <tuple>

namespace media {
namespace {
void require(bool ok, const char* why) { if (!ok) throw Failure(grpc::StatusCode::INVALID_ARGUMENT, why); }
int64_t sample_us(uint64_t sample) {
  require(sample <= UINT64_C(100000000000000), "speech timeline exceeds limit");
  return static_cast<int64_t>((sample / 16000) * 1000000 + (sample % 16000) * 1000000 / 16000);
}
WindowPolicy policy(const mediapb::SpeechOptions& o) {
  const auto& w = o.windowing(); const auto& b = w.speech_boundary();
  return {w.window_ms() * 16, (w.strategy() == mediapb::CHUNKING_STRATEGY_SPEECH_AWARE ? b.max_window_ms() : w.window_ms()) * 16,
          w.overlap_ms() * 16, b.min_pause_ms() * 16, b.speech_padding_ms() * 16,
          w.strategy() == mediapb::CHUNKING_STRATEGY_SPEECH_AWARE, b.no_pause_policy() == mediapb::NO_PAUSE_POLICY_FAIL};
}
class Track {
 public:
  Track(const std::shared_ptr<grpc::Channel>& channel, const std::string& id, uint32_t track, mediapb::SpeechOptions options,
        const std::string& vad_weights, const std::string& machine, grpc::ServerContext& parent, EmitProcessed emit)
      : emit_(std::move(emit)), context_(grpc::ClientContext::FromServerContext(parent)) {
    stub_ = mediapb::SpeechWorkerService::NewStub(channel);
    wire_ = stub_->Transcribe(context_.get());
    try {
    mediapb::TranscribeRequest request;
    request.mutable_open()->set_stream_id(id); request.mutable_open()->set_track_id(track);
    *request.mutable_open()->mutable_options() = options;
    request.mutable_open()->mutable_options()->clear_windowing();
    mediapb::TranscribeResponse response;
    if (!wire_->Write(request) || !wire_->Read(&response)) failed();
    require(response.has_opened() && response.opened().accepted().stream_id() == id &&
            response.opened().accepted().track_id() == track &&
            response.opened().accepted().options().SerializeAsString() == request.open().options().SerializeAsString() &&
            response.opened().producer().model_id() == options.model_id() &&
            response.opened().producer().model_sha256().size() == 32 &&
            response.opened().producer().configuration_sha256().size() == 32 &&
            valid_uuid(response.opened().producer().machine_id()), "invalid worker handshake");
    track_ = track;
    overlap_ = policy(options).overlap_samples;
    if (options.reconcile_transcript()) {
      reconciler_ = std::make_unique<Reconciler>([this](mediapb::TranscriptSegment segment) {
        mediapb::StreamEvent event;
        *event.mutable_producer() = selector_;
        event.mutable_producer()->set_service("grpc-asr-reconciliation");
        event.mutable_producer()->clear_model_id(); event.mutable_producer()->clear_model_sha256();
        event.mutable_producer()->set_configuration_sha256(sha256("reconciliation-v1:timing-tolerance-us=350000"));
        *segment.mutable_producer() = event.producer();
        *segment.add_contributors() = producer_; *segment.add_contributors() = selector_;
        *event.mutable_transcript() = std::move(segment);
        emit_(std::move(event));
      });
    }
    producer_ = response.opened().producer();
    selector_.set_machine_id(machine); selector_.set_service("grpc-asr-windowing");
    selector_.set_app_version(MEDIA_APP_VERSION); selector_.set_build_id(MEDIA_BUILD_ID);
    selector_.set_execution_backend("cpu");
    selector_.set_configuration_sha256(sha256(options.windowing().SerializeAsString() + ":vad_threshold=0.5;vad_frame_samples=512"));
    windows_ = std::make_unique<Windowing>(policy(options), [&](SpeechWindow window) { infer(std::move(window)); });
    if (options.windowing().strategy() == mediapb::CHUNKING_STRATEGY_SPEECH_AWARE) {
      vad_ = std::make_unique<VoiceActivity>(vad_weights, [&](auto samples, float probability) {
        windows_->append(samples, probability >= 0.5f);
      });
      selector_.set_model_id("silero-v5.1.2"); selector_.set_model_sha256(vad_->model_hash());
      selector_.set_library_version(SpeechModel::library_version());
    }
    } catch (...) { context_->TryCancel(); if (!done_) wire_->Finish(); done_ = true; throw; }
  }
  ~Track() { if (!done_ && context_) { context_->TryCancel(); if (wire_) wire_->Finish(); } }
  void audio(const mediapb::AudioChunk& audio) {
    require(audio.sample_rate() == 16000 && audio.channels() == 1 && audio.pcm_f32le().size() % 4 == 0,
            "pipeline requires mono 16 kHz float PCM");
    require(audio.first_sample() == samples_ && audio.chunk_number() == chunks_ + 1, "noncontiguous decoded speech audio");
    try {
      if (audio.discontinuity()) {
        if (vad_) vad_->discontinuity();
        windows_->discontinuity();
        time_offset_.reset();
      }
      if (!time_offset_ && audio.has_presentation_time_us()) {
        require(audio.presentation_time_us() >= -(INT64_C(1) << 62) && audio.presentation_time_us() <= (INT64_C(1) << 62),
                "source presentation timestamp exceeds limit");
        time_offset_ = audio.presentation_time_us() - sample_us(audio.first_sample());
      }
      std::vector<float> samples;
      const auto& data = audio.pcm_f32le(); samples.reserve(data.size() / 4);
      for (size_t i = 0; i < data.size(); i += 4) {
        uint32_t bits = 0;
        for (int b = 0; b < 4; ++b) bits |= static_cast<uint32_t>(static_cast<unsigned char>(data[i+b])) << (b*8);
        samples.push_back(std::bit_cast<float>(bits));
      }
      if (vad_) vad_->append(samples); else windows_->append(samples, true);
      samples_ += samples.size(); ++chunks_;
    } catch (const std::length_error&) { throw Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "no pause before speech window limit"); }
      catch (const std::invalid_argument&) { throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "invalid decoded speech samples"); }
  }
  void finish() {
    try {
      if (vad_) vad_->finish();
      windows_->finish();
    } catch (const std::length_error&) {
      throw Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "no pause before speech window limit");
    }
    mediapb::TranscribeRequest request; request.mutable_finish()->set_window_count(count_);
    mediapb::TranscribeResponse response;
    if (!wire_->Write(request)) failed();
    wire_->WritesDone();
    if (!wire_->Read(&response)) failed();
    require(response.has_closed() && response.closed().window_count() == count_, "worker close accounting mismatch");
    require(!wire_->Read(&response), "unexpected worker output after close");
    const auto status = wire_->Finish(); done_ = true;
    if (!status.ok()) throw Failure(status.error_code(), "speech worker failed at close");
    if (reconciler_) reconciler_->finish();
  }
 private:
  [[noreturn]] void failed() {
    const auto status = wire_->Finish(); done_ = true;
    throw Failure(status.ok() ? grpc::StatusCode::DATA_LOSS : status.error_code(), "speech worker stream failed");
  }
  void infer(SpeechWindow window) {
    mediapb::TranscribeRequest request;
    auto* input = request.mutable_window();
    input->set_window_number(window.number); input->set_first_sample(window.first_sample);
    input->set_first_new_sample(window.first_new_sample);
    input->set_boundary(static_cast<mediapb::InferenceBoundary>(static_cast<int>(window.boundary) + 1));
    for (float sample : window.samples) {
      const uint32_t bits = std::bit_cast<uint32_t>(sample);
      for (int b = 0; b < 4; ++b) input->mutable_pcm_f32le()->push_back(static_cast<char>(bits >> (b*8)));
    }
    mediapb::TranscribeResponse response;
    if (!wire_->Write(request) || !wire_->Read(&response)) failed();
    require(response.has_hypothesis() && response.hypothesis().window_number() == window.number &&
            response.hypothesis().track_id() == track_ &&
            response.hypothesis().first_sample() == window.first_sample &&
            response.hypothesis().first_new_sample() == window.first_new_sample &&
            response.hypothesis().boundary() == input->boundary() &&
            response.hypothesis().producer().SerializeAsString() == producer_.SerializeAsString() &&
            response.hypothesis().sample_count() == window.samples.size(), "worker hypothesis identity mismatch");
    mediapb::StreamEvent event;
    *event.mutable_producer() = producer_;
    *event.mutable_hypothesis() = response.hypothesis();
    if (time_offset_) {
      event.mutable_hypothesis()->set_presentation_time_us(sample_us(window.first_sample) + *time_offset_);
      for (auto& segment : *event.mutable_hypothesis()->mutable_segments()) {
        segment.set_start_us(segment.start_us() + *time_offset_); segment.set_end_us(segment.end_us() + *time_offset_);
        for (auto& token : *segment.mutable_tokens()) {
          if (token.has_start_us()) token.set_start_us(token.start_us() + *time_offset_);
          if (token.has_end_us()) token.set_end_us(token.end_us() + *time_offset_);
        }
      }
    }
    *event.mutable_hypothesis()->add_contributors() = selector_;
    // Preserve the raw hypothesis before publishing any derived revisions.
    const auto hypothesis = event.hypothesis();
    emit_(std::move(event));
    if (reconciler_) reconciler_->accept(hypothesis,
        window.boundary == WindowBoundary::Fixed || window.boundary == WindowBoundary::Forced ? overlap_ : 0);
    ++count_;
  }
  EmitProcessed emit_;
  std::unique_ptr<grpc::ClientContext> context_;
  std::unique_ptr<mediapb::SpeechWorkerService::Stub> stub_;
  std::unique_ptr<grpc::ClientReaderWriter<mediapb::TranscribeRequest, mediapb::TranscribeResponse>> wire_;
  mediapb::Producer producer_, selector_;
  std::unique_ptr<Windowing> windows_;
  std::unique_ptr<VoiceActivity> vad_;
  std::unique_ptr<Reconciler> reconciler_;
  uint32_t overlap_ = 0;
  uint64_t count_ = 0, samples_ = 0, chunks_ = 0;
  bool done_ = false;
  uint32_t track_ = 0;
  std::optional<int64_t> time_offset_;
};
class Pipeline final : public Processing {
 public:
  Pipeline(std::shared_ptr<grpc::Channel> channel, std::string id, mediapb::OpenStream options, std::string vad,
           std::string machine, grpc::ServerContext& parent, EmitProcessed emit)
      : channel_(std::move(channel)), id_(std::move(id)), vad_(std::move(vad)), machine_(std::move(machine)),
        options_(std::move(options)), parent_(parent), emit_(std::move(emit)) {}
  void audio(const mediapb::AudioChunk& audio) override {
    for (int i = 0; i < options_.speech_size(); ++i) {
      const auto key = std::make_pair(audio.track_id(), i);
      if (!tracks_.contains(key)) {
        if (tracks_.size() >= 32) throw Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "speech track/task limit reached");
        tracks_[key] = std::make_unique<Track>(channel_, id_, audio.track_id(), options_.speech(i), vad_, machine_, parent_, emit_);
      }
      tracks_.at(key)->audio(audio);
    }
  }
  void finish() override {
    if (tracks_.empty()) throw Failure(grpc::StatusCode::FAILED_PRECONDITION, "requested speech but no audio was decoded");
    for (auto& [key, track] : tracks_) track->finish();
  }
 private:
  std::shared_ptr<grpc::Channel> channel_;
  std::string id_, vad_, machine_;
  mediapb::OpenStream options_;
  grpc::ServerContext& parent_;
  EmitProcessed emit_;
  std::map<std::pair<uint32_t, int>, std::unique_ptr<Track>> tracks_;
};
}
SpeechPipelineFactory::SpeechPipelineFactory(std::string target, std::string vad, std::string machine)
    : vad_weights_(std::move(vad)), machine_id_(machine.empty() ? uuid() : std::move(machine)) {
  require(!target.empty() && valid_uuid(machine_id_), "invalid speech routing configuration");
  std::promise<std::shared_ptr<grpc::Channel>> ready;
  ready.set_value(grpc::CreateChannel(target, grpc::InsecureChannelCredentials()));
  channel_ = ready.get_future().share();
  if (!vad_weights_.empty()) { VoiceActivity validate(vad_weights_, [](auto, auto) {}); }
}
SpeechPipelineFactory::SpeechPipelineFactory(std::shared_future<std::shared_ptr<grpc::Channel>> channel,
    std::string vad, std::string machine)
    : channel_(std::move(channel)), vad_weights_(std::move(vad)), machine_id_(std::move(machine)) {
  require(channel_.valid() && valid_uuid(machine_id_), "invalid in-process speech routing configuration");
  if (!vad_weights_.empty()) { VoiceActivity validate(vad_weights_, [](auto, auto) {}); }
}

void SpeechPipelineFactory::resolve(mediapb::OpenStream& request) const {
  require(request.speech_size() > 0 && request.speech_size() <= 4, "one to four speech tasks are supported");
  for (auto& options : *request.mutable_speech()) {
    require(!options.model_id().empty() && options.model_id().size() <= 256, "speech model ID required");
    if (!options.has_reconcile_transcript()) options.set_reconcile_transcript(true);
    if (options.reconcile_transcript()) options.set_word_timestamps(true);
    auto* w = options.mutable_windowing();
    if (!w->has_window_ms()) w->set_window_ms(10000);
    if (!w->has_overlap_ms()) w->set_overlap_ms(1000);
    if (w->strategy() == mediapb::CHUNKING_STRATEGY_UNSPECIFIED) w->set_strategy(mediapb::CHUNKING_STRATEGY_FIXED);
    if (options.task() == mediapb::SPEECH_TASK_UNSPECIFIED) options.set_task(mediapb::SPEECH_TASK_TRANSCRIBE);
    require(w->window_ms() > 0 && w->window_ms() <= 30000 && w->overlap_ms() < w->window_ms(), "invalid speech window durations");
    require(w->window_ms()-w->overlap_ms() >= minimum_advance_ms,
            "configured speech windows must advance by at least 100 ms");
    if (w->strategy() == mediapb::CHUNKING_STRATEGY_SPEECH_AWARE) {
      require(w->has_speech_boundary() && w->speech_boundary().vad_model_id() == "silero-v5.1.2", "qualified VAD model ID required");
      if (vad_weights_.empty()) throw Failure(grpc::StatusCode::UNIMPLEMENTED, "VAD weights are not configured");
      const auto& b = w->speech_boundary();
      require(b.max_window_ms() >= w->window_ms() && b.max_window_ms() <= 30000 &&
              b.min_pause_ms() <= 30000 && b.speech_padding_ms() <= 15000 &&
              (b.no_pause_policy() == mediapb::NO_PAUSE_POLICY_SPLIT_WITH_OVERLAP || b.no_pause_policy() == mediapb::NO_PAUSE_POLICY_FAIL),
              "invalid speech pause policy");
    } else require(w->strategy() == mediapb::CHUNKING_STRATEGY_FIXED && !w->has_speech_boundary(), "invalid chunking strategy");
    try { Windowing validate(policy(options), [](auto) {}); }
    catch (const std::invalid_argument& e) { throw Failure(grpc::StatusCode::INVALID_ARGUMENT, e.what()); }
  }
}
std::unique_ptr<Processing> SpeechPipelineFactory::create(const std::string& id, const mediapb::OpenStream& request,
    grpc::ServerContext& parent, EmitProcessed emit) const {
  return std::make_unique<Pipeline>(channel_.get(), id, request, vad_weights_, machine_id_, parent, std::move(emit));
}
}  // namespace media
