#pragma once
#include "speech.h"
#include <atomic>
#include <chrono>
namespace media {
class SpeechWorker final : public mediapb::SpeechWorkerService::Service {
 public:
  SpeechWorker(SpeechBackend& model, std::string machine_id = {},
               std::chrono::milliseconds input_idle_timeout = std::chrono::seconds(120));
  grpc::Status GetModelInfo(grpc::ServerContext*, const mediapb::GetModelInfoRequest*, mediapb::GetModelInfoResponse*) override;
  grpc::Status Transcribe(grpc::ServerContext*,
      grpc::ServerReaderWriter<mediapb::TranscribeResponse, mediapb::TranscribeRequest>*) override;
 private:
  static constexpr unsigned max_streams = 32;
  static constexpr unsigned max_window_samples = 480000;
  SpeechBackend& model_;
  std::chrono::milliseconds input_idle_timeout_;
  mediapb::Producer producer_;
  std::atomic_flag busy_ = ATOMIC_FLAG_INIT;
  std::atomic<unsigned> streams_{0};
};
}  // namespace media
