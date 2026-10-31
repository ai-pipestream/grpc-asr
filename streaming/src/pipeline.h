#pragma once
#include "processing.h"
#include <future>
#include <grpcpp/channel.h>
namespace media {
class SpeechPipelineFactory final : public ProcessingFactory {
 public:
  static constexpr uint32_t minimum_advance_ms = 100;
  std::optional<uint32_t> minimum_window_advance_ms() const override { return minimum_advance_ms; }
  SpeechPipelineFactory(std::string target, std::string vad_weights = {}, std::string machine_id = {});
  SpeechPipelineFactory(std::shared_future<std::shared_ptr<grpc::Channel>>, std::string vad_weights, std::string machine_id);
  void resolve(mediapb::OpenStream&) const override;
  std::unique_ptr<Processing> create(const std::string&, const mediapb::OpenStream&, grpc::ServerContext&, EmitProcessed) const override;
 private:
  std::shared_future<std::shared_ptr<grpc::Channel>> channel_;
  std::string vad_weights_, machine_id_;
};
}  // namespace media
