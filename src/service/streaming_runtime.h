#pragma once
#include "engine/stream_backend.h"
#include "pipeline.h"
#include "service.h"
#include "worker.h"
namespace asr {
// Counts what the streaming handlers report as unexpected failures in this
// process's metrics, as grpc_asr_streaming_unexpected_failures_total{handler}.
// The runtime installs it; it is a function of its own so a test can too.
void count_streaming_failures_in_metrics();
// Lifetime owner for the additional typed streaming interfaces.
class StreamingRuntime {
 public:
  StreamingRuntime(engine::ModelPool&, const Config&);
  void connect(std::shared_ptr<grpc::Channel> channel);
  void register_services(grpc::ServerBuilder&);
 private:
  ::media::Registry registry_;
  std::promise<std::shared_ptr<grpc::Channel>> channel_;
  std::unique_ptr<::media::SpeechBackend> backend_;
  std::unique_ptr<::media::SpeechWorker> worker_;
  std::unique_ptr<::media::SpeechPipelineFactory> pipeline_;
  std::unique_ptr<::media::Service> service_;
};
}
