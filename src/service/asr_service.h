#pragma once

#include <atomic>

#include "ai/pipestream/asr/v1/asr_service.grpc.pb.h"
#include "config.h"
#include "engine/model_pool.h"

namespace asr {

// Synchronous gRPC service. Each Transcribe stream runs on its own RPC
// thread; heavy work is bounded by the model pool's per-model state
// free-list, acquired only after the request has fully validated.
class AsrServiceImpl final : public ai::pipestream::asr::v1::AsrService::Service {
  public:
    AsrServiceImpl(const Config& config, engine::ModelPool& pool);

    grpc::Status Transcribe(
        grpc::ServerContext* context,
        grpc::ServerReaderWriter<ai::pipestream::asr::v1::TranscribeResponse,
                                 ai::pipestream::asr::v1::TranscribeRequest>* stream) override;

    grpc::Status GetServiceInfo(
        grpc::ServerContext* context,
        const ai::pipestream::asr::v1::GetServiceInfoRequest* request,
        ai::pipestream::asr::v1::GetServiceInfoResponse* response) override;

    // Metrics counters, printed by the interval line in main. transcribed
    // counts OK streams; rejected counts client-caused failures; failed
    // counts server-caused ones. audio_ms accumulates transcribed media
    // time.
    std::atomic<long> transcribed{0};
    std::atomic<long> rejected{0};
    std::atomic<long> failed{0};
    std::atomic<long> audio_ms{0};

  private:
    const Config& config_;
    engine::ModelPool& pool_;
};

}  // namespace asr
