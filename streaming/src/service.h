#pragma once
#include "registry.h"
#include "processing.h"
#include "http.h"

namespace media {
class Service final : public mediapb::MediaService::Service {
 public:
  explicit Service(Registry& registry, const ProcessingFactory* processing = nullptr, HttpPolicy http = {})
      : registry_(registry), processing_(processing), http_(std::move(http)) {}
  grpc::Status GetServiceInfo(grpc::ServerContext*, const mediapb::GetServiceInfoRequest*, mediapb::GetServiceInfoResponse*) override;
  grpc::Status Ingest(grpc::ServerContext*, grpc::ServerReaderWriter<mediapb::IngestResponse, mediapb::IngestRequest>*) override;
  grpc::Status Subscribe(grpc::ServerContext*, const mediapb::SubscribeRequest*, grpc::ServerWriter<mediapb::SubscribeResponse>*) override;
  grpc::Status Pull(grpc::ServerContext*, const mediapb::PullRequest*, grpc::ServerWriter<mediapb::PullResponse>*) override;
 private:
  Registry& registry_;
  const ProcessingFactory* processing_;
  HttpPolicy http_;
  grpc::Status process(grpc::ServerContext*, const std::function<bool(mediapb::IngestRequest*)>&,
                       const std::function<bool(const mediapb::IngestResponse&)>&, bool http = false);
};
}  // namespace media
