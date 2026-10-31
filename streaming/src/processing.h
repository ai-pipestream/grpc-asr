#pragma once
#include "registry.h"
#include <functional>
#include <optional>
namespace media {
using EmitProcessed = std::function<void(mediapb::StreamEvent)>;
class Processing {
 public:
  virtual ~Processing() = default;
  virtual void audio(const mediapb::AudioChunk&) = 0;
  virtual void finish() = 0;
};
class ProcessingFactory {
 public:
  virtual ~ProcessingFactory() = default;
  virtual std::optional<uint32_t> minimum_window_advance_ms() const { return std::nullopt; }
  virtual void resolve(mediapb::OpenStream&) const = 0;
  virtual std::unique_ptr<Processing> create(const std::string& id, const mediapb::OpenStream&,
                                            grpc::ServerContext&, EmitProcessed) const = 0;
};
}  // namespace media
