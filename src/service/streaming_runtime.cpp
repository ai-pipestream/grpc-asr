#include "service/streaming_runtime.h"
#include <grpcpp/server_builder.h>
#include "parakeet_model.h"
#include <cstdlib>
#include <fstream>
namespace asr {
namespace {
std::string env(const char* name, std::string fallback = {}) {
  const char* value = std::getenv(name);
  return value && *value ? value : fallback;
}
std::string weights(const std::string& path, size_t cap) {
  if (path.empty()) return {};
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  const auto length = file.tellg();
  if (!file || length <= 0 || static_cast<uint64_t>(length) > cap)
    throw std::runtime_error("invalid streaming model file: " + path);
  std::string data(static_cast<size_t>(length), '\0');
  file.seekg(0);
  if (!file.read(data.data(), length)) throw std::runtime_error("cannot read streaming model file");
  return data;
}
}
StreamingRuntime::StreamingRuntime(engine::ModelPool& pool, const Config& config)
    : registry_({}, env("GRPC_ASR_MACHINE_ID")) {
  const auto kind = env("GRPC_ASR_STREAM_BACKEND", "whisper");
  if (kind == "whisper") {
    const auto id = env("GRPC_ASR_STREAM_MODEL", pool.model_names().front());
    if (!pool.has_model(id)) throw std::invalid_argument("GRPC_ASR_STREAM_MODEL is not loaded in the pool");
    backend_ = std::make_unique<engine::StreamBackend>(pool, config, id);
  } else if (kind == "parakeet") {
    auto data = weights(env("GRPC_ASR_PARAKEET_MODEL_PATH"), UINT64_C(8589934592));
    backend_ = std::make_unique<::media::ParakeetModel>(std::span<const char>(data),
        env("GRPC_ASR_STREAM_MODEL", "parakeet"), config.threads);
  } else throw std::invalid_argument("GRPC_ASR_STREAM_BACKEND must be whisper or parakeet");
  worker_ = std::make_unique<::media::SpeechWorker>(*backend_, registry_.machine_id());
  // The public worker and media APIs share an in-process gRPC channel.
  // It becomes ready immediately after the canonical server starts.
  auto vad = weights(env("GRPC_ASR_VAD_MODEL_PATH"), 4 * 1024 * 1024);
  const auto target = env("GRPC_ASR_WORKER_TARGET");
  if (target.empty())
    pipeline_ = std::make_unique<::media::SpeechPipelineFactory>(channel_.get_future().share(),
        std::move(vad), registry_.machine_id());
  else
    pipeline_ = std::make_unique<::media::SpeechPipelineFactory>(target, std::move(vad), registry_.machine_id());
  ::media::HttpPolicy http;
  const auto origins = env("GRPC_ASR_HTTP_ORIGINS");
  for (size_t start = 0; start < origins.size();) {
    const auto end = origins.find(',', start);
    const auto origin = origins.substr(start, end == std::string::npos ? end : end - start);
    if (origin != ::media::HttpInput::origin(origin))
      throw std::invalid_argument("GRPC_ASR_HTTP_ORIGINS requires canonical scheme://host:port origins");
    http.allowed_origins.insert(origin);
    if (end == std::string::npos) break;
    start = end + 1;
    if (start == origins.size()) throw std::invalid_argument("empty HTTP origin");
  }
  const auto allow_private = env("GRPC_ASR_HTTP_ALLOW_PRIVATE", "0");
  if (allow_private != "0" && allow_private != "1")
    throw std::invalid_argument("GRPC_ASR_HTTP_ALLOW_PRIVATE must be 0 or 1");
  http.allow_private_addresses = allow_private == "1";
  service_ = std::make_unique<::media::Service>(registry_, pipeline_.get(), std::move(http));
}
void StreamingRuntime::connect(std::shared_ptr<grpc::Channel> channel) {
  channel_.set_value(std::move(channel));
}
void StreamingRuntime::register_services(grpc::ServerBuilder& builder) {
  builder.RegisterService(worker_.get());
  builder.RegisterService(service_.get());
}
}
