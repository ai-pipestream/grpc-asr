#include "engine/stream_backend.h"
#include "worker.h"
#include "fixture.h"
#include <chrono>
#include <filesystem>
#include <print>
#include <grpcpp/grpcpp.h>
int main() {
  const char* path = env_or_null("GRPC_ASR_TEST_MODEL");
  if (!path || !std::filesystem::exists(path)) return 77;
  asr::Config config;
  config.backend = "cpu"; config.concurrency = 1;
  config.models_dir = std::filesystem::path(path).parent_path(); config.models = {"tiny.en"};
  asr::engine::ModelPool pool(config);
  asr::engine::StreamBackend backend(pool, config, "tiny.en");
  ::media::mediapb::SpeechOptions options;
  options.set_model_id("tiny.en"); options.set_language("en");
  backend.validate_options(options);
  if (backend.model_hash().size() != 32 || backend.model_hash() != pool.model_hash("tiny.en")) return 1;
  ::media::SpeechWindow window{1, 0, 0, ::media::WindowBoundary::Fixed, std::vector<float>(16000, 0)};
  {
    auto occupied = pool.acquire("tiny.en");
    auto start = std::chrono::steady_clock::now();
    bool cancelled = false;
    try { backend.infer(window, 0, options, [&] {
      return std::chrono::steady_clock::now() - start > std::chrono::milliseconds(40);
    }); } catch (const ::media::Failure& e) { cancelled = e.code == grpc::StatusCode::CANCELLED; }
    if (!cancelled || std::chrono::steady_clock::now() - start > std::chrono::seconds(1)) return 2;
  }
  backend.infer(window, 0, options, [] { return false; });
  // Borrowed adapter destruction must neither free nor retain a pooled state.
  auto lease = pool.acquire("tiny.en", {}, std::chrono::steady_clock::now() + std::chrono::seconds(1));
  ::media::SpeechWorker worker(backend);
  ::media::mediapb::GetModelInfoRequest request;
  ::media::mediapb::GetModelInfoResponse response;
  grpc::ServerContext context;
  if (!worker.GetModelInfo(&context, &request, &response).ok() ||
      response.producer().execution_backend() != "cpu" ||
      response.producer().model_sha256() != backend.model_hash()) return 3;
  std::println("shared pool ownership, cancellation, inference and provenance passed");
}
