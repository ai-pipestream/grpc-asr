// A handler that answers INTERNAL with a fixed message must say the cause
// somewhere. These drive the real handlers into an exception that is not a
// Failure and read what the server said about it.
#include "failure_report.h"
#include "service.h"
#include "worker.h"
#include <grpcpp/grpcpp.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace std::chrono_literals;
namespace mediapb = media::mediapb;
namespace {
void check(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

// A model whose inference throws what a real one throws when the device
// runs out of memory: a plain std::runtime_error, not a Failure.
class FailingBackend final : public media::SpeechBackend {
 public:
  bool typed = false;
  void validate_options(const mediapb::SpeechOptions&) const override {}
  std::vector<mediapb::TranscriptSegment> infer(const media::SpeechWindow&, uint32_t,
      const mediapb::SpeechOptions&, const std::function<bool()>&) override {
    if (typed) throw media::Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "typed refusal");
    throw std::runtime_error("ggml_gallocr_alloc_graph: failed to allocate\ncuda buffer of 512 MiB");
  }
  const std::string& model_hash() const override { return hash_; }
  const std::string& model_id() const override { return id_; }
  int threads() const override { return 1; }
  std::optional<bool> multilingual_capability() const override { return false; }
  bool supports_translation() const override { return false; }
  bool supports_language_hint() const override { return true; }
  bool reports_language() const override { return false; }
  mediapb::SpeechBackendKind backend_kind() const override {
    return mediapb::SPEECH_BACKEND_KIND_WHISPER;
  }
  std::string engine_version() const override { return "test-engine/1"; }
 private:
  std::string hash_ = std::string(32, 'a');
  std::string id_ = "failing";
};

// A processing stage whose construction throws after the stream has opened,
// so the handler has a stream id to name.
class FailingFactory final : public media::ProcessingFactory {
 public:
  void resolve(mediapb::OpenStream&) const override {}
  std::unique_ptr<media::Processing> create(const std::string&, const mediapb::OpenStream&,
      grpc::ServerContext&, media::EmitProcessed) const override {
    throw std::runtime_error("speech pipeline could not start: worker channel refused");
  }
};

struct Server {
  std::unique_ptr<grpc::Server> instance;
  int port = 0;
  explicit Server(grpc::Service& service) {
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    instance = builder.BuildAndStart();
    check(instance != nullptr, "test server bind failed");
  }
  ~Server() { instance->Shutdown(std::chrono::system_clock::now() + 1s); instance->Wait(); }
  std::shared_ptr<grpc::Channel> channel() const {
    return grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials());
  }
};

// Runs `body` with stderr redirected into a pipe and returns what was
// written to it.
template <typename Body>
std::string stderr_of(Body body) {
  std::fflush(stderr);
  int fds[2];
  check(::pipe(fds) == 0, "pipe failed");
  const int saved = ::dup(STDERR_FILENO);
  check(saved >= 0 && ::dup2(fds[1], STDERR_FILENO) >= 0, "stderr redirect failed");
  ::close(fds[1]);
  try { body(); } catch (...) {
    std::fflush(stderr); ::dup2(saved, STDERR_FILENO); ::close(saved); ::close(fds[0]);
    throw;
  }
  std::fflush(stderr);
  ::dup2(saved, STDERR_FILENO);
  ::close(saved);
  std::string text;
  char buffer[4096];
  for (ssize_t n; (n = ::read(fds[0], buffer, sizeof buffer)) > 0;) text.append(buffer, static_cast<size_t>(n));
  ::close(fds[0]);
  return text;
}

std::atomic<uint64_t> observed_worker{0};
void observe(media::FailureSite site) noexcept {
  if (site == media::FailureSite::SpeechWorker) observed_worker.fetch_add(1);
}

// One worker call: open a track, send one window of silence.
grpc::Status worker_call(mediapb::SpeechWorkerService::Stub& stub, const std::string& stream_id) {
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + 10s);
  auto rpc = stub.Transcribe(&context);
  mediapb::TranscribeRequest request;
  request.mutable_open()->set_stream_id(stream_id);
  request.mutable_open()->mutable_options()->set_model_id("failing");
  check(rpc->Write(request), "worker open failed");
  mediapb::TranscribeResponse response;
  check(rpc->Read(&response) && response.has_opened(), "worker handshake missing");
  request.Clear();
  auto* window = request.mutable_window();
  window->set_window_number(1);
  window->set_boundary(mediapb::INFERENCE_BOUNDARY_FIXED);
  window->set_pcm_f32le(std::string(16000 * 4, '\0'));
  check(rpc->Write(request), "worker window failed");
  rpc->WritesDone();
  while (rpc->Read(&response)) {}
  return rpc->Finish();
}

void worker_says_why() {
  FailingBackend backend;
  media::SpeechWorker worker(backend);
  Server server(worker);
  auto stub = mediapb::SpeechWorkerService::NewStub(server.channel());
  const std::string stream_id = media::uuid();
  const uint64_t before = media::unexpected_failures(media::FailureSite::SpeechWorker);
  media::set_failure_observer(&observe);
  grpc::Status status;
  const std::string log = stderr_of([&] { status = worker_call(*stub, stream_id); });
  check(status.error_code() == grpc::StatusCode::INTERNAL && status.error_message() == "speech worker failed",
        "the client still gets the fixed message, got: " + status.error_message());
  check(log.find("speech_worker") != std::string::npos, "the log names the handler: " + log);
  check(log.find(stream_id) != std::string::npos, "the log names the stream: " + log);
  check(log.find("ggml_gallocr_alloc_graph: failed to allocate cuda buffer of 512 MiB") != std::string::npos,
        "the log carries the cause on one line: " + log);
  check(std::count(log.begin(), log.end(), '\n') == 1, "one line per failure: " + log);
  check(media::unexpected_failures(media::FailureSite::SpeechWorker) == before + 1,
        "the failure is counted for its handler");
  check(observed_worker.load() == 1, "the observer is told");

  // A Failure is an answer the handler chose, with its own code and
  // message for the client: nothing unexpected, nothing to report.
  backend.typed = true;
  const std::string quiet = stderr_of([&] { status = worker_call(*stub, media::uuid()); });
  check(status.error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED, "a typed failure keeps its code");
  check(quiet.empty(), "a typed failure is not logged as unexpected: " + quiet);
  check(media::unexpected_failures(media::FailureSite::SpeechWorker) == before + 1,
        "a typed failure is not counted as unexpected");
  media::set_failure_observer(nullptr);
}

void ingest_says_why() {
  FailingFactory factory;
  media::Registry registry;
  media::Service service(registry, &factory);
  Server server(service);
  auto stub = mediapb::MediaService::NewStub(server.channel());
  const uint64_t before = media::unexpected_failures(media::FailureSite::IngestProcessing);
  grpc::Status status;
  std::string stream_id;
  const std::string log = stderr_of([&] {
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 10s);
    auto rpc = stub->Ingest(&context);
    mediapb::IngestRequest request;
    request.mutable_open()->mutable_direct()->set_name("test-media");
    request.mutable_open()->mutable_direct()->set_media_type("application/octet-stream");
    request.mutable_open()->add_speech()->set_model_id("failing");
    check(rpc->Write(request), "ingest open failed");
    rpc->WritesDone();
    mediapb::IngestResponse response;
    while (rpc->Read(&response))
      if (response.event().has_opened()) stream_id = response.event().cursor().stream_id();
    status = rpc->Finish();
  });
  check(status.error_code() == grpc::StatusCode::INTERNAL && status.error_message() == "ingress processing failed",
        "the client still gets the fixed message, got: " + status.error_message());
  check(!stream_id.empty(), "the stream had opened");
  check(log.find("ingest_processing") != std::string::npos, "the log names the handler: " + log);
  check(log.find(stream_id) != std::string::npos, "the log names the stream: " + log);
  check(log.find("worker channel refused") != std::string::npos, "the log carries the cause: " + log);
  check(media::unexpected_failures(media::FailureSite::IngestProcessing) == before + 1,
        "the failure is counted for its handler");
}

void every_site_reports() {
  for (size_t index = 0; index < media::failure_site_names.size(); ++index) {
    const auto site = static_cast<media::FailureSite>(index);
    const uint64_t before = media::unexpected_failures(site);
    const std::string log = stderr_of([&] {
      media::report_unexpected_failure(site, {}, std::runtime_error("cause"));
    });
    check(log == "grpc-asr streaming: " + std::string(media::failure_site_names[index]) +
                     " failed with an unexpected error (stream -): cause\n",
          "a report without a stream says so: " + log);
    check(media::unexpected_failures(site) == before + 1, "each site has its own count");
  }
}
}  // namespace

int main() {
  try {
    worker_says_why();
    ingest_says_why();
    every_site_reports();
    std::cout << "unexpected handler failures are logged and counted\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
