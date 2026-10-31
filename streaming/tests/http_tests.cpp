#include "http.h"
#include "http_fixture.h"
#include "registry.h"
#include "service.h"
#include "media_fixture.h"
#include <grpcpp/grpcpp.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>

using namespace std::chrono_literals;
void check(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void failure(grpc::StatusCode code, const std::function<void()>& fn) {
  try { fn(); } catch (const media::Failure& error) {
    check(error.code == code, "wrong HTTP failure code");
    check(std::string(error.what()).find("secret") == std::string::npos, "HTTP error exposed URL secrets");
    return;
  }
  throw std::runtime_error("HTTP failure was silently accepted");
}
int main() {
  try {
    check(media::HttpInput::origin("https://EXAMPLE.com/path?q=value") == "https://example.com:443", "HTTP origin normalization failed");
    for (const auto& bad : {"file:///etc/passwd", "http://user:pass@example.com/", "http://@example.com/", "http://example.com/#"})
      failure(grpc::StatusCode::INVALID_ARGUMENT, [&] { media::HttpInput::origin(bad); });
    failure(grpc::StatusCode::PERMISSION_DENIED, [] { media::HttpInput denied("http://example.com/", {}, [] { return false; }); });
    {
      std::string data(512 * 1024, '\0');
      for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<char>(i % 251);
      Fixture server([&](int fd, Fixture& f) {
        send_bytes(fd, "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(data.size()) + "\r\n\r\n");
        send_bytes(fd, data.substr(0, 4096));
        while (!f.release && !f.stop) std::this_thread::sleep_for(1ms);
        if (!f.stop) send_bytes(fd, data.substr(4096));
      });
      media::HttpInput input(server.url, server.policy(), [] { return false; });
      char buffer[37]; std::string received;
      auto n = input.read(buffer, sizeof(buffer)); check(n > 0 && !server.release, "HTTP waited for complete body");
      received.append(buffer, n); server.release = true;
      while ((n = input.read(buffer, sizeof(buffer)))) {
        received.append(buffer, n); check(input.buffered_bytes() <= 65536, "HTTP buffering exceeded bound");
      }
      check(received == data && input.status_code() == 200, "HTTP byte preservation failed");
    }
    {
      Fixture server([](int fd, Fixture&) { send_bytes(fd, "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nabcd\r\n4\r\nefgh\r\n0\r\n\r\n"); });
      media::HttpInput input(server.url, server.policy(), [] { return false; });
      char buffer[32]; std::string result; size_t n;
      while ((n = input.read(buffer, sizeof(buffer)))) result.append(buffer, n);
      check(result == "abcdefgh", "HTTP transfer framing leaked into media");
    }
    for (const auto& response : {std::string("HTTP/1.1 404 Missing\r\nContent-Length: 4\r\n\r\noops"),
                                std::string("HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/\r\nContent-Length: 0\r\n\r\n")}) {
      Fixture server([&](int fd, Fixture&) { send_bytes(fd, response); });
      media::HttpInput input(server.url, server.policy(), [] { return false; });
      failure(grpc::StatusCode::FAILED_PRECONDITION, [&] { char out[32]; input.read(out, sizeof(out)); });
    }
    {
      Fixture server([](int fd, Fixture&) { send_bytes(fd, "HTTP/1.1 200 OK\r\nContent-Encoding: gzip\r\nContent-Length: 4\r\n\r\noops"); });
      media::HttpInput input(server.url, server.policy(), [] { return false; });
      failure(grpc::StatusCode::UNIMPLEMENTED, [&] { char out[32]; input.read(out, sizeof(out)); });
    }
    {
      Fixture server([](int fd, Fixture&) { send_bytes(fd, "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort"); });
      media::HttpInput input(server.url, server.policy(), [] { return false; });
      failure(grpc::StatusCode::UNAVAILABLE, [&] { char out[32]; while (input.read(out, sizeof(out))) {} });
    }
    {
      Fixture server([](int, Fixture&) {});
      media::HttpInput input(server.url, server.policy(false), [] { return false; });
      failure(grpc::StatusCode::PERMISSION_DENIED, [&] { char out[32]; input.read(out, sizeof(out)); });
    }
    {
      Fixture server([](int fd, Fixture& f) {
        send_bytes(fd, "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n");
        while (!f.stop) std::this_thread::sleep_for(1ms);
      });
      std::atomic<bool> cancel{false};
      media::HttpInput input(server.url, server.policy(), [&] { return cancel.load(); });
      std::jthread canceller([&] { std::this_thread::sleep_for(50ms); cancel = true; });
      const auto start = std::chrono::steady_clock::now();
      failure(grpc::StatusCode::CANCELLED, [&] { char out[32]; input.read(out, sizeof(out)); });
      check(std::chrono::steady_clock::now() - start < 2s, "HTTP cancellation was not responsive");
    }
    {
      const auto data = wav_fixture(16000, 1, 6);
      Fixture http([&](int fd, Fixture& f) {
        send_bytes(fd, "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(data.size()) + "\r\n\r\n");
        send_bytes(fd, data.substr(0, data.size() / 2));
        while (!f.release && !f.stop) std::this_thread::sleep_for(1ms);
        if (!f.stop) send_bytes(fd, data.substr(data.size() / 2));
      });
      media::Registry registry;
      Fixture missing([](int fd, Fixture&) { send_bytes(fd, "HTTP/1.1 404 Missing\r\nContent-Length: 4\r\n\r\noops"); });
      auto policy = http.policy();
      policy.allowed_origins.insert(media::HttpInput::origin(missing.url));
      media::Service service(registry, nullptr, policy);
      grpc::ServerBuilder builder; int port = 0;
      builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
      builder.RegisterService(&service);
      struct Guard {
        std::unique_ptr<grpc::Server> server;
        ~Guard() { if (server) { server->Shutdown(); server->Wait(); } }
      } guard{builder.BuildAndStart()};
      check(guard.server != nullptr, "HTTP gRPC server failed to bind");
      auto client = media::mediapb::MediaService::NewStub(grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
      grpc::ClientContext info_context; info_context.set_deadline(std::chrono::system_clock::now() + 5s);
      media::mediapb::GetServiceInfoResponse discovery;
      check(client->GetServiceInfo(&info_context, {}, &discovery).ok() && discovery.http_pull_configured() &&
            !discovery.speech_processing_configured(), "HTTP discovery does not match configured ingress");
      grpc::ClientContext context; context.set_deadline(std::chrono::system_clock::now() + 10s);
      media::mediapb::PullRequest request;
      request.mutable_open()->mutable_http()->set_url(http.url);
      request.mutable_open()->mutable_decode()->set_emit_audio(true);
      auto rpc = client->Pull(&context, request);
      media::mediapb::PullResponse response;
      std::string source, id; size_t samples = 0;
      bool info = false, closed = false;
      while (rpc->Read(&response)) {
        const auto& event = response.event();
        if (event.has_opened()) id = event.cursor().stream_id();
        if (event.has_http_info()) {
          info = event.http_info().origin() == media::HttpInput::origin(http.url) && event.http_info().status_code() == 200 &&
                 event.producer().library_version().starts_with("libcurl/");
        }
        if (event.has_source()) source += event.source().data();
        if (event.has_audio()) { http.release = true; samples += event.audio().pcm_f32le().size() / 4; }
        if (event.has_closed()) closed = event.closed().status_code() == 0 && event.closed().source_sha256() == media::sha256(data);
      }
      check(rpc->Finish().ok() && info && closed && samples == 96000 && source == data, "live HTTP-to-gRPC decode failed");
      media::mediapb::SubscribeRequest hydrate; hydrate.add_stream_ids(id); hydrate.mutable_hydration()->set_metadata(true);
      auto snapshot = registry.subscribe(hydrate);
      media::mediapb::SubscribeResponse item;
      check(snapshot->next(item, 1s) && item.has_snapshot_begin(), "HTTP snapshot begin missing");
      check(snapshot->next(item, 1s) && item.snapshot().http_info().status_code() == 200 && item.snapshot().has_closed(),
            "HTTP metadata or terminal state not hydrated");
      grpc::ClientContext failed_context; failed_context.set_deadline(std::chrono::system_clock::now() + 5s);
      request.mutable_open()->mutable_http()->set_url(missing.url);
      auto failed = client->Pull(&failed_context, request);
      bool error_info = false, error_close = false, error_body = false;
      while (failed->Read(&response)) {
        const auto& event = response.event();
        if (event.has_http_info()) error_info = event.http_info().status_code() == 404;
        if (event.has_source()) error_body = true;
        if (event.has_closed()) error_close = event.closed().status_code() == grpc::StatusCode::FAILED_PRECONDITION;
      }
      check(failed->Finish().error_code() == grpc::StatusCode::FAILED_PRECONDITION && error_info && error_close && !error_body,
            "failed HTTP response lost metadata or leaked error body into media");
    }
    std::cout << "HTTP streaming, integrity, framing, policy, errors, cancellation and live gRPC decoding tests passed\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
