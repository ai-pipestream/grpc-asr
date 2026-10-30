#include "service.h"
#include "decoder.h"

#include <openssl/evp.h>

#include <array>
#include <cstring>
#include <limits>

namespace media {
namespace {
class Digest {
 public:
  Digest() : context_(EVP_MD_CTX_new(), EVP_MD_CTX_free) {
    if (!context_ || EVP_DigestInit_ex(context_.get(), EVP_sha256(), nullptr) != 1)
      throw std::runtime_error("digest initialization failed");
  }
  void append(const std::string& data) {
    if (EVP_DigestUpdate(context_.get(), data.data(), data.size()) != 1)
      throw std::runtime_error("digest update failed");
  }
  std::string finish() {
    std::array<unsigned char, EVP_MAX_MD_SIZE> output{};
    unsigned int size = 0;
    if (EVP_DigestFinal_ex(context_.get(), output.data(), &size) != 1)
      throw std::runtime_error("digest finalization failed");
    return {reinterpret_cast<const char*>(output.data()), size};
  }
 private:
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context_;
};
}  // namespace

grpc::Status Service::GetServiceInfo(grpc::ServerContext*, const mediapb::GetServiceInfoRequest*, mediapb::GetServiceInfoResponse* info) {
  *info->mutable_producer() = registry_.producer();
  info->set_server_epoch(registry_.epoch());
  const auto& limits = registry_.limits();
  info->set_max_chunk_bytes(limits.max_chunk_bytes);
  info->set_retention_bytes(limits.retention_bytes);
  info->set_subscriber_queue_bytes(limits.subscriber_bytes);
  info->set_max_streams(limits.max_streams);
  info->set_max_subscribers(limits.max_subscribers);
  info->set_http_pull_configured(!http_.allowed_origins.empty());
  info->set_speech_processing_configured(processing_ != nullptr);
  if (processing_) {
    if (const auto advance = processing_->minimum_window_advance_ms())
      info->set_minimum_window_advance_ms(*advance);
  }
  return grpc::Status::OK;
}

grpc::Status Service::Ingest(grpc::ServerContext* context, grpc::ServerReaderWriter<mediapb::IngestResponse, mediapb::IngestRequest>* wire) {
  return process(context, [&](auto* request) { return wire->Read(request); },
                 [&](const auto& response) { return wire->Write(response); });
}

grpc::Status Service::Pull(grpc::ServerContext* context, const mediapb::PullRequest* request,
                           grpc::ServerWriter<mediapb::PullResponse>* wire) {
  try {
    if (!request->has_open() || !request->open().has_http())
      throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "Pull requires an HTTP source");
    if (!request->open().http().credential_profile().empty())
      throw Failure(grpc::StatusCode::UNIMPLEMENTED, "HTTP credential profiles are not implemented");
    HttpInput input(request->open().http().url(), http_, [&] { return context->IsCancelled(); });
    auto accepted = request->open();
    if (accepted.stream_id().empty()) accepted.set_stream_id(uuid());
    bool opened = false, described = false;
    uint64_t count = 0, bytes = 0;
    auto describe = [&] {
      if (described || input.status_code() == 0) return;
      mediapb::StreamEvent event;
      event.mutable_http_info()->set_origin(HttpInput::origin(accepted.http().url()));
      event.mutable_http_info()->set_status_code(input.status_code());
      mediapb::PullResponse response;
      *response.mutable_event() = registry_.derived(accepted.stream_id(), std::move(event), HttpInput::library_version());
      if (!wire->Write(response)) throw Failure(grpc::StatusCode::CANCELLED, "HTTP source response disconnected");
      described = true;
    };
    return process(context, [&](mediapb::IngestRequest* message) {
      message->Clear();
      if (!opened) { *message->mutable_open() = accepted; opened = true; return true; }
      std::string buffer(registry_.limits().max_chunk_bytes, '\0');
      size_t n;
      try { n = input.read(buffer.data(), buffer.size()); }
      catch (...) { describe(); throw; }
      describe();
      if (!n) {
        message->mutable_finish()->set_chunk_count(count); message->mutable_finish()->set_byte_count(bytes);
      } else {
        if (count == std::numeric_limits<uint64_t>::max() || n > std::numeric_limits<uint64_t>::max() - bytes)
          throw Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "HTTP source accounting overflow");
        buffer.resize(n);
        auto* chunk = message->mutable_chunk();
        chunk->set_chunk_number(++count); chunk->set_byte_offset(bytes); chunk->set_data(std::move(buffer)); bytes += n;
      }
      return true;
    }, [&](const mediapb::IngestResponse& response) {
      mediapb::PullResponse output; *output.mutable_event() = response.event(); return wire->Write(output);
    }, true);
  } catch (const Failure& error) { return {error.code, error.what()}; }
    catch (const std::exception&) { return {grpc::StatusCode::INTERNAL, "HTTP pull initialization failed"}; }
}

grpc::Status Service::process(grpc::ServerContext* context,
    const std::function<bool(mediapb::IngestRequest*)>& read,
    const std::function<bool(const mediapb::IngestResponse&)>& write, bool http) {
  std::string id;
  uint64_t count = 0, bytes = 0;
  grpc::Status status;
  bool explicit_finish = false;
  mediapb::FinishStream finish;
  auto send = [&](const mediapb::StreamEvent& event) {
    mediapb::IngestResponse response;
    *response.mutable_event() = event;
    if (!write(response)) throw Failure(grpc::StatusCode::CANCELLED, "ingress response disconnected");
  };
  try {
    Digest digest;
    try {
      mediapb::IngestRequest request;
      if (!read(&request) || !request.has_open())
        throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "first message must open a stream");
      auto accepted = request.open();
      if (!accepted.speech().empty()) {
        if (!processing_) throw Failure(grpc::StatusCode::UNIMPLEMENTED, "speech worker is not configured");
        processing_->resolve(accepted);
      }
      const auto opened = registry_.open(accepted, processing_ != nullptr, http);
      auto options = accepted.decode();
      const bool decoding = accepted.has_decode() || !accepted.speech().empty();
      id = opened.cursor().stream_id();
      send(opened);
      std::unique_ptr<Processing> processing;
      if (!accepted.speech().empty()) {
        processing = processing_->create(id, accepted, *context, [&](mediapb::StreamEvent event) {
          send(registry_.derived(id, std::move(event), ""));
        });
        options.set_emit_audio(true);
      }
      std::string buffered;
      size_t offset = 0;
      auto next_chunk = [&]() -> bool {
        if (explicit_finish) return false;
        if (!read(&request)) throw Failure(grpc::StatusCode::CANCELLED, "input ended without FinishStream");
        if (request.has_finish()) {
          finish = request.finish();
          explicit_finish = true;
          return false;
        }
        if (!request.has_chunk()) throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "expected a source chunk or finish");
        auto chunk = request.chunk();
        if (chunk.data().empty()) throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "empty source chunk");
        if (chunk.data().size() > registry_.limits().max_chunk_bytes)
          throw Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "source chunk exceeds negotiated limit");
        if (count == std::numeric_limits<uint64_t>::max() || chunk.chunk_number() != count + 1 || chunk.byte_offset() != bytes)
          throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "noncontiguous source chunk number or byte offset");
        if (chunk.data().size() > std::numeric_limits<uint64_t>::max() - bytes)
          throw Failure(grpc::StatusCode::RESOURCE_EXHAUSTED, "source byte counter overflow");
        const auto hash = sha256(chunk.data());
        if (!chunk.sha256().empty() && chunk.sha256() != hash)
          throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "source chunk checksum mismatch");
        chunk.set_sha256(hash);
        digest.append(chunk.data());
        ++count;
        bytes += chunk.data().size();
        send(registry_.source(id, chunk));
        buffered = std::move(*chunk.mutable_data());
        offset = 0;
        return true;
      };
      if (decoding) {
        const auto library_version = decoder_version();
        decode_media([&](uint8_t* output, int capacity) -> int {
          if (offset == buffered.size() && !next_chunk()) return 0;
          const auto size = std::min(buffered.size() - offset, static_cast<size_t>(capacity));
          std::memcpy(output, buffered.data() + offset, size);
          offset += size;
          return static_cast<int>(size);
        }, [&](mediapb::StreamEvent event) {
          if (event.has_audio() && processing) processing->audio(event.audio());
          if (!event.has_audio() || accepted.decode().emit_audio())
            send(registry_.derived(id, std::move(event), library_version));
        }, [&] { return context->IsCancelled(); }, options);
      }
      // Some finite formats stop at their declared data length. Preserve any
      // remaining original bytes and still require the explicit input trailer.
      while (next_chunk()) {}
      if (!explicit_finish) throw Failure(grpc::StatusCode::CANCELLED, "input ended without FinishStream");
      if (finish.chunk_count() != count || finish.byte_count() != bytes)
        throw Failure(grpc::StatusCode::INVALID_ARGUMENT, "finish accounting mismatch");
      if (processing) processing->finish();
    } catch (const Failure& error) {
      status = {error.code, error.what()};
    } catch (const std::exception&) {
      status = {grpc::StatusCode::INTERNAL, "ingress processing failed"};
    }
    if (id.empty()) return status;
    const auto hash = digest.finish();
    if (status.ok() && !finish.sha256().empty() && finish.sha256() != hash)
      status = {grpc::StatusCode::INVALID_ARGUMENT, "complete source checksum mismatch"};
    if (context->IsCancelled()) status = {grpc::StatusCode::CANCELLED, "ingress cancelled"};
    mediapb::StreamClosed closed;
    closed.set_status_code(status.error_code());
    closed.set_message(status.error_message());
    closed.set_source_chunk_count(count);
    closed.set_source_byte_count(bytes);
    closed.set_source_sha256(hash);
    const auto event = registry_.close(id, closed);
    // Publish terminal state for observers even if the originating client left.
    mediapb::IngestResponse response;
    *response.mutable_event() = event;
    if (!write(response) && status.ok())
      return {grpc::StatusCode::CANCELLED, "terminal response disconnected"};
    return status;
  } catch (const std::exception&) {
    return {grpc::StatusCode::INTERNAL, "ingress finalization failed"};
  }
}

grpc::Status Service::Subscribe(grpc::ServerContext* context, const mediapb::SubscribeRequest* request, grpc::ServerWriter<mediapb::SubscribeResponse>* wire) {
  try {
    auto subscription = registry_.subscribe(*request);
    mediapb::SubscribeResponse response;
    while (!context->IsCancelled()) {
      if (subscription->next(response, std::chrono::milliseconds(50))) {
        if (!wire->Write(response)) break;
      } else if (subscription->stopped()) {
        return subscription->status();
      }
    }
    subscription->cancel();
    return {grpc::StatusCode::CANCELLED, "subscription disconnected"};
  } catch (const Failure& error) {
    return {error.code, error.what()};
  } catch (const std::exception&) {
    return {grpc::StatusCode::INTERNAL, "subscription failed"};
  }
}
}  // namespace media
