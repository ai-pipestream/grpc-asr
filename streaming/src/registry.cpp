#include "registry.h"

#include <grpc/grpc.h>

#include <openssl/evp.h>
#include <openssl/rand.h>

#include <algorithm>
#include <array>
#include <cstdint>

namespace media {
namespace {
size_t charge(const google::protobuf::MessageLite& message) {
  // Include a fixed allowance for queue nodes and protobuf object bookkeeping.
  return message.ByteSizeLong() + 512;
}
void require(bool condition, grpc::StatusCode code, const char* message) {
  if (!condition) throw Failure(code, message);
}
}  // namespace

std::string uuid() {
  std::array<unsigned char, 16> data{};
  if (RAND_bytes(data.data(), data.size()) != 1) throw std::runtime_error("UUID entropy failed");
  data[6] = (data[6] & 0x0f) | 0x40;
  data[8] = (data[8] & 0x3f) | 0x80;
  const char* hex = "0123456789abcdef";
  std::string result;
  for (size_t i = 0; i < data.size(); ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) result += '-';
    result += hex[data[i] >> 4];
    result += hex[data[i] & 15];
  }
  return result;
}

bool valid_uuid(const std::string& value) {
  if (value.size() != 36) return false;
  bool nonzero = false;
  for (size_t i = 0; i < value.size(); ++i) {
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      if (value[i] != '-') return false;
    } else {
      const char c = value[i];
      if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
      nonzero |= c != '0';
    }
  }
  return nonzero;
}

std::string sha256(const std::string& bytes) {
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int length = 0;
  if (EVP_Digest(bytes.data(), bytes.size(), digest.data(), &length, EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("SHA-256 failed");
  return {reinterpret_cast<const char*>(digest.data()), length};
}

Subscription::Subscription(const mediapb::SubscribeRequest& request, size_t limit)
    : ids_(request.stream_ids().begin(), request.stream_ids().end()),
      kinds_(request.payload_kinds().begin(), request.payload_kinds().end()), limit_(limit) {}

bool Subscription::accepts(const std::string& id) const { return ids_.empty() || ids_.contains(id); }
bool Subscription::accepts(const mediapb::StreamEvent& event) const {
  if (!accepts(event.cursor().stream_id())) return false;
  if (kinds_.empty() || event.has_opened() || event.has_closed()) return true;
  return (event.has_source() && kinds_.contains(mediapb::PAYLOAD_KIND_SOURCE)) ||
         (event.has_audio() && kinds_.contains(mediapb::PAYLOAD_KIND_AUDIO)) ||
         (event.has_transcript() && kinds_.contains(mediapb::PAYLOAD_KIND_TRANSCRIPT)) ||
         (event.has_media_info() && kinds_.contains(mediapb::PAYLOAD_KIND_MEDIA_INFO)) ||
         (event.has_packet() && kinds_.contains(mediapb::PAYLOAD_KIND_PACKET)) ||
         (event.has_hypothesis() && kinds_.contains(mediapb::PAYLOAD_KIND_HYPOTHESIS)) ||
         (event.has_http_info() && kinds_.contains(mediapb::PAYLOAD_KIND_HTTP_INFO));
}
bool Subscription::push(mediapb::SubscribeResponse response) {
  std::lock_guard lock(mutex_);
  if (stopped_) return false;
  const auto size = charge(response);
  if (size > limit_ || bytes_ > limit_ - size) {
    stopped_ = true;
    status_ = {grpc::StatusCode::RESOURCE_EXHAUSTED, "subscriber queue exhausted; resume using last delivered cursor"};
    pending_.clear();
    bytes_ = 0;
    changed_.notify_all();
    return false;
  }
  bytes_ += size;
  pending_.push_back(std::move(response));
  changed_.notify_one();
  return true;
}
bool Subscription::next(mediapb::SubscribeResponse& response, std::chrono::milliseconds wait) {
  std::unique_lock lock(mutex_);
  changed_.wait_for(lock, wait, [&] { return stopped_ || !pending_.empty(); });
  if (pending_.empty()) return false;
  bytes_ -= charge(pending_.front());
  response = std::move(pending_.front());
  pending_.pop_front();
  return true;
}
void Subscription::cancel() {
  std::lock_guard lock(mutex_);
  stopped_ = true;
  pending_.clear();
  bytes_ = 0;
  changed_.notify_all();
}
bool Subscription::stopped() const { std::lock_guard lock(mutex_); return stopped_; }
grpc::Status Subscription::status() const { std::lock_guard lock(mutex_); return status_; }

Registry::Registry(Limits limits, std::string machine_id) : limits_(limits), epoch_(uuid()) {
  if (!limits.max_chunk_bytes || !limits.retention_bytes || !limits.subscriber_bytes ||
      !limits.max_streams || !limits.max_subscribers)
    throw std::invalid_argument("registry limits must be positive");
  if (!machine_id.empty() && !valid_uuid(machine_id)) throw std::invalid_argument("invalid machine UUID");
  producer_.set_machine_id(machine_id.empty() ? uuid() : machine_id);
  producer_.set_service("grpc-asr");
  producer_.set_app_version(MEDIA_APP_VERSION);
  producer_.set_build_id(MEDIA_BUILD_ID);
  producer_.set_execution_backend("cpu");
  producer_.set_library_version(std::string("grpc/") + grpc_version_string());
  producer_.set_configuration_sha256(sha256(
      "max_chunk_bytes=" + std::to_string(limits.max_chunk_bytes) +
      ";retention_bytes=" + std::to_string(limits.retention_bytes) +
      ";subscriber_bytes=" + std::to_string(limits.subscriber_bytes) +
      ";max_streams=" + std::to_string(limits.max_streams) +
      ";max_subscribers=" + std::to_string(limits.max_subscribers)));
}

mediapb::StreamEvent Registry::publish(const std::string& id, Stream& stream, mediapb::StreamEvent event) {
  auto* cursor = event.mutable_cursor();
  cursor->set_server_epoch(epoch_);
  cursor->set_stream_id(id);
  cursor->set_stream_incarnation(stream.incarnation);
  cursor->set_event_sequence(++stream.sequence);
  if (!event.has_producer()) *event.mutable_producer() = stream.producer;
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now);
  event.mutable_emitted_at()->set_seconds(seconds.count());
  event.mutable_emitted_at()->set_nanos(std::chrono::duration_cast<std::chrono::nanoseconds>(now - seconds).count());
  *stream.snapshot.mutable_through() = *cursor;
  const auto size = charge(event);
  while (!stream.history.empty() && (size > limits_.retention_bytes || stream.history_bytes > limits_.retention_bytes - size)) {
    stream.history_bytes -= charge(stream.history.front());
    stream.history.pop_front();
  }
  if (size <= limits_.retention_bytes) {
    stream.history.push_back(event);
    stream.history_bytes += size;
  }
  stream.snapshot.set_first_retained_sequence(stream.history.empty() ? stream.sequence + 1 : stream.history.front().cursor().event_sequence());
  // Keep overflowed listeners counted until their RPC releases them. A listener
  // may still be blocked in a transport write, and must not free capacity early.
  std::erase_if(subscribers_, [](const auto& weak) { return weak.expired(); });
  for (const auto& weak : subscribers_) {
    if (auto s = weak.lock(); s && s->accepts(event)) {
      mediapb::SubscribeResponse response;
      *response.mutable_event() = event;
      s->push(std::move(response));
    }
  }
  return event;
}

mediapb::StreamEvent Registry::open(const mediapb::OpenStream& request, bool speech_enabled, bool http_enabled) {
  require(request.source_case() != mediapb::OpenStream::SOURCE_NOT_SET, grpc::StatusCode::INVALID_ARGUMENT, "source is required");
  require(request.has_direct() || (http_enabled && request.has_http()), grpc::StatusCode::UNIMPLEMENTED, "HTTP input requires the Pull RPC");
  require(request.speech().empty() || speech_enabled, grpc::StatusCode::UNIMPLEMENTED, "speech worker is not connected in this build");
  require(request.decode().audio_track_ids_size() <= 64, grpc::StatusCode::RESOURCE_EXHAUSTED, "too many selected audio tracks");
  require(request.decode().audio_track_ids().empty() || request.decode().emit_audio() || !request.speech().empty(),
          grpc::StatusCode::INVALID_ARGUMENT, "audio track selection requires emit_audio");
  require(request.direct().name().size() <= 1024 && request.direct().media_type().size() <= 256,
          grpc::StatusCode::INVALID_ARGUMENT, "source description too long");
  const auto id = request.stream_id().empty() ? uuid() : request.stream_id();
  require(valid_uuid(id), grpc::StatusCode::INVALID_ARGUMENT, "stream_id must be a non-nil lowercase UUID");
  std::lock_guard lock(mutex_);
  require(!streams_.contains(id), grpc::StatusCode::ALREADY_EXISTS, "stream UUID is active or retained");
  if (streams_.size() >= limits_.max_streams) {
    auto oldest = streams_.end();
    for (auto candidate = streams_.begin(); candidate != streams_.end(); ++candidate)
      if (candidate->second.closed && (oldest == streams_.end() || candidate->second.closed_at < oldest->second.closed_at))
        oldest = candidate;
    if (oldest != streams_.end()) streams_.erase(oldest);
  }
  require(streams_.size() < limits_.max_streams, grpc::StatusCode::RESOURCE_EXHAUSTED, "registry stream limit reached");
  auto& stream = streams_[id];
  stream.incarnation = uuid();
  stream.producer = producer_;
  mediapb::OpenStream processing;
  if (request.has_decode()) *processing.mutable_decode() = request.decode();
  *processing.mutable_speech() = request.speech();
  stream.producer.set_configuration_sha256(sha256(producer_.configuration_sha256() + processing.SerializeAsString()));
  mediapb::StreamEvent event;
  auto* opened = event.mutable_opened();
  *opened->mutable_source() = request.direct();
  if (request.has_http()) opened->mutable_source()->set_name("HTTP response entity");
  opened->set_max_chunk_bytes(limits_.max_chunk_bytes);
  opened->set_retention_bytes(limits_.retention_bytes);
  if (request.has_decode()) *opened->mutable_decode() = request.decode();
  *opened->mutable_speech() = request.speech();
  *stream.snapshot.mutable_opened() = *opened;
  *stream.snapshot.add_producers() = stream.producer;
  return publish(id, stream, std::move(event));
}

mediapb::StreamEvent Registry::source(const std::string& id, const mediapb::SourceChunk& chunk) {
  std::lock_guard lock(mutex_);
  auto it = streams_.find(id);
  require(it != streams_.end(), grpc::StatusCode::NOT_FOUND, "unknown stream");
  require(!it->second.closed, grpc::StatusCode::FAILED_PRECONDITION, "stream is closed");
  mediapb::StreamEvent event;
  *event.mutable_source() = chunk;
  return publish(id, it->second, std::move(event));
}

mediapb::StreamEvent Registry::close(const std::string& id, const mediapb::StreamClosed& closed) {
  std::lock_guard lock(mutex_);
  auto it = streams_.find(id);
  require(it != streams_.end(), grpc::StatusCode::NOT_FOUND, "unknown stream");
  require(!it->second.closed, grpc::StatusCode::FAILED_PRECONDITION, "stream is already closed");
  it->second.closed = true;
  it->second.closed_at = std::chrono::steady_clock::now();
  *it->second.snapshot.mutable_closed() = closed;
  mediapb::StreamEvent event;
  *event.mutable_closed() = closed;
  return publish(id, it->second, std::move(event));
}

mediapb::StreamEvent Registry::derived(const std::string& id, mediapb::StreamEvent event, const std::string& library_version) {
  require(event.has_media_info() || event.has_audio() || event.has_packet() || event.has_hypothesis() || event.has_transcript() || event.has_http_info(),
          grpc::StatusCode::INVALID_ARGUMENT, "unsupported derived payload");
  require(event.ByteSizeLong() <= 512 * 1024, grpc::StatusCode::RESOURCE_EXHAUSTED, "derived event exceeds limit");
  std::lock_guard lock(mutex_);
  auto it = streams_.find(id);
  require(it != streams_.end(), grpc::StatusCode::NOT_FOUND, "unknown stream");
  require(!it->second.closed, grpc::StatusCode::FAILED_PRECONDITION, "stream is closed");
  if (!event.has_hypothesis() && !event.has_transcript()) {
    *event.mutable_producer() = it->second.producer;
    event.mutable_producer()->set_library_version(library_version);
  } else if (event.has_hypothesis()) {
    require(event.has_producer() && event.producer().model_sha256().size() == 32,
            grpc::StatusCode::INVALID_ARGUMENT, "speech event requires native model provenance");
    *event.mutable_hypothesis()->mutable_producer() = event.producer();
  }
  if (event.has_transcript()) {
    const auto& t = event.transcript();
    require(valid_uuid(t.segment_id()) && t.revision() > 0 && event.has_producer() &&
            t.producer().SerializeAsString() == event.producer().SerializeAsString() &&
            valid_uuid(t.producer().machine_id()) && t.producer().configuration_sha256().size() == 32 &&
            t.contributors_size() > 0 && t.contributors(0).model_sha256().size() == 32,
            grpc::StatusCode::INVALID_ARGUMENT, "transcript requires identity and processing provenance");
    for (const auto& prior : it->second.snapshot.transcripts())
      if (prior.segment_id() == t.segment_id())
        require(!prior.final() && t.revision() == prior.revision() + 1 && t.track_id() == prior.track_id(),
                grpc::StatusCode::INVALID_ARGUMENT, "invalid transcript revision");
  }
  bool found = false;
  for (const auto& existing : it->second.snapshot.producers())
    found |= existing.SerializeAsString() == event.producer().SerializeAsString();
  if (!found) {
    require(it->second.snapshot.producers_size() < 16, grpc::StatusCode::RESOURCE_EXHAUSTED, "too many producers");
    *it->second.snapshot.add_producers() = event.producer();
  }
  if (event.has_media_info()) *it->second.snapshot.mutable_media_info() = event.media_info();
  if (event.has_http_info()) *it->second.snapshot.mutable_http_info() = event.http_info();
  auto& snapshot = it->second.snapshot;
  auto* hypotheses = snapshot.mutable_hypotheses();
  auto* transcripts = snapshot.mutable_transcripts();
  if (event.has_hypothesis()) *hypotheses->Add() = event.hypothesis();
  if (event.has_transcript()) {
    for (int i = 0; i < transcripts->size(); ++i)
      if (transcripts->Get(i).segment_id() == event.transcript().segment_id()) {
        transcripts->DeleteSubrange(i, 1); break;
      }
    *transcripts->Add() = event.transcript();
  }
  while (hypotheses->size() > 16) { hypotheses->DeleteSubrange(0, 1); snapshot.set_result_state_truncated(true); }
  while (transcripts->size() > 64) { transcripts->DeleteSubrange(0, 1); snapshot.set_result_state_truncated(true); }
  while (snapshot.ByteSizeLong() > limits_.retention_bytes && (hypotheses->size() || transcripts->size())) {
    if (hypotheses->size()) hypotheses->DeleteSubrange(0, 1);
    else transcripts->DeleteSubrange(0, 1);
    snapshot.set_result_state_truncated(true);
  }
  return publish(id, it->second, std::move(event));
}

std::shared_ptr<Subscription> Registry::subscribe(const mediapb::SubscribeRequest& request) {
  require(request.stream_ids_size() <= static_cast<int>(limits_.max_streams), grpc::StatusCode::RESOURCE_EXHAUSTED, "too many selected streams");
  require(request.hydration().after_size() <= static_cast<int>(limits_.max_streams), grpc::StatusCode::RESOURCE_EXHAUSTED, "too many replay cursors");
  for (const auto& id : request.stream_ids()) require(valid_uuid(id), grpc::StatusCode::INVALID_ARGUMENT, "invalid stream UUID");
  for (int kind : request.payload_kinds()) require(kind >= mediapb::PAYLOAD_KIND_SOURCE && kind <= mediapb::PAYLOAD_KIND_HTTP_INFO, grpc::StatusCode::INVALID_ARGUMENT, "invalid payload kind");
  const auto& hydration = request.hydration();
  require(hydration.after().empty() || hydration.retained_events(), grpc::StatusCode::INVALID_ARGUMENT, "replay cursors require retained_events");
  auto sub = std::make_shared<Subscription>(request, limits_.subscriber_bytes);
  std::lock_guard lock(mutex_);
  std::erase_if(subscribers_, [](const auto& weak) { return weak.expired(); });
  require(subscribers_.size() < limits_.max_subscribers, grpc::StatusCode::RESOURCE_EXHAUSTED, "subscriber limit reached");
  std::map<std::string, uint64_t> after;
  for (const auto& cursor : hydration.after()) {
    require(cursor.server_epoch() == epoch_, grpc::StatusCode::OUT_OF_RANGE, "replay epoch unavailable");
    require(sub->accepts(cursor.stream_id()), grpc::StatusCode::INVALID_ARGUMENT, "cursor outside stream selection");
    auto it = streams_.find(cursor.stream_id());
    require(it != streams_.end(), grpc::StatusCode::OUT_OF_RANGE, "replay stream unavailable");
    require(cursor.stream_incarnation() == it->second.incarnation,
            grpc::StatusCode::OUT_OF_RANGE, "replay stream incarnation unavailable or missing");
    require(cursor.event_sequence() <= it->second.sequence, grpc::StatusCode::OUT_OF_RANGE, "cursor ahead of stream");
    require(cursor.event_sequence() >= it->second.snapshot.first_retained_sequence() - 1, grpc::StatusCode::OUT_OF_RANGE, "requested history evicted");
    require(after.emplace(cursor.stream_id(), cursor.event_sequence()).second, grpc::StatusCode::INVALID_ARGUMENT, "duplicate replay cursor");
  }
  auto enqueue = [&](mediapb::SubscribeResponse response) {
    require(sub->push(std::move(response)), grpc::StatusCode::RESOURCE_EXHAUSTED, "hydration exceeds subscriber budget; narrow selection or resume later");
  };
  if (hydration.metadata() || hydration.retained_events()) {
    const auto snapshot_id = uuid();
    mediapb::SubscribeResponse begin;
    begin.mutable_snapshot_begin()->set_snapshot_id(snapshot_id);
    for (const auto& [id, stream] : streams_) if (sub->accepts(id))
      *begin.mutable_snapshot_begin()->add_through() = stream.snapshot.through();
    enqueue(std::move(begin));
    for (const auto& [id, stream] : streams_) {
      if (!sub->accepts(id)) continue;
      if (hydration.metadata()) {
        mediapb::SubscribeResponse response;
        *response.mutable_snapshot() = stream.snapshot;
        enqueue(std::move(response));
      }
      if (hydration.retained_events()) for (const auto& event : stream.history) {
        auto found = after.find(id);
        if ((found == after.end() || event.cursor().event_sequence() > found->second) && sub->accepts(event)) {
          mediapb::SubscribeResponse response;
          *response.mutable_event() = event;
          enqueue(std::move(response));
        }
      }
    }
    mediapb::SubscribeResponse end;
    end.mutable_snapshot_end()->set_snapshot_id(snapshot_id);
    enqueue(std::move(end));
  }
  subscribers_.push_back(sub);
  return sub;
}
}  // namespace media
