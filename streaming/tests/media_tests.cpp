#include "service.h"
#include "media_fixture.h"
extern "C" {
#include <libavcodec/packet.h>
#include <libavutil/display.h>
}

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace std::chrono_literals;
namespace mediapb = media::mediapb;

void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void fails(grpc::StatusCode code, const std::function<void()>& call) {
  try { call(); } catch (const media::Failure& failure) {
    check(failure.code == code, "wrong failure status");
    return;
  }
  throw std::runtime_error("expected failure was accepted");
}
mediapb::OpenStream open_request(const std::string& id = {}) {
  mediapb::OpenStream result;
  result.set_stream_id(id);
  result.mutable_direct()->set_name("test-media");
  result.mutable_direct()->set_media_type("application/octet-stream");
  return result;
}
mediapb::SourceChunk chunk(uint64_t index, const std::string& data) {
  mediapb::SourceChunk result;
  result.set_chunk_number(index);
  result.set_byte_offset((index - 1) * data.size());
  result.set_data(data);
  return result;
}
mediapb::SubscribeResponse next(const std::shared_ptr<media::Subscription>& sub) {
  mediapb::SubscribeResponse response;
  check(sub->next(response, 1s), "missing subscription response");
  return response;
}

void uuid_selection_and_future_streams() {
  media::Registry registry;
  const auto wanted = media::uuid();
  check(media::valid_uuid(wanted), "generated UUID invalid");
  check(!media::valid_uuid("00000000-0000-0000-0000-000000000000"), "nil UUID accepted");
  mediapb::SubscribeRequest filter;
  filter.add_stream_ids(wanted);
  auto selected = registry.subscribe(filter);
  auto global = registry.subscribe({});
  const auto other = registry.open(open_request());
  check(next(global).event().cursor().stream_id() == other.cursor().stream_id(), "global missing new stream");
  mediapb::SubscribeResponse response;
  check(!selected->next(response, 0ms), "UUID filter leaked another stream");
  registry.open(open_request(wanted));
  check(next(selected).event().cursor().stream_id() == wanted, "future UUID not selected");
  check(next(global).event().cursor().stream_id() == wanted, "global missing second new stream");
  fails(grpc::StatusCode::ALREADY_EXISTS, [&] { registry.open(open_request(wanted)); });
  fails(grpc::StatusCode::INVALID_ARGUMENT, [&] { registry.open(open_request("not-a-uuid")); });
  filter.add_stream_ids("bad");
  fails(grpc::StatusCode::INVALID_ARGUMENT, [&] { registry.subscribe(filter); });
}

void snapshot_live_atomicity() {
  media::Limits limits;
  limits.retention_bytes = 8 * 1024 * 1024;
  limits.subscriber_bytes = 8 * 1024 * 1024;
  media::Registry registry(limits);
  const auto opened = registry.open(open_request());
  const auto id = opened.cursor().stream_id();
  for (int i = 1; i <= 20; ++i) registry.source(id, chunk(i, "x"));
  std::atomic<bool> start{false};
  std::jthread writer([&] {
    while (!start.load()) std::this_thread::yield();
    for (int i = 21; i <= 200; ++i) registry.source(id, chunk(i, "x"));
    registry.close(id, {});
  });
  mediapb::SubscribeRequest request;
  request.mutable_hydration()->set_metadata(true);
  request.mutable_hydration()->set_retained_events(true);
  start.store(true);
  auto sub = registry.subscribe(request);
  writer.join();
  const auto begin = next(sub);
  check(begin.has_snapshot_begin(), "snapshot did not begin");
  check(begin.snapshot_begin().through_size() == 1, "snapshot missing high water mark");
  const auto watermark = begin.snapshot_begin().through(0).event_sequence();
  const auto state = next(sub);
  check(state.has_snapshot(), "snapshot state missing");
  check(state.snapshot().through().event_sequence() == watermark, "state not at frozen boundary");
  uint64_t expected = 1;
  bool live = false;
  while (expected <= 202) {
    const auto response = next(sub);
    if (response.has_snapshot_end()) {
      check(!live, "duplicate snapshot end");
      check(response.snapshot_end().snapshot_id() == begin.snapshot_begin().snapshot_id(), "snapshot ID mismatch");
      check(expected == watermark + 1, "hydration missing events");
      live = true;
      continue;
    }
    check(response.has_event(), "unexpected payload");
    check(response.event().cursor().event_sequence() == expected++, "gap or duplicate across snapshot boundary");
    check(live ? response.event().cursor().event_sequence() > watermark : response.event().cursor().event_sequence() <= watermark,
          "event on wrong side of snapshot boundary");
  }
  // If the producer finished before registration, SnapshotEnd follows all events.
  if (!live) check(next(sub).has_snapshot_end(), "missing final snapshot boundary");
}

void replay_eviction_and_filtering() {
  media::Limits limits;
  limits.retention_bytes = 2500;
  media::Registry registry(limits);
  const auto opened = registry.open(open_request());
  const auto id = opened.cursor().stream_id();
  for (int i = 1; i <= 8; ++i) registry.source(id, chunk(i, std::string(500, 'a')));
  mediapb::SubscribeRequest request;
  request.mutable_hydration()->set_retained_events(true);
  auto* cursor = request.mutable_hydration()->add_after();
  *cursor = opened.cursor();
  fails(grpc::StatusCode::OUT_OF_RANGE, [&] { registry.subscribe(request); });
  cursor->set_event_sequence(9);
  auto sub = registry.subscribe(request);
  check(next(sub).has_snapshot_begin(), "resume missing boundary");
  check(next(sub).has_snapshot_end(), "resume replayed old events");
  registry.source(id, chunk(9, "b"));
  check(next(sub).event().cursor().event_sequence() == 10, "resume missing next event");
  cursor->set_server_epoch(media::uuid());
  fails(grpc::StatusCode::OUT_OF_RANGE, [&] { registry.subscribe(request); });
  mediapb::SubscribeRequest filtered;
  filtered.add_payload_kinds(mediapb::PAYLOAD_KIND_TRANSCRIPT);
  auto only_results = registry.subscribe(filtered);
  registry.source(id, chunk(10, "b"));
  mediapb::SubscribeResponse response;
  check(!only_results->next(response, 0ms), "payload filter leaked source bytes");
  registry.close(id, {});
  check(next(only_results).event().has_closed(), "filter hid lifecycle event");
  fails(grpc::StatusCode::FAILED_PRECONDITION, [&] { registry.source(id, chunk(11, "b")); });
}

void subscriber_backpressure_isolation() {
  media::Limits limits;
  limits.subscriber_bytes = 4000;
  media::Registry registry(limits);
  const auto id = registry.open(open_request()).cursor().stream_id();
  auto slow = registry.subscribe({});
  auto fast = registry.subscribe({});
  for (int i = 1; i <= 10; ++i) {
    registry.source(id, chunk(i, std::string(500, 'x')));
    check(next(fast).event().source().chunk_number() == static_cast<uint64_t>(i), "slow subscriber blocked fast one");
  }
  check(slow->stopped(), "slow subscriber grew without bound");
  check(slow->status().error_code() == grpc::StatusCode::RESOURCE_EXHAUSTED, "slow subscriber failed silently");
}

void closed_stream_reclamation() {
  media::Limits limits; limits.max_streams = 2;
  media::Registry registry(limits);
  const auto active = registry.open(open_request());
  const auto old = registry.open(open_request());
  const auto old_id = old.cursor().stream_id();
  auto listener = registry.subscribe({});
  mediapb::SubscribeRequest selected; selected.add_stream_ids(old_id);
  auto delayed = registry.subscribe(selected);
  registry.close(old_id, {});
  check(next(listener).event().has_closed(), "closed stream not delivered");
  fails(grpc::StatusCode::ALREADY_EXISTS, [&] {registry.open(open_request(old_id));});
  mediapb::SubscribeRequest hydration; hydration.add_stream_ids(old_id); hydration.mutable_hydration()->set_metadata(true);
  auto retained = registry.subscribe(hydration); next(retained);
  check(next(retained).snapshot().has_closed(), "closed state not retained before capacity pressure");
  for (int i=0; i<20; ++i) {
    const auto opened = registry.open(open_request());
    check(next(listener).event().cursor().SerializeAsString()==opened.cursor().SerializeAsString(), "new stream not delivered after reclamation");
    registry.close(opened.cursor().stream_id(), {}); next(listener);
  }
  const auto reopened = registry.open(open_request(old_id)); next(listener);
  check(reopened.cursor().stream_incarnation()!=old.cursor().stream_incarnation(), "reopened UUID reused incarnation");
  const auto queued_old = next(delayed).event();
  const auto queued_new = next(delayed).event();
  check(queued_old.has_closed() && queued_old.cursor().stream_incarnation()==old.cursor().stream_incarnation() &&
        queued_new.has_opened() && queued_new.cursor().stream_incarnation()==reopened.cursor().stream_incarnation(),
        "eviction changed queued identity or stopped future UUID subscription");
  mediapb::SubscribeRequest replay; replay.add_stream_ids(old_id); replay.mutable_hydration()->set_retained_events(true);
  *replay.mutable_hydration()->add_after() = old.cursor();
  fails(grpc::StatusCode::OUT_OF_RANGE, [&] {registry.subscribe(replay);});
  *replay.mutable_hydration()->mutable_after(0) = reopened.cursor();
  auto valid = registry.subscribe(replay);
  check(next(valid).has_snapshot_begin() && next(valid).has_snapshot_end(), "fresh incarnation did not resume");
  replay.mutable_hydration()->mutable_after(0)->clear_stream_incarnation();
  fails(grpc::StatusCode::OUT_OF_RANGE, [&] {registry.subscribe(replay);});
  registry.source(active.cursor().stream_id(), chunk(1,"alive"));
  check(next(listener).event().source().data()=="alive", "reclamation removed active stream");
  fails(grpc::StatusCode::RESOURCE_EXHAUSTED, [&] {registry.open(open_request());});
}

void transcript_revisions() {
  media::Limits limits; limits.retention_bytes = 4096;
  media::Registry registry(limits);
  const auto id = registry.open(open_request()).cursor().stream_id();
  const auto segment_id = media::uuid();
  mediapb::StreamEvent event;
  auto* producer = event.mutable_producer(); producer->set_machine_id(media::uuid());
  producer->set_service("test-reconciler"); producer->set_configuration_sha256(std::string(32, 'c'));
  auto* t = event.mutable_transcript(); t->set_segment_id(segment_id); t->set_revision(1); t->set_text("first");
  *t->mutable_producer() = *producer;
  t->add_contributors()->set_model_sha256(std::string(32, 'm'));
  registry.derived(id, event, "");
  t->set_revision(3);
  fails(grpc::StatusCode::INVALID_ARGUMENT, [&] {registry.derived(id, event, "");});
  t->set_revision(2); t->set_final(true); t->set_text("final"); registry.derived(id, event, "");
  t->set_revision(3);
  fails(grpc::StatusCode::INVALID_ARGUMENT, [&] {registry.derived(id, event, "");});
  mediapb::SubscribeRequest request; request.add_stream_ids(id); request.mutable_hydration()->set_metadata(true);
  auto sub = registry.subscribe(request); next(sub);
  const auto snapshot = next(sub).snapshot();
  check(snapshot.transcripts_size()==1 && snapshot.transcripts(0).revision()==2 && snapshot.transcripts(0).text()=="final",
        "hydration did not retain latest final transcript");
  for (int i=0; i<30; ++i) {
    t->set_segment_id(media::uuid()); t->set_revision(1); t->set_text(std::string(600,'x'));
    registry.derived(id, event, "");
  }
  auto bounded = registry.subscribe(request); next(bounded);
  const auto state = next(bounded).snapshot();
  check(state.result_state_truncated() && state.transcripts_size()>0 && state.transcripts_size()<30 &&
        state.ByteSizeLong()<=limits.retention_bytes, "transcript retention exceeded byte budget");
}

void bounded_registry_and_capabilities() {
  {
    media::Limits bounded;
    bounded.retention_bytes = 2048;
    media::Registry results(bounded);
    const auto id = results.open(open_request()).cursor().stream_id();
    mediapb::SubscribeRequest filter;
    filter.add_payload_kinds(mediapb::PAYLOAD_KIND_HYPOTHESIS);
    auto live = results.subscribe(filter);
    for (uint64_t i = 1; i <= 10; ++i) {
      mediapb::StreamEvent event;
      event.mutable_producer()->set_model_id("test-model");
      event.mutable_producer()->set_model_sha256(std::string(32, 'm'));
      event.mutable_hypothesis()->set_window_number(i);
      event.mutable_hypothesis()->add_segments()->set_text(std::string(300, 'x'));
      results.derived(id, std::move(event), "");
      check(next(live).event().hypothesis().window_number() == i, "hypothesis subscription filter lost output");
    }
    mediapb::SubscribeRequest request;
    request.add_stream_ids(id); request.mutable_hydration()->set_metadata(true);
    auto replay = results.subscribe(request);
    check(next(replay).has_snapshot_begin(), "hypothesis snapshot begin absent");
    const auto snapshot = next(replay).snapshot();
    check(snapshot.result_state_truncated() && snapshot.hypotheses_size() > 0 && snapshot.hypotheses_size() < 10,
          "hypothesis hydration was not bounded");
    check(snapshot.ByteSizeLong() <= bounded.retention_bytes, "hypothesis snapshot exceeded retention budget");
    check(snapshot.hypotheses(snapshot.hypotheses_size() - 1).producer().model_sha256() == std::string(32, 'm'),
          "hypothesis hydration lost native provenance");
  }
  // Provenance identifies processing settings, independently of stream identity.
  media::Registry provenance;
  const auto relay = provenance.open(open_request());
  const auto another_relay = provenance.open(open_request());
  auto decode_request = open_request();
  decode_request.mutable_decode()->set_emit_audio(true);
  const auto decoded = provenance.open(decode_request);
  check(relay.producer().configuration_sha256() == another_relay.producer().configuration_sha256(),
        "stream identity changed processing digest");
  check(relay.producer().configuration_sha256() != decoded.producer().configuration_sha256(),
        "decode options absent from processing digest");
  mediapb::StreamEvent audio;
  audio.mutable_audio();
  const auto derived = provenance.derived(decoded.cursor().stream_id(), std::move(audio), "test-decoder/1");
  check(derived.producer().configuration_sha256() == decoded.producer().configuration_sha256(),
        "derived event lost processing digest");
  media::Limits limits;
  limits.max_subscribers = 1;
  limits.max_streams = 1;
  media::Registry registry(limits);
  mediapb::OpenStream http;
  http.mutable_http()->set_url("http://127.0.0.1/example");
  fails(grpc::StatusCode::UNIMPLEMENTED, [&] { registry.open(http); });
  auto speech = open_request();
  speech.add_speech()->set_model_id("tiny");
  fails(grpc::StatusCode::UNIMPLEMENTED, [&] { registry.open(speech); });
  fails(grpc::StatusCode::INVALID_ARGUMENT, [&] { registry.open({}); });
  const auto opened = registry.open(open_request());
  check(opened.producer().build_id().size() == 64, "missing build identity");
  check(opened.producer().configuration_sha256().size() == 32, "missing configuration identity");
  fails(grpc::StatusCode::RESOURCE_EXHAUSTED, [&] { registry.open(open_request()); });
  auto sub = registry.subscribe({});
  fails(grpc::StatusCode::RESOURCE_EXHAUSTED, [&] { registry.subscribe({}); });
  sub->cancel();
  // Stopped does not free a slot while the RPC still owns the mailbox.
  fails(grpc::StatusCode::RESOURCE_EXHAUSTED, [&] { registry.subscribe({}); });
  sub.reset();
  sub = registry.subscribe({});
  check(!sub->stopped(), "released subscription slot not reusable");
}

class RunningServer {
 public:
  media::Registry registry;
  media::Service service{registry};
  std::unique_ptr<grpc::Server> server;
  std::unique_ptr<mediapb::MediaService::Stub> client;
  RunningServer() {
    if (const char* target = std::getenv("MEDIA_TEST_TARGET"); target && *target) {
      auto channel = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
      check(channel->WaitForConnected(std::chrono::system_clock::now() + 5s), "external test server unavailable");
      client = mediapb::MediaService::NewStub(channel);
      return;
    }
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    server = builder.BuildAndStart();
    check(server != nullptr && port != 0, "test server failed to bind");
    client = mediapb::MediaService::NewStub(grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
  }
  ~RunningServer() { if (server) server->Shutdown(std::chrono::system_clock::now() + 1s); }
};

void subscriber_disconnect_isolation() {
  RunningServer running;
  grpc::ClientContext input_context, cancelled_context, observer_context;
  for (auto* context : {&input_context, &cancelled_context, &observer_context})
    context->set_deadline(std::chrono::system_clock::now()+5s);
  auto input = running.client->Ingest(&input_context);
  mediapb::IngestRequest request; *request.mutable_open() = open_request();
  check(input->Write(request), "isolation input open failed");
  mediapb::IngestResponse response;
  check(input->Read(&response) && response.event().has_opened(), "isolation handshake missing");
  mediapb::SubscribeRequest selection; selection.add_stream_ids(response.event().cursor().stream_id());
  selection.mutable_hydration()->set_metadata(true);
  auto cancelled = running.client->Subscribe(&cancelled_context, selection);
  auto observer = running.client->Subscribe(&observer_context, selection);
  mediapb::SubscribeResponse item;
  for (auto* listener : {cancelled.get(), observer.get()}) {
    check(listener->Read(&item) && item.has_snapshot_begin(), "isolation snapshot begin missing");
    check(listener->Read(&item) && item.has_snapshot(), "isolation snapshot missing");
    check(listener->Read(&item) && item.has_snapshot_end(), "isolation snapshot end missing");
  }
  cancelled_context.TryCancel(); while (cancelled->Read(&item)) {}
  check(cancelled->Finish().error_code()==grpc::StatusCode::CANCELLED, "listener cancellation failed");
  request.Clear(); *request.mutable_chunk() = chunk(1, "still flowing");
  check(input->Write(request), "cancelled listener stopped input");
  check(input->Read(&response) && response.event().has_source(), "input response stopped after listener cancellation");
  check(observer->Read(&item) && item.event().source().data()=="still flowing", "other listener lost data");
  input_context.TryCancel(); while (input->Read(&response)) {}
  check(input->Finish().error_code()==grpc::StatusCode::CANCELLED, "input cancellation ignored");
  check(observer->Read(&item) && item.event().has_closed() &&
        item.event().closed().status_code()==grpc::StatusCode::CANCELLED &&
        item.event().closed().source_sha256()==media::sha256("still flowing"), "observer lost cancelled input accounting");
  observer_context.TryCancel(); while (observer->Read(&item)) {} observer->Finish();
}

void wire_streaming_and_integrity() {
  RunningServer running;
  grpc::ClientContext info_context;
  info_context.set_deadline(std::chrono::system_clock::now() + 5s);
  mediapb::GetServiceInfoResponse info;
  check(running.client->GetServiceInfo(&info_context, {}, &info).ok(), "ingress discovery failed");
  check(media::valid_uuid(info.producer().machine_id()) && media::valid_uuid(info.server_epoch()) &&
        !info.producer().app_version().empty() && !info.producer().build_id().empty() &&
        info.max_chunk_bytes() > 0 && info.retention_bytes() > 0 && info.subscriber_queue_bytes() > 0 &&
        info.max_streams() > 0 && info.max_subscribers() > 0 && !info.has_ui(), "incomplete ingress discovery");
  if (running.server)
    check(!info.http_pull_configured() && !info.speech_processing_configured(), "unconfigured feature advertised");
  grpc::ClientContext context;
  context.set_deadline(std::chrono::system_clock::now() + 5s);
  auto stream = running.client->Ingest(&context);
  mediapb::IngestRequest request;
  *request.mutable_open() = open_request();
  check(stream->Write(request), "open write failed");
  mediapb::IngestResponse response;
  check(stream->Read(&response) && response.event().has_opened(), "missing handshake before input");
  check(response.event().producer().machine_id()==info.producer().machine_id() &&
        response.event().cursor().server_epoch()==info.server_epoch() &&
        response.event().opened().max_chunk_bytes()==info.max_chunk_bytes() &&
        response.event().opened().retention_bytes()==info.retention_bytes(), "discovery differs from handshake");
  const auto id = response.event().cursor().stream_id();
  const std::string data("\x00\xff\x01media", 8);
  request.Clear();
  *request.mutable_chunk() = chunk(1, data);
  check(stream->Write(request), "chunk write failed");
  // Do not finish until the response arrives: catches buffering-until-EOF.
  check(stream->Read(&response) && response.event().has_source(), "source did not stream before finish");
  check(response.event().source().data() == data, "source bytes changed");
  check(response.event().source().sha256() == media::sha256(data), "chunk checksum wrong");
  request.Clear();
  request.mutable_finish()->set_chunk_count(1);
  request.mutable_finish()->set_byte_count(data.size());
  request.mutable_finish()->set_sha256(media::sha256(data));
  check(stream->Write(request), "finish write failed");
  stream->WritesDone();
  check(stream->Read(&response) && response.event().has_closed(), "missing terminal event");
  check(response.event().closed().status_code() == 0, "successful stream reported failure");
  check(response.event().closed().source_sha256() == media::sha256(data), "full digest wrong");
  check(!stream->Read(&response), "events after terminal");
  check(stream->Finish().ok(), "successful ingest failed");

  grpc::ClientContext subscriber_context;
  subscriber_context.set_deadline(std::chrono::system_clock::now() + 5s);
  mediapb::SubscribeRequest subscribe;
  subscribe.add_stream_ids(id);
  subscribe.mutable_hydration()->set_metadata(true);
  auto listener = running.client->Subscribe(&subscriber_context, subscribe);
  mediapb::SubscribeResponse hydrated;
  check(listener->Read(&hydrated) && hydrated.has_snapshot_begin(), "wire hydration missing begin");
  check(listener->Read(&hydrated) && hydrated.has_snapshot(), "wire hydration missing state");
  check(hydrated.snapshot().has_closed(), "closed stream not hydratable");
  check(hydrated.snapshot().closed().source_byte_count() == data.size(), "hydrated accounting wrong");
  check(listener->Read(&hydrated) && hydrated.has_snapshot_end(), "wire hydration missing end");
  subscriber_context.TryCancel();
  while (listener->Read(&hydrated)) {}
  check(listener->Finish().error_code() == grpc::StatusCode::CANCELLED, "subscription cancellation failed");
}

void malformed_wire_input() {
  RunningServer running;
  for (int scenario = 0; scenario < 6; ++scenario) {
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 5s);
    auto stream = running.client->Ingest(&context);
    mediapb::IngestRequest request;
    *request.mutable_open() = open_request();
    check(stream->Write(request), "malformed test open failed");
    mediapb::IngestResponse response;
    check(stream->Read(&response) && response.event().has_opened(), "malformed test handshake failed");
    request.Clear();
    if (scenario == 0) {
      *request.mutable_chunk() = chunk(2, "bad-order");
    } else if (scenario == 1) {
      *request.mutable_chunk() = chunk(1, "bad-hash");
      request.mutable_chunk()->set_sha256("wrong");
    } else if (scenario == 2) {
      request.mutable_finish()->set_byte_count(1);
    } else if (scenario == 3) {
      request.mutable_finish()->set_sha256(std::string(32, 'x'));
    } else if (scenario == 4) {
      *request.mutable_open() = open_request();
    }
    if (scenario != 5) check(stream->Write(request), "malformed test write failed");
    stream->WritesDone();
    check(stream->Read(&response) && response.event().has_closed(), "failure omitted terminal event");
    check(response.event().closed().source_byte_count() == 0, "invalid bytes were accepted");
    check(!stream->Read(&response), "failure produced events after close");
    const auto status = stream->Finish();
    check(status.error_code() == (scenario == 5 ? grpc::StatusCode::CANCELLED : grpc::StatusCode::INVALID_ARGUMENT), "wrong wire failure status");
  }
}

void wire_live_decoding() {
  RunningServer running;
  int32_t matrix[9]; av_display_rotation_set(matrix,90);
  const std::string rotation(reinterpret_cast<const char*>(matrix),sizeof matrix);
  // FFmpeg probes 64 KiB of PCM WAV for embedded SPDIF during header parsing.
  // Use six seconds so the first half includes that probe, while holding back
  // three full seconds until actual PCM arrives. NUT has no such startup probe.
  for (const auto& fixture : {std::pair{wav_fixture(16000, 1, 6), uint64_t{96000}},
                              std::pair{nut_fixture(), uint64_t{48000}},
                              std::pair{matroska_fixture().bytes, uint64_t{48000}},
                              std::pair{fragmented_mp4_fixture({},90), uint64_t{0}}}) {
    const auto& data = fixture.first;
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + 5s);
    auto stream = running.client->Ingest(&context);
    mediapb::IngestRequest request;
    *request.mutable_open() = open_request();
    request.mutable_open()->mutable_decode()->set_emit_audio(true);
    request.mutable_open()->mutable_decode()->set_emit_packets(true);
    check(stream->Write(request), "decode open failed");
    mediapb::IngestResponse response;
    check(stream->Read(&response) && response.event().has_opened(), "decode handshake absent");
    check(response.event().opened().decode().emit_audio(), "decode options not acknowledged");
    const auto id = response.event().cursor().stream_id();
    const size_t split = data.size() / 2;
    request.Clear();
    *request.mutable_chunk() = chunk(1, data.substr(0, split));
    check(stream->Write(request), "decode first upload failed");
    uint64_t audio_samples = 0;
    std::string reconstructed;
    bool saw_audio = false, saw_metadata = false, saw_packet = false, closed = false;
    bool saw_rotation = false;
    std::string latest_metadata;
    auto observe = [&](const mediapb::StreamEvent& event) {
      if (event.has_source()) reconstructed += event.source().data();
      if (event.has_audio()) {
        saw_audio = true;
        check(event.audio().first_sample() == audio_samples, "wire audio sample sequence gap");
        audio_samples += event.audio().pcm_f32le().size() / 4;
        check(event.producer().library_version().find("libavcodec/") != std::string::npos, "decoder provenance absent");
      }
      saw_metadata |= event.has_media_info();
      if (event.has_media_info()) {
        latest_metadata=event.media_info().SerializeAsString();
        for (const auto& track:event.media_info().tracks())
          for (const auto& side:track.side_data()) if(side.native_type()==AV_PKT_DATA_DISPLAYMATRIX) {
            check(side.data()==rotation,"wire display matrix changed bytes");
            saw_rotation=true;
          }
      }
      saw_packet |= event.has_packet();
      if (event.has_closed()) {
        closed = true;
        check(event.closed().status_code() == 0, "decode terminal reported failure");
        check(event.closed().source_sha256() == media::sha256(data), "decode lost original source hash");
      }
    };
    // Hold the second half until decoded audio is observed, not just byte echo.
    while (!saw_audio && stream->Read(&response)) observe(response.event());
    check(saw_audio, "decoded audio did not arrive during upload");
    request.Clear();
    *request.mutable_chunk() = chunk(2, data.substr(split));
    request.mutable_chunk()->set_byte_offset(split);
    check(stream->Write(request), "decode second upload failed");
    request.Clear();
    request.mutable_finish()->set_chunk_count(2);
    request.mutable_finish()->set_byte_count(data.size());
    check(stream->Write(request), "decode finish failed");
    stream->WritesDone();
    while (stream->Read(&response)) observe(response.event());
    check(stream->Finish().ok(), "wire decode failed");
    // AAC adds bounded frame padding/encoder delay; PCM fixtures remain exact.
    const bool duration_ok = fixture.second ? audio_samples == fixture.second :
        audio_samples >= 96000 && audio_samples <= 97024;
    check(closed && saw_metadata && saw_packet && duration_ok, "wire decode missing output");
    check(reconstructed == data, "decoded path changed original source bytes");
    if (!fixture.second) check(saw_rotation,"wire track rotation missing");

    grpc::ClientContext hydration_context;
    hydration_context.set_deadline(std::chrono::system_clock::now() + 5s);
    mediapb::SubscribeRequest subscription;
    subscription.add_stream_ids(id);
    subscription.mutable_hydration()->set_metadata(true);
    auto listener = running.client->Subscribe(&hydration_context, subscription);
    mediapb::SubscribeResponse item;
    check(listener->Read(&item) && item.has_snapshot_begin(), "media hydration missing begin");
    check(listener->Read(&item) && item.has_snapshot(), "media hydration missing state");
    check(item.snapshot().has_media_info() && item.snapshot().producers_size() == 2, "media hydration missing metadata or producer");
    check(item.snapshot().media_info().SerializeAsString()==latest_metadata,"hydration changed native track metadata");
    hydration_context.TryCancel();
    while (listener->Read(&item)) {}
    check(listener->Finish().error_code() == grpc::StatusCode::CANCELLED, "media hydration cancel failed");
  }
}

int main() {
  try {
    const auto cases = {uuid_selection_and_future_streams, snapshot_live_atomicity,
                        transcript_revisions, closed_stream_reclamation,
                        replay_eviction_and_filtering, subscriber_backpressure_isolation,
                        bounded_registry_and_capabilities, subscriber_disconnect_isolation, wire_streaming_and_integrity,
                        malformed_wire_input, wire_live_decoding};
    for (auto test : cases) test();
    std::cout << cases.size() << " media protocol tests passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
